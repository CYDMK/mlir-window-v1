#!/usr/bin/env python3
"""
Generate TPU-MLIR model_transform and model_deploy commands
for PaddleOCR DET and REC ONNX models.
This edited version runs ONNX -> MLIR -> BMODEL directly from the script.

Features:
- Interactive path selection: manual / GUI browser / directory search
- Quantize: F32, F16, BF16, INT8, INT4
- Configurable input shape (default [1, 3, 640, 640])
- Results saved to results/<run_id>_<model>.txt
- Cross-platform: Windows, macOS, Linux
- Auto start Docker Desktop on Windows if Docker daemon is not running
- Auto create/start Docker container if missing/stopped
"""

from __future__ import annotations

import ast
import dataclasses
import datetime
import io
import os
import platform
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Optional

import yaml
import onnx
from onnx import TensorProto, helper

# Optional GUI (tkinter)
_TKINTER_AVAILABLE = False
try:
    import tkinter as _tk
    from tkinter import filedialog as _filedialog
    from tkinter import ttk as _ttk
    from tkinter import messagebox as _messagebox
    # Verify it actually works (headless Linux will fail here)
    _root_test = _tk.Tk()
    _root_test.withdraw()
    _root_test.destroy()
    _TKINTER_AVAILABLE = True
except Exception:
    pass


# ============================================================
# CONSTANTS
# ============================================================

QUANTIZE_OPTIONS = ["F32", "F16", "BF16", "INT8", "INT4"]
QUANTIZE_NEEDS_CALIB: set[str] = {"INT8", "INT4"}
PROCESSOR_OPTIONS = ["bm1684x", "bm1688", "cv186x"]
DEFAULT_INPUT_SHAPE = [1, 3, 640, 640]
DEFAULT_CALIB_INPUT_NUM = 100
DOCKER_CONTAINER_NAME = "model_conversion"
DOCKER_IMAGE = "sophgo/tpuc_dev:latest"
DOCKER_WORKDIR = "/workspace"
DOCKER_START_TIMEOUT_SEC = 180


@dataclasses.dataclass
class ModelConfig:
    model_type: str
    config_path: Optional[Path]
    onnx_path: Path
    input_shape: list
    quantize: str
    processor: str
    keep_aspect_ratio: bool
    output_names: Optional[list]
    calib_mode: str          # "none" | "existing" | "run"
    calibration_table: Optional[str]
    calib_dataset: Optional[Path]
    calib_input_num: int
    yolo_pt_path: Optional[Path] = None


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_number(value):
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        raise ValueError(f"Cannot parse number from: {value!r}")
    text = value.strip()
    allowed_chars = set("0123456789.+-*/() ")
    if not set(text) <= allowed_chars:
        raise ValueError(f"Unsafe numeric expression in YAML: {text!r}")
    node = ast.parse(text, mode="eval")
    return float(eval(compile(node, "<safe_number>", "eval"), {"__builtins__": {}}, {}))


def fmt_float(x):
    return f"{x:.10g}"


def comma_list(values):
    return ",".join(fmt_float(x) for x in values)


def shape_to_tpu_mlir(shape):
    return "[[" + ",".join(str(x) for x in shape) + "]]"


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dtype_to_str(dtype):
    return TensorProto.DataType.Name(dtype)


# ============================================================
# ONNX HELPERS
# ============================================================

def get_value_info_shape(value_info):
    shape = []
    tensor_type = value_info.type.tensor_type
    for dim in tensor_type.shape.dim:
        if dim.dim_value > 0:
            shape.append(dim.dim_value)
        elif dim.dim_param:
            shape.append(str(dim.dim_param))
        else:
            shape.append("?")
    return shape


def is_dynamic_shape(shape):
    return any(not isinstance(x, int) for x in shape)


def get_real_model_inputs(model):
    initializer_names = {x.name for x in model.graph.initializer}
    return [x for x in model.graph.input if x.name not in initializer_names]


def get_onnx_info(onnx_path):
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    real_inputs = get_real_model_inputs(model)
    if not real_inputs:
        raise RuntimeError("No real runtime inputs found in ONNX model.")
    if len(real_inputs) > 1:
        print("WARNING: More than one real ONNX input found.", file=sys.stderr)
        print("The script will use the first input only.", file=sys.stderr)
        for inp in real_inputs:
            print(f"  input: {inp.name}", file=sys.stderr)
    input_info = real_inputs[0]
    input_tensor_type = input_info.type.tensor_type
    outputs = []
    for out in model.graph.output:
        tensor_type = out.type.tensor_type
        outputs.append({
            "name": out.name,
            "dtype": dtype_to_str(tensor_type.elem_type),
            "shape": get_value_info_shape(out),
        })
    return {
        "input_name": input_info.name,
        "input_dtype": dtype_to_str(input_tensor_type.elem_type),
        "input_shape": get_value_info_shape(input_info),
        "output_names": [x["name"] for x in outputs],
        "outputs": outputs,
    }


# ============================================================
# YAML TRANSFORM HELPERS
# ============================================================

def find_transform(config, section_name, transform_name):
    section = config.get(section_name, {})
    dataset = section.get("dataset", {})
    transforms = dataset.get("transforms", [])
    for item in transforms:
        if isinstance(item, dict) and transform_name in item:
            return item[transform_name]
    return None


def find_transform_prefer_eval(config, transform_name):
    result = find_transform(config, "Eval", transform_name)
    if result is not None:
        return result
    return find_transform(config, "Train", transform_name)


def get_decode_img_mode(config):
    decode_cfg = find_transform_prefer_eval(config, "DecodeImage")
    if decode_cfg is None:
        raise RuntimeError("DecodeImage not found in Eval/Train transforms.")
    img_mode = decode_cfg.get("img_mode", "BGR").lower()
    if img_mode not in {"bgr", "rgb", "gray"}:
        raise RuntimeError(f"Unsupported img_mode: {img_mode}")
    return img_mode


# ============================================================
# DET PREPROCESS
# ============================================================

def get_det_preprocess(config):
    img_mode = get_decode_img_mode(config)
    norm_cfg = find_transform_prefer_eval(config, "NormalizeImage")
    to_chw_cfg = find_transform_prefer_eval(config, "ToCHWImage")
    if norm_cfg is None:
        raise RuntimeError("DET config requires NormalizeImage but it was not found.")
    norm_scale = safe_number(norm_cfg.get("scale", 1.0))
    mean = [safe_number(x) for x in norm_cfg.get("mean", [])]
    std = [safe_number(x) for x in norm_cfg.get("std", [])]
    if len(mean) != len(std):
        raise RuntimeError(f"mean/std length mismatch: mean={mean}, std={std}")
    if len(mean) not in {1, 3}:
        raise RuntimeError(f"Expected 1 or 3 channels, got mean={mean}, std={std}")
    tpu_mean = [m / norm_scale for m in mean]
    tpu_scale = [norm_scale / s for s in std]
    channel_format = "nchw" if to_chw_cfg is not None else "nhwc"
    return {
        "pixel_format": img_mode,
        "channel_format": channel_format,
        "mean": tpu_mean,
        "scale": tpu_scale,
    }


# ============================================================
# REC PREPROCESS
# ============================================================

def get_rec_image_shape(config):
    rec_resize_cfg = find_transform_prefer_eval(config, "RecResizeImg")
    if rec_resize_cfg is not None:
        image_shape = rec_resize_cfg.get("image_shape")
        if image_shape:
            return [int(x) for x in image_shape]
    global_shape = config.get("Global", {}).get("d2s_train_image_shape")
    if global_shape:
        return [int(x) for x in global_shape]
    raise RuntimeError(
        "REC config requires RecResizeImg.image_shape or Global.d2s_train_image_shape."
    )


def get_rec_preprocess(config):
    img_mode = get_decode_img_mode(config)
    image_shape = get_rec_image_shape(config)
    c = int(image_shape[0])
    tpu_mean = [127.5] * c
    tpu_scale = [1.0 / 127.5] * c
    return {
        "pixel_format": img_mode,
        "channel_format": "nchw",
        "mean": tpu_mean,
        "scale": tpu_scale,
        "image_shape": image_shape,
    }


def get_yolo_preprocess():
    """Default YOLO preprocessing for exported YOLOv8 ONNX models.

    HBB and OBB use the same preprocessing for conversion:
    RGB image, NCHW, scale 1/255, no mean subtraction.
    """
    return {
        "pixel_format": "rgb",
        "channel_format": "nchw",
        "mean": [0.0, 0.0, 0.0],
        "scale": [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0],
    }

def is_yolo_model(model_type: str) -> bool:
    return model_type in {"yolo_hbb", "yolo_obb"}


def is_pt_file(path: Path) -> bool:
    return path.suffix.lower() == ".pt"

def is_onnx_file(path: Path) -> bool:
    return path.suffix.lower() == ".onnx"

def export_yolo_pt_to_onnx(
    pt_path: Path,
    imgsz: int,
    out_dir: Path,
    log_file: Path,
    model_type: str = "",
) -> Path:
    """Export YOLO .pt to ONNX using ultralytics in the current Python env.

    For YOLO OBB only, add a Transpose node at the output:
        1 x 6 x 8400  ->  1 x 8400 x 6

    YOLO HBB will keep the original output shape.
    """
    pt_path = pt_path.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    expected_onnx = pt_path.with_suffix(".onnx")

    code = (
        "from ultralytics import YOLO\n"
        "model = YOLO(r\'\'\'{}\'\'\')\n".format(str(pt_path)) +
        "model.export(format='onnx', imgsz={}, opset=12, simplify=False, dynamic=False)\n".format(int(imgsz))
    )
    cmd = [sys.executable, "-c", code]

    print("\n  Exporting YOLO PT -> ONNX ...")
    print(f"  PT   : {pt_path}")
    print(f"  ONNX : {expected_onnx}")

    t_start = datetime.datetime.now()
    with log_file.open("w", encoding="utf-8") as lf:
        lf.write("$ " + " ".join(cmd) + "\n")
        lf.write(f"started: {t_start.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        kwargs = dict(
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(pt_path.parent),
        )

        if platform.system() == "Windows":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        proc = subprocess.Popen(cmd, **kwargs)

        for line in proc.stdout:
            print(line, end="", flush=True)
            lf.write(line)

        proc.wait()
        elapsed = (datetime.datetime.now() - t_start).total_seconds()
        lf.write(f"\n[exit code: {proc.returncode}] [elapsed: {elapsed:.1f}s]\n")

    if proc.returncode != 0:
        raise RuntimeError(f"YOLO export failed. See log: {log_file}")

    if not expected_onnx.is_file():
        candidates = sorted(
            pt_path.parent.glob("*.onnx"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )

        if candidates:
            expected_onnx = candidates[0]
        else:
            raise RuntimeError(f"YOLO export finished but ONNX not found: {expected_onnx}")

    # Save ONNX copy to this run output folder
    saved_onnx = out_dir / expected_onnx.name

    if expected_onnx.resolve() != saved_onnx.resolve():
        shutil.copy2(str(expected_onnx), str(saved_onnx))
    else:
        print("  ONNX already exists in output folder, skip copy.")

    print(f"  [OK] Saved ONNX copy: {saved_onnx}")

    # Add transpose only for YOLO OBB
    is_obb = (
        model_type == "yolo_obb"
        or "yolo_obb" in str(out_dir).lower()
        or "obb" in pt_path.stem.lower()
    )

    if is_obb:
        print("\n  Adding output transpose for YOLO OBB...")
        print("  Target shape: 1 x 8400 x 6")

        model = onnx.load(str(saved_onnx))

        if not model.graph.output:
            raise RuntimeError("ONNX model has no graph output to transpose.")

        old_output_name = model.graph.output[0].name
        new_output_name = "output_transpose"

        transpose_node = helper.make_node(
            "Transpose",
            inputs=[old_output_name],
            outputs=[new_output_name],
            perm=[0, 2, 1],
            name="Transpose_YOLO_OBB_Output",
        )

        model.graph.node.append(transpose_node)
        model.graph.output[0].name = new_output_name

        out_shape = model.graph.output[0].type.tensor_type.shape
        del out_shape.dim[:]
        for dim_value in [1, 8400, 6]:
            dim = out_shape.dim.add()
            dim.dim_value = dim_value

        onnx.save(model, str(saved_onnx))

        print("  [OK] YOLO OBB output transposed")
        print("  Output should be:")
        print("      output_transpose : 1 x 8400 x 6")
    else:
        print("\n  [SKIP] Output transpose skipped")
        print("  Reason: model is not YOLO OBB")

    return saved_onnx.resolve()


# ============================================================
# COMMAND PRINTERS
# ============================================================

def print_model_info(model_type, config_path, onnx_path, onnx_info, input_shape, preprocess, file=None):
    f = file or sys.stdout
    print(f"# ===== {model_type.upper()} INFO =====", file=f)
    print(f"# config         : {config_path}", file=f)
    print(f"# onnx           : {onnx_path}", file=f)
    print(f"# input name     : {onnx_info['input_name']}", file=f)
    print(f"# input dtype    : {onnx_info['input_dtype']}", file=f)
    print(f"# onnx shape     : {onnx_info['input_shape']}", file=f)
    print(f"# used shape     : {input_shape}", file=f)
    print(f"# pixel_format   : {preprocess['pixel_format']}", file=f)
    print(f"# channel_format : {preprocess['channel_format']}", file=f)
    print(f"# mean           : {comma_list(preprocess['mean'])}", file=f)
    print(f"# scale          : {comma_list(preprocess['scale'])}", file=f)
    for i, out in enumerate(onnx_info["outputs"]):
        print(f"# output {i} name : {out['name']}", file=f)
        print(f"# output {i} dtype: {out['dtype']}", file=f)
        print(f"# output {i} shape: {out['shape']}", file=f)
    print(file=f)


def print_model_transform_command(
    model_name, onnx_path, input_shape, preprocess,
    keep_aspect_ratio, output_names, test_input, test_result, mlir_path, file=None,
):
    f = file or sys.stdout
    parts = [
        ("kv", "--model_name", model_name),
        ("kv", "--model_def", onnx_path),
        ("kv", "--input_shapes", shape_to_tpu_mlir(input_shape)),
        ("kv", "--mean", comma_list(preprocess["mean"])),
        ("kv", "--scale", comma_list(preprocess["scale"])),
    ]
    if keep_aspect_ratio:
        parts.append(("flag", "--keep_aspect_ratio", None))
    parts.append(("kv", "--pixel_format", preprocess["pixel_format"]))
    if output_names:
        parts.append(("kv", "--output_names", ",".join(output_names)))
    if test_input:
        parts.append(("kv", "--test_input", test_input))
    if test_result:
        parts.append(("kv", "--test_result", test_result))
    parts.append(("kv", "--mlir", mlir_path))

    print("# ----- model_transform -----", file=f)
    print("model_transform \\", file=f)
    for idx, item in enumerate(parts):
        item_type, key, value = item
        is_last = (idx == len(parts) - 1)
        line = f"    {key}" if item_type == "flag" else f"    {key} {value}"
        if not is_last:
            line += " \\"
        print(line, file=f)
    print(file=f)


def print_model_deploy_command(
    mlir_path, quantize, processor,
    bmodel_path, calibration_table=None, file=None,
):
    """Print deploy command that creates .bmodel directly.

    No --test_input / --test_reference is emitted because some exported
    PaddleOCR ONNX/MLIR files use input names that do not match auto-generated
    NPZ keys. Removing them avoids AssertionError during deploy.
    """
    f = file or sys.stdout
    quantize_upper = quantize.upper()
    parts: list[tuple[str, str, str]] = [
        ("kv", "--mlir", mlir_path),
        ("kv", "--quantize", quantize_upper),
    ]
    if quantize_upper in QUANTIZE_NEEDS_CALIB:
        if calibration_table is None:
            raise RuntimeError(
                f"{quantize_upper} requires a calibration table.\n"
                "Please provide the calibration table path."
            )
        parts.append(("kv", "--calibration_table", calibration_table))
    parts.extend([
        ("kv", "--processor", processor),
        ("kv", "--model", bmodel_path),
    ])
    print("# ----- model_deploy -----", file=f)
    print("model_deploy \\", file=f)
    for idx, item in enumerate(parts):
        _, key, value = item
        is_last = (idx == len(parts) - 1)
        line = f"    {key} {value}"
        if not is_last:
            line += " \\"  # trailing backslash for generated shell/bat command
        print(line, file=f)
    print(file=f)

def print_calibration_command(mlir_path, dataset, input_num, output_table, file=None):
    f = file or sys.stdout
    print("# ----- run_calibration -----", file=f)
    print(f"run_calibration {mlir_path} \\", file=f)
    print(f"    --dataset {dataset} \\", file=f)
    print(f"    --input_num {input_num} \\", file=f)
    print(f"    -o {output_table}", file=f)
    print(file=f)


def _find_model_transform() -> bool:
    """Check if model_transform is available locally (cross-platform)."""
    if shutil.which("model_transform") is not None:
        return True
    # On Windows, also check for model_transform.exe or model_transform.cmd
    if platform.system() == "Windows":
        for ext in (".exe", ".cmd", ".bat"):
            if shutil.which(f"model_transform{ext}") is not None:
                return True
    return False


# ============================================================
# RUN ID
# ============================================================

def generate_run_id() -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:6].upper()
    return f"{ts}_{uid}"


# ============================================================
# INTERACTIVE PATH SELECTION
# ============================================================

def _browse_file_gui(title: str, extension: str) -> Optional[str]:
    if not _TKINTER_AVAILABLE:
        return None
    try:
        root = _tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        if extension:
            ext_upper = extension.lstrip(".").upper()
            filetypes = [(f"{ext_upper} files", f"*{extension}"), ("All files", "*.*")]
        else:
            filetypes = [("All files", "*.*")]
        path = _filedialog.askopenfilename(title=title, filetypes=filetypes)
        root.destroy()
        return path if path else None
    except Exception:
        return None


def _search_files(base_dir: str, extension: str) -> list[Path]:
    base = Path(base_dir)
    if not base.is_dir():
        print(f"  Directory not found: {base_dir}")
        return []
    if extension:
        return sorted(base.rglob(f"*{extension}"))
    return sorted(p for p in base.rglob("*") if p.is_file())


def select_file(prompt: str, extension: str) -> Optional[Path]:
    """
    Let user pick a file via:
      1) manual path entry
      2) GUI file dialog  (if tkinter available)
      3) recursive search in a directory
      0) skip / cancel
    Returns a Path or None.
    """
    while True:
        print(f"\n  {prompt}")
        menu: list[tuple[str, str]] = [("manual", "Enter path manually")]
        if _TKINTER_AVAILABLE:
            menu.append(("browse", "Browse with file dialog"))
        menu.append(("search", "Search in a directory"))
        menu.append(("skip",   "Skip"))

        for i, (_, label) in enumerate(menu, 1):
            print(f"    {i}) {label}")

        raw = input("  Select: ").strip()
        try:
            idx = int(raw) - 1
        except ValueError:
            print("  Invalid choice. Try again.")
            continue

        if not (0 <= idx < len(menu)):
            print("  Invalid choice. Try again.")
            continue

        action = menu[idx][0]

        if action == "manual":
            raw_path = input("  Path: ").strip().strip('"').strip("'")
            p = Path(raw_path)
            if p.is_file():
                return p
            print(f"  File not found: {raw_path}")

        elif action == "browse":
            result = _browse_file_gui(title=prompt, extension=extension)
            if result:
                return Path(result)
            print("  No file selected.")

        elif action == "search":
            raw_dir = input("  Directory to search: ").strip().strip('"').strip("'")
            found = _search_files(raw_dir, extension)
            if not found:
                ext_hint = f"*{extension}" if extension else "files"
                print(f"  No {ext_hint} found in: {raw_dir}")
                continue
            print(f"\n  Found {len(found)} file(s):")
            for i, fp in enumerate(found, 1):
                print(f"    {i}) {fp}")
            sel = input("  Select number (0 to cancel): ").strip()
            try:
                n = int(sel)
                if 1 <= n <= len(found):
                    return found[n - 1]
            except ValueError:
                pass
            print("  Cancelled.")

        elif action == "skip":
            return None


def select_directory(prompt: str) -> Optional[Path]:
    """Let user pick a directory via manual entry, GUI dialog, or skip."""
    while True:
        print(f"\n  {prompt}")
        menu: list[tuple[str, str]] = [("manual", "Enter path manually")]
        if _TKINTER_AVAILABLE:
            menu.append(("browse", "Browse with file dialog"))
        menu.append(("skip", "Skip"))

        for i, (_, label) in enumerate(menu, 1):
            print(f"    {i}) {label}")

        raw = input("  Select: ").strip()
        try:
            idx = int(raw) - 1
        except ValueError:
            print("  Invalid choice. Try again.")
            continue

        if not (0 <= idx < len(menu)):
            print("  Invalid choice. Try again.")
            continue

        action = menu[idx][0]

        if action == "manual":
            raw_path = input("  Path: ").strip().strip('"').strip("'")
            p = Path(raw_path)
            if p.is_dir():
                return p
            print(f"  Directory not found: {raw_path}")

        elif action == "browse":
            try:
                root = _tk.Tk()
                root.withdraw()
                root.attributes("-topmost", True)
                path = _filedialog.askdirectory(title=prompt)
                root.destroy()
                if path:
                    return Path(path)
                print("  No directory selected.")
            except Exception:
                print("  GUI not available.")

        elif action == "skip":
            return None


# ============================================================
# INTERACTIVE OPTION MENUS
# ============================================================

def _pick_option(prompt: str, options: list[str], default_idx: int = 0) -> str:
    print(f"\n  {prompt}")
    for i, opt in enumerate(options, 1):
        marker = "  <-- default" if (i - 1) == default_idx else ""
        print(f"    {i}) {opt}{marker}")
    while True:
        raw = input(f"  Select [1-{len(options)}] (Enter = default): ").strip()
        if raw == "":
            return options[default_idx]
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except ValueError:
            pass
        print("  Invalid choice.")


def select_quantize() -> str:
    labels = []
    for q in QUANTIZE_OPTIONS:
        note = "  *needs calibration table*" if q in QUANTIZE_NEEDS_CALIB else ""
        labels.append(f"{q}{note}")
    choice = _pick_option("Quantize", labels, default_idx=1)  # default F16
    return choice.split()[0]


def select_processor() -> str:
    return _pick_option("Processor", PROCESSOR_OPTIONS, default_idx=0)


def select_input_shape(onnx_shape: list, model_type: str, config) -> list[int]:
    print("\n  Input Shape")
    print(f"    ONNX detected : {onnx_shape}")

    if model_type == "det" or is_yolo_model(model_type):
        default_shape = list(DEFAULT_INPUT_SHAPE)
    else:
        try:
            c, h, w = get_rec_image_shape(config)
            default_shape = [1, c, h, w]
        except Exception:
            default_shape = list(DEFAULT_INPUT_SHAPE)

    if is_dynamic_shape(onnx_shape):
        print("    (dynamic shape — must be overridden manually)")

    default_str = ",".join(str(x) for x in default_shape)
    print(f"    Default       : [{default_str}]")
    print("    Format        : N,C,H,W  e.g. 1,3,640,640")

    while True:
        raw = input(f"  Shape [{default_str}]: ").strip()
        if not raw:
            return default_shape
        try:
            parts = [int(x.strip()) for x in raw.strip("[]()").split(",")]
            if len(parts) == 4 and all(x > 0 for x in parts):
                return parts
            print("  Must be exactly 4 positive integers.")
        except ValueError:
            print("  Invalid format. Example: 1,3,640,640")


# ============================================================
# COMMAND ARG BUILDERS  (for subprocess — no test flags)
# ============================================================

def _transform_argv(
    model_name: str,
    onnx_path: str,
    input_shape: list,
    preprocess: dict,
    keep_aspect_ratio: bool,
    output_names: Optional[list[str]],
    mlir_path: str,
) -> list[str]:
    argv = [
        "model_transform",
        "--model_name", model_name,
        "--model_def", onnx_path,
        "--input_shapes", shape_to_tpu_mlir(input_shape),
        "--mean", comma_list(preprocess["mean"]),
        "--scale", comma_list(preprocess["scale"]),
    ]
    if keep_aspect_ratio:
        argv.append("--keep_aspect_ratio")
    argv += ["--pixel_format", preprocess["pixel_format"]]
    if output_names:
        argv += ["--output_names", ",".join(output_names)]
    argv += ["--mlir", mlir_path]
    return argv


def _deploy_argv(
    mlir_path: str,
    quantize: str,
    processor: str,
    bmodel_path: str,
    calibration_table: Optional[str],
) -> list[str]:
    quantize_upper = quantize.upper()
    argv = [
        "model_deploy",
        "--mlir", mlir_path,
        "--quantize", quantize_upper,
    ]
    if quantize_upper in QUANTIZE_NEEDS_CALIB:
        if calibration_table is None:
            raise RuntimeError(f"{quantize_upper} requires a calibration table.")
        argv += ["--calibration_table", calibration_table]
    argv += ["--processor", processor, "--model", bmodel_path]
    return argv


def _calibration_argv(mlir_path: str, dataset: str, input_num: int, output_table: str) -> list[str]:
    return [
        "run_calibration",
        mlir_path,
        "--dataset", dataset,
        "--input_num", str(input_num),
        "-o", output_table,
    ]


# ============================================================
# DOCKER HELPERS
# ============================================================



def _run_quiet(argv: list[str], timeout: int = 20) -> subprocess.CompletedProcess:
    """Run a command without opening an extra console window on Windows."""
    kwargs = dict(capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    if platform.system() == "Windows":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    return subprocess.run(argv, **kwargs)


def _docker_daemon_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        r = _run_quiet(["docker", "info"], timeout=10)
        return r.returncode == 0
    except Exception:
        return False


def _start_docker_desktop_windows() -> None:
    """Try to start Docker Desktop on Windows. Safe no-op on other OS."""
    if platform.system() != "Windows":
        return

    candidates = [
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Docker" / "Docker" / "Docker Desktop.exe",
        Path(os.environ.get("LocalAppData", "")) / "Docker" / "Docker Desktop.exe",
    ]

    for exe in candidates:
        if exe.is_file():
            print(f"\n  Docker Desktop is not running. Starting Docker Desktop...")
            try:
                subprocess.Popen([str(exe)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                print(f"  WARNING: could not start Docker Desktop automatically: {e}")
            return

    print("\n  WARNING: Docker Desktop is not running and Docker Desktop.exe was not found.")
    print("  Please open Docker Desktop manually, then run this script again.")


def ensure_docker_ready() -> bool:
    """Ensure Docker CLI + daemon are available; auto-start Docker Desktop on Windows."""
    if shutil.which("docker") is None:
        print("\n  ERROR: docker command not found. Please install Docker Desktop first.")
        return False

    if _docker_daemon_ready():
        return True

    _start_docker_desktop_windows()

    print(f"  Waiting for Docker daemon (up to {DOCKER_START_TIMEOUT_SEC}s)...")
    import time
    deadline = time.time() + DOCKER_START_TIMEOUT_SEC
    while time.time() < deadline:
        if _docker_daemon_ready():
            print("  Docker daemon is ready.")
            return True
        time.sleep(3)

    print("\n  ERROR: Docker daemon is still not ready.")
    print("  Open Docker Desktop and wait until it says Engine running, then run again.")
    return False


def _container_exists(name: str) -> bool:
    try:
        r = _run_quiet(["docker", "inspect", name], timeout=10)
        return r.returncode == 0
    except Exception:
        return False


def _container_running(name: str) -> bool:
    try:
        r = _run_quiet(["docker", "inspect", "-f", "{{.State.Running}}", name], timeout=10)
        return r.returncode == 0 and r.stdout.strip().lower() == "true"
    except Exception:
        return False


def _container_has_model_transform(name: str) -> bool:
    try:
        chk = _run_quiet(
            ["docker", "exec", name, "sh", "-c", "which model_transform || command -v model_transform"],
            timeout=15,
        )
        return chk.returncode == 0
    except Exception:
        return False


def ensure_tpumlir_container(workspace_host: Path) -> Optional[str]:
    """
    Start Docker Desktop if needed, then start/create a TPU-MLIR container.
    The host workspace is mounted to /workspace.
    """
    if not ensure_docker_ready():
        return None

    name = DOCKER_CONTAINER_NAME
    workspace_host = workspace_host.resolve()

    if _container_exists(name):
        if not _container_running(name):
            print(f"\n  Docker: starting existing container '{name}'...")
            r = _run_quiet(["docker", "start", name], timeout=30)
            if r.returncode != 0:
                print(f"  ERROR: failed to start container '{name}':")
                print(r.stderr or r.stdout)
                return None
        else:
            print(f"\n  Docker: container '{name}' is already running.")

        if not _container_has_model_transform(name):
            print(f"  ERROR: container '{name}' does not have model_transform.")
            print(f"  Remove it and rerun, or create it from image: {DOCKER_IMAGE}")
            print(f"  Command: docker rm -f {name}")
            return None
        return name

    print(f"\n  Docker: container '{name}' not found. Creating it...")
    print(f"  Image    : {DOCKER_IMAGE}")
    print(f"  Mount    : {workspace_host} -> {DOCKER_WORKDIR}")

    r = _run_quiet([
        "docker", "run", "-dit", "--privileged",
        "--name", name,
        "-v", f"{workspace_host}:{DOCKER_WORKDIR}",
        DOCKER_IMAGE,
        "bash",
    ], timeout=120)

    if r.returncode != 0:
        print("\n  ERROR: failed to create Docker container.")
        print(r.stderr or r.stdout)
        print("\n  If the image is missing, pull it first:")
        print(f"    docker pull {DOCKER_IMAGE}")
        return None

    if not _container_has_model_transform(name):
        print(f"  ERROR: newly created container '{name}' does not have model_transform.")
        return None

    print(f"  Docker: created and started container '{name}'.")
    return name

def find_docker_tpumlir_container() -> Optional[str]:
    """Return name of the first running Docker container that has model_transform."""
    if shutil.which("docker") is None:
        return None
    try:
        r = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            return None
        for name in r.stdout.strip().splitlines():
            if not name:
                continue
            # Use 'sh -c' so this works from a Windows host against a Linux container
            chk = subprocess.run(
                ["docker", "exec", name, "sh", "-c", "which model_transform || command -v model_transform"],
                capture_output=True, text=True, timeout=10,
            )
            if chk.returncode == 0:
                return name
    except Exception:
        pass
    return None


def host_to_container_path(container: str, host_path: Path) -> Optional[str]:
    """Translate an absolute host path to its equivalent path inside the container."""
    try:
        r = subprocess.run(
            ["docker", "inspect", container,
             "--format", "{{range .Mounts}}{{.Source}}|{{.Destination}}\n{{end}}"],
            capture_output=True, text=True, timeout=10,
        )
        # Normalize to forward slashes for comparison (handles Windows paths like C:\foo)
        host_str = str(host_path.resolve()).replace("\\", "/")

        # On Windows, Docker Desktop may expose drive as /c/ or /host_mnt/c/
        # Build alternative representations to try
        host_variants = [host_str]
        if len(host_str) >= 2 and host_str[1] == ":":
            drive = host_str[0].lower()
            rest = host_str[2:]  # e.g. /Users/foo
            host_variants.append(f"/{drive}{rest}")            # /c/Users/foo
            host_variants.append(f"/host_mnt/{drive}{rest}")   # /host_mnt/c/Users/foo
            host_variants.append(f"/mnt/{drive}{rest}")        # /mnt/c/Users/foo (WSL2)

        for line in r.stdout.strip().splitlines():
            if "|" not in line:
                continue
            src, dst = line.split("|", 1)
            src_norm = src.replace("\\", "/")
            for hv in host_variants:
                if hv.startswith(src_norm):
                    rel = hv[len(src_norm):]
                    return dst.rstrip("/") + "/" + rel.lstrip("/")
    except Exception:
        pass
    return None


# ============================================================
# COMMAND RUNNER
# ============================================================

def run_command(
    argv: list[str],
    cwd: Path,
    log_file: Path,
    docker_container: Optional[str] = None,
    docker_cwd: Optional[str] = None,
) -> tuple[int, float]:
    """
    Run a command locally or via docker exec.
    Streams output to screen and saves to log_file.
    Returns (exit_code, elapsed_seconds).
    """
    if docker_container:
        full_argv = ["docker", "exec", "-w", docker_cwd or str(cwd), docker_container] + argv
    else:
        full_argv = argv

    cmd_display = " ".join(full_argv)
    print(f"\n  $ {cmd_display}")
    print(f"  (cwd: {docker_cwd or cwd})\n")

    t_start = datetime.datetime.now()

    with log_file.open("w", encoding="utf-8") as lf:
        lf.write(f"$ {cmd_display}\n")
        lf.write(f"started: {t_start.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        popen_kwargs: dict = dict(
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(cwd),
        )
        # Windows: suppress extra console window that pops up for subprocesses
        if platform.system() == "Windows":
            popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        try:
            proc = subprocess.Popen(full_argv, **popen_kwargs)
        except FileNotFoundError:
            elapsed = (datetime.datetime.now() - t_start).total_seconds()
            msg = (
                f"ERROR: command not found — '{full_argv[0]}'\n"
                f"Make sure Docker is running or TPU-MLIR is activated.\n"
            )
            print(f"  {msg}")
            lf.write(msg)
            return 1, elapsed

        for line in proc.stdout:
            print(line, end="", flush=True)
            lf.write(line)

        proc.wait()
        elapsed = (datetime.datetime.now() - t_start).total_seconds()
        lf.write(f"\n[exit code: {proc.returncode}] [elapsed: {elapsed:.1f}s]\n")

    return proc.returncode, elapsed


def _fmt_elapsed(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}  ({seconds:.1f}s)"


def _write_timing_log(path: Path, timing: dict[str, float]) -> None:
    total = sum(timing.values())
    col = max(len(k) for k in timing) + 2
    lines = [
        "=== Timing Report ===",
        f"  {'Step':<{col}}  Duration",
        "  " + "-" * (col + 26),
    ]
    for step, elapsed in timing.items():
        lines.append(f"  {step:<{col}}  {_fmt_elapsed(elapsed)}")
    lines += [
        "  " + "-" * (col + 26),
        f"  {'TOTAL':<{col}}  {_fmt_elapsed(total)}",
    ]
    report = "\n".join(lines)
    print(f"\n{report}")
    path.write_text(report + "\n", encoding="utf-8")
    print(f"  timing.log    : saved ({path})")


# ============================================================
# MODEL GENERATION
# ============================================================

def _collect_config_terminal(model_type: str) -> Optional[ModelConfig]:
    """Collect all user inputs interactively via terminal. Returns None if user skips."""
    sep = "=" * 58
    print(f"\n{sep}")
    print(f"  Configure {model_type.upper()} model")
    print(sep)

    config_path = None
    yolo_pt_path: Optional[Path] = None
    yaml_config = None

    if not is_yolo_model(model_type):
        config_path = select_file(f"Select Config YAML  [{model_type.upper()}]", ".yml")
        if config_path is None:
            print(f"  Skipping {model_type.upper()} — no config file.")
            return None
        onnx_path = select_file(f"Select ONNX model   [{model_type.upper()}]", ".onnx")
        if onnx_path is None:
            print(f"  Skipping {model_type.upper()} — no ONNX file.")
            return None
        print(f"\n  Loading config : {config_path}")
        yaml_config = load_yaml(str(config_path))
        print(f"  Loading ONNX   : {onnx_path}")
        onnx_info = get_onnx_info(str(onnx_path))
        detected_shape = onnx_info["input_shape"]
    else:
        model_path = select_file(f"Select YOLO model .pt or .onnx [{model_type.upper()}]", "")
        if model_path is None:
            print(f"  Skipping {model_type.upper()} — no YOLO model file.")
            return None
        if is_pt_file(model_path):
            yolo_pt_path = model_path
            onnx_path = model_path.with_suffix(".onnx")
            detected_shape = DEFAULT_INPUT_SHAPE
            print(f"\n  YOLO PT selected : {model_path}")
            print(f"  ONNX will be exported to: {onnx_path}")
        elif is_onnx_file(model_path):
            onnx_path = model_path
            print(f"\n  Loading YOLO ONNX: {onnx_path}")
            onnx_info = get_onnx_info(str(onnx_path))
            detected_shape = onnx_info["input_shape"]
        else:
            print("  ERROR: YOLO model must be .pt or .onnx")
            return None

    input_shape = select_input_shape(
        onnx_shape=detected_shape,
        model_type=model_type,
        config=yaml_config or {},
    )

    quantize = select_quantize()

    calib_mode = "none"
    calibration_table: Optional[str] = None
    calib_dataset: Optional[Path] = None
    calib_input_num = DEFAULT_CALIB_INPUT_NUM

    if quantize in QUANTIZE_NEEDS_CALIB:
        print(f"\n  Calibration for {quantize}")
        print(f"    1) Use existing calibration table")
        print(f"    2) Run calibration with a dataset folder")
        print(f"    3) Skip  (WARNING: deploy will fail)")
        while True:
            raw_cal = input("  Select [1-3]: ").strip()
            if raw_cal == "1":
                cal_path = select_file(f"Select calibration table for {quantize}", "")
                if cal_path:
                    calibration_table = str(cal_path)
                    calib_mode = "existing"
                else:
                    print(f"  WARNING: no calibration table selected.")
                break
            elif raw_cal == "2":
                ds = select_directory("Dataset folder for calibration")
                if ds:
                    calib_dataset = ds
                    raw_n = input(
                        f"  Number of images [default {DEFAULT_CALIB_INPUT_NUM}]: "
                    ).strip()
                    if raw_n:
                        try:
                            calib_input_num = max(1, int(raw_n))
                        except ValueError:
                            print(f"  Invalid number — using {DEFAULT_CALIB_INPUT_NUM}.")
                    calib_mode = "run"
                else:
                    print(f"  WARNING: no dataset selected.")
                break
            elif raw_cal == "3":
                print(f"  WARNING: {quantize} without calibration — deploy step will fail.")
                break
            else:
                print("  Invalid choice.")

    processor = select_processor()

    kar_raw = input("\n  Keep aspect ratio? [y/N]: ").strip().lower()
    keep_aspect_ratio = kar_raw in {"y", "yes"}

    output_names: Optional[list] = None
    ask_out = input("  Specify output names manually? [y/N]: ").strip().lower()
    if ask_out in {"y", "yes"}:
        raw_names = input("  Output names (comma-separated): ").strip()
        if raw_names:
            output_names = [x.strip() for x in raw_names.split(",")]

    return ModelConfig(
        model_type=model_type,
        config_path=config_path,
        onnx_path=onnx_path,
        input_shape=input_shape,
        quantize=quantize,
        processor=processor,
        keep_aspect_ratio=keep_aspect_ratio,
        output_names=output_names,
        calib_mode=calib_mode,
        calibration_table=calibration_table,
        calib_dataset=calib_dataset,
        calib_input_num=calib_input_num,
        yolo_pt_path=yolo_pt_path,
    )


def _run_pipeline(cfg: ModelConfig, run_id: str) -> None:
    """Execute model_transform → (run_calibration) → model_deploy for one ModelConfig."""
    model_type = cfg.model_type
    config_path = cfg.config_path
    onnx_path = cfg.onnx_path
    input_shape = cfg.input_shape
    quantize = cfg.quantize
    processor = cfg.processor
    keep_aspect_ratio = cfg.keep_aspect_ratio
    output_names = cfg.output_names
    calibration_table = cfg.calibration_table
    run_calib = (cfg.calib_mode == "run")
    calib_dataset_host = cfg.calib_dataset
    calib_input_num = cfg.calib_input_num
    yolo_pt_path = cfg.yolo_pt_path

    sep = "=" * 58
    print(f"\n{sep}")
    print(f"  {model_type.upper()} pipeline  (Run ID: {run_id})")
    print(sep)

    yaml_config = None

    # ============================================================
    # OUTPUT / LOG FOLDERS
    # ============================================================
    # Create these BEFORE YOLO export, because YOLO .pt -> .onnx
    # needs yolo_export_log.
    base_results = Path(__file__).resolve().parent / "results"

    # Output model folder: keep generated model files here
    out_dir = base_results / f"{run_id}_{model_type}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Central log folder: collect all logs from every run here
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Centralized log file paths
    transform_log = log_dir / f"{run_id}_{model_type}_transform.log"
    deploy_log = log_dir / f"{run_id}_{model_type}_deploy.log"
    calibration_log = log_dir / f"{run_id}_{model_type}_calibration.log"
    timing_log = log_dir / f"{run_id}_{model_type}_timing.log"
    yolo_export_log = log_dir / f"{run_id}_{model_type}_pt_to_onnx.log"

    # YOLO can start from .pt. Export .pt -> .onnx first, then continue ONNX -> MLIR -> BMODEL.
    if is_yolo_model(model_type) and yolo_pt_path is not None:
        try:
            onnx_path = export_yolo_pt_to_onnx(
                pt_path=yolo_pt_path,
                imgsz=int(input_shape[2]),
                out_dir=out_dir,
                log_file=yolo_export_log,
                model_type=model_type,
            )
        except Exception as e:
            print(f"\n  ERROR: YOLO PT -> ONNX export failed: {e}")
            print(f"  See log: {yolo_export_log}")
            return

    if is_yolo_model(model_type):
        print(f"\n  Loading YOLO ONNX : {onnx_path}")
        onnx_info = get_onnx_info(str(onnx_path))
        preprocess = get_yolo_preprocess()
        model_name = model_type
    else:
        print(f"\n  Loading config : {config_path}")
        yaml_config = load_yaml(str(config_path))
        print(f"  Loading ONNX   : {onnx_path}")
        onnx_info = get_onnx_info(str(onnx_path))
        if model_type == "det":
            preprocess = get_det_preprocess(yaml_config)
        else:
            preprocess = get_rec_preprocess(yaml_config)
        model_name = f"ppocr_{model_type}"

    onnx_stem = onnx_path.stem
    mlir_path = f"{onnx_stem}.mlir"
    quantize_lower = quantize.lower()
    bmodel_path = f"{onnx_stem}_{processor}_{quantize_lower}.bmodel"
    calibration_table_name = f"{onnx_stem}_cali_table"

    docker_container: Optional[str] = None
    docker_out_dir: Optional[str] = None
    onnx_cmd_path: str
    cal_cmd_path: Optional[str] = None
    calib_dataset_cmd_path: Optional[str] = None

    if _find_model_transform():
        onnx_cmd_path = str(onnx_path.resolve())
        if calibration_table:
            cal_cmd_path = str(Path(calibration_table).resolve())
        elif run_calib:
            cal_cmd_path = str(out_dir / calibration_table_name)
            calib_dataset_cmd_path = str(calib_dataset_host.resolve())
    else:
        docker_container = ensure_tpumlir_container(Path(__file__).resolve().parent)

        if docker_container is None:
            onnx_cmd_path = str(onnx_path.resolve())
            if calibration_table:
                cal_cmd_path = str(Path(calibration_table).resolve())
            elif run_calib:
                cal_cmd_path = str(out_dir / calibration_table_name)
                calib_dataset_cmd_path = str(calib_dataset_host.resolve())
        else:
            print(f"\n  Docker: using container '{docker_container}'")

            # Prefer running inside the mounted ONNX folder (e.g. /workspace).
            # This produces the .mlir and .bmodel directly in the mounted folder,
            # so you do not need to run the generated command manually.
            container_onnx = host_to_container_path(docker_container, onnx_path.resolve())
            if container_onnx is None:
                docker_out_dir = host_to_container_path(docker_container, out_dir)
                if docker_out_dir is None:
                    print(f"\n  WARNING: ONNX/results folder is not mounted in Docker.")
                    print(f"  Host ONNX : {onnx_path.resolve()}")
                    print(f"  Host output: {out_dir}")

                    # ============================================================
                    # AUTO USE DOCKER WORKSPACE
                    # ============================================================
                    # Do not ask the user to type /workspace every time.
                    # The script uses /workspace automatically because the Docker
                    # container is normally mounted as:
                    #     D:\\BUU\\model_conversion  ->  /workspace
                    docker_out_dir = DOCKER_WORKDIR

                    print(f"  Auto Docker container folder:")
                    print(f"      {docker_out_dir}")
                print(f"  ONNX is outside Docker mount — checking host results folder...")
                dst_onnx = out_dir / onnx_path.name
                if onnx_path.resolve() != dst_onnx.resolve():
                    print(f"  Copying ONNX to host results folder...")
                    shutil.copy2(str(onnx_path), str(dst_onnx))
                else:
                    print(f"  ONNX already exists in output folder, skip copy.")
                onnx_cmd_path = f"{docker_out_dir.rstrip('/')}/{onnx_path.name}"
            else:
                onnx_cmd_path = container_onnx
                docker_out_dir = container_onnx.rsplit('/', 1)[0] if '/' in container_onnx else '/workspace'

            if calibration_table:
                cal_file = Path(calibration_table)
                container_cal = host_to_container_path(docker_container, cal_file.resolve())
                if container_cal is None:
                    print(f"  Calibration table outside Docker mount — copying to host results folder...")
                    shutil.copy2(str(cal_file), str(out_dir / cal_file.name))
                    cal_cmd_path = f"{docker_out_dir.rstrip('/')}/{cal_file.name}"
                else:
                    cal_cmd_path = container_cal
            elif run_calib:
                cal_cmd_path = f"{docker_out_dir.rstrip('/')}/{calibration_table_name}"
                container_ds = host_to_container_path(
                    docker_container, calib_dataset_host.resolve()
                )
                if container_ds is None:
                    print(f"\n  ERROR: dataset folder is not mounted in the container.")
                    print(f"  Mount the dataset directory into Docker before running.")
                    return
                calib_dataset_cmd_path = container_ds

    # --- Build and save commands.txt ---
    is_windows = platform.system() == "Windows"
    if is_windows:
        script_header = (
            f"@echo off\n"
            f"REM Run ID   : {run_id}\n"
            f"REM Model    : {model_type.upper()}\n"
            f"REM Date     : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"REM Platform : {platform.system()} {platform.release()} ({platform.machine()})\n"
            f"REM Usage    : run commands.bat  (or execute in Docker terminal)\n\n"
        )
        script_filename = "commands.bat"
    else:
        script_header = (
            f"#!/usr/bin/env bash\n"
            f"# Run ID   : {run_id}\n"
            f"# Model    : {model_type.upper()}\n"
            f"# Date     : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"# Platform : {platform.system()} {platform.release()} ({platform.machine()})\n"
            f"# Usage    : bash commands.txt\n\n"
            f"set -e\n\n"
        )
        script_filename = "commands.txt"

    buf = io.StringIO()
    buf.write(script_header)
    print_model_info(
        model_type=model_type,
        config_path=str(config_path),
        onnx_path=onnx_cmd_path,
        onnx_info=onnx_info,
        input_shape=input_shape,
        preprocess=preprocess,
        file=buf,
    )
    print_model_transform_command(
        model_name=model_name,
        onnx_path=onnx_cmd_path,
        input_shape=input_shape,
        preprocess=preprocess,
        keep_aspect_ratio=keep_aspect_ratio,
        output_names=output_names,
        test_input=None,
        test_result=None,
        mlir_path=mlir_path,
        file=buf,
    )
    if run_calib:
        print_calibration_command(
            mlir_path=mlir_path,
            dataset=calib_dataset_cmd_path,
            input_num=calib_input_num,
            output_table=calibration_table_name,
            file=buf,
        )
    print_model_deploy_command(
        mlir_path=mlir_path,
        quantize=quantize,
        processor=processor,
        bmodel_path=bmodel_path,
        calibration_table=cal_cmd_path,
        file=buf,
    )
    commands_text = buf.getvalue()
    (out_dir / script_filename).write_text(commands_text, encoding="utf-8")

    divider = "-" * 58
    print(f"\n{divider}")
    print(commands_text, end="")
    print(divider)
    print(f"\n  Output folder : {out_dir}")
    print(f"  {script_filename}  : saved")

    if not _find_model_transform() and docker_container is None:
        print(f"\n  TPU-MLIR not found locally or in Docker.")
        if is_windows:
            if shutil.which("docker") is None:
                print(f"\n  [ERROR] Docker not found. Install Docker Desktop for Windows:")
                print(f"    https://www.docker.com/products/docker-desktop/")
            else:
                print(f"\n  [ERROR] No running Docker container with TPU-MLIR was found.")
                print(f"  Start your TPU-MLIR container and mount the results folder, e.g.:")
                print(f"    docker run -it --name tpumlir -v {out_dir.parent}:/workspace <image>")
                print(f"  Then re-run this script.")
            print(f"\n  Commands saved to: {out_dir / script_filename}")
            print(f"  Run them manually inside the container: bash {script_filename}")
        else:
            print(f"  Run commands manually on a Linux machine with TPU-MLIR:")
            print(f"    source /path/to/tpu-mlir/envsetup.sh")
            print(f"    cd {out_dir}/")
            print(f"    bash {script_filename}")
        return

    timing: dict[str, float] = {}

    print(f"\n{'='*58}")
    print(f"  Running model_transform ...")
    print(f"{'='*58}")
    rc_transform, t_transform = run_command(
        argv=_transform_argv(
            model_name=model_name,
            onnx_path=onnx_cmd_path,
            input_shape=input_shape,
            preprocess=preprocess,
            keep_aspect_ratio=keep_aspect_ratio,
            output_names=output_names,
            mlir_path=mlir_path,
        ),
        cwd=out_dir,
        log_file=transform_log,
        docker_container=docker_container,
        docker_cwd=docker_out_dir,
    )
    timing["model_transform"] = t_transform
    print(f"\n  model_transform {'OK' if rc_transform == 0 else 'FAILED'}  [{_fmt_elapsed(t_transform)}]")
    if rc_transform != 0:
        print(f"  See log: {transform_log}")
        _write_timing_log(timing_log, timing)
        return

    if run_calib:
        print(f"\n{'='*58}")
        print(f"  Running run_calibration  (images: {calib_input_num}) ...")
        print(f"{'='*58}")
        rc_calib, t_calib = run_command(
            argv=_calibration_argv(
                mlir_path=mlir_path,
                dataset=calib_dataset_cmd_path,
                input_num=calib_input_num,
                output_table=calibration_table_name,
            ),
            cwd=out_dir,
            log_file=calibration_log,
            docker_container=docker_container,
            docker_cwd=docker_out_dir,
        )
        timing["run_calibration"] = t_calib
        print(f"\n  run_calibration {'OK' if rc_calib == 0 else 'FAILED'}  [{_fmt_elapsed(t_calib)}]")
        if rc_calib != 0:
            print(f"  See log: {calibration_log}")
            _write_timing_log(timing_log, timing)
            return

    print(f"\n{'='*58}")
    print(f"  Running model_deploy ...")
    print(f"{'='*58}")
    rc_deploy, t_deploy = run_command(
        argv=_deploy_argv(
            mlir_path=mlir_path,
            quantize=quantize,
            processor=processor,
            bmodel_path=bmodel_path,
            calibration_table=cal_cmd_path,
        ),
        cwd=out_dir,
        log_file=deploy_log,
        docker_container=docker_container,
        docker_cwd=docker_out_dir,
    )
    timing["model_deploy"] = t_deploy
    print(f"\n  model_deploy {'OK' if rc_deploy == 0 else 'FAILED'}  [{_fmt_elapsed(t_deploy)}]")
    if rc_deploy != 0:
        print(f"  See log: {deploy_log}")
        _write_timing_log(timing_log, timing)
        return

    # If the model was generated inside Docker, copy it back to the Windows/host
    # results folder automatically so it appears in File Explorer.
    if docker_container and docker_out_dir:
        try:
            container_bmodel = f"{docker_out_dir.rstrip('/')}/{bmodel_path}"
            host_bmodel = out_dir / bmodel_path

            cp_cmd = [
                "docker",
                "cp",
                f"{docker_container}:{container_bmodel}",
                str(host_bmodel),
            ]
            print(f"\n  Copying bmodel from Docker ...")
            print(f"  $ {' '.join(cp_cmd)}")

            cp_result = subprocess.run(
                cp_cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )

            if cp_result.returncode == 0:
                print("  docker cp OK")
                print(f"    {container_bmodel}")
                print(f"    -> {host_bmodel}")
            else:
                print("  WARNING: docker cp failed")
                if cp_result.stdout:
                    print(cp_result.stdout.strip())
                if cp_result.stderr:
                    print(cp_result.stderr.strip())
                print("  You can copy manually with:")
                print(f"    docker cp {docker_container}:{container_bmodel} {host_bmodel}")

        except Exception as e:
            print("\n  WARNING: failed to copy bmodel from Docker")
            print(f"    {e}")
            print("  You can copy manually with:")
            print(f"    docker cp {docker_container}:{docker_out_dir.rstrip('/')}/{bmodel_path} {out_dir / bmodel_path}")

    _write_timing_log(timing_log, timing)
    print(f"\n  Output: {out_dir / bmodel_path}")

    # ============================================================
    # AUTO OPEN RESULT FOLDER
    # ============================================================
    try:
        print("\n  Opening result folder...")
        if platform.system() == "Windows":
            os.startfile(str(out_dir))
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", str(out_dir)])
        else:
            subprocess.Popen(["xdg-open", str(out_dir)])
    except Exception as e:
        print(f"\n  WARNING: failed to open result folder: {e}")


def configure_and_generate(model_type: str, run_id: str) -> None:
    """Terminal-mode entry point: collect config then run pipeline."""
    cfg = _collect_config_terminal(model_type)
    if cfg is not None:
        _run_pipeline(cfg, run_id)


# ============================================================
# TKINTER GUI
# ============================================================

class _ModelPanel:
    """Form fields for one model type (rec or det) inside a parent frame."""

    def __init__(self, parent, model_type: str):
        self.model_type = model_type
        self.frame = _ttk.Frame(parent, padding=12)
        self._build()

    # ---- browse helpers ----

    def _browse_file(self, var, extension, title):
        if extension:
            ext_up = extension.lstrip(".").upper()
            ft = [(f"{ext_up} files", f"*{extension}"), ("All files", "*.*")]
        else:
            ft = [("All files", "*.*")]
        path = _filedialog.askopenfilename(
            title=title, filetypes=ft,
            parent=self.frame.winfo_toplevel(),
        )
        if path:
            var.set(path)

    def _browse_dir(self, var, title):
        path = _filedialog.askdirectory(
            title=title, parent=self.frame.winfo_toplevel()
        )
        if path:
            var.set(path)

    # ---- build ----

    def _build(self):
        f = self.frame
        pad = {"padx": 4, "pady": 3}

        def lbl(row, text):
            _ttk.Label(f, text=text, anchor="w").grid(
                row=row, column=0, sticky="w", **pad
            )

        def entry(row, var, width=48):
            e = _ttk.Entry(f, textvariable=var, width=width)
            e.grid(row=row, column=1, sticky="ew", **pad)
            return e

        def browse_btn(row, cmd):
            b = _ttk.Button(f, text="Browse…", command=cmd, width=9)
            b.grid(row=row, column=2, **pad)
            return b

        f.columnconfigure(1, weight=1)

        # Config YAML
        row = 0
        lbl(row, "Config YAML" if not is_yolo_model(self.model_type) else "Config YAML (unused)")
        self.config_var = _tk.StringVar()
        entry(row, self.config_var)
        browse_btn(row, lambda: self._browse_file(
            self.config_var, ".yml", f"Select Config YAML [{self.model_type.upper()}]"
        ))

        # Model file
        row += 1
        lbl(row, "YOLO .pt / .onnx" if is_yolo_model(self.model_type) else "ONNX Model")
        self.onnx_var = _tk.StringVar()
        entry(row, self.onnx_var)
        browse_btn(row, lambda: self._browse_file(
            self.onnx_var, "" if is_yolo_model(self.model_type) else ".onnx",
            f"Select {'YOLO .pt or .onnx' if is_yolo_model(self.model_type) else 'ONNX Model'} [{self.model_type.upper()}]"
        ))

        # Input Shape
        row += 1
        lbl(row, "Input Shape")
        default_s = "1,3,32,320" if self.model_type == "rec" else "1,3,640,640"
        self.shape_var = _tk.StringVar(value=default_s)
        entry(row, self.shape_var, width=20)
        _ttk.Label(f, text="N,C,H,W", foreground="gray").grid(
            row=row, column=2, sticky="w", **pad
        )

        # Quantize
        row += 1
        lbl(row, "Quantize")
        self.quant_var = _tk.StringVar(value="F16")
        cb = _ttk.Combobox(
            f, textvariable=self.quant_var,
            values=QUANTIZE_OPTIONS, state="readonly", width=8,
        )
        cb.grid(row=row, column=1, sticky="w", **pad)
        cb.bind("<<ComboboxSelected>>", self._on_quant_change)

        # Calibration mode
        row += 1
        lbl(row, "Calibration")
        self.calib_mode_var = _tk.StringVar(value="none")
        cf = _ttk.Frame(f)
        cf.grid(row=row, column=1, columnspan=2, sticky="w", **pad)
        for val, label in [("none", "None"), ("existing", "Existing table"), ("run", "Run calibration")]:
            _ttk.Radiobutton(
                cf, text=label, variable=self.calib_mode_var,
                value=val, command=self._on_calib_change,
            ).pack(side="left", padx=6)

        # Calibration table (existing)
        row += 1
        self._cal_lbl = _ttk.Label(f, text="  Cal. Table", anchor="w")
        self._cal_lbl.grid(row=row, column=0, sticky="w", **pad)
        self.cal_table_var = _tk.StringVar()
        self._cal_entry = entry(row, self.cal_table_var)
        self._cal_btn = browse_btn(row, lambda: self._browse_file(
            self.cal_table_var, "", "Select Calibration Table"
        ))

        # Dataset folder (run)
        row += 1
        self._ds_lbl = _ttk.Label(f, text="  Dataset", anchor="w")
        self._ds_lbl.grid(row=row, column=0, sticky="w", **pad)
        self.dataset_var = _tk.StringVar()
        self._ds_entry = entry(row, self.dataset_var)
        self._ds_btn = browse_btn(row, lambda: self._browse_dir(
            self.dataset_var, "Select Dataset Folder"
        ))

        # Image count
        row += 1
        self._imgn_lbl = _ttk.Label(f, text="  Images", anchor="w")
        self._imgn_lbl.grid(row=row, column=0, sticky="w", **pad)
        self.input_num_var = _tk.IntVar(value=DEFAULT_CALIB_INPUT_NUM)
        self._imgn_entry = _ttk.Entry(f, textvariable=self.input_num_var, width=8)
        self._imgn_entry.grid(row=row, column=1, sticky="w", **pad)

        # Processor
        row += 1
        lbl(row, "Processor")
        self.proc_var = _tk.StringVar(value=PROCESSOR_OPTIONS[0])
        _ttk.Combobox(
            f, textvariable=self.proc_var,
            values=PROCESSOR_OPTIONS, state="readonly", width=10,
        ).grid(row=row, column=1, sticky="w", **pad)

        # Keep aspect ratio
        row += 1
        lbl(row, "Keep Aspect Ratio")
        self.kar_var = _tk.BooleanVar(value=False)
        _ttk.Checkbutton(f, variable=self.kar_var).grid(
            row=row, column=1, sticky="w", **pad
        )

        # Output names
        row += 1
        lbl(row, "Output Names")
        self.out_names_var = _tk.StringVar()
        entry(row, self.out_names_var)
        _ttk.Label(f, text="optional, comma-sep", foreground="gray").grid(
            row=row, column=2, sticky="w", **pad
        )

        self._on_quant_change()

    # ---- callbacks ----

    def _on_quant_change(self, _event=None):
        needs = self.quant_var.get() in QUANTIZE_NEEDS_CALIB
        if not needs:
            self.calib_mode_var.set("none")
        for w in (
            self._cal_lbl, self._cal_entry, self._cal_btn,
            self._ds_lbl, self._ds_entry, self._ds_btn,
            self._imgn_lbl, self._imgn_entry,
        ):
            w.configure(state="normal" if needs else "disabled")
        self._on_calib_change()

    def _on_calib_change(self, *_):
        mode = self.calib_mode_var.get()
        show_cal = mode == "existing"
        show_run = mode == "run"
        for w in (self._cal_lbl, self._cal_entry, self._cal_btn):
            w.configure(state="normal" if show_cal else "disabled")
        for w in (self._ds_lbl, self._ds_entry, self._ds_btn,
                  self._imgn_lbl, self._imgn_entry):
            w.configure(state="normal" if show_run else "disabled")

    # ---- validation & extraction ----

    def validate(self) -> Optional[str]:
        tag = self.model_type.upper()
        if not is_yolo_model(self.model_type) and not Path(self.config_var.get()).is_file():
            return f"[{tag}] Config YAML not found"
        model_file = Path(self.onnx_var.get())
        if not model_file.is_file():
            return f"[{tag}] model file not found"
        if is_yolo_model(self.model_type) and model_file.suffix.lower() not in {".pt", ".onnx"}:
            return f"[{tag}] YOLO model must be .pt or .onnx"
        if (not is_yolo_model(self.model_type)) and model_file.suffix.lower() != ".onnx":
            return f"[{tag}] ONNX model must be .onnx"
        try:
            parts = [int(x) for x in self.shape_var.get().strip("[]()").split(",")]
            if len(parts) != 4 or not all(x > 0 for x in parts):
                raise ValueError
        except ValueError:
            return f"[{tag}] Invalid input shape (need N,C,H,W)"
        mode = self.calib_mode_var.get()
        if self.quant_var.get() in QUANTIZE_NEEDS_CALIB:
            if mode == "existing" and not Path(self.cal_table_var.get()).exists():
                return f"[{tag}] Calibration table not found"
            if mode == "run" and not Path(self.dataset_var.get()).is_dir():
                return f"[{tag}] Dataset folder not found"
        return None

    def get_config(self) -> ModelConfig:
        shape = [int(x) for x in self.shape_var.get().strip("[]()").split(",")]
        raw_names = self.out_names_var.get().strip()
        out_names = [x.strip() for x in raw_names.split(",")] if raw_names else None
        mode = self.calib_mode_var.get()
        quant = self.quant_var.get()
        if quant not in QUANTIZE_NEEDS_CALIB:
            mode = "none"
        model_file = Path(self.onnx_var.get())
        yolo_pt = model_file if is_yolo_model(self.model_type) and model_file.suffix.lower() == ".pt" else None
        onnx_file = model_file.with_suffix(".onnx") if yolo_pt else model_file
        return ModelConfig(
            model_type=self.model_type,
            config_path=None if is_yolo_model(self.model_type) else Path(self.config_var.get()),
            onnx_path=onnx_file,
            input_shape=shape,
            quantize=quant,
            processor=self.proc_var.get(),
            keep_aspect_ratio=self.kar_var.get(),
            output_names=out_names,
            calib_mode=mode,
            calibration_table=self.cal_table_var.get() or None,
            calib_dataset=Path(self.dataset_var.get()) if mode == "run" else None,
            calib_input_num=self.input_num_var.get(),
            yolo_pt_path=yolo_pt,
        )


class TkConfigApp:
    """Main tkinter window: collects one or two ModelConfigs then closes."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.result: list[ModelConfig] = []

        self.root = _tk.Tk()
        self.root.title(f"Model Conversion — Run ID: {run_id}")
        self.root.resizable(True, False)
        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}

        # Model type row
        top = _ttk.Frame(self.root, padding=(8, 6))
        top.pack(fill="x")
        _ttk.Label(top, text="Model Type:", font=("", 10, "bold")).pack(
            side="left", **pad
        )
        self.model_type_var = _tk.StringVar(value="det")
        for val, label in [("rec", "Paddle REC"), ("det", "Paddle DET"), ("yolo_hbb", "YOLO HBB"), ("yolo_obb", "YOLO OBB"), ("both", "Paddle Both")]:
            _ttk.Radiobutton(
                top, text=label,
                variable=self.model_type_var, value=val,
                command=self._on_model_type_change,
            ).pack(side="left", padx=6)

        # Notebook with Paddle / YOLO tabs
        self.notebook = _ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=4)

        self.panels: dict[str, _ModelPanel] = {}
        for mt, tab_text in [("rec", "Paddle REC"), ("det", "Paddle DET"), ("yolo_hbb", "YOLO HBB"), ("yolo_obb", "YOLO OBB")]:
            panel = _ModelPanel(self.notebook, mt)
            self.notebook.add(panel.frame, text=f"  {tab_text}  ")
            self.panels[mt] = panel

        # Buttons
        btn_row = _ttk.Frame(self.root, padding=(8, 6))
        btn_row.pack(fill="x")
        _ttk.Button(btn_row, text="Cancel", command=self.root.destroy, width=10).pack(
            side="right", padx=4
        )
        _ttk.Button(btn_row, text="Run", command=self._on_run, width=12).pack(
            side="right", padx=4
        )

        self._on_model_type_change()

    def _on_model_type_change(self):
        mt = self.model_type_var.get()
        # Enable only selected tab, except Paddle Both enables REC + DET.
        enabled = {"rec"} if mt == "rec" else {"det"} if mt == "det" else {"yolo_hbb"} if mt == "yolo_hbb" else {"yolo_obb"} if mt == "yolo_obb" else {"rec", "det"}
        tab_order = ["rec", "det", "yolo_hbb", "yolo_obb"]
        for i, name in enumerate(tab_order):
            self.notebook.tab(i, state="normal" if name in enabled else "disabled")
        self.notebook.select(tab_order.index(next(iter(enabled))))

    def _on_run(self):
        mt = self.model_type_var.get()
        targets = ["rec", "det"] if mt == "both" else [mt]

        errors = [self.panels[t].validate() for t in targets]
        errors = [e for e in errors if e]
        if errors:
            _messagebox.showerror("Validation Error", "\n".join(errors), parent=self.root)
            return

        self.result = [self.panels[t].get_config() for t in targets]
        self.root.destroy()

    def run(self) -> list[ModelConfig]:
        self.root.mainloop()
        return self.result


# ============================================================
# MENU
# ============================================================

def show_menu() -> str:
    print("\n#### Model lists ####")
    print("  1) rec")
    print("  2) det")
    print("  3) yolo_hbb")
    print("  4) yolo_obb")
    print("  5) paddle both")
    print("  6) exit")
    while True:
        choice = input("Enter choice: ").strip().lower()
        if choice in {"1", "rec"}:
            return "rec"
        if choice in {"2", "det"}:
            return "det"
        if choice in {"3", "yolo_hbb", "hbb"}:
            return "yolo_hbb"
        if choice in {"4", "yolo_obb", "obb"}:
            return "yolo_obb"
        if choice in {"5", "both"}:
            return "both"
        if choice in {"6", "exit", "q", "quit"}:
            return "exit"
        print("  Invalid input. Please enter 1, 2, 3, 4, 5, or 6.")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    run_id = generate_run_id()
    print(f"Run ID   : {run_id}")
    print(f"Platform : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"GUI      : {'tkinter available' if _TKINTER_AVAILABLE else 'not available — using terminal mode'}")

    if _TKINTER_AVAILABLE:
        try:
            app = TkConfigApp(run_id)
            configs = app.run()
        except Exception as e:
            print(f"GUI closed: {e}")
            return
        if not configs:
            print("Cancelled.")
            return
        for cfg in configs:
            _run_pipeline(cfg, run_id)
    else:
        while True:
            target = show_menu()
            if target == "exit":
                print("Exit program.")
                break
            if target in {"rec", "both"}:
                configure_and_generate("rec", run_id)
            if target in {"det", "both"}:
                configure_and_generate("det", run_id)
            if target == "yolo_hbb":
                configure_and_generate("yolo_hbb", run_id)
            if target == "yolo_obb":
                configure_and_generate("yolo_obb", run_id)
            print()


if __name__ == "__main__":
    main()
