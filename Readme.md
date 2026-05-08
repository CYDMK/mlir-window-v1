# TPU-MLIR Auto Converter Developer Guide

เครื่องมือนี้ใช้สำหรับแปลงโมเดลจาก ONNX / YOLO `.pt` ไปเป็น `.bmodel` สำหรับใช้งานบน Sophgo TPU ผ่าน TPU-MLIR

รองรับ:

* PaddleOCR DET
* PaddleOCR REC
* YOLO HBB
* YOLO OBB

---

# 1. Pipeline Overview

## PaddleOCR

```text
PaddleOCR ONNX
↓
Read YAML Config
↓
Extract Preprocess
↓
model_transform
↓
model_deploy
↓
BMODEL
```

## YOLO

```text
YOLO .pt
↓
Ultralytics export ONNX
↓
Save ONNX copy
↓
If YOLO OBB: transpose output
↓
model_transform
↓
model_deploy
↓
BMODEL
```

---

# 2. Main Features

* Auto export YOLO `.pt` เป็น `.onnx`
* Auto convert ONNX → MLIR → BMODEL
* Auto start Docker Desktop
* Auto create Docker container
* Auto use `/workspace`
* Auto copy `.bmodel` from Docker to host
* Auto open result folder after finished
* Save logs inside each run folder
* Generate `commands.bat` / `commands.txt`
* YOLO OBB output transpose:

  * from `1 x 6 x 8400`
  * to `1 x 8400 x 6`

---

# 3. Project Structure

```text
model_conversion/
├── checkandauto_auto_open_folder.py
├── open_gui.bat
├── env_auto/
└── results/
    └── 20260508_XXXXXX_yolo_obb/
        ├── best.onnx
        ├── best.mlir
        ├── best_bm1684x_f16.bmodel
        ├── commands.bat
        └── logs/
            ├── 20260508_XXXXXX_yolo_obb_pt_to_onnx.log
            ├── 20260508_XXXXXX_yolo_obb_transform.log
            ├── 20260508_XXXXXX_yolo_obb_deploy.log
            └── 20260508_XXXXXX_yolo_obb_timing.log
```

---

# 4. Required Environment

## Python

แนะนำ Python 3.9 - 3.11

## Python Packages

```bash
pip install ultralytics onnx pyyaml numpy opencv-python protobuf ml_dtypes typing_extensions
```

## Docker

ต้องติดตั้ง Docker Desktop และใช้ image:

```bash
docker pull sophgo/tpuc_dev:latest
```

Container name ที่ script ใช้:

```text
model_conversion
```

Docker workdir:

```text
/workspace
```

---

# 5. Main Constants

```python
QUANTIZE_OPTIONS = ["F32", "F16", "BF16", "INT8", "INT4"]
QUANTIZE_NEEDS_CALIB = {"INT8", "INT4"}

PROCESSOR_OPTIONS = ["bm1684x", "bm1688", "cv186x"]

DEFAULT_INPUT_SHAPE = [1, 3, 640, 640]
DEFAULT_CALIB_INPUT_NUM = 100

DOCKER_CONTAINER_NAME = "model_conversion"
DOCKER_IMAGE = "sophgo/tpuc_dev:latest"
DOCKER_WORKDIR = "/workspace"
```

---

# 6. Important Data Class

```python
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
    calib_mode: str
    calibration_table: Optional[str]
    calib_dataset: Optional[Path]
    calib_input_num: int
    yolo_pt_path: Optional[Path] = None
```

ใช้เก็บค่าทั้งหมดของโมเดลก่อนส่งเข้า `_run_pipeline()`

---

# 7. Supported Model Types

```python
def is_yolo_model(model_type: str) -> bool:
    return model_type in {"yolo_hbb", "yolo_obb"}
```

Model type ที่ใช้ใน GUI / terminal:

```text
det
rec
yolo_hbb
yolo_obb
both
```

---

# 8. YOLO Export Logic

ฟังก์ชันหลัก:

```python
export_yolo_pt_to_onnx()
```

หน้าที่:

1. รับ `.pt`
2. ใช้ `ultralytics.YOLO`
3. export เป็น `.onnx`
4. copy ONNX ไปไว้ใน run folder
5. ถ้าเป็น YOLO OBB จะเพิ่ม Transpose node

คำสั่ง export ภายใน:

```python
model.export(
    format="onnx",
    imgsz=640,
    opset=12,
    simplify=False,
    dynamic=False
)
```

---

# 9. YOLO OBB Transpose Logic

เฉพาะ `yolo_obb` จะทำ:

```text
1 x 6 x 8400
↓
1 x 8400 x 6
```

ผ่าน ONNX node:

```python
Transpose(
    perm=[0, 2, 1]
)
```

Node name:

```text
Transpose_YOLO_OBB_Output
```

Output ใหม่:

```text
output_transpose
```

HBB จะไม่ transpose

---

# 10. Docker Logic

## Check Docker

```python
ensure_docker_ready()
```

ทำหน้าที่:

* ตรวจว่า `docker` command มีไหม
* ตรวจว่า Docker daemon running ไหม
* ถ้า Windows แล้ว Docker ยังไม่เปิด จะเปิด Docker Desktop ให้เอง
* รอ Docker start ตาม timeout

## Create Container

```python
ensure_tpumlir_container()
```

ถ้าไม่มี container จะสร้าง:

```bash
docker run -dit --privileged ^
  --name model_conversion ^
  -v <project_folder>:/workspace ^
  sophgo/tpuc_dev:latest ^
  bash
```

---

# 11. Host to Container Path Mapping

ฟังก์ชัน:

```python
host_to_container_path()
```

ใช้แปลง path จาก Windows เช่น:

```text
D:\BUU\model_conversion\results\xxx\best.onnx
```

เป็น path ใน Docker:

```text
/workspace/results/xxx/best.onnx
```

ถ้า map ไม่ได้ script จะใช้:

```python
DOCKER_WORKDIR = "/workspace"
```

อัตโนมัติ

---

# 12. TPU-MLIR Commands

## model_transform

สร้างโดย:

```python
_transform_argv()
```

ตัวอย่าง:

```bash
model_transform \
  --model_name yolo_obb \
  --model_def /workspace/results/xxx/best.onnx \
  --input_shapes [[1,3,640,640]] \
  --mean 0,0,0 \
  --scale 0.003921568627,0.003921568627,0.003921568627 \
  --pixel_format rgb \
  --mlir best.mlir
```

## model_deploy

สร้างโดย:

```python
_deploy_argv()
```

ตัวอย่าง:

```bash
model_deploy \
  --mlir best.mlir \
  --quantize F16 \
  --processor bm1684x \
  --model best_bm1684x_f16.bmodel
```

---

# 13. Quantization

## F16 / F32 / BF16

ไม่ต้องใช้ calibration

## INT8 / INT4

ต้องใช้ calibration table หรือ dataset

ถ้าเลือก `run calibration` จะเรียก:

```bash
run_calibration
```

---

# 14. Logs

Logs จะถูกเก็บใน run folder:

```text
results/<run_id>_<model_type>/logs/
```

ตัวอย่าง:

```text
logs/
├── pt_to_onnx.log
├── transform.log
├── deploy.log
├── calibration.log
└── timing.log
```

---

# 15. Timing Report

หลังจบ script จะสร้าง timing log:

```text
=== Timing Report ===
Step              Duration
model_transform   00:00:15
model_deploy      00:00:40
TOTAL             00:00:55
```

---

# 16. Output Files

หลัง convert สำเร็จจะได้:

```text
best.onnx
best.mlir
best_bm1684x_f16.bmodel
commands.bat
logs/
```

และ script จะเปิด result folder ให้อัตโนมัติ

---

# 17. GUI Flow

GUI ใช้ `tkinter`

Class หลัก:

```python
TkConfigApp
_ModelPanel
```

แต่ละ panel ใช้เก็บ:

* model path
* input shape
* quantize
* calibration
* processor
* keep aspect ratio
* output names

---

# 18. Developer Notes

## จุดที่ควรแก้ถ้าจะเพิ่ม model type ใหม่

1. เพิ่ม model type ใน GUI:

```python
("new_model", "New Model")
```

2. เพิ่ม preprocess function

```python
get_new_model_preprocess()
```

3. เพิ่ม logic ใน `_run_pipeline()`

```python
if model_type == "new_model":
    ...
```

---

# 19. Common Errors

## Docker not running

แก้โดยเปิด Docker Desktop หรือให้ script เปิดให้เอง

## model_transform not found

แปลว่า container ไม่ใช่ TPU-MLIR image ที่ถูกต้อง

แก้:

```bash
docker rm -f model_conversion
docker pull sophgo/tpuc_dev:latest
```

## ONNX output shape ไม่ตรง

สำหรับ YOLO OBB ให้เช็คใน Netron ว่า output เป็น:

```text
output_transpose
1 x 8400 x 6
```

## INT8 deploy fail

มักเกิดจากไม่มี calibration table

---

# 20. Recommended Dev Improvements

* Add model info export เป็น `model_info.txt`
* Add bmodel verification ด้วย `model_tool --info`
* Add checkbox สำหรับ dynamic ONNX export
* Add option save transposed ONNX เป็นไฟล์ใหม่
* Add progress bar ใน GUI
* Add auto install missing packages

---

# 21. Run Command

```bash
python checkandauto.py
```

หรือผ่าน `.bat`:

```bat
set SCRIPT=checkandauto.py
python "%SCRIPT%"
```

---

# 22. Summary

เครื่องมือนี้ช่วยให้ Dev แปลงโมเดลเป็น `.bmodel` ได้ครบในขั้นตอนเดียว:

```text
Select model
↓
Export ONNX if needed
↓
Patch ONNX if OBB
↓
Run TPU-MLIR
↓
Save logs
↓
Copy bmodel
↓
Open result folder
```
