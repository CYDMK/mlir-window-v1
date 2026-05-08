@echo off
setlocal enabledelayedexpansion
title BModel GUI Auto ENV Launcher

cd /d "%~dp0"

echo ==========================================
echo   BModel Conversion Auto ENV Launcher
echo ==========================================
echo Project folder: %cd%
echo.

REM ==========================================================
REM SCRIPT
REM ==========================================================

set SCRIPT=checkandauto.py

REM ==========================================================
REM ENV
REM ==========================================================

set DEFAULT_ENV=env_auto
set ENV_NAME=
set ENV_ACTIVATE=

echo [1] Scanning environment folders containing "env"...

set COUNT=0

for /d %%D in (*env*) do (
    set /a COUNT+=1
    echo     Found: %%D

    if exist "%%D\Scripts\activate.bat" (
        echo     [OK] activate.bat found in %%D

        call "%%D\Scripts\activate.bat"

        echo     Checking libraries in %%D...

        python -c "import ultralytics, onnx, yaml, numpy, tkinter, cv2; print('LIB_OK')" >nul 2>&1

        if not errorlevel 1 (
            set ENV_NAME=%%D
            set ENV_ACTIVATE=%%D\Scripts\activate.bat
            goto :env_ready
        ) else (
            echo     [SKIP] Missing libraries in %%D
        )
    )
)

echo.
echo [2] No complete env found. Creating new env: %DEFAULT_ENV%

if not exist "%DEFAULT_ENV%" (
    py -m venv "%DEFAULT_ENV%"

    if errorlevel 1 (
        echo [ERROR] Failed to create env.
        pause
        exit /b 1
    )
)

set ENV_NAME=%DEFAULT_ENV%
set ENV_ACTIVATE=%DEFAULT_ENV%\Scripts\activate.bat

echo.
echo [3] Activating new env...
call "%ENV_ACTIVATE%"

echo.
echo [4] Installing required libraries...

python -m pip install --upgrade pip

pip install ultralytics onnx pyyaml numpy opencv-python protobuf ml_dtypes typing_extensions

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install libraries.
    pause
    exit /b 1
)

goto :env_ready


:env_ready

echo.
echo [5] Environment selected:
echo     %ENV_NAME%

echo.
echo [6] Activating environment...

call "%ENV_ACTIVATE%"

echo     [OK] Activated

echo.
echo [7] Checking Python command...

python --version

if errorlevel 1 (
    echo     [ERROR] python not working
    pause
    exit /b 1
)

echo     [OK] Python ready

echo.
echo [8] Checking required libraries one by one...

python -c "import ultralytics; print('[OK] ultralytics / YOLO')" || goto :lib_error
python -c "import onnx; print('[OK] onnx')" || goto :lib_error
python -c "import yaml; print('[OK] yaml / PyYAML')" || goto :lib_error
python -c "import numpy; print('[OK] numpy')" || goto :lib_error
python -c "import tkinter; print('[OK] tkinter')" || goto :lib_error
python -c "import cv2; print('[OK] cv2 / opencv-python')" || goto :lib_error
python -c "import google.protobuf; print('[OK] protobuf')" || goto :lib_error
python -c "import ml_dtypes; print('[OK] ml_dtypes')" || goto :lib_error
python -c "import typing_extensions; print('[OK] typing_extensions')" || goto :lib_error

echo.
echo [9] Checking script file...

if not exist "%SCRIPT%" (
    echo     [ERROR] Cannot find %SCRIPT%
    echo.
    echo     Please put:
    echo         %SCRIPT%
    echo.
    echo     In this folder:
    echo         %cd%
    pause
    exit /b 1
)

echo     [OK] %SCRIPT% found

echo.
echo [10] Checking Docker command...

docker --version

if errorlevel 1 (
    echo     [ERROR] Docker command not found.
    echo.
    echo     Please install Docker Desktop.
    pause
    exit /b 1
)

echo     [OK] Docker command ready

echo.
echo [11] Checking Docker engine...

docker info >nul 2>&1

if errorlevel 1 (
    echo     [WARN] Docker engine is not running.
    echo.
    echo     Trying to open Docker Desktop...

    start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"

    echo.
    echo     Waiting Docker Desktop 60 seconds...

    timeout /t 60 /nobreak

    docker info >nul 2>&1

    if errorlevel 1 (
        echo     [ERROR] Docker engine still not running.
        pause
        exit /b 1
    )
)

echo     [OK] Docker engine running

echo.
echo [12] Checking container model_conversion...

docker ps -a --format "{{.Names}}" | findstr /x "model_conversion" >nul

if errorlevel 1 (
    echo     [WARN] Container model_conversion not found.
    echo     Python script will create it automatically.
) else (
    echo     [OK] Container model_conversion exists
)

echo.
echo [13] Starting GUI...
echo.

python "%SCRIPT%"

echo.
echo ==========================================
echo Program finished.
echo ==========================================

pause
exit /b 0


:lib_error

echo.
echo [ERROR] Missing library in selected env:
echo     %ENV_NAME%

echo.
echo Try installing:

echo     pip install ultralytics onnx pyyaml numpy opencv-python protobuf ml_dtypes typing_extensions

pause
exit /b 1