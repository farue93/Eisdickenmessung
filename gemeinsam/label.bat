@echo off
cd /d "%~dp0"
set "PY=C:\Users\Fabi\Desktop\VSC260216\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" label_tool.py
if errorlevel 1 (
  echo.
  echo Falls "namedWindow not implemented": headless-OpenCV aktiv. Fix:
  echo   "%PY%" -m pip uninstall -y opencv-python-headless
  echo   "%PY%" -m pip install opencv-python==4.13.0.92
)
pause
