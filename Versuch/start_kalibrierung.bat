@echo off
cd /d "%~dp0"
rem Erst das Projekt-venv, sonst der py-Starter, sonst System-Python.
set "PY=C:\Users\Fabi\Desktop\VSC260216\.venv\Scripts\python.exe"
if not exist "%PY%" (
  set "PY=python"
  where py >nul 2>&1 && set "PY=py"
)
"%PY%" kalibrierung_schachbrett.py %*
if errorlevel 1 (
  echo.
  echo Fehlt ein Paket? Dann:  "%PY%" -m pip install numpy opencv-python
)
pause
