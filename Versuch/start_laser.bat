@echo off
cd /d "%~dp0"
set "PY=C:\Users\Fabi\Desktop\VSC260216\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" live_laser.py
if errorlevel 1 pause
