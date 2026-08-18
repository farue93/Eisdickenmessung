@echo off
cd /d "%~dp0"
set "PY=C:\Users\Fabi\Desktop\VSC260216\.venv\Scripts\python.exe"
if exist "%PY%" goto start
set "PY=py"
where /q py && goto start
set "PY=python"
:start
"%PY%" messung_flaeche.py
if errorlevel 1 pause
