@echo off
cd /d "%~dp0"
set "PY=C:\Users\Fabi\Desktop\VSC260216\.venv\Scripts\python.exe"
if exist "%PY%" goto start
set "PY=py"
where /q py && goto start
set "PY=python"
:start
echo ==== Module, Modelle, Rechenwerk, Kalibrierkette ====
"%PY%" trockentest.py module
echo.
echo ==== Ordnerwache ====
"%PY%" trockentest.py wache
echo.
echo Fenster mit einer Taste schliessen.
pause >nul
