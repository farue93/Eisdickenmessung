@echo off
cd /d "%~dp0"
rem Erst das Projekt-venv, sonst der py-Starter, sonst System-Python.
rem schachbrett_drucken.py braucht KEINE Zusatzpakete, laeuft also ueberall.
set "PY=C:\Users\Fabi\Desktop\VSC260216\.venv\Scripts\python.exe"
if not exist "%PY%" (
  set "PY=python"
  where py >nul 2>&1 && set "PY=py"
)
"%PY%" schachbrett_drucken.py %*
pause
