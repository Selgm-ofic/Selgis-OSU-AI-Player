@echo off
setlocal
cd /d "%~dp0"
title Semga Live (Preview Mode)

set "PY_CMD="

python --version >nul 2>&1
if not errorlevel 1 (
    set "PY_CMD=python"
    goto :found_python
)

py --version >nul 2>&1
if not errorlevel 1 (
    set "PY_CMD=py"
    goto :found_python
)

if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set "PY_CMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    goto :found_python
)
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set "PY_CMD=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    goto :found_python
)
if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
    set "PY_CMD=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    goto :found_python
)
if exist "C:\Python312\python.exe" (
    set "PY_CMD=C:\Python312\python.exe"
    goto :found_python
)
if exist "C:\Python311\python.exe" (
    set "PY_CMD=C:\Python311\python.exe"
    goto :found_python
)
if exist "C:\Python310\python.exe" (
    set "PY_CMD=C:\Python310\python.exe"
    goto :found_python
)

echo ======================================================================
echo  [ERROR] Python is not installed or not added to PATH.
echo ======================================================================
echo  Please install Python 3.11: https://www.python.org/downloads/
echo  IMPORTANT: Check "Add python.exe to PATH" during installation.
echo ======================================================================
echo.
pause
exit /b 1

:found_python
echo ========================================================
echo   Starting Semga Live in preview mode...
echo ========================================================
echo.

"%PY_CMD%" "%~dp0run_live.py" --preview %*
if errorlevel 1 (
    echo.
    echo ======================================================================
    echo  [ERROR] Application exited with error code %errorlevel%.
    echo ======================================================================
    echo.
    pause
    exit /b %errorlevel%
)
