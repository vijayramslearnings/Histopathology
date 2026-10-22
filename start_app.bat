@echo off
title PathoVision AI - Digital Pathology Suite
cd /d "%~dp0"

echo ======================================================================
echo    Starting PathoVision AI Suite (Backend + Frontend)
echo ======================================================================
echo.

python start_app.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] An error occurred while running start_app.py.
    echo Ensure Python is installed and added to PATH.
    pause
)
