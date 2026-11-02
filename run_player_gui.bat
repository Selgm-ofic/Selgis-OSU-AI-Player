@echo off
chcp 65001 >nul
title Selgis Player GUI (Normis Edition)
cd /d "%~dp0"
echo ========================================================
echo   Запуск легкого интерфейса плеера (Normis GUI)
echo ========================================================
echo.
python normis_gui.py %*
if errorlevel 1 (
    echo.
    echo [ОШИБКА] Не удалось запустить normis_gui.py
    pause
)
