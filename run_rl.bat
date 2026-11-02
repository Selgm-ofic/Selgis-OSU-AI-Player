@echo off
chcp 65001 >nul
title Selgis RL Lab
cd /d "%~dp0"
echo ========================================================
echo   Запуск симуляции и лаборатории RL Lab
echo ========================================================
echo.
python main.py rl %*
if errorlevel 1 (
    echo.
    echo [ОШИБКА] Не удалось запустить RL Lab
    pause
)
