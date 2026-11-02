@echo off
chcp 65001 >nul
title osu! UNet Policy Live Bot
cd /d "%~dp0"
echo ========================================================
echo   Запуск нейросетевого бота osu! (Модель: UNet)
echo ========================================================
echo.
python run_live.py
if errorlevel 1 (
    echo.
    echo [ОШИБКА] Не удалось запустить скрипт.
    echo Убедитесь, что Python установлен и зависимости установлены:
    echo    pip install -r requirements.txt
    echo.
)
pause
