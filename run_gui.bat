@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ===================================================
echo   Запуск Osu! AI Assistant (v8 Engine)
echo ===================================================
python main.py gui
pause
