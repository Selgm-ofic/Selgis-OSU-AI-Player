@echo off
chcp 65001 >nul
title osu! UNet Relax Live Bot (Mouse Aim Only)
cd /d "%~dp0"
python run_live.py --no-clicks
pause
