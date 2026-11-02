"""
Semga Inference App — Автоматический установщик зависимостей
"""
import os
import shutil
import subprocess
import sys
import time

def print_banner():
    print("=" * 70)
    print("      SELGIS OSU AI PLAYER — УСТАНОВКА ЗАВИСИМОСТЕЙ")
    print("=" * 70)
    print(f"[INFO] Python: {sys.version.split()[0]} ({sys.executable})")
    print()

def run_pip(args, desc):
    print(f"--> {desc}...")
    cmd = [sys.executable, "-m", "pip"] + args
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"[!] Предупреждение при выполнении: {' '.join(cmd)}")
    return res.returncode == 0

def has_nvidia_gpu():
    if shutil.which("nvidia-smi"):
        try:
            res = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return res.returncode == 0
        except Exception:
            return False
    return False

def main():
    print_banner()

    # 1. Upgrade pip
    print("[1/5] Обновление пакетного менеджера pip...")
    run_pip(["install", "--upgrade", "pip", "--quiet"], "Обновление pip")
    print("[OK] Pip готов к работе.\n")

    # 2. Check NVIDIA GPU and install PyTorch
    print("[2/5] Проверка графического процессора и установка PyTorch...")
    has_gpu = has_nvidia_gpu()
    if has_gpu:
        print("[INFO] Обнаружена видеокарта NVIDIA!")
        print("[INFO] Устанавливаем PyTorch с поддержкой аппаратного ускорения CUDA (cu121)...")
        run_pip(["install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu121"], "Установка PyTorch (CUDA)")
    else:
        print("[INFO] Видеокарта NVIDIA не обнаружена.")
        print("[INFO] Устанавливаем стандартную версию PyTorch (CPU режим)...")
        run_pip(["install", "torch", "torchvision"], "Установка PyTorch (CPU)")
    print()

    # 3. Install core libraries
    print("[3/5] Установка вспомогательных библиотек (OpenCV, MSS, DXCam, Pillow, PyYAML, PyWin32)...")
    core_pkgs = [
        "opencv-python>=4.5.0",
        "numpy>=1.23.0",
        "pillow>=9.0.0",
        "pyyaml>=6.0",
        "mss>=9.0.0",
        "dxcam>=0.1.1",
        "pywin32>=306"
    ]
    run_pip(["install"] + core_pkgs, "Установка зависимостей интерфейса и захвата")
    print()

    # 4. Install Selgis ML framework
    print("[4/5] Установка фреймворка машинного обучения Selgis-ML...")
    try:
        import selgis
        print(f"[OK] Selgis-ML уже установлен (версия {getattr(selgis, '__version__', 'dev')})")
    except ImportError:
        print("[INFO] Selgis-ML не найден. Устанавливаем с GitHub (https://github.com/Selgm-ofic/Selgis-ML)...")
        run_pip(["install", "git+https://github.com/Selgm-ofic/Selgis-ML.git"], "Установка Selgis-ML с GitHub")
    print()

    # 5. Self-test
    print("[5/5] Итоговое тестирование установленных компонентов...")
    try:
        import torch
        import cv2
        import PIL
        import yaml
        import mss
        import dxcam
        import win32gui
        import selgis

        cuda_ok = torch.cuda.is_available()
        device_str = torch.cuda.get_device_name(0) if cuda_ok else "CPU режим"

        print("=" * 70)
        print("  [УСПЕХ] ВСЕ ЗАВИСИМОСТИ УСПЕШНО УСТАНОВЛЕНЫ И ГОТОВЫ К РАБОТЕ!")
        print(f"  • PyTorch:    {torch.__version__}")
        print(f"  • Аппаратное ускорение: {'ВКЛЮЧЕНО (GPU: ' + device_str + ')' if cuda_ok else 'ВЫКЛЮЧЕНО (CPU)'}")
        print(f"  • Selgis-ML:  {getattr(selgis, '__version__', 'OK')}")
        print(f"  • OpenCV:     {cv2.__version__}")
        print(f"  • Pillow:     {PIL.__version__}")
        print("=" * 70)
        print("\nТеперь вы можете закрыть это окно и запустить:")
        print("  ==> run_player_gui.bat (Нормис GUI)")
        print("  ==> run_gui.bat        (R&D GUI)\n")
    except Exception as e:
        print("=" * 70)
        print(f"  [ОШИБКА] При проверке библиотек возникло исключение: {e}")
        print("  Попробуйте запустить скрипт от имени Администратора.")
        print("=" * 70)

    input("Нажмите Enter для выхода...")

if __name__ == "__main__":
    main()
