"""
osu! UNet Policy Live Runner with Interactive Console Setup & Argument Support.
"""
from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

# Ensure stdout and stderr support UTF-8 safely on Windows console
if sys.platform == "win32":
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import yaml

from osu_bot.live import RelaxLive

try:
    import win32api
    import win32gui
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False

try:
    import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False

try:
    import dxcam
    HAS_DXCAM = True
except ImportError:
    HAS_DXCAM = False

try:
    import msvcrt
    HAS_MSVCRT = True
except ImportError:
    HAS_MSVCRT = False


DEFAULT_CONFIG_NAME = "config.yaml"


def detect_monitors() -> List[Dict]:
    monitors: List[Dict] = []

    if HAS_WIN32:
        try:
            enum_mons = win32api.EnumDisplayMonitors()
            for idx, hmon in enumerate(enum_mons, start=1):
                info = win32api.GetMonitorInfo(hmon[0])
                rect = info.get("Monitor", (0, 0, 1920, 1080))
                is_primary = bool(info.get("Flags", 0) & 1)
                dev_name = info.get("Device", f"\\\\.\\DISPLAY{idx}")
                friendly_name = "Generic Monitor"
                try:
                    dev = win32api.EnumDisplayDevices(dev_name, 0)
                    friendly_name = dev.DeviceString or "Generic Monitor"
                except Exception:
                    pass

                width = rect[2] - rect[0]
                height = rect[3] - rect[1]
                monitors.append({
                    "index": idx,
                    "name": friendly_name,
                    "device": dev_name,
                    "width": width,
                    "height": height,
                    "left": rect[0],
                    "top": rect[1],
                    "right": rect[2],
                    "bottom": rect[3],
                    "is_primary": is_primary,
                })
            if monitors:
                return monitors
        except Exception:
            pass

    if HAS_MSS:
        try:
            with mss.MSS() as sct:
                for idx, m in enumerate(sct.monitors):
                    if idx == 0:
                        continue
                    monitors.append({
                        "index": idx,
                        "name": m.get("name", f"Display {idx}"),
                        "device": f"Display {idx}",
                        "width": m.get("width", 1920),
                        "height": m.get("height", 1080),
                        "left": m.get("left", 0),
                        "top": m.get("top", 0),
                        "right": m.get("left", 0) + m.get("width", 1920),
                        "bottom": m.get("top", 0) + m.get("height", 1080),
                        "is_primary": bool(m.get("is_primary", idx == 1)),
                    })
            if monitors:
                return monitors
        except Exception:
            pass

    return [{
        "index": 1,
        "name": "Primary Display",
        "device": "\\\\.\\DISPLAY1",
        "width": 1920,
        "height": 1080,
        "left": 0,
        "top": 0,
        "right": 1920,
        "bottom": 1080,
        "is_primary": True,
    }]


def find_osu_window_rect() -> Optional[Tuple[int, int, int, int]]:
    if not HAS_WIN32:
        return None

    candidates = []

    def enum_cb(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if "osu!" in title:
                candidates.append((hwnd, title))
        return True

    try:
        win32gui.EnumWindows(enum_cb, None)
    except Exception:
        return None

    if not candidates:
        return None

    target_hwnd = candidates[0][0]
    try:
        client_rect = win32gui.GetClientRect(target_hwnd)
        p1 = win32gui.ClientToScreen(target_hwnd, (client_rect[0], client_rect[1]))
        p2 = win32gui.ClientToScreen(target_hwnd, (client_rect[2], client_rect[3]))
        left, top, right, bottom = p1[0], p1[1], p2[0], p2[1]
        if (right - left) >= 100 and (bottom - top) >= 100:
            return left, top, right, bottom
    except Exception:
        return None
    return None


def get_available_models(base_dir: Path) -> List[Path]:
    models: dict[str, Path] = {}
    for candidate_dir in [base_dir / "data" / "models", base_dir / "models"]:
        if candidate_dir.is_dir():
            for p in sorted(candidate_dir.glob("*.pt"), key=lambda x: x.name):
                if p.name not in models:
                    models[p.name] = p
    return [models[k] for k in sorted(models.keys())]


def load_configuration(cfg_path: Path) -> dict:
    default_cfg = {
        "dxgi": {"monitor": 1},
        "playfield": {"left": 295, "top": 35, "right": 1625, "bottom": 1075},
        "runtime": {"target_fps": 60},
        "settings": {
            "selected_model": "unet_policy_v8_5.pt",
            "clicks_enabled": True,
            "confidence": 0.60,
            "smooth": 0.0,
        },
    }
    if not cfg_path.exists():
        return default_cfg

    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            return default_cfg

        for section, sec_data in default_cfg.items():
            if section not in loaded or not isinstance(loaded[section], dict):
                loaded[section] = sec_data
            else:
                for k, v in sec_data.items():
                    if k not in loaded[section]:
                        loaded[section][k] = v
        return loaded
    except Exception:
        return default_cfg


def save_configuration(cfg_path: Path, cfg: dict):
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def resolve_model_path(base_dir: Path, requested: Optional[str]) -> Path:
    search_dirs = [base_dir / "data" / "models", base_dir / "models"]
    if requested:
        p = Path(requested)
        if p.is_file():
            return p
        for sdir in search_dirs:
            cand = sdir / requested
            if cand.is_file():
                return cand

    candidates = []
    for sdir in search_dirs:
        candidates.extend([
            sdir / "unet_policy_v8_5.pt",
            sdir / "unet_policy_v8_5_best_xy.pt",
            sdir / "unet_policy_v8_2.pt",
            sdir / "unet_policy_v8_2_best_xy.pt",
            sdir / "unet_policy_v7_2.pt",
        ])
    for c in candidates:
        if c.is_file():
            return c

    all_models = get_available_models(base_dir)
    if all_models:
        return all_models[0]

    raise FileNotFoundError(f"Model weights not found. Searched in: {search_dirs}")


def print_header(title: str = "Semga Live -- Bazyach CLI Runner"):
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def display_current_settings(cfg: dict, weights_path: Path):
    d = cfg.get("dxgi", {})
    pf = cfg.get("playfield", {})
    sett = cfg.get("settings", {})
    rt = cfg.get("runtime", {})

    mon = d.get("monitor", 1)
    fl, ft = pf.get("left", 0), pf.get("top", 0)
    fr, fb = pf.get("right", 1920), pf.get("bottom", 1080)
    w = max(0, fr - fl)
    h = max(0, fb - ft)
    clicks = sett.get("clicks_enabled", True)
    conf = sett.get("confidence", 0.60)
    smooth = sett.get("smooth", 0.0)
    fps = rt.get("target_fps", 60)

    click_mode = "ON (Z / X active)" if clicks else "OFF (Relax, mouse aim only)"

    print("\n[CURRENT CONFIGURATION]:")
    print(f"  - Model:         {weights_path.name}")
    print(f"  - Monitor Index: {mon}")
    print(f"  - Playfield ROI: [{fl}, {ft}, {fr}, {fb}] (size: {w}x{h} px)")
    print(f"  - Clicks (Z/X):  {click_mode}")
    print(f"  - Confidence:    {conf:.2f}")
    print(f"  - Smoothing:     {smooth:.2f}")
    print(f"  - Target FPS:    {fps} FPS")


def interactive_select_monitor(cfg: dict) -> bool:
    print("\n--- [MONITOR DETECTION AND SELECTION] ---")
    monitors = detect_monitors()
    current_idx = cfg.get("dxgi", {}).get("monitor", 1)

    print(f"Found monitors: {len(monitors)}\n")
    for m in monitors:
        active_mark = " [*ACTIVE*]" if m["index"] == current_idx else ""
        primary_mark = " (Primary)" if m["is_primary"] else ""
        print(f"  [{m['index']}] {m['name']} - {m['width']}x{m['height']} "
              f"at ({m['left']}, {m['top']}){primary_mark}{active_mark}")

    print("\nEnter monitor index to capture (or press Enter to cancel): ", end="", flush=True)
    choice = input().strip()
    if not choice:
        return False

    try:
        val = int(choice)
        available_indices = [m["index"] for m in monitors]
        if val in available_indices:
            cfg.setdefault("dxgi", {})["monitor"] = val
            print(f"[OK] Monitor {val} selected.")
            return True
        else:
            print(f"[!] Invalid monitor index. Available: {available_indices}")
    except ValueError:
        print("[!] Input was not a number.")
    return False


def interactive_toggle_clicks(cfg: dict) -> bool:
    sett = cfg.setdefault("settings", {})
    cur = sett.get("clicks_enabled", True)
    sett["clicks_enabled"] = not cur
    status = "ENABLED (Z/X active)" if sett["clicks_enabled"] else "DISABLED (Relax: mouse aim only)"
    print(f"\n[OK] Clicks are now: {status}")
    return True


def interactive_adjust_confidence(cfg: dict) -> bool:
    sett = cfg.setdefault("settings", {})
    cur = sett.get("confidence", 0.60)
    print("\n--- [CONFIDENCE THRESHOLD SETTING] ---")
    print(f"Current value: {cur:.2f} (recommended: 0.40 - 0.80, default: 0.60)")
    print("Enter new value [0.10 - 1.00] (or press Enter to cancel): ", end="", flush=True)
    val_str = input().strip()
    if not val_str:
        return False
    try:
        val = float(val_str.replace(",", "."))
        if 0.05 <= val <= 1.0:
            sett["confidence"] = round(val, 2)
            print(f"[OK] Confidence threshold set to: {sett['confidence']:.2f}")
            return True
        else:
            print("[!] Value must be between 0.10 and 1.00.")
    except ValueError:
        print("[!] Invalid number format.")
    return False


def interactive_adjust_fps(cfg: dict) -> bool:
    rt = cfg.setdefault("runtime", {})
    cur = int(rt.get("target_fps", 60))
    print("\n--- [TARGET FPS REGULATION] ---")
    print(f"Current target FPS: {cur}")
    print("Presets:")
    print("  1. 30 FPS   (Low-spec laptops)")
    print("  2. 60 FPS   (Standard 60Hz)")
    print("  3. 120 FPS  (High-refresh 120Hz)")
    print("  4. 144 FPS  (Gaming 144Hz)")
    print("  5. 240 FPS  (Maximum performance)")
    print("  6. Custom FPS value")
    print("  0. Cancel")
    print("Select option [0-6] (or enter FPS directly): ", end="", flush=True)
    val_str = input().strip()
    if not val_str or val_str == "0":
        return False

    preset_map = {"1": 30, "2": 60, "3": 120, "4": 144, "5": 240}
    if val_str in preset_map:
        rt["target_fps"] = preset_map[val_str]
        print(f"[OK] Target FPS set to: {rt['target_fps']}")
        return True

    if val_str == "6":
        print("Enter custom target FPS [10 - 360]: ", end="", flush=True)
        val_str = input().strip()

    try:
        val = int(val_str)
        if 10 <= val <= 360:
            rt["target_fps"] = val
            print(f"[OK] Target FPS set to: {rt['target_fps']}")
            return True
        else:
            print("[!] FPS must be between 10 and 360.")
    except ValueError:
        print("[!] Invalid number format.")
    return False


def interactive_adjust_smooth(cfg: dict) -> bool:
    sett = cfg.setdefault("settings", {})
    cur = sett.get("smooth", 0.0)
    print("\n--- [MOUSE SMOOTHING FACTOR] ---")
    print(f"Current value: {cur:.2f} [0.0 = Raw input, 0.5 = High smoothing]")
    print("Enter new value [0.00 - 0.90] (or press Enter to cancel): ", end="", flush=True)
    val_str = input().strip()
    if not val_str:
        return False
    try:
        val = float(val_str.replace(",", "."))
        if 0.0 <= val <= 0.90:
            sett["smooth"] = round(val, 2)
            print(f"[OK] Smoothing factor set to: {sett['smooth']:.2f}")
            return True
        else:
            print("[!] Value must be between 0.0 and 0.90.")
    except ValueError:
        print("[!] Invalid number format.")
    return False


def interactive_select_model(base_dir: Path, cfg: dict) -> bool:
    print("\n--- [NEURAL NETWORK MODEL SELECTION] ---")
    models = get_available_models(base_dir)
    if not models:
        print("[!] No model weights (.pt) found in models/ directory.")
        return False

    sett = cfg.setdefault("settings", {})
    current_name = sett.get("selected_model", "")

    for idx, p in enumerate(models, start=1):
        active_mark = " [*SELECTED*]" if p.name == current_name else ""
        print(f"  [{idx}] {p.name}{active_mark}")

    print("\nEnter model index (or press Enter to cancel): ", end="", flush=True)
    choice = input().strip()
    if not choice:
        return False

    try:
        idx = int(choice)
        if 1 <= idx <= len(models):
            chosen = models[idx - 1]
            sett["selected_model"] = chosen.name
            print(f"[OK] Selected model: {chosen.name}")
            return True
        else:
            print("[!] Invalid model index.")
    except ValueError:
        print("[!] Input was not a number.")
    return False


def interactive_calibrate_roi(cfg: dict) -> bool:
    pf = cfg.setdefault("playfield", {})
    cur_l = pf.get("left", 295)
    cur_t = pf.get("top", 35)
    cur_r = pf.get("right", 1625)
    cur_b = pf.get("bottom", 1075)

    while True:
        w = max(0, cur_r - cur_l)
        h = max(0, cur_b - cur_t)
        print("\n--- [ROI CALIBRATION (osu! Playfield Bounds)] ---")
        print(f"Current bounds: Left={cur_l}, Top={cur_t}, Right={cur_r}, Bottom={cur_b} ({w}x{h} px)")
        print("  1. Auto-detect active osu! game window (Client rect via win32gui)")
        print("  2. Preset: 1920x1080 osu! 4:3 Fullscreen [295, 35, 1625, 1075]")
        print("  3. Preset: 1920x1080 Fullscreen 16:9 [0, 0, 1920, 1080]")
        print("  4. Preset: 2560x1440 osu! 4:3 Fullscreen [393, 47, 2167, 1433]")
        print("  5. Preset: 2560x1440 Fullscreen 16:9 [0, 0, 2560, 1440]")
        print("  6. Enter coordinates manually (Left, Top, Right, Bottom)")
        print("  0. Back to main menu")
        print("Select option [0-6]: ", end="", flush=True)

        choice = input().strip()
        if choice == "0" or not choice:
            break
        elif choice == "1":
            coords = find_osu_window_rect()
            if coords is not None:
                cur_l, cur_t, cur_r, cur_b = coords
                pf["left"], pf["top"], pf["right"], pf["bottom"] = cur_l, cur_t, cur_r, cur_b
                print(f"[OK] osu! window detected! Coordinates updated: [{cur_l}, {cur_t}, {cur_r}, {cur_b}]")
                return True
            else:
                print("[!] osu! window was not found among visible Windows applications.")
                print("    Make sure osu! is running and not minimized.")
        elif choice == "2":
            cur_l, cur_t, cur_r, cur_b = 295, 35, 1625, 1075
            pf["left"], pf["top"], pf["right"], pf["bottom"] = cur_l, cur_t, cur_r, cur_b
            print("[OK] Applied preset: 1920x1080 (4:3 Standard).")
            return True
        elif choice == "3":
            cur_l, cur_t, cur_r, cur_b = 0, 0, 1920, 1080
            pf["left"], pf["top"], pf["right"], pf["bottom"] = cur_l, cur_t, cur_r, cur_b
            print("[OK] Applied preset: 1920x1080 (16:9 Full).")
            return True
        elif choice == "4":
            cur_l, cur_t, cur_r, cur_b = 393, 47, 2167, 1433
            pf["left"], pf["top"], pf["right"], pf["bottom"] = cur_l, cur_t, cur_r, cur_b
            print("[OK] Applied preset: 2560x1440 (4:3 Standard).")
            return True
        elif choice == "5":
            cur_l, cur_t, cur_r, cur_b = 0, 0, 2560, 1440
            pf["left"], pf["top"], pf["right"], pf["bottom"] = cur_l, cur_t, cur_r, cur_b
            print("[OK] Applied preset: 2560x1440 (16:9 Full).")
            return True
        elif choice == "6":
            try:
                print(f"Left [{cur_l}]: ", end="", flush=True)
                l_s = input().strip()
                new_l = int(l_s) if l_s else cur_l

                print(f"Top [{cur_t}]: ", end="", flush=True)
                t_s = input().strip()
                new_t = int(t_s) if t_s else cur_t

                print(f"Right [{cur_r}]: ", end="", flush=True)
                r_s = input().strip()
                new_r = int(r_s) if r_s else cur_r

                print(f"Bottom [{cur_b}]: ", end="", flush=True)
                b_s = input().strip()
                new_b = int(b_s) if b_s else cur_b

                if new_r > new_l + 50 and new_b > new_t + 50:
                    cur_l, cur_t, cur_r, cur_b = new_l, new_t, new_r, new_b
                    pf["left"], pf["top"], pf["right"], pf["bottom"] = cur_l, cur_t, cur_r, cur_b
                    print(f"[OK] Custom coordinates set: [{cur_l}, {cur_t}, {cur_r}, {cur_b}]")
                    return True
                else:
                    print("[!] Error: Width and height must each be at least 50 pixels.")
            except ValueError:
                print("[!] Invalid number format.")
    return False


def run_interactive_menu(cfg: dict, cfg_path: Path, base_dir: Path) -> dict:
    has_unsaved_changes = False

    while True:
        model_name = cfg.get("settings", {}).get("selected_model", "")
        try:
            weights_path = resolve_model_path(base_dir, model_name)
        except Exception:
            weights_path = Path("models/unet_policy_v8_5.pt")

        print_header("Semga Live -- Bazyach Setup Console")
        display_current_settings(cfg, weights_path)

        if has_unsaved_changes:
            print("\n  >>> WARNING: You have unsaved changes! Select [6] to save to config.yaml <<<")

        print("\n[ACTIONS MENU]:")
        print("  1. Select Monitor (List available monitors and set active index)")
        print("  2. Toggle Clicks (Z/X active <-> Relax aim-only)")
        print("  3. Adjust Confidence Threshold (0.10 - 1.00)")
        print("  4. Configure / Calibrate Playfield ROI (Auto-detect / Presets / Manual)")
        print("  5. Select Neural Network Model")
        print("  6. Adjust Target FPS (30, 60, 120, 144, 240)")
        print("  7. Adjust Mouse Smoothing Factor")
        print("  8. Save settings to config.yaml")
        print("  9. Start Bot with Current Settings")
        print("  0. Exit")
        print("\nSelect action [0-9]: ", end="", flush=True)

        choice = input().strip()
        if choice == "1":
            if interactive_select_monitor(cfg):
                has_unsaved_changes = True
        elif choice == "2":
            if interactive_toggle_clicks(cfg):
                has_unsaved_changes = True
        elif choice == "3":
            if interactive_adjust_confidence(cfg):
                has_unsaved_changes = True
        elif choice == "4":
            if interactive_calibrate_roi(cfg):
                has_unsaved_changes = True
        elif choice == "5":
            if interactive_select_model(base_dir, cfg):
                has_unsaved_changes = True
        elif choice == "6":
            if interactive_adjust_fps(cfg):
                has_unsaved_changes = True
        elif choice == "7":
            if interactive_adjust_smooth(cfg):
                has_unsaved_changes = True
        elif choice == "8":
            save_configuration(cfg_path, cfg)
            has_unsaved_changes = False
            print(f"[OK] Configuration successfully saved to: {cfg_path}")
        elif choice == "9":
            if has_unsaved_changes:
                save_configuration(cfg_path, cfg)
                print(f"[INFO] Settings automatically saved to: {cfg_path}")
            print("\n[INFO] Starting bot inference...")
            break
        elif choice == "0":
            print("\nExiting.")
            sys.exit(0)
        else:
            print("[!] Unknown command.")
        time.sleep(0.4)

    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description="osu! UNet Policy Live Runner (Bazyach Edition)")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_NAME, help="Path to config.yaml")
    parser.add_argument("--weights", type=str, default=None, help="Path to policy weights (.pt)")
    parser.add_argument("--no-clicks", action="store_true", help="Disable key presses (relax mode: mouse aim only)")
    parser.add_argument("--clicks", action="store_true", help="Explicitly enable clicks (Z/X)")
    parser.add_argument("--smooth", type=float, default=None, help="Mouse smoothing factor [0.0 = raw, 0.5 = smooth]")
    parser.add_argument("--confidence", type=float, default=None, help="Confidence threshold [0.10 - 1.00]")
    parser.add_argument("--monitor", type=int, default=None, help="Monitor index to capture (1, 2, ...)")
    parser.add_argument("--preview", action="store_true", help="Show OpenCV debug preview window")
    parser.add_argument("--menu", "--setup", "-s", dest="setup_mode", action="store_true",
                        help="Open interactive CLI settings menu")
    parser.add_argument("--auto-roi", action="store_true", help="Auto-detect osu! window client rect on startup")
    parser.add_argument("--no-wait", action="store_true", help="Skip quick launch menu / countdown prompt")
    return parser.parse_args()


def countdown_prompt_for_menu(timeout_seconds: float = 3.0) -> bool:
    if not HAS_MSVCRT or not sys.stdin.isatty():
        return False

    print("\nPress any key within 3 seconds to open the Settings Menu...", end="", flush=True)
    t_end = time.time() + timeout_seconds
    while time.time() < t_end:
        if msvcrt.kbhit():
            msvcrt.getch()
            print("\n[OK] Opening interactive Settings Menu.")
            return True
        time.sleep(0.05)
    print("\n[INFO] Quick launching with current configuration...")
    return False


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = base_dir / args.config

    cfg = load_configuration(cfg_path)

    # If --setup or --menu flag passed, immediately enter configuration menu
    if args.setup_mode:
        cfg = run_interactive_menu(cfg, cfg_path, base_dir)
    elif not args.no_wait:
        # Check if user wants to open menu during 3s grace countdown
        print_header("Semga Live -- Bazyach CLI Runner")
        model_name = cfg.get("settings", {}).get("selected_model", "")
        try:
            cur_weights = resolve_model_path(base_dir, args.weights or model_name)
        except Exception:
            cur_weights = Path("models/unet_policy_v8_5.pt")
        display_current_settings(cfg, cur_weights)
        if countdown_prompt_for_menu(timeout_seconds=3.0):
            cfg = run_interactive_menu(cfg, cfg_path, base_dir)

    # Apply auto-ROI if requested
    if args.auto_roi:
        coords = find_osu_window_rect()
        if coords:
            cfg.setdefault("playfield", {})
            cfg["playfield"]["left"], cfg["playfield"]["top"], cfg["playfield"]["right"], cfg["playfield"]["bottom"] = coords
            print(f"[INFO] Auto-detected osu! ROI: {coords}")

    # Override config with explicit CLI parameters
    if args.monitor is not None:
        cfg.setdefault("dxgi", {})["monitor"] = int(args.monitor)

    sett = cfg.setdefault("settings", {})
    if args.no_clicks:
        sett["clicks_enabled"] = False
    elif args.clicks:
        sett["clicks_enabled"] = True

    if args.confidence is not None:
        sett["confidence"] = float(args.confidence)

    if args.smooth is not None:
        sett["smooth"] = float(args.smooth)

    # Determine final model weights path
    req_weights = args.weights or sett.get("selected_model")
    weights_path = resolve_model_path(base_dir, req_weights)
    sett["selected_model"] = weights_path.name

    clicks_enabled = bool(sett.get("clicks_enabled", True))
    confidence_val = float(sett.get("confidence", 0.60))
    smooth_val = float(sett.get("smooth", 0.0))

    # Save any command-line overrides if running in standard mode
    save_configuration(cfg_path, cfg)

    print("\n" + "=" * 72)
    print(f"  Starting RelaxLive: {weights_path.name}")
    print(f"  Monitor: {cfg.get('dxgi', {}).get('monitor', 1)} | Clicks: {'ON' if clicks_enabled else 'OFF'} | Confidence: {confidence_val:.2f} | Smoothing: {smooth_val:.2f}")
    pf = cfg.get("playfield", {})
    print(f"  ROI: [{pf.get('left')}, {pf.get('top')}, {pf.get('right')}, {pf.get('bottom')}]")
    print("=" * 72)

    bot = RelaxLive(
        config_dict=cfg,
        weights_path=str(weights_path),
        clicks_enabled=clicks_enabled,
        smooth=smooth_val,
        show_preview=args.preview,
        confidence_thresh=confidence_val,
    )
    bot.run()


if __name__ == "__main__":
    main()
