"""Рекордер сессий для обучения детектора (stdlib-only, без pip-зависимостей).

Сценарий с хоткеем (рекомендуется, работает прямо из lazer):
  1. python main.py record --beatmap <путь.osz> --diff <сложность>
  2. Скрипт встает в засаду: "Нажми F9 чтобы начать"
  3. Alt-Tab в lazer, запускаешь карту, жмешь F9 — бип 880Гц = запись пошла
  4. Играешь. В конце снова F9 — бип 440Гц = стоп + сохранение

Старый режим с countdown остался: добавь --countdown 3.

Глобальный хоткей — через WinAPI RegisterHotKey (ctypes, без пакета
`keyboard`). Окно предпросмотра в hotkey-режиме по умолчанию ВЫКЛЮЧЕНО,
чтобы cv2 не уводил фокус из игры; включи через --preview при желании.
"""
import os
import sys
import time
import json
import ctypes
from ctypes import wintypes
import threading
import argparse
from datetime import datetime

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from capture.manager import CaptureManager, load_cfg

VK_CODES = {"F1": 0x70, "F2": 0x71, "F3": 0x72, "F4": 0x73,
            "F5": 0x74, "F6": 0x75, "F7": 0x76, "F8": 0x77, "F9": 0x78,
            "F10": 0x79, "F11": 0x7A, "F12": 0x7B}


def beep(freq: int, ms: int = 200):
    try:
        import winsound
        winsound.Beep(freq, ms)
    except Exception:
        print("\a", end="", flush=True)


class HotkeyArm:
    """Глобальный хоткей старт/стоп через RegisterHotKey. Потокобезопасные флаги."""

    def __init__(self, name: str = "F1"):
        self.vk = VK_CODES.get(name.upper(), 0x70)
        self.name = name.upper()
        self.pressed = threading.Event()   # одно нажатие = сигнал
        self._stop = threading.Event()
        self.ready = threading.Event()
        self._state_lock = threading.Lock()
        self._down = False
        self._callback = None
        self.hook_enabled = False
        self._t = None

    def _set_key_state(self, is_down: bool):
        with self._state_lock:
            if is_down and not self._down:
                self.pressed.set()
            self._down = is_down

    def _loop(self):
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        kernel32.GetModuleHandleW.restype = ctypes.c_void_p
        WM_KEYDOWN, WM_KEYUP = 0x0100, 0x0101
        WM_SYSKEYDOWN, WM_SYSKEYUP = 0x0104, 0x0105
        WH_KEYBOARD_LL, PM_REMOVE = 13, 1

        class KBDLLHOOKSTRUCT(ctypes.Structure):
            _fields_ = [("vkCode", wintypes.DWORD), ("scanCode", wintypes.DWORD),
                        ("flags", wintypes.DWORD), ("time", wintypes.DWORD),
                        ("dwExtraInfo", ctypes.c_void_p)]

        callback_type = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, ctypes.c_int,
                                           wintypes.WPARAM, wintypes.LPARAM)
        user32.SetWindowsHookExW.argtypes = [ctypes.c_int, callback_type,
                                             ctypes.c_void_p, wintypes.DWORD]
        user32.SetWindowsHookExW.restype = ctypes.c_void_p
        user32.CallNextHookEx.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                          wintypes.WPARAM, wintypes.LPARAM]
        user32.CallNextHookEx.restype = ctypes.c_ssize_t
        user32.UnhookWindowsHookEx.argtypes = [ctypes.c_void_p]
        user32.UnhookWindowsHookEx.restype = wintypes.BOOL

        @callback_type
        def keyboard_proc(code, message, data):
            if code >= 0:
                event = ctypes.cast(data, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
                if event.vkCode == self.vk:
                    if message in (WM_KEYDOWN, WM_SYSKEYDOWN):
                        self._set_key_state(True)
                    elif message in (WM_KEYUP, WM_SYSKEYUP):
                        self._set_key_state(False)
            return user32.CallNextHookEx(None, code, message, data)

        self._callback = keyboard_proc
        module = kernel32.GetModuleHandleW(None)
        hook = user32.SetWindowsHookExW(WH_KEYBOARD_LL, keyboard_proc, module, 0)
        self.hook_enabled = bool(hook)
        self._set_key_state(bool(user32.GetAsyncKeyState(self.vk) & 0x8000))
        self.ready.set()
        try:
            msg = wintypes.MSG()
            while not self._stop.is_set():
                while user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE):
                    user32.TranslateMessage(ctypes.byref(msg))
                    user32.DispatchMessageW(ctypes.byref(msg))
                # Polling remains active as a fallback if another application
                # blocks or delays the low-level hook.
                self._set_key_state(bool(user32.GetAsyncKeyState(self.vk) & 0x8000))
                time.sleep(0.006)
        finally:
            if hook:
                user32.UnhookWindowsHookEx(hook)

    def start(self):
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        if not self.ready.wait(2.0):
            raise RuntimeError("hotkey polling thread did not initialize")

    def wait_press(self, timeout=None) -> bool:
        ok = self.pressed.wait(timeout)
        self.pressed.clear()
        return ok

    def close(self):
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=0.3)


def start_stdin_control(arm: HotkeyArm):
    """Allow the GUI to trigger the same start/stop event without focus."""
    def loop():
        for line in sys.stdin:
            if line.strip().lower() == "toggle":
                arm.pressed.set()
    threading.Thread(target=loop, daemon=True).start()


def save_session(sess, stamps, meta):
    os.makedirs(sess, exist_ok=True)
    np.save(os.path.join(sess, "stamps.npy"), np.array(stamps, dtype=np.float64))
    meta.update({"duration_ms": float(stamps[-1]) if stamps else 0.0,
                 "frames": len(stamps)})
    with open(os.path.join(sess, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Saved: {sess} ({len(stamps)} frames)")


def run_countdown(man, cfg, sess, meta, rec, seconds: int, preview: bool):
    print(f"Countdown {seconds}с — переключись в lazer и сними паузу!")
    for i in range(seconds, 0, -1):
        print(f"  {i}...")
        time.sleep(1.0)
    return run_capture(man, sess, meta, rec, preview)


def run_capture(man, sess, meta, rec, preview: bool):
    os.makedirs(sess, exist_ok=True)
    vw = None
    stamps = []
    t0 = time.perf_counter()
    beep(880)
    print("REC... F9 или Q в окне = стоп" if preview else "REC... нажми хоткей еще раз = стоп")
    try:
        while True:
            frame, _ = man.read()
            if frame is None:
                continue
            roi = man.get_playfield(frame)
            if vw is None:
                h, w = roi.shape[:2]
                vw = cv2.VideoWriter(os.path.join(sess, "video.mp4"),
                                     cv2.VideoWriter_fourcc(*rec.get("video_codec", "mp4v")),
                                     int(rec.get("fps", 60)), (w, h))
            vw.write(roi)
            stamps.append((time.perf_counter() - t0) * 1000.0)
            if preview:
                small = cv2.resize(roi, (640, 360))
                cv2.imshow("REC (Q=stop)", small)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            elif HotkeyArm.wait_global_stop():
                break
    except KeyboardInterrupt:
        pass
    finally:
        if vw is not None:
            vw.release()
        cv2.destroyAllWindows()
    beep(440)
    return stamps


# Глобальный флаг стопа для capture-цикла без окна (ставит hotkey-поток)
_STOP_EVENT = threading.Event()


def _patch_wait():
    HotkeyArm.wait_global_stop = staticmethod(_STOP_EVENT.is_set)


_patch_wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--source", default=None, choices=["obs", "dxgi", "auto"])
    ap.add_argument("--beatmap", default="")
    ap.add_argument("--diff", default="")
    ap.add_argument("--name", default=None)
    ap.add_argument("--output-dir", default=None,
                    help="recording root directory (overrides config)")
    ap.add_argument("--delay", type=float, default=0.0)
    ap.add_argument("--countdown", type=int, default=None,
                    help="старый режим: старт через N сек без хоткея")
    ap.add_argument("--no-hotkey", action="store_true",
                    help="не регистрировать глобальный хоткей")
    ap.add_argument("--hotkey", default=None, help="хоткей старт/стоп (F1..F12)")
    ap.add_argument("--preview", action="store_true",
                    help="показывать окно предпросмотра (уводит фокус из игры!)")
    ap.add_argument("--stdin-control", action="store_true",
                    help="accept 'toggle' lines from the native GUI")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    if args.source:
        cfg["source"] = args.source
    rec = cfg.get("recording", {})
    hotkey_name = (args.hotkey or rec.get("hotkey", "F1")).upper()
    out_dir = args.output_dir or rec.get("dir", "data/recordings")
    name = args.name or datetime.now().strftime("%Y%m%d_%H%M%S")
    sess = os.path.join(out_dir, name)

    man = CaptureManager(cfg)
    if not man.open():
        print("FAIL: no capture source")
        return

    meta = {
        "session": name,
        "beatmap_osz": args.beatmap,
        "diff": args.diff,
        "skin": cfg.get("skin", {}).get("name", "?"),
        "source": man.mode,
        "delay_ms": float(args.delay) * 1000.0,
        "mapping": cfg.get("mapping", {}),
        "playfield": cfg.get("playfield", {}),
        "preempt_ms": cfg.get("beatmaps", {}).get("preempt_ms", 1200.0),
    }

    try:
        if args.countdown is not None or args.no_hotkey:
            # Классика без хоткея
            stamps = run_countdown(man, cfg, sess, meta, rec,
                                   args.countdown or int(rec.get("countdown_sec", 3)),
                                   preview=True)
        else:
            arm = HotkeyArm(hotkey_name)
            arm.start()
            if args.stdin_control:
                start_stdin_control(arm)
            print(f"HOTKEY READY: {hotkey_name} "
                  f"(hook={'yes' if arm.hook_enabled else 'no'}, polling=yes)",
                  flush=True)
            print(f"ARMED source={man.mode}. Alt-Tab в lazer, запусти карту и нажми {hotkey_name} (старт).")
            print(f"Повторный {hotkey_name} — стоп+сохранение. Ctrl+C — отмена.")
            while not arm.wait_press(timeout=5.0):
                print(f"[armed] waiting for {hotkey_name} or GUI toggle...",
                      flush=True)
            _STOP_EVENT.clear()
            # Второй поток: тот же хоткей теперь = стоп
            stopper = threading.Thread(target=lambda: (arm.wait_press(), _STOP_EVENT.set()),
                                       daemon=True)
            stopper.start()
            stamps = run_capture(man, sess, meta, rec, preview=args.preview)
            arm.close()
    finally:
        man.close()

    if stamps:
        save_session(sess, stamps, meta)
    else:
        print("Пусто, нечего сохранять.")


if __name__ == "__main__":
    main()
