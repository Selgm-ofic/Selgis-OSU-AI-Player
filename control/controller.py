"""Управление: мышь (SendInput absolute) + клавиши Z/X (K1/K2).

Сглаживание: экспоненциальное + aim-ahead компенсация лага камеры:
  target_pred = target + velocity * lag_comp_s
lag_comp_s калибруется (по умолчанию 0.03 = 30мс под OBS).
"""
import ctypes
from ctypes import wintypes
import time
import numpy as np

INPUT_MOUSE, INPUT_KEYBOARD = 0, 1
MOUSEEVENTF_MOVE, MOUSEEVENTF_ABSOLUTE = 0x0001, 0x8000
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.c_size_t)]


class _INPUT(ctypes.Structure):
    class _U(ctypes.Union):
        _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT)]
    _fields_ = [("type", wintypes.DWORD), ("u", _U)]


class Controller:
    VK = {"z": 0x5A, "x": 0x58, "f2": 0x71}
    SCANCODES = {"z": 0x2C, "x": 0x2D, "f2": 0x3C}

    def __init__(self, lag_comp_s: float = 0.0, smooth: float = 0.0):
        self.u32 = ctypes.windll.user32
        self.u32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(_INPUT), ctypes.c_int]
        self.u32.SendInput.restype = wintypes.UINT
        self.sw = self.u32.GetSystemMetrics(0)
        self.sh = self.u32.GetSystemMetrics(1)
        self.lag = lag_comp_s
        self.smooth = smooth
        self._keys = {}
        self._px, self._py = None, None

    def _send(self, inp: _INPUT):
        sent = self.u32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))
        if sent != 1:
            raise ctypes.WinError(ctypes.get_last_error())

    def move(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0):
        # aim-ahead + сглаживание
        tx = x + vx * self.lag
        ty = y + vy * self.lag
        if self._px is not None:
            tx = self._px + (tx - self._px) * (1.0 - self.smooth)
            ty = self._py + (ty - self._py) * (1.0 - self.smooth)
        self._px, self._py = tx, ty
        nx = int(np.clip(tx, 0, self.sw - 1) * 65535 // max(self.sw, 1))
        ny = int(np.clip(ty, 0, self.sh - 1) * 65535 // max(self.sh, 1))
        inp = _INPUT()
        inp.type = INPUT_MOUSE
        inp.u.mi = _MOUSEINPUT(nx, ny, 0, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, 0, 0)
        self._send(inp)

    def position(self):
        """Return the actual Windows cursor position in screen pixels."""
        point = wintypes.POINT()
        if not self.u32.GetCursorPos(ctypes.byref(point)):
            raise ctypes.WinError(ctypes.get_last_error())
        return float(point.x), float(point.y)

    def set_key(self, name: str, down: bool):
        key = name.lower()
        if self._keys.get(key) == down:
            return
        self._keys[key] = down
        scan = self.SCANCODES.get(key, 0)
        flags = KEYEVENTF_SCANCODE if scan else 0
        if not down:
            flags |= KEYEVENTF_KEYUP
        inp = _INPUT()
        inp.type = INPUT_KEYBOARD
        inp.u.ki = _KEYBDINPUT(0 if scan else self.VK.get(key, 0), scan, flags, 0, 0)
        self._send(inp)

    def release_all(self):
        for k in list(self._keys):
            self.set_key(k, False)

    def press_retry(self):
        self.set_key("f2", True)
        time.sleep(0.03)
        self.set_key("f2", False)
