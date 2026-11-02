"""
Standalone Mouse and Keyboard Controller for osu! Bot (SendInput API)
"""
import ctypes
from ctypes import wintypes
import numpy as np

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000
KEYEVENTF_KEYUP = 0x0002

SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t)
    ]


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t)
    ]


class _INPUT(ctypes.Structure):
    class _U(ctypes.Union):
        _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT)]
    _fields_ = [("type", wintypes.DWORD), ("u", _U)]


class Controller:
    VK = {"z": 0x5A, "x": 0x58, "f2": 0x71, "f10": 0x79}

    def __init__(self, smooth: float = 0.0):
        self.u32 = ctypes.windll.user32
        self.u32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(_INPUT), ctypes.c_int]
        self.u32.SendInput.restype = wintypes.UINT
        self.smooth = float(smooth)
        self._keys: dict[str, bool] = {}
        self._px: float | None = None
        self._py: float | None = None
        self._ensure_desktop()

    def _ensure_desktop(self):
        try:
            hdesk = self.u32.OpenInputDesktop(0, False, 0x01FF)
            if hdesk:
                self.u32.SetThreadDesktop(hdesk)
        except Exception:
            pass

    def _send(self, inp: _INPUT):
        sent = self.u32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))
        if sent != 1:
            self._ensure_desktop()
            sent = self.u32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))
        if sent != 1:
            raise ctypes.WinError(ctypes.get_last_error())

    def move(self, x: float, y: float):
        tx, ty = float(x), float(y)
        if self._px is not None and self.smooth > 0.0:
            tx = self._px + (tx - self._px) * (1.0 - self.smooth)
            ty = self._py + (ty - self._py) * (1.0 - self.smooth)
        self._px, self._py = tx, ty

        vx = self.u32.GetSystemMetrics(SM_XVIRTUALSCREEN)
        vy = self.u32.GetSystemMetrics(SM_YVIRTUALSCREEN)
        vw = self.u32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
        vh = self.u32.GetSystemMetrics(SM_CYVIRTUALSCREEN)

        nx = int(np.clip(tx - vx, 0, vw - 1) * 65535 // max(vw - 1, 1))
        ny = int(np.clip(ty - vy, 0, vh - 1) * 65535 // max(vh - 1, 1))

        inp = _INPUT()
        inp.type = INPUT_MOUSE
        inp.u.mi = _MOUSEINPUT(
            nx,
            ny,
            0,
            MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK,
            0,
            0,
        )
        self._send(inp)

    def position(self) -> tuple[float, float]:
        point = wintypes.POINT()
        if not self.u32.GetCursorPos(ctypes.byref(point)):
            raise ctypes.WinError(ctypes.get_last_error())
        return float(point.x), float(point.y)

    def set_key(self, name: str, down: bool):
        key = name.lower()
        if self._keys.get(key) == down:
            return
        self._keys[key] = down
        if key not in self.VK:
            return
        inp = _INPUT()
        inp.type = INPUT_KEYBOARD
        inp.u.ki = _KEYBDINPUT(self.VK[key], 0, 0 if down else KEYEVENTF_KEYUP, 0, 0)
        self._send(inp)

    def release_all(self):
        for k in list(self._keys):
            self.set_key(k, False)
