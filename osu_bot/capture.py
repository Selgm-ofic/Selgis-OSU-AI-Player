"""
High-performance screen capture module (DXcam with MSS fallback).
Supports dynamic runtime ROI changes without restarting capture pipeline.
"""
import ctypes
from ctypes import wintypes
import time
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np


def list_monitors() -> List[Dict[str, Any]]:
    """
    Enumerate all connected displays using WinAPI (EnumDisplayMonitors / EnumDisplayDevices).
    Returns a list of dicts with:
      - index: 1-based display index
      - name: Device string or generic name
      - device: Display device identifier (e.g. \\\\.\\DISPLAY1)
      - width: Monitor width in pixels
      - height: Monitor height in pixels
      - left: Virtual desktop left offset
      - top: Virtual desktop top offset
      - is_primary: True if this is the primary monitor
    """
    monitors: List[Dict[str, Any]] = []
    try:
        import win32api
        import win32con

        enum_mons = win32api.EnumDisplayMonitors()
        for idx, (hMon, _, _) in enumerate(enum_mons):
            info = win32api.GetMonitorInfo(hMon)
            dev_name = info.get("Device", "")
            is_primary = bool(info.get("Flags", 0) & win32con.MONITORINFOF_PRIMARY)
            name = f"Display {idx + 1}"
            try:
                dev = win32api.EnumDisplayDevices(dev_name, 0)
                if dev.DeviceString:
                    name = dev.DeviceString
            except Exception:
                pass
            l, t, r, b = info["Monitor"]
            monitors.append({
                "index": idx + 1,
                "name": name,
                "device": dev_name,
                "width": r - l,
                "height": b - t,
                "left": l,
                "top": t,
                "is_primary": is_primary,
            })
    except Exception:
        # Fallback using mss or user32 if pywin32 is not present
        try:
            import mss
            with mss.MSS() as sct:
                for idx, m in enumerate(sct.monitors[1:], start=1):
                    monitors.append({
                        "index": idx,
                        "name": m.get("name", f"Display {idx}"),
                        "device": m.get("unique_id", f"\\\\.\\DISPLAY{idx}"),
                        "width": m["width"],
                        "height": m["height"],
                        "left": m["left"],
                        "top": m["top"],
                        "is_primary": m.get("is_primary", idx == 1),
                    })
        except Exception:
            pass
    return monitors


class DXGICapture:
    def __init__(self, monitor: int = 1):
        self.monitor = int(monitor)
        self._camera = None
        self._sct = None
        self._mon = None
        self._region: Optional[Tuple[int, int, int, int]] = None
        self._t0 = time.perf_counter()
        self._n = 0
        self.fps = 0.0
        self._backend = "none"

    def set_region(self, left: int, top: int, width: int, height: int):
        new_region = (
            int(left),
            int(top),
            int(left) + max(16, int(width)),
            int(top) + max(16, int(height)),
        )
        if self._region == new_region:
            return
        self._region = new_region
        if self._camera is not None and getattr(self._camera, "is_capturing", False):
            try:
                self._camera.stop()
                self._camera.start(target_fps=120, video_mode=True, region=self._region)
            except Exception:
                pass

    def _ensure_desktop(self):
        try:
            user32 = ctypes.windll.user32
            hdesk = user32.OpenInputDesktop(0, False, 0x01FF)
            if hdesk:
                user32.SetThreadDesktop(hdesk)
        except Exception:
            pass

    def open(self) -> bool:
        self.close()
        self._ensure_desktop()
        try:
            import dxcam
            output_idx = max(0, self.monitor - 1)
            self._camera = dxcam.create(
                output_idx=output_idx,
                output_color="BGR",
                backend="dxgi",
                processor_backend="numpy",
            )
            self._camera.start(target_fps=120, video_mode=True, region=self._region)
            time.sleep(0.04)
            frame = self._camera.get_latest_frame()
            if frame is None and self._region is not None:
                frame = self._camera.grab(region=self._region)
            if frame is None or frame.shape[1] < 10:
                raise RuntimeError("DXcam returned no frame")
            self._backend = "dxcam"
            self._reset_stats()
            return True
        except Exception:
            self._release_dxcam()

        try:
            import mss
            self._ensure_desktop()
            self._sct = mss.MSS()
            monitors = self._sct.monitors
            idx = self.monitor if 0 < self.monitor < len(monitors) else 1
            self._mon = monitors[idx]
            target = self._mss_target()
            shot = self._sct.grab(target)
            if shot is None or shot.width < 10:
                raise RuntimeError("mss returned no frame")
            self._backend = "mss"
            self._reset_stats()
            return True
        except Exception:
            self.close()
            return False

    def _reset_stats(self):
        self._t0 = time.perf_counter()
        self._n = 0
        self.fps = 0.0

    def _mss_target(self):
        if self._region is None:
            return self._mon
        left, top, right, bottom = self._region
        mon_left = self._mon.get("left", 0) if self._mon else 0
        mon_top = self._mon.get("top", 0) if self._mon else 0
        return {
            "left": mon_left + left,
            "top": mon_top + top,
            "width": right - left,
            "height": bottom - top,
        }

    def _tick(self):
        self._n += 1
        now = time.perf_counter()
        elapsed = now - self._t0
        if elapsed >= 1.0:
            self.fps = self._n / elapsed
            self._n = 0
            self._t0 = now
        return now

    def read(self) -> Tuple[Optional[np.ndarray], float]:
        try:
            if self._camera is not None:
                frame = self._camera.get_latest_frame()
                if frame is None and self._region is not None:
                    frame = self._camera.grab(region=self._region)
                if frame is not None:
                    return frame, self._tick()
            if self._sct is not None:
                shot = self._sct.grab(self._mss_target())
                frame = cv2.cvtColor(np.asarray(shot), cv2.COLOR_BGRA2BGR)
                return frame, self._tick()
        except Exception:
            pass
        return None, time.perf_counter()

    def _release_dxcam(self):
        try:
            if self._camera is not None:
                if getattr(self._camera, "is_capturing", False):
                    self._camera.stop()
                self._camera.release()
        except Exception:
            pass
        self._camera = None

    def close(self):
        self._release_dxcam()
        try:
            if self._sct is not None:
                self._sct.close()
        except Exception:
            pass
        self._sct = None
        self._backend = "none"


class CaptureManager:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        d = cfg.get("dxgi", {})
        self.dxgi = DXGICapture(monitor=d.get("monitor", 1))

    def set_playfield(self, left: int, top: int, right: int, bottom: int):
        self.cfg["playfield"] = {
            "left": int(left),
            "top": int(top),
            "right": int(right),
            "bottom": int(bottom)
        }
        w = max(16, int(right) - int(left))
        h = max(16, int(bottom) - int(top))
        self.dxgi.set_region(int(left), int(top), w, h)

    def open(self) -> bool:
        pf = self.cfg.get("playfield", {})
        l = int(pf.get("left", 0))
        t = int(pf.get("top", 0))
        w = int(pf.get("right", 1920)) - l
        h = int(pf.get("bottom", 1080)) - t
        self.dxgi.set_region(l, t, w, h)
        return self.dxgi.open()

    def read(self) -> Tuple[Optional[np.ndarray], float]:
        return self.dxgi.read()

    def get_playfield(self, frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if frame is None:
            return None
        pf = self.cfg.get("playfield", {})
        l = int(pf.get("left", 0))
        t = int(pf.get("top", 0))
        w = int(pf.get("right", 1920)) - l
        h = int(pf.get("bottom", 1080)) - t
        if abs(frame.shape[1] - w) <= 2 and abs(frame.shape[0] - h) <= 2:
            return frame
        f_h, f_w = frame.shape[:2]
        cl = max(0, min(l, f_w - 10))
        ct = max(0, min(t, f_h - 10))
        cr = max(cl + 10, min(int(pf.get("right", f_w)), f_w))
        cb = max(ct + 10, min(int(pf.get("bottom", f_h)), f_h))
        return frame[ct:cb, cl:cr]

    def close(self):
        self.dxgi.close()
