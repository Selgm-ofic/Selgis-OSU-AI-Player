import ctypes
import time
from typing import Optional, Tuple
import cv2
import numpy as np


class DXGICapture:
    def __init__(self, monitor: int = 1):
        self.monitor = monitor
        self._camera = None
        self._sct = None
        self._mon = None
        self._region = None
        self._t0 = time.perf_counter()
        self._n = 0
        self.fps = 0.0
        self._backend = "none"

    def set_region(self, left: int, top: int, width: int, height: int):
        self._region = (
            int(left),
            int(top),
            int(left) + max(10, int(width)),
            int(top) + max(10, int(height)),
        )
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
            output_idx = max(0, int(self.monitor) - 1)
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
        except Exception as exc:
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
        return {
            "left": left,
            "top": top,
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

    def read(self) -> Tuple[Optional[object], float]:
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

    @property
    def mode(self) -> str:
        return f"dxgi:{self._backend}:mon{self.monitor}"

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
