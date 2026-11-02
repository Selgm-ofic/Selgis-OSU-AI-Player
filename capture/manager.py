import time
from typing import Optional, Tuple
import cv2

from .obs import OBSCapture
from .dxgi import DXGICapture


class CaptureManager:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        src = cfg.get("source", "auto")
        o = cfg.get("obs", {})
        d = cfg.get("dxgi", {})
        self.obs = OBSCapture(
            width=o.get("width", 1920), height=o.get("height", 1080),
            fps=o.get("fps", 60), try_indices=o.get("try_indices", [1, 0, 2, 3]),
        )
        self.dxgi = DXGICapture(monitor=d.get("monitor", 1))
        self._active = None
        self._wanted = src

    def open(self) -> bool:
        pf = self.cfg.get("playfield", {})
        l = int(pf.get("left", 0)); t = int(pf.get("top", 0))
        w = int(pf.get("right", 1920)) - l
        h = int(pf.get("bottom", 1080)) - t
        self.dxgi.set_region(l, t, w, h)
        if self._wanted == "obs":
            order = [self.obs]
        elif self._wanted == "dxgi":
            order = [self.dxgi]
        else:
            order = [self.dxgi, self.obs]
        for cap in order:
            if cap.open():
                self._active = cap
                if cap is self.dxgi:
                    pf = self.cfg.get("playfield", {})
                    l = int(pf.get("left", 0)); t = int(pf.get("top", 0))
                    w = int(pf.get("right", 1920)) - l
                    h = int(pf.get("bottom", 1080)) - t
                    try:
                        mon = self.dxgi._mon or {}
                        l += int(mon.get("left", 0)); t += int(mon.get("top", 0))
                    except Exception:
                        pass
                    self.dxgi.set_region(l, t, w, h)
                    self._dxgi_roi_only = True
                else:
                    self._dxgi_roi_only = False
                return True
        return False

    def read(self):
        if self._active is None:
            return None, time.perf_counter()
        return self._active.read()

    @property
    def mode(self) -> str:
        return self._active.mode if self._active else "none"

    @property
    def fps(self) -> float:
        return self._active.fps if self._active else 0.0

    def get_playfield(self, frame) -> Optional[object]:
        if frame is None:
            return None
        if getattr(self, "_dxgi_roi_only", False) and self._active is self.dxgi:
            return frame
        pf = self.cfg.get("playfield", {})
        h, w = frame.shape[:2]
        l = max(0, min(int(pf.get("left", 0)), w - 10))
        t = max(0, min(int(pf.get("top", 0)), h - 10))
        r = max(l + 10, min(int(pf.get("right", w)), w))
        b = max(t + 10, min(int(pf.get("bottom", h)), h))
        return frame[t:b, l:r]

    def preprocess(self, roi) -> Tuple[object, Tuple[float, float]]:
        pp = self.cfg.get("preprocess", {})
        iw, ih = int(pp.get("infer_width", 640)), int(pp.get("infer_height", 360))
        h, w = roi.shape[:2]
        tiny = cv2.resize(roi, (iw, ih), interpolation=cv2.INTER_LINEAR)
        return tiny, (iw / max(w, 1), ih / max(h, 1))

    def close(self):
        self.obs.close()
        self.dxgi.close()
        self._active = None


def load_cfg(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
