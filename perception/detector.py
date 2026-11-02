"""YOLO-детектор для hot-loop: CUDA FP16, batch=1, без градиентов.

Вход: BGR-кадр 640x360 (выход CaptureManager.preprocess).
Выход: список Det (schema.py), координаты нормализованы 0..1.
urgency здесь 0 — его считает tracker по динамике бокса (рост =
приближение approach-кольца). Трекинг — Фаза 3.
"""
import time
import numpy as np
from typing import List

from perception.schema import Det, CLASS_NAMES


class YoloDetector:
    def __init__(self, weights: str, device: int = 0, conf: float = 0.35,
                 iou: float = 0.5, imgsz: int = 640, half: bool = True,
                 input_mode: str = "gray_bgr", agnostic_nms: bool = False,
                 temporal_lag_frames: int = 1):
        from ultralytics import YOLO
        from ultralytics.utils import LOGGER
        LOGGER.setLevel("ERROR")
        self.model = YOLO(weights)
        self.device = device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        use_half = half
        try:
            import torch
            if not (torch.cuda.is_available() and device >= 0):
                use_half = False
            else:
                # Input geometry is fixed in the live loop, so cuDNN can cache
                # the fastest convolution algorithms after warmup.
                torch.backends.cudnn.benchmark = True
        except Exception:
            use_half = False
        self.half = use_half
        self.input_mode = input_mode
        self.agnostic_nms = agnostic_nms
        self.previous_gray = None
        self.temporal_lag_frames = max(1, int(temporal_lag_frames))
        self.gray_history = []
        self._n = 0
        self._ms = 0.0

    def warmup(self, n: int = 10):
        dummy = np.zeros((360, 640, 3), np.uint8)
        for _ in range(n):
            self.predict(dummy)

    def predict(self, frame: np.ndarray) -> List[Det]:
        if self.input_mode == "temporal3":
            from perception.temporal import temporal_frame
            previous = (self.gray_history[0] if self.gray_history else None)
            frame, current_gray = temporal_frame(frame, previous)
            self.gray_history.append(current_gray)
            if len(self.gray_history) > self.temporal_lag_frames:
                self.gray_history.pop(0)
        elif self.input_mode == "gray_bgr":
            import cv2
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return self.predict_preprocessed(frame)

    def predict_preprocessed(self, frame: np.ndarray) -> List[Det]:
        """Infer an image whose channels are already in the training format."""
        t0 = time.perf_counter()
        res = self.model.predict(frame, device=self.device, half=self.half,
                                 conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                                 verbose=False, max_det=32,
                                 agnostic_nms=self.agnostic_nms)[0]
        dt = (time.perf_counter() - t0) * 1000.0
        self._n += 1
        self._ms += dt
        out: List[Det] = []
        if res.boxes is not None and len(res.boxes) > 0:
            xywhn = res.boxes.xywhn.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            conf = res.boxes.conf.cpu().numpy()
            for (cx, cy, w, h), c, cf in zip(xywhn, cls, conf):
                name = CLASS_NAMES[c] if 0 <= c < len(CLASS_NAMES) else str(c)
                out.append(Det(cls=name, x=float(cx), y=float(cy),
                               w=float(w), h=float(h), conf=float(cf)))
        return out

    @property
    def avg_ms(self) -> float:
        return self._ms / max(1, self._n)

    def reset_stats(self):
        self._n = 0
        self._ms = 0.0

    def reset_sequence(self):
        self.previous_gray = None
        self.gray_history.clear()


def load_detector(cfg: dict) -> YoloDetector:
    dc = cfg.get("detector", {})
    return YoloDetector(
        weights=dc.get("weights", "data/runs/detect/osu_nano/weights/best.pt"),
        device=int(dc.get("device", 0)),
        conf=float(dc.get("preview_conf", dc.get("conf", 0.35))),
        iou=float(dc.get("iou", 0.5)),
        imgsz=int(dc.get("imgsz", 640)),
        input_mode=str(dc.get("input_mode", "gray_bgr")),
        agnostic_nms=bool(dc.get("agnostic_nms", False)),
        temporal_lag_frames=int(dc.get("temporal_lag_frames", 1)),
    )
