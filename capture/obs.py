import threading
import time
from typing import Optional, Tuple
import cv2


class OBSCapture:
    def __init__(self, width=1920, height=1080, fps=60, try_indices=(1, 0, 2, 3)):
        self.width = width
        self.height = height
        self.fps_target = fps
        self.try_indices = list(try_indices)
        self.cap: Optional[cv2.VideoCapture] = None
        self.index: int = -1
        self._t0 = time.perf_counter()
        self._n = 0
        self.fps = 0.0
        self._latest_frame = None
        self._latest_ts = time.perf_counter()
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

    def _worker(self):
        while not self._stop.is_set():
            if self.cap is None:
                break
            ok, frame = self.cap.read()
            ts = time.perf_counter()
            if ok and frame is not None:
                with self._lock:
                    self._latest_frame = frame
                    self._latest_ts = ts
                    self._n += 1
                    dt = ts - self._t0
                    if dt >= 1.0:
                        self.fps = self._n / dt
                        self._n = 0
                        self._t0 = ts
            else:
                time.sleep(0.001)

    def open(self) -> bool:
        self.close()
        for idx in self.try_indices:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                continue
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            except Exception:
                pass
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps_target)
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                self.cap = cap
                self.index = idx
                self._latest_frame = frame
                self._latest_ts = time.perf_counter()
                self._t0 = time.perf_counter()
                self._n = 0
                self._stop.clear()
                self._thread = threading.Thread(target=self._worker, daemon=True)
                self._thread.start()
                return True
            cap.release()
        return False

    def read(self) -> Tuple[Optional[object], float]:
        with self._lock:
            frame = self._latest_frame
            ts = self._latest_ts
        if frame is not None:
            return frame, ts
        return None, time.perf_counter()

    @property
    def mode(self) -> str:
        return f"obs:{self.index}"

    def close(self):
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=0.5)
        self._thread = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._latest_frame = None
