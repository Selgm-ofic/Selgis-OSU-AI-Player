"""Трекер ByteTrack-lite: IoU-ассоциация + constant-velocity фильтр.

Зачем: YOLO дает боксы без памяти. Трекер держит id объекта между кадрами,
сглаживает позицию и считает urgency из динамики площади бокса:
растущий бокс = схлопывающееся approach-кольцо = скоро бить.

Выход: список Track (id, cls, x, y, vx, vy, urgency, age).
Координаты — нормализованные 0..1 относительно infer-кадра.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict
from perception.schema import Det


def _iou(a: Det, b: Det) -> float:
    ax1, ay1 = a.x - a.w / 2, a.y - a.h / 2
    ax2, ay2 = a.x + a.w / 2, a.y + a.h / 2
    bx1, by1 = b.x - b.w / 2, b.y - b.h / 2
    bx2, by2 = b.x + b.w / 2, b.y + b.h / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a.w * a.h + b.w * b.h - inter
    return inter / max(union, 1e-9)


@dataclass
class Track:
    tid: int
    cls: str
    x: float
    y: float
    w: float
    h: float
    vx: float = 0.0
    vy: float = 0.0
    urgency: float = 0.0   # 0..1, 1 = бить сейчас
    age: int = 0
    missed: int = 0
    conf: float = 0.0
    t0: float = 0.0        # время рождения трека (сек, perf_counter)
    visual_timing: bool = False

    def predict(self, dt: float):
        self.x += self.vx * dt
        self.y += self.vy * dt


class Tracker:
    def __init__(self, iou_thresh: float = 0.3, max_missed: int = 5,
                 preempt_s: float = 1.2):
        self.iou_thresh = iou_thresh
        self.max_missed = max_missed
        self.preempt_s = preempt_s
        self._next = 0
        self.tracks: Dict[int, Track] = {}

    def update(self, dets: List[Det], dt: float, now: float) -> List[Track]:
        dt = max(dt, 1e-3)
        for tr in self.tracks.values():
            tr.predict(dt)

        used = [False] * len(dets)
        # жадное сопоставление по IoU (N мало — венгерка не нужна)
        pairs = []
        for tid, tr in self.tracks.items():
            pseudo = Det(cls=tr.cls, x=tr.x, y=tr.y, w=tr.w, h=tr.h, conf=0)
            for i, d in enumerate(dets):
                if used[i] or d.cls != tr.cls:
                    continue
                v = _iou(pseudo, d)
                if v >= self.iou_thresh:
                    pairs.append((v, tid, i))
        pairs.sort(reverse=True)
        matched_tr, matched_dt = set(), set()
        for _, tid, i in pairs:
            if tid in matched_tr or i in matched_dt:
                continue
            matched_tr.add(tid); matched_dt.add(i)
            used[i] = True
            tr = self.tracks[tid]
            nx, ny = dets[i].x, dets[i].y
            tr.vx = 0.7 * tr.vx + 0.3 * (nx - tr.x) / dt
            tr.vy = 0.7 * tr.vy + 0.3 * (ny - tr.y) / dt
            tr.x, tr.y = nx, ny
            tr.w, tr.h = dets[i].w, dets[i].h
            tr.conf = dets[i].conf
            tr.age += 1
            tr.missed = 0
            tr.urgency = (float(dets[i].urgency) if dets[i].urgency is not None
                          else self._urgency(tr, now))
            tr.visual_timing = dets[i].urgency is not None

        for tid in list(self.tracks):
            if tid not in matched_tr:
                self.tracks[tid].missed += 1
                if self.tracks[tid].missed > self.max_missed:
                    del self.tracks[tid]

        for i, d in enumerate(dets):
            if not used[i]:
                tr = Track(tid=self._next, cls=d.cls, x=d.x, y=d.y,
                           w=d.w, h=d.h, conf=d.conf, age=1, t0=now)
                tr.urgency = (float(d.urgency) if d.urgency is not None
                              else self._urgency(tr, now))
                tr.visual_timing = d.urgency is not None
                self.tracks[self._next] = tr
                self._next += 1

        return sorted(self.tracks.values(), key=lambda t: t.urgency, reverse=True)

    def _urgency(self, tr: Track, now: float) -> float:
        """Время жизни трека / preempt. Объект появляется примерно за preempt
        до удара (детектор видит его с появления), значит доля прожитого
        времени ~ готовность бить. Грубо, но работает без кольца approach
        в детекторе; политику доучим на реальных данных (Фаза 3)."""
        if tr.cls == "slider_tail":
            return 0.0
        return float(np.clip((now - tr.t0) / max(self.preempt_s, 0.2), 0.0, 1.0))

    def reset(self):
        self.tracks.clear()
        self._next = 0
