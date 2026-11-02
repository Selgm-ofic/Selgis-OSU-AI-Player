"""Predict hit deadline from measured approach-ring phase velocity."""
from collections import defaultdict, deque

import numpy as np


class ApproachClock:
    def __init__(self, history=10, stale_s=0.25):
        self.history = defaultdict(lambda: deque(maxlen=history))
        self.stale_s = stale_s

    def update(self, tracks, now):
        live_ids = set()
        estimates = []
        for track in tracks:
            if (track.cls not in ("circle", "slider_head") or
                    not track.visual_timing or track.missed != 0):
                continue
            live_ids.add(track.tid)
            points = self.history[track.tid]
            points.append((float(now), float(track.urgency)))
            if len(points) < 3 or points[-1][0] - points[0][0] < 0.025:
                continue
            times = np.asarray([p[0] for p in points], dtype=np.float64)
            phases = np.asarray([p[1] for p in points], dtype=np.float64)
            times -= times[-1]
            centered = times - times.mean()
            denom = float(np.dot(centered, centered))
            if denom <= 1e-9:
                continue
            slope = float(np.dot(centered, phases - phases.mean()) / denom)
            if slope <= 0.08:
                continue
            tth_ms = (1.0 - phases[-1]) / slope * 1000.0
            if -30.0 <= tth_ms <= 500.0:
                estimates.append((tth_ms, track.tid, track.x, track.y))
        for tid in list(self.history):
            points = self.history[tid]
            if tid not in live_ids and points and now - points[-1][0] > self.stale_s:
                del self.history[tid]
        return min(estimates, default=None, key=lambda item: item[0])

    def reset(self):
        self.history.clear()
