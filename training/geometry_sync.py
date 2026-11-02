"""YOLO-free alignment of an Auto recording to its beatmap clock.

The recorder stores a temporal motion channel.  At a hit, the cursor, hit
flash, disappearing circle, or changing slider head produces a local motion
peak at the exact coordinate from the .osu file.  Aggregating hundreds of
these weak peaks gives a robust recording-to-map offset without asking an old
detector to create labels for a new one.
"""
from dataclasses import dataclass
import json
import os

import cv2
import numpy as np

from perception.beatmap import Beatmap, OSU_H, OSU_W
from training.action_timeline import TimeTransform, object_end_time


@dataclass(frozen=True)
class GeometrySyncResult:
    transform: TimeTransform
    uncertainty_ms: float
    score: float
    runner_up_score: float
    anchors: int
    covered_objects: int
    total_objects: int
    record_start_ms: float
    record_end_ms: float
    method: str = "temporal_hit_geometry_v1"

    def is_safe(self, max_uncertainty_ms=20.0, min_coverage=0.80):
        coverage = self.covered_objects / max(1, self.total_objects)
        return self.uncertainty_ms <= max_uncertainty_ms and coverage >= min_coverage


def _load_motion(session, size=(160, 90)):
    stamps = np.load(os.path.join(session, "vision_stamps.npy")).astype(np.float64)
    folder = os.path.join(session, "vision_frames")
    frames = []
    for index in range(len(stamps)):
        packed = cv2.imread(os.path.join(folder, f"{index:06d}.png"),
                            cv2.IMREAD_COLOR)
        if packed is None:
            raise RuntimeError(f"missing temporal frame {index:06d}.png")
        motion = cv2.resize(packed[:, :, 1], size, interpolation=cv2.INTER_AREA)
        # Max-pooling makes cursor/hit flashes tolerant to a few pixels of
        # playfield mapping error while preserving timing.
        motion = cv2.dilate(motion, np.ones((5, 5), np.uint8), iterations=1)
        frames.append(motion)
    if len(stamps) < 30 or not np.all(np.diff(stamps) > 0):
        raise RuntimeError("temporal timestamps are missing or non-monotonic")
    return stamps, np.asarray(frames, dtype=np.uint8)


def _anchor_arrays(bm: Beatmap, width, height):
    usable = [obj for obj in bm.objects if not obj.is_spinner]
    if len(usable) < 20:
        raise RuntimeError("too few non-spinner hit objects for geometry sync")
    times = np.asarray([obj.time for obj in usable], dtype=np.float64)
    xs = np.clip(np.rint([obj.x / OSU_W * (width - 1) for obj in usable]),
                 0, width - 1).astype(np.int32)
    ys = np.clip(np.rint([obj.y / OSU_H * (height - 1) for obj in usable]),
                 0, height - 1).astype(np.int32)
    # Circles give a cleaner disappearance pulse than long slider heads.
    weights = np.asarray([1.0 if obj.is_circle else 0.72 for obj in usable],
                         dtype=np.float64)
    return times, xs, ys, weights


def _scores(offsets, scale, stamps, signal, times, weights, anchor_ids=None):
    ids = np.arange(len(times)) if anchor_ids is None else np.asarray(anchor_ids)
    event_times = times[ids]
    columns = ids
    out = np.full(len(offsets), -np.inf, dtype=np.float64)
    for oi, offset in enumerate(offsets):
        record_ms = (event_times - offset) / scale
        valid = ((record_ms >= stamps[0] + 300.0) &
                 (record_ms <= stamps[-1] - 300.0))
        if valid.sum() < 16:
            continue
        r = record_ms[valid]
        cols = columns[valid]
        idx = np.clip(np.searchsorted(stamps, r), 1, len(stamps) - 1)
        before = np.clip(np.searchsorted(stamps, r - 220.0), 0, len(stamps) - 1)
        after = np.clip(np.searchsorted(stamps, r + 220.0), 0, len(stamps) - 1)
        hit = signal[idx, cols].astype(np.float64)
        neighbourhood = 0.5 * (
            signal[before, cols].astype(np.float64) +
            signal[after, cols].astype(np.float64))
        # A small absolute term helps sliders; the contrast term makes the
        # broad approach-circle motion plateau unable to win by itself.
        evidence = np.clip(hit - 0.55 * neighbourhood, -40.0, 180.0) + 0.08 * hit
        w = weights[cols]
        out[oi] = np.sum(evidence * w) / np.sum(w)
    return out


def estimate_geometry_sync(session: str, bm: Beatmap, scale=1.0):
    stamps, motion = _load_motion(session)
    times, xs, ys, weights = _anchor_arrays(bm, motion.shape[2], motion.shape[1])
    # F x N lookup table: each column is motion at one known hit coordinate.
    signal = motion[:, ys, xs]

    # A beatmap may contain a long audio intro (30+ seconds) before its first
    # object while the user starts recording directly at gameplay.  Derive the
    # upper bound from the map instead of assuming that map time starts near 0.
    offset_low = -15000.0
    offset_high = max(15000.0, float(times[0] + 15000.0))
    coarse_offsets = np.arange(offset_low, offset_high + 0.1, 20.0)
    coarse_scores = _scores(
        coarse_offsets, scale, stamps, signal, times, weights)
    coarse_best = float(coarse_offsets[int(np.argmax(coarse_scores))])
    fine_offsets = np.arange(coarse_best - 80.0, coarse_best + 80.1, 1.0)
    fine_scores = _scores(fine_offsets, scale, stamps, signal, times, weights)
    best_index = int(np.argmax(fine_scores))
    best_offset = float(fine_offsets[best_index])
    best_score = float(fine_scores[best_index])

    separated = np.abs(fine_offsets - best_offset) >= 25.0
    runner_up = float(np.max(fine_scores[separated])) if separated.any() else best_score

    # Independent time slices detect a false global peak and clock drift.
    order = np.argsort(times)
    local_offsets = []
    for group in np.array_split(order, min(7, max(3, len(order) // 50))):
        if len(group) < 12:
            continue
        candidates = np.arange(best_offset - 90.0, best_offset + 90.1, 2.0)
        scores = _scores(candidates, scale, stamps, signal, times, weights, group)
        local_offsets.append(float(candidates[int(np.argmax(scores))]))
    local_offsets = np.asarray(local_offsets, dtype=np.float64)
    local_spread = (float(np.std(local_offsets))
                    if len(local_offsets) > 1 else float("inf"))

    # Bootstrap across objects estimates peak stability independently of map
    # section.  This rejects maps where menu/background motion made a peak.
    rng = np.random.default_rng(7)
    bootstrap = []
    candidates = np.arange(best_offset - 60.0, best_offset + 60.1, 2.0)
    sample_size = max(24, int(len(times) * 0.65))
    for _ in range(16):
        ids = rng.choice(len(times), size=sample_size, replace=False)
        scores = _scores(candidates, scale, stamps, signal, times, weights, ids)
        bootstrap.append(float(candidates[int(np.argmax(scores))]))
    bootstrap_spread = float(np.std(bootstrap))
    uncertainty = max(4.0, 0.5 * local_spread, bootstrap_spread)

    transform = TimeTransform(offset_ms=best_offset, scale=float(scale))
    record_hit_times = (times - best_offset) / scale
    covered = int(np.sum((record_hit_times >= stamps[0]) &
                         (record_hit_times <= stamps[-1])))
    result = GeometrySyncResult(
        transform=transform,
        uncertainty_ms=uncertainty,
        score=best_score,
        runner_up_score=runner_up,
        anchors=len(times),
        covered_objects=covered,
        total_objects=len(bm.objects),
        record_start_ms=max(float(stamps[0]),
                            transform.to_record(bm.objects[0].time - bm.preempt_ms)),
        record_end_ms=min(float(stamps[-1]), transform.to_record(
            max(object_end_time(obj) for obj in bm.objects) + 250.0)),
    )
    return result


def save_geometry_sync(session, bm, result):
    payload = {
        "version": 1,
        "beatmap": {"path": bm.path, "title": bm.title,
                    "version": bm.version},
        "clock": {"formula": "beatmap_ms = scale * recording_ms + offset_ms",
                  "scale": result.transform.scale,
                  "offset_ms": result.transform.offset_ms,
                  "uncertainty_ms": result.uncertainty_ms,
                  "method": result.method,
                  "safe": result.is_safe()},
        "quality": {"score": result.score,
                    "runner_up_score": result.runner_up_score,
                    "anchors": result.anchors,
                    "covered_objects": result.covered_objects,
                    "total_objects": result.total_objects},
        "usable_recording_ms": [result.record_start_ms, result.record_end_ms],
    }
    with open(os.path.join(session, "detector_sync.json"), "w",
              encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return payload
