"""Align detected gameplay frames with a beatmap timeline.

The synchronizer searches an affine mapping::

    beatmap_ms = scale * recording_ms + offset_ms

using object class and normalized position agreement.  Each recording is
aligned independently, so different beatmaps and dropped video frames do not
share a fragile global frame offset.
"""

from dataclasses import dataclass
from collections import Counter, defaultdict
import math
from statistics import median, pstdev
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

from perception.beatmap import Beatmap, OSU_H, OSU_W, objects_at
from training.action_timeline import TimeTransform


@dataclass(frozen=True)
class ObservedObject:
    cls: str
    x: float
    y: float
    confidence: float = 1.0
    w: float = 0.05
    h: float = 0.05
    # Optional approach-circle estimate in beatmap milliseconds.  Object
    # presence gives frame-level alignment; this value removes the long
    # visibility plateau when a skin makes the approach ring measurable.
    time_to_hit_ms: Optional[float] = None
    approach_progress: Optional[float] = None


@dataclass(frozen=True)
class FrameObservation:
    record_ms: float
    objects: Tuple[ObservedObject, ...]


@dataclass(frozen=True)
class SyncResult:
    transform: TimeTransform
    score: float
    runner_up_score: float
    observations_used: int
    max_time_uncertainty_ms: float = float("inf")
    method: str = "visual_presence"

    @property
    def margin(self) -> float:
        return self.score - self.runner_up_score

    def is_click_safe(self, max_error_ms: float = 17.0) -> bool:
        """Whether alignment is precise enough to create click targets."""
        return self.max_time_uncertainty_ms <= max_error_ms


def _compatible(observed: str, expected: str) -> bool:
    if observed == expected:
        return True
    hit_heads = {"circle", "slider_head"}
    if observed in hit_heads and expected in hit_heads:
        return True
    slider = {"slider_head", "slider_ball", "slider_tail"}
    return observed in slider and expected in slider


def alignment_score(
    observations: Sequence[FrameObservation],
    bm: Beatmap,
    transform: TimeTransform,
    preempt_ms: float = 1200.0,
    distance_sigma: float = 0.055,
    timing_sigma_ms: float = 35.0,
) -> float:
    """Score class/position agreement for one candidate time transform."""
    total = 0.0
    used_frames = 0
    sigma2 = max(1e-9, distance_sigma * distance_sigma)
    for frame in observations:
        if not frame.objects:
            continue
        expected = objects_at(bm, transform.to_map(frame.record_ms), preempt_ms)
        map_ms = transform.to_map(frame.record_ms)
        expected_norm = [
            (obj, cls, x / OSU_W, y / OSU_H) for obj, cls, x, y in expected
        ]
        if not expected_norm:
            total -= 0.35
            used_frames += 1
            continue
        available = set(range(len(expected_norm)))
        frame_score = 0.0
        for det in sorted(frame.objects, key=lambda item: item.confidence, reverse=True):
            best = None
            for index in available:
                obj, cls, x, y = expected_norm[index]
                if not _compatible(det.cls, cls):
                    continue
                distance2 = (det.x - x) ** 2 + (det.y - y) ** 2
                if best is None or distance2 < best[0]:
                    best = (distance2, index)
            if best is None:
                frame_score -= 0.25 * det.confidence
                continue
            distance2, index = best
            available.remove(index)
            position_score = math.exp(-distance2 / (2.0 * sigma2))
            timing_score = 1.0
            observed_tth = det.time_to_hit_ms
            if observed_tth is None and det.approach_progress is not None:
                observed_tth = (1.0 - det.approach_progress) * preempt_ms
            if observed_tth is not None:
                obj = expected_norm[index][0]
                expected_tth = obj.time - map_ms
                residual = observed_tth - expected_tth
                timing_score = math.exp(
                    -(residual * residual) / (2.0 * timing_sigma_ms * timing_sigma_ms)
                )
            frame_score += det.confidence * position_score * timing_score
        frame_score -= 0.12 * len(available)
        total += frame_score / max(1, len(frame.objects), len(expected_norm))
        used_frames += 1
    return total / max(1, used_frames)


def estimate_sync(
    observations: Sequence[FrameObservation],
    bm: Beatmap,
    offset_range_ms: Tuple[float, float] = (-15000.0, 15000.0),
    coarse_step_ms: float = 50.0,
    fine_step_ms: float = 5.0,
    scales: Sequence[float] = (0.749, 0.75, 0.751,
                               0.999, 1.0, 1.001,
                               1.499, 1.5, 1.501),
    preempt_ms: float = 1200.0,
    uncertainty_score_drop: float = 0.005,
) -> SyncResult:
    """Estimate recording-to-map time alignment with coarse-to-fine search."""
    if not observations or not any(frame.objects for frame in observations):
        raise ValueError("at least one non-empty frame observation is required")
    low, high = offset_range_ms
    if high < low or coarse_step_ms <= 0 or fine_step_ms <= 0:
        raise ValueError("invalid synchronization search range")

    ranked: List[Tuple[float, TimeTransform]] = []
    for scale in scales:
        offset = float(low)
        while offset <= high + 1e-9:
            transform = TimeTransform(offset_ms=offset, scale=float(scale))
            ranked.append((alignment_score(observations, bm, transform, preempt_ms), transform))
            offset += coarse_step_ms
    ranked.sort(key=lambda item: item[0], reverse=True)
    coarse_best = ranked[0][1]

    fine: List[Tuple[float, TimeTransform]] = []
    fine_low = coarse_best.offset_ms - coarse_step_ms
    fine_high = coarse_best.offset_ms + coarse_step_ms
    for scale in scales:
        offset = fine_low
        while offset <= fine_high + 1e-9:
            transform = TimeTransform(offset_ms=offset, scale=float(scale))
            fine.append((alignment_score(observations, bm, transform, preempt_ms), transform))
            offset += fine_step_ms
    fine.sort(key=lambda item: item[0], reverse=True)
    best_score, best_transform = fine[0]
    runner_up = next(
        (score for score, transform in fine[1:]
         if abs(transform.offset_ms - best_transform.offset_ms) >= coarse_step_ms / 2),
        fine[1][0] if len(fine) > 1 else best_score,
    )
    # Scores often have a flat top: a circle is visible for ~1.35 seconds, so
    # "same objects are present" is not a click-accurate time signal.  Retain
    # every near-optimal transform and report the worst disagreement at the
    # recording endpoints.  Dataset generation can then refuse unsafe labels.
    near = [
        transform for score, transform in fine
        if score >= best_score - max(0.0, uncertainty_score_drop)
    ]
    record_times = [frame.record_ms for frame in observations if frame.objects]
    endpoints = (min(record_times), max(record_times))
    uncertainty = 0.0
    for transform in near:
        for record_ms in endpoints:
            uncertainty = max(
                uncertainty,
                abs(transform.to_map(record_ms) - best_transform.to_map(record_ms)),
            )
    uncertainty += fine_step_ms / 2.0
    return SyncResult(
        transform=best_transform,
        score=best_score,
        runner_up_score=runner_up,
        observations_used=sum(bool(frame.objects) for frame in observations),
        max_time_uncertainty_ms=uncertainty,
    )


def estimate_auto_anchor_sync(
    observations: Sequence[FrameObservation],
    bm: Beatmap,
    scale: float = 1.0,
    distance_limit: float = 0.07,
    search_range_ms: Tuple[float, float] = (-15000.0, 15000.0),
) -> Optional[SyncResult]:
    """Align Auto play from bright cursor-on-hit circle observations."""
    rows = []
    low, high = search_range_ms
    hittable = [(index, obj) for index, obj in enumerate(bm.objects)
                if not obj.is_spinner]
    for frame in observations:
        for det in frame.objects:
            if (det.cls not in ("circle", "slider_head") or
                    det.approach_progress is None or det.approach_progress < .99):
                continue
            for index, obj in hittable:
                distance = math.hypot(det.x - obj.x / OSU_W,
                                      det.y - obj.y / OSU_H)
                if distance > distance_limit:
                    continue
                offset = obj.time - scale * frame.record_ms
                if low <= offset <= high:
                    rows.append((offset, index))
    if len(rows) < 20:
        return None
    bins = Counter(round(offset / 10.0) * 10.0 for offset, _ in rows)
    peak, peak_votes = bins.most_common(1)[0]
    by_object = defaultdict(list)
    for offset, index in rows:
        if abs(offset - peak) <= 180.0:
            by_object[index].append(offset)
    object_offsets = [median(values) for values in by_object.values()
                      if len(values) >= 2]
    if len(object_offsets) < 12:
        return None
    centre = median(object_offsets)
    mad = median(abs(value - centre) for value in object_offsets)
    cutoff = max(35.0, 3.0 * 1.4826 * mad)
    inliers = [value for value in object_offsets if abs(value - centre) <= cutoff]
    if len(inliers) < 12:
        return None
    centre = float(median(inliers))
    spread = float(pstdev(inliers)) if len(inliers) > 1 else 0.0
    uncertainty = max(8.5, 1.253 * spread / math.sqrt(len(inliers)) + 5.0)
    return SyncResult(
        transform=TimeTransform(offset_ms=centre, scale=float(scale)),
        score=float(len(inliers)), runner_up_score=float(peak_votes),
        observations_used=len(inliers),
        max_time_uncertainty_ms=uncertainty,
        method="auto_bright_anchor",
    )


def identify_beatmap(
    observations: Sequence[FrameObservation],
    candidates: Mapping[str, Beatmap],
    **sync_kwargs,
) -> Tuple[str, SyncResult]:
    """Choose the map whose best alignment has the highest observation score."""
    if not candidates:
        raise ValueError("at least one candidate beatmap is required")
    results = {
        name: estimate_sync(observations, bm, **sync_kwargs)
        for name, bm in candidates.items()
    }
    return max(results.items(), key=lambda item: item[1].score)
