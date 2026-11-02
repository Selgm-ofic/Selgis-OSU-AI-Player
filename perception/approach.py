"""Approach-ring phase estimator for the project's high-contrast B/W skin."""
import cv2
import numpy as np

from perception.schema import HIT_CLASSES


def estimate_progress(frame, det, box_scale=1.15):
    """Return 0..1 approach progress, or None when no reliable ring is found."""
    if det.cls not in HIT_CLASSES:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    height, width = gray.shape[:2]
    cx, cy = det.x * width, det.y * height
    base_radius = min(det.w * width, det.h * height) / (2.0 * max(box_scale, 0.1))
    if base_radius < 3.0:
        return None
    angles = np.linspace(0.0, np.pi * 2.0, 96, endpoint=False)
    radii = np.linspace(base_radius * 1.03, base_radius * 3.65, 72)
    xs = np.rint(cx + radii[:, None] * np.cos(angles)[None, :]).astype(np.int32)
    ys = np.rint(cy + radii[:, None] * np.sin(angles)[None, :]).astype(np.int32)
    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    pixels = gray[np.clip(ys, 0, height - 1),
                  np.clip(xs, 0, width - 1)].astype(np.float32)
    pixels[~valid] = 0.0
    counts = np.maximum(1, valid.sum(axis=1))
    means = pixels.sum(axis=1) / counts
    bright_fraction = ((pixels > 135.0) & valid).sum(axis=1) / counts
    scores = means * (0.35 + bright_fraction)
    scores[valid.sum(axis=1) < 48] = 0.0
    best_index = int(np.argmax(scores))
    if scores[best_index] < 42.0:
        return None
    ratio = radii[best_index] / base_radius
    return float(np.clip(1.0 - (ratio - 1.0) / 2.5, 0.0, 1.0))


def annotate_approach(frame, detections, box_scale=1.15):
    for det in detections:
        # Geometry detections already encode a conservative hit-now marker
        # from the bright Auto cursor/body. Applying the radial estimator to
        # their synthetic fixed box produced convincing but false phases.
        if det.source == "geometry":
            continue
        progress = estimate_progress(frame, det, box_scale)
        if progress is not None:
            det.urgency = progress
    return detections
