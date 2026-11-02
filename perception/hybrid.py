"""Geometry fallback for the fixed high-contrast osu! skin.

The synthetic YOLO has a sizeable real-lazer domain gap.  This detector uses
concentric circle evidence to recover hit-object centres and is merged with
YOLO rather than replacing learned spinner/cursor/slider classes.
"""
import cv2
import numpy as np

from perception.schema import Det


def _cluster(circles, merge_px=24.0):
    groups = []
    for x, y, radius in sorted(circles, key=lambda row: row[2]):
        group = next((item for item in groups
                      if (item[0] - x) ** 2 + (item[1] - y) ** 2 < merge_px ** 2), None)
        if group is None:
            groups.append([float(x), float(y), float(radius), 1])
        else:
            count = group[3]
            group[0] = (group[0] * count + x) / (count + 1)
            group[1] = (group[1] * count + y) / (count + 1)
            group[2] = min(group[2], float(radius))
            group[3] += 1
    return groups


def detect_hit_circles(frame):
    """Return conservative circle centres supported by >=2 concentric edges."""
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.25, minDist=22,
        param1=90, param2=24, minRadius=10, maxRadius=65,
    )
    if circles is None:
        return []
    raw = np.round(circles[0]).astype(int)
    result = []
    for x, y, radius, support in _cluster(raw):
        x, y = int(round(x)), int(round(y))
        if support < 2 or x < 12 or y < 12 or x >= width - 12 or y >= height - 12:
            continue
        patch = gray[max(0, y - 8):min(height, y + 9),
                     max(0, x - 8):min(width, x + 9)]
        centre = float(patch.mean()) if patch.size else 0.0
        # Slider curves form circular arcs too, but their empty centres are
        # nearly black.  Hit circles on this skin retain a grey/white body.
        if centre < 12.0:
            continue
        confidence = min(.88, .48 + .07 * support + .0015 * centre)
        result.append(Det(
            "circle", x / width, y / height,
            54.0 / width, 50.0 / height, confidence,
            urgency=1.0 if centre >= 65.0 else None,
            source="geometry",
        ))
    return result


def augment_detections(frame, detections, prefer_geometry=False):
    """Merge geometry hits with YOLO, correcting overlapping tail mistakes."""
    geometry = detect_hit_circles(frame)
    if prefer_geometry and geometry:
        # Synchronization needs reliable hit centres, not speculative slider
        # sub-classes. Keep only unambiguous non-hit YOLO classes alongside it.
        return geometry + [det for det in detections if det.cls in ("spinner", "cursor")]
    merged = list(detections)
    for hit in geometry:
        nearest = None
        for index, det in enumerate(merged):
            distance = ((det.x - hit.x) ** 2 + (det.y - hit.y) ** 2) ** .5
            if distance < .045 and (nearest is None or distance < nearest[0]):
                nearest = (distance, index)
        if nearest is None:
            merged.append(hit)
            continue
        det = merged[nearest[1]]
        if det.cls in ("circle", "slider_head"):
            det.conf = max(det.conf, hit.conf)
        elif det.cls == "slider_tail":
            # The current synthetic model systematically calls bright real
            # hit heads slider_tail.  Concentric-body evidence corrects it.
            merged[nearest[1]] = hit
        # Keep YOLO slider_ball/spinner/cursor semantics when they are present.
    return merged
