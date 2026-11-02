"""Фичи для политики из треков + курсора.

Вектор (MAX_TRACKS=8 треков x 6 + 4):
  на трек: [x, y, vx*0.1, vy*0.1, urgency, cls_id/5]
  хвост: [cursor_x, cursor_y, cursor_vx*0.1, clicking]
Все координаты 0..1 (infer-кадр). Сортировка треков — по urgency (трекер).
"""
import numpy as np
from typing import List, Tuple
from perception.tracker import Track

MAX_TRACKS = 8
CLS_ORDER = ["circle", "slider_head", "slider_ball", "slider_tail", "spinner", "cursor"]
# per-track layout v3: [x, y, vx*0.1, vy*0.1, w, h, urgency, conf, cls0..5] = 14
# keeps absolute position + box size (CS) + visual-timing + class one-hot.
PER_TRACK_V2 = 6
PER_TRACK_V3 = 14
CLS_DIM = len(CLS_ORDER)


def _track_to_v2(t: Track):
    try:
        cid = CLS_ORDER.index(t.cls) / max(len(CLS_ORDER) - 1, 1)
    except ValueError:
        cid = 0.0
    return [t.x, t.y, t.vx * 0.1, t.vy * 0.1, t.urgency, cid]


def _track_to_v3(t: Track):
    vec = [t.x, t.y, t.vx * 0.1, t.vy * 0.1,
           float(t.w), float(t.h), float(t.urgency), float(t.conf)]
    onehot = [0.0] * CLS_DIM
    try:
        onehot[CLS_ORDER.index(t.cls)] = 1.0
    except ValueError:
        pass
    vec.extend(onehot)
    return vec


def build_features(tracks: List[Track],
                   cursor: Tuple[float, float, float, float, bool]) -> np.ndarray:
    """Legacy 52-dim builder — kept for loading old datasets/weights."""
    cx, cy, cvx, cvy, clicking = cursor
    feats = []
    for i in range(MAX_TRACKS):
        if i < len(tracks):
            feats += _track_to_v2(tracks[i])
        else:
            feats += [0.0] * PER_TRACK_V2
    feats += [cx, cy, cvx * 0.1, float(clicking)]
    return np.array(feats, dtype=np.float32)


def build_features_v3(tracks: List[Track],
                      cursor: Tuple[float, float, float, float, bool]) -> np.ndarray:
    cx, cy, cvx, cvy, clicking = cursor
    feats = []
    for i in range(MAX_TRACKS):
        if i < len(tracks):
            feats += _track_to_v3(tracks[i])
        else:
            feats += [0.0] * PER_TRACK_V3
    feats += [cx, cy, cvx * 0.1, cvy * 0.1, float(clicking)]
    return np.array(feats, dtype=np.float32)


FEATURE_DIM = MAX_TRACKS * PER_TRACK_V2 + 4          # 52, legacy
FEATURE_DIM_V3 = MAX_TRACKS * PER_TRACK_V3 + 5       # 117, v3
# default for new training; model loader accepts both
DEFAULT_FEATURE_DIM = FEATURE_DIM_V3
