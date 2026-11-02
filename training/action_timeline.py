"""Continuous-time action targets derived from an osu! beatmap.

Clicks are events on the beatmap clock, not labels attached to video frame
numbers.  A recording-specific :class:`TimeTransform` maps capture timestamps
onto that clock.  This keeps labels stable when capture FPS jitters or frames
are dropped.
"""

from bisect import bisect_right
from dataclasses import dataclass
import math
from typing import Iterable, List, Sequence, Tuple

from perception.beatmap import Beatmap, HitObject, OSU_H, OSU_W, slider_ball_pos


@dataclass(frozen=True)
class TimeTransform:
    """Affine mapping from recording time to beatmap time, in milliseconds."""

    offset_ms: float = 0.0
    scale: float = 1.0

    def to_map(self, record_ms: float) -> float:
        return self.scale * float(record_ms) + self.offset_ms

    def to_record(self, map_ms: float) -> float:
        return (float(map_ms) - self.offset_ms) / self.scale


@dataclass(frozen=True)
class KeyEvent:
    time_ms: float
    key: str
    down: bool
    object_index: int


@dataclass(frozen=True)
class ActionTarget:
    record_ms: float
    map_ms: float
    cursor_x: float
    cursor_y: float
    z_down: bool
    x_down: bool
    next_hit_ms: float
    hold: bool
    object_index: int


def object_end_time(obj: HitObject, click_hold_ms: float = 36.0) -> float:
    if obj.is_slider or obj.is_spinner:
        return max(obj.time, obj.end_time)
    return obj.time + click_hold_ms


def build_key_schedule(bm: Beatmap, click_hold_ms: float = 36.0) -> List[KeyEvent]:
    """Create exact down/up events, alternating Z/X between hit objects."""
    events: List[KeyEvent] = []
    for index, obj in enumerate(bm.objects):
        key = "z" if index % 2 == 0 else "x"
        events.append(KeyEvent(obj.time, key, True, index))
        events.append(KeyEvent(object_end_time(obj, click_hold_ms), key, False, index))
    # Release before press at the same timestamp to avoid a stuck key.
    events.sort(key=lambda e: (e.time_ms, e.down, e.object_index))
    return events


def _smoothstep(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def _endpoint(obj: HitObject) -> Tuple[float, float]:
    if obj.is_slider:
        return slider_ball_pos(obj, object_end_time(obj))
    return obj.x, obj.y


def cursor_target_at(
    bm: Beatmap,
    map_ms: float,
    move_lead_ms: float = 450.0,
    spinner_radius: float = 70.0,
    spinner_rps: float = 5.0,
) -> Tuple[float, float, int]:
    """Return an ideal normalized cursor target and its owning object index.

    Slider paths come from the beatmap. Spinner motion is deterministic. Between
    objects, a smooth minimum-jerk-like interpolation moves toward the next hit.
    """
    objects = bm.objects
    if not objects:
        return 0.5, 0.5, -1

    starts = [o.time for o in objects]
    current = max(0, bisect_right(starts, map_ms) - 1)
    obj = objects[current]

    if obj.is_slider and obj.time <= map_ms <= obj.end_time:
        x, y = slider_ball_pos(obj, map_ms)
        return x / OSU_W, y / OSU_H, current
    if obj.is_spinner and obj.time <= map_ms <= obj.end_time:
        angle = (map_ms - obj.time) / 1000.0 * spinner_rps * math.tau
        x = obj.x + spinner_radius * math.cos(angle)
        y = obj.y + spinner_radius * math.sin(angle)
        return x / OSU_W, y / OSU_H, current

    next_index = bisect_right(starts, map_ms)
    if next_index >= len(objects):
        x, y = _endpoint(objects[-1])
        return x / OSU_W, y / OSU_H, len(objects) - 1

    nxt = objects[next_index]
    if current == next_index:
        return nxt.x / OSU_W, nxt.y / OSU_H, next_index

    prev = objects[next_index - 1] if next_index > 0 else nxt
    px, py = _endpoint(prev)
    t_start = max(object_end_time(prev), nxt.time - move_lead_ms)
    if map_ms <= t_start:
        return px / OSU_W, py / OSU_H, next_index - 1
    if map_ms >= nxt.time:
        return nxt.x / OSU_W, nxt.y / OSU_H, next_index
    alpha = _smoothstep((map_ms - t_start) / max(1.0, nxt.time - t_start))
    x = px + (nxt.x - px) * alpha
    y = py + (nxt.y - py) * alpha
    return x / OSU_W, y / OSU_H, next_index


def _key_state(events: Sequence[KeyEvent], map_ms: float) -> Tuple[bool, bool]:
    z_down = False
    x_down = False
    for event in events:
        if event.time_ms > map_ms:
            break
        if event.key == "z":
            z_down = event.down
        else:
            x_down = event.down
    return z_down, x_down


def build_action_targets(
    bm: Beatmap,
    record_timestamps_ms: Iterable[float],
    transform: TimeTransform,
    click_hold_ms: float = 36.0,
) -> List[ActionTarget]:
    """Sample ideal state labels while retaining exact click times separately."""
    events = build_key_schedule(bm, click_hold_ms)
    starts = [o.time for o in bm.objects]
    targets: List[ActionTarget] = []
    for record_ms in record_timestamps_ms:
        map_ms = transform.to_map(record_ms)
        cursor_x, cursor_y, owner = cursor_target_at(bm, map_ms)
        z_down, x_down = _key_state(events, map_ms)
        next_index = bisect_right(starts, map_ms)
        if next_index < len(starts):
            next_hit_ms = starts[next_index] - map_ms
        elif owner >= 0:
            next_hit_ms = starts[owner] - map_ms
        else:
            next_hit_ms = float("inf")

        hold = owner >= 0 and (
            bm.objects[owner].is_slider or bm.objects[owner].is_spinner
        ) and bm.objects[owner].time <= map_ms <= bm.objects[owner].end_time

        # During an active slider, next_hit_ms reflects time to slider completion or next repeat/tick
        if hold and bm.objects[owner].is_slider:
            slider_rem = bm.objects[owner].end_time - map_ms
            next_hit_ms = min(next_hit_ms, slider_rem)

        targets.append(ActionTarget(
            record_ms=float(record_ms),
            map_ms=map_ms,
            cursor_x=min(1.0, max(0.0, cursor_x)),
            cursor_y=min(1.0, max(0.0, cursor_y)),
            z_down=z_down,
            x_down=x_down,
            next_hit_ms=next_hit_ms,
            hold=hold,
            object_index=owner,
        ))
    return targets

