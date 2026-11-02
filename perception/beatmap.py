"""Парсер .osu / .osz для гибридной авторазметки.

Работает и со stable, и с lazer: формат карт одинаковый.
Из карты берем ground-truth: позиции кругов/голов слайдеров,
кривые слайдеров (для ball), длительности спиннеров.
Vision остается главным входом в игре — парсер нужен только
для генерации YOLO-боксов без ручной разметки.

Ч/б скин никак не влияет на парсер (только на картинку),
но упрощает жизнь детектору: меньше дисперсии цвета.
"""
import os
import re
import math
import zipfile
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

OSU_W, OSU_H = 512, 384  # координатная система osu! (stable и lazer)


@dataclass
class TimingPoint:
    time: float          # ms
    beat_length: float   # ms per beat (отрицательные = inherited, игнорим для скорости)
    meter: int = 4
    uninherited: bool = True


@dataclass
class HitObject:
    x: float
    y: float
    time: float          # ms от начала карты
    type: int            # битовые флаги: 1 circle, 2 slider, 8 spinner
    curve_type: str = ""       # B/P/L/C для слайдеров
    curve_points: List[Tuple[float, float]] = field(default_factory=list)
    slides: int = 0
    length: float = 0.0        # длина слайдера в osu-px
    end_time: float = 0.0      # для спиннера/ слайдера (посчитаем после парсинга таймингов)
    pixel_length: float = 0.0
    stack_height: int = 0
    raw_x: float = 0.0
    raw_y: float = 0.0

    @property
    def is_circle(self) -> bool:
        return bool(self.type & 1)

    @property
    def is_slider(self) -> bool:
        return bool(self.type & 2)

    @property
    def is_spinner(self) -> bool:
        return bool(self.type & 8)


@dataclass
class Beatmap:
    path: str
    title: str = ""
    artist: str = ""
    version: str = ""
    circle_size: float = 4.0
    approach_rate: float = 5.0
    slider_multiplier: float = 1.4
    slider_tick_rate: float = 1.0
    stack_leniency: float = 0.7
    timing: List[TimingPoint] = field(default_factory=list)
    objects: List[HitObject] = field(default_factory=list)

    @property
    def preempt_ms(self) -> float:
        """osu! AR-to-preempt conversion in map-clock milliseconds."""
        ar = min(11.0, max(-5.0, self.approach_rate))
        if ar < 5.0:
            return 1200.0 + 120.0 * (5.0 - ar)
        return 1200.0 - 150.0 * (ar - 5.0)

    def beat_length_at(self, time_ms: float) -> float:
        """Активный beatLength (uninherited) на момент времени."""
        bl = -1.0
        for tp in self.timing:
            if tp.time <= time_ms and tp.uninherited:
                bl = tp.beat_length
        return bl if bl > 0 else 500.0

    def slider_velocity_at(self, time_ms: float) -> float:
        """Inherited timing-point slider velocity multiplier at map time."""
        velocity = 1.0
        for tp in self.timing:
            if tp.time > time_ms:
                break
            if tp.uninherited:
                velocity = 1.0
            elif tp.beat_length < 0:
                velocity = min(10.0, max(0.1, -100.0 / tp.beat_length))
        return velocity

    def finalize_sliders(self):
        """Посчитать end_time слайдеров и спиннеров."""
        for o in self.objects:
            if o.is_slider:
                bl = self.beat_length_at(o.time)
                # длительность одного прохода слайдера в ms
                sv = self.slider_velocity_at(o.time)
                dur = (o.length / (self.slider_multiplier * 100.0 * sv)) * bl
                o.end_time = o.time + dur * max(1, o.slides)
            # спиннер: end_time уже лежит в поле (5-й параметр)


def _parse_section(lines: List[str], name: str) -> List[str]:
    out, active = [], False
    for ln in lines:
        s = ln.strip()
        if s.startswith("["):
            active = s.lower() == f"[{name.lower()}]"
            continue
        if active:
            if s == "" or s.startswith("//"):
                continue
            out.append(s)
    return out


def parse_osu_file(path: str) -> Beatmap:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    bm = Beatmap(path=path)

    for ln in lines:
        s = ln.strip()
        if s.startswith("Title:"):
            bm.title = s.split(":", 1)[1].strip()
        elif s.startswith("Artist:"):
            bm.artist = s.split(":", 1)[1].strip()
        elif s.startswith("Version:"):
            bm.version = s.split(":", 1)[1].strip()
        elif s.startswith("CircleSize:"):
            try:
                bm.circle_size = float(s.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif s.startswith("ApproachRate:"):
            try:
                bm.approach_rate = float(s.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif s.startswith("SliderMultiplier:"):
            try:
                bm.slider_multiplier = float(s.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif s.startswith("SliderTickRate:"):
            try:
                bm.slider_tick_rate = float(s.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif s.startswith("StackLeniency:"):
            try:
                bm.stack_leniency = float(s.split(":", 1)[1].strip())
            except ValueError:
                pass

    for row in _parse_section(lines, "TimingPoints"):
        p = row.split(",")
        if len(p) < 2:
            continue
        try:
            t = float(p[0]); bl = float(p[1])
            uninherited = len(p) < 7 or p[6].strip() == "1"
            meter = int(p[2]) if len(p) > 2 else 4
            bm.timing.append(TimingPoint(t, bl, meter, uninherited))
        except ValueError:
            continue
    bm.timing.sort(key=lambda tp: tp.time)

    for row in _parse_section(lines, "HitObjects"):
        p = row.split(",")
        if len(p) < 5:
            continue
        try:
            x, y, t = float(p[0]), float(p[1]), float(p[2])
            typ = int(p[3])
        except ValueError:
            continue
        o = HitObject(x=x, y=y, time=t, type=typ, raw_x=x, raw_y=y)
        if o.is_slider and len(p) > 5:
            # формат: curveType|p1|p2...,slides,length
            curve_raw = p[5]
            parts = curve_raw.split("|")
            o.curve_type = parts[0] if parts else ""
            for pt in parts[1:]:
                if ":" in pt:
                    try:
                        px, py = pt.split(":")
                        o.curve_points.append((float(px), float(py)))
                    except ValueError:
                        pass
            try:
                o.slides = int(p[6]) if len(p) > 6 else 1
                o.length = float(p[7]) if len(p) > 7 else 0.0
            except ValueError:
                pass
        if o.is_spinner and len(p) > 5:
            try:
                o.end_time = float(p[5])
                o.x, o.y = OSU_W / 2, OSU_H / 2  # спиннер всегда по центру
                o.raw_x, o.raw_y = o.x, o.y
            except ValueError:
                pass
        bm.objects.append(o)

    bm.objects.sort(key=lambda o: o.time)
    bm.finalize_sliders()
    apply_stacking(bm)
    return bm


def apply_stacking(bm: Beatmap):
    """osu! stable/lazer stacking calculation (v6+).
    Calculates diagonal visual stack offset for overlapping hit objects."""
    objects = bm.objects
    if not objects:
        return
    for o in objects:
        o.stack_height = 0
    preempt = bm.preempt_ms
    stack_leniency = getattr(bm, "stack_leniency", 0.7)
    stack_threshold = int(preempt) * stack_leniency

    extended_end = len(objects) - 1
    for i in range(len(objects) - 1, -1, -1):
        stack_base = i
        for n in range(stack_base + 1, len(objects)):
            sb = objects[stack_base]
            if sb.is_spinner:
                break
            on = objects[n]
            if on.is_spinner:
                continue
            end_t = sb.end_time if sb.is_slider else sb.time
            if on.time - end_t > stack_threshold:
                break
            sb_end = slider_tail_pos(sb) if sb.is_slider else (sb.x, sb.y)
            if math.hypot(sb.x - on.x, sb.y - on.y) < 3.0 or (sb.is_slider and math.hypot(sb_end[0] - on.x, sb_end[1] - on.y) < 3.0):
                stack_base = n
                on.stack_height = 0
        if stack_base > extended_end:
            extended_end = stack_base
            if extended_end == len(objects) - 1:
                break

    extended_start = 0
    for i in range(extended_end, 0, -1):
        n = i
        obj_i = objects[i]
        if obj_i.stack_height != 0 or obj_i.is_spinner:
            continue
        if obj_i.is_circle:
            while True:
                n -= 1
                if n < 0:
                    break
                obj_n = objects[n]
                if obj_n.is_spinner:
                    continue
                end_t = obj_n.end_time if obj_n.is_slider else obj_n.time
                if int(obj_i.time) - int(end_t) > stack_threshold:
                    break
                if n < extended_start:
                    obj_n.stack_height = 0
                    extended_start = n
                obj_n_end = slider_tail_pos(obj_n) if obj_n.is_slider else (obj_n.x, obj_n.y)
                if obj_n.is_slider and math.hypot(obj_n_end[0] - obj_i.x, obj_n_end[1] - obj_i.y) < 3.0:
                    offset = obj_i.stack_height - obj_n.stack_height + 1
                    for j in range(n + 1, i + 1):
                        oj = objects[j]
                        if math.hypot(obj_n_end[0] - oj.x, obj_n_end[1] - oj.y) < 3.0:
                            oj.stack_height -= offset
                    break
                if math.hypot(obj_n.x - obj_i.x, obj_n.y - obj_i.y) < 3.0:
                    obj_n.stack_height = obj_i.stack_height + 1
                    obj_i = obj_n
        elif obj_i.is_slider:
            while True:
                n -= 1
                if n < 0:
                    break
                obj_n = objects[n]
                if obj_n.is_spinner:
                    continue
                if obj_i.time - obj_n.time > stack_threshold:
                    break
                obj_n_end = slider_tail_pos(obj_n) if obj_n.is_slider else (obj_n.x, obj_n.y)
                if math.hypot(obj_n_end[0] - obj_i.x, obj_n_end[1] - obj_i.y) < 3.0:
                    obj_n.stack_height = obj_i.stack_height + 1
                    obj_i = obj_n

    scale = (54.4 - 4.48 * bm.circle_size) / 64.0
    stack_offset_factor = scale * -6.4
    for o in objects:
        if o.stack_height != 0:
            offset = o.stack_height * stack_offset_factor
            o.x += offset
            o.y += offset
            if o.is_slider:
                o.curve_points = [(px + offset, py + offset) for (px, py) in o.curve_points]
                o._pixel_path = None


def parse_osz(path: str, keep_dir: Optional[str] = None) -> List[Beatmap]:
    """Распаковать .osz, вернуть все сложности (.osu внутри)."""
    out_dir = keep_dir or tempfile.mkdtemp(prefix="osz_")
    with zipfile.ZipFile(path, "r") as z:
        z.extractall(out_dir)
    maps = []
    for root, _, files in os.walk(out_dir):
        for fn in files:
            if fn.lower().endswith(".osu"):
                try:
                    maps.append(parse_osu_file(os.path.join(root, fn)))
                except Exception as e:
                    print(f"[beatmap] skip {fn}: {e}")
    return maps


# ─── Позиция шара слайдера (для класса slider_ball) ──────────────
def _bezier(controls, t):
    points = [list(point) for point in controls]
    while len(points) > 1:
        points = [[a[0] + (b[0] - a[0]) * t,
                   a[1] + (b[1] - a[1]) * t]
                  for a, b in zip(points[:-1], points[1:])]
    return tuple(points[0])


def _raw_slider_path(o: HitObject):
    points = [(o.x, o.y)] + o.curve_points
    if len(points) < 2:
        return points
    curve = o.curve_type.upper()
    if curve == "P" and len(points) == 3:
        (ax, ay), (bx, by), (cx, cy) = points
        d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) > 1e-6:
            aa, bb, cc = ax*ax + ay*ay, bx*bx + by*by, cx*cx + cy*cy
            ux = (aa*(by-cy) + bb*(cy-ay) + cc*(ay-by)) / d
            uy = (aa*(cx-bx) + bb*(ax-cx) + cc*(bx-ax)) / d
            start = math.atan2(ay-uy, ax-ux)
            middle = math.atan2(by-uy, bx-ux)
            end = math.atan2(cy-uy, cx-ux)
            ccw = (end - start) % math.tau
            delta = ccw if (middle - start) % math.tau <= ccw else ccw - math.tau
            radius = math.hypot(ax-ux, ay-uy)
            samples = max(24, int(abs(delta) * radius / 2.0) + 1)
            return [(ux + radius * math.cos(start + delta*i/samples),
                     uy + radius * math.sin(start + delta*i/samples))
                    for i in range(samples + 1)]
    if curve == "B":
        segments, start = [], 0
        for index in range(1, len(points)):
            if points[index] == points[index - 1]:
                if index - start >= 2:
                    segments.append(points[start:index])
                start = index
        if len(points) - start >= 2:
            segments.append(points[start:])
        path = []
        for controls in segments or [points]:
            samples = max(20, 12 * (len(controls) - 1))
            section = [_bezier(controls, i / samples) for i in range(samples + 1)]
            path.extend(section if not path else section[1:])
        return path
    if curve == "C" and len(points) >= 2:
        padded = [points[0], *points, points[-1]]
        path = []
        for index in range(1, len(padded) - 2):
            p0, p1, p2, p3 = padded[index - 1:index + 3]
            for step in range(21):
                t = step / 20.0
                t2, t3 = t*t, t*t*t
                x = 0.5 * ((2*p1[0]) + (-p0[0]+p2[0])*t +
                           (2*p0[0]-5*p1[0]+4*p2[0]-p3[0])*t2 +
                           (-p0[0]+3*p1[0]-3*p2[0]+p3[0])*t3)
                y = 0.5 * ((2*p1[1]) + (-p0[1]+p2[1])*t +
                           (2*p0[1]-5*p1[1]+4*p2[1]-p3[1])*t2 +
                           (-p0[1]+3*p1[1]-3*p2[1]+p3[1])*t3)
                path.append((x, y))
        return path
    return points


def _slider_path(o: HitObject):
    cached = getattr(o, "_pixel_path", None)
    if cached is not None:
        return cached
    raw = _raw_slider_path(o)
    if len(raw) < 2:
        o._pixel_path = raw
        return raw
    output = [raw[0]]
    remaining = max(0.0, o.pixel_length or o.length)
    for a, b in zip(raw[:-1], raw[1:]):
        length = math.hypot(b[0]-a[0], b[1]-a[1])
        if length <= 1e-9:
            continue
        if length >= remaining:
            ratio = remaining / length
            output.append((a[0] + (b[0]-a[0])*ratio,
                           a[1] + (b[1]-a[1])*ratio))
            remaining = 0.0
            break
        output.append(b)
        remaining -= length
    if remaining > 1e-6 and len(output) >= 2:
        a, b = output[-2], output[-1]
        length = math.hypot(b[0]-a[0], b[1]-a[1])
        if length > 1e-9:
            output.append((b[0] + (b[0]-a[0])/length*remaining,
                           b[1] + (b[1]-a[1])/length*remaining))
    o._pixel_path = output
    return output


def slider_ball_pos(o: HitObject, time_ms: float) -> Tuple[float, float]:
    """Грубая позиция ball: линейная интерполяция по полилинии
    head -> curve_points. Точности хватает для YOLO-боксов
    (пиксель-в-пиксель не нужен, детектор доучит по картинке)."""
    if not o.is_slider or o.end_time <= o.time:
        return o.x, o.y
    prog = (time_ms - o.time) / max(1.0, o.end_time - o.time)
    # туда-обратно при slides > 1
    seg = prog * max(1, o.slides)
    leg = seg % 2.0
    if leg > 1.0:
        leg = 2.0 - leg
    pts = _slider_path(o)
    if len(pts) < 2:
        return o.x, o.y
    # длины сегментов полилинии
    seg_lens = []
    total = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        L = ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
        seg_lens.append(L)
        total += L
    if total <= 0:
        return o.x, o.y
    target = leg * total
    for (a, b), L in zip(zip(pts[:-1], pts[1:]), seg_lens):
        if target <= L:
            k = target / max(L, 1e-6)
            return a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k
        target -= L
    return pts[-1]


def objects_at(bm: Beatmap, time_ms: float,
               preempt: float = 1200.0) -> List[Tuple[HitObject, str, float, float]]:
    """Активные объекты в момент time_ms.
    Возвращает (obj, yolo_cls, x_osu, y_osu).
    preempt — за сколько ms до удара круг уже виден (approach)."""
    active = []
    for o in bm.objects:
        if o.is_circle:
            if o.time - preempt <= time_ms <= o.time + 150:
                active.append((o, "circle", o.x, o.y))
        elif o.is_slider:
            if o.time - preempt <= time_ms <= o.end_time + 150:
                if time_ms < o.time:
                    active.append((o, "slider_head", o.x, o.y))
                elif time_ms <= o.end_time:
                    bx, by = slider_ball_pos(o, time_ms)
                    active.append((o, "slider_ball", bx, by))
                else:
                    tx, ty = slider_tail_pos(o)
                    active.append((o, "slider_tail", tx, ty))
        elif o.is_spinner:
            if o.time <= time_ms <= o.end_time:
                active.append((o, "spinner", o.x, o.y))
    return active


def slider_tail_pos(o: HitObject) -> Tuple[float, float]:
    pts = _slider_path(o)
    if o.slides % 2 == 0 and len(pts) >= 1:
        return pts[0]  # четное число проходов — возврат в голову
    return pts[-1] if pts else (o.x, o.y)


def circle_radius_osu_px(bm: Beatmap) -> float:
    """Радиус хит-круга в osu-пикселях по формуле игры: r = 54.4 - 4.48*CS."""
    return max(8.0, 54.4 - 4.48 * bm.circle_size)
