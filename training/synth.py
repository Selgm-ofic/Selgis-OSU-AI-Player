"""Офлайн-генератор YOLO-датасета прямо из .osu — без OBS и записи.

Идея: вместо того чтобы играть карту и писать экран (неудобно + синхра
плавает), рендерим карту сами в стиле ч/б скина сразу в infer-размере
640x360. Тайминг идеальный по построению: кадр в момент t содержит ровно
те объекты, что должны быть видны в момент t. Боксы — те же, что в
autolabel (2r из CS, спиннер — квадрат), классы — schema.py.

Проекция osu(512x384) -> infer(640x360): тот же stretch-маппинг, что в
autolabel при osu_field == playfield ROI. Детектор, обученный на таких
кадрах, потом доучивается на 1-2 реальных записях (fine-tune на настоящий
рендер lazer + фон).

Использование:
  python main.py synth --osz "C:/Test OSU/lexycat - glitter.osz" --diff shine
  python main.py synth --all                 # все .osz из beatmaps.dir
"""
import os
import sys
import glob
import math
import random
import argparse
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from capture.manager import load_cfg
from perception.beatmap import (parse_osu_file, parse_osz, objects_at,
                                slider_tail_pos, slider_ball_pos,
                                circle_radius_osu_px, OSU_W, OSU_H)
from perception.schema import CLASS_TO_ID


def osu_to_px(xo, yo, box_osu, iw=640, ih=360):
    """Stretch 512x384 -> 640x360. Возвращает (cx, cy, rx, ry) в пикселях infer."""
    cx = xo / OSU_W * iw
    cy = yo / OSU_H * ih
    rx = box_osu / OSU_W * iw / 2.0
    ry = box_osu / OSU_H * ih / 2.0
    return cx, cy, rx, ry


class BWRenderer:
    """Рендер под черно-белый скин: темный фон, белые круги с черной окантовкой."""

    def __init__(self, iw=640, ih=360, seed=0):
        self.iw, self.ih = iw, ih
        self.rng = random.Random(seed)

    def background(self, shade: int) -> np.ndarray:
        img = np.full((self.ih, self.iw, 3), shade, np.uint8)
        # легкий вертикальный градиент как дим фона в игре
        grad = np.linspace(0, 8, self.ih, dtype=np.int16)[:, None, None]
        img = np.clip(img.astype(np.int16) + grad, 0, 255).astype(np.uint8)
        return img

    def circle(self, img, cx, cy, r, progress: float, dim: float = 1.0):
        """progress 0..1: 1 = момент удара. Кольцо approach схлопывается к кругу."""
        white = int(235 * dim)
        # approach-кольцо
        if progress < 1.0:
            ar = r * (1.0 + (1.0 - progress) * 2.5)
            cv2.circle(img, (int(cx), int(cy)), int(ar), (white,) * 3, 2)
        # тело круга: белая заливка + черная окантовка
        cv2.circle(img, (int(cx), int(cy)), int(r), (white,) * 3, -1)
        cv2.circle(img, (int(cx), int(cy)), int(r), (10,) * 3, 2)
        # внутренняя тень-центр
        cv2.circle(img, (int(cx), int(cy)), max(2, int(r * 0.35)), (25,) * 3, -1)

    def slider_track(self, img, pts, thick: int, dim: float = 1.0):
        if len(pts) < 2:
            return
        arr = np.array(pts, np.int32)
        cv2.polylines(img, [arr], False, (20,) * 3, thick + 4, cv2.LINE_AA)      # черная подложка
        cv2.polylines(img, [arr], False, (int(200 * dim),) * 3, thick, cv2.LINE_AA)  # светлый трек

    def spinner(self, img, cx, cy, R, t_norm: float):
        # концентрические кольца + спицы (вид сверху, ч/б)
        for k, rr in enumerate([R, R * 0.66, R * 0.33]):
            cv2.circle(img, (int(cx), int(cy)), int(rr), (220,) * 3, 2)
        ang = t_norm * 6.28 * 3
        x2 = cx + R * math.cos(ang)
        y2 = cy + R * math.sin(ang)
        cv2.line(img, (int(cx), int(cy)), (int(x2), int(y2)), (220,) * 3, 3)


def render_map_to_yolo(bm, cfg: dict, tag: str, stride: int = 3,
                       max_frames: int = 1500, seed: int = 7) -> dict:
    pp = cfg.get("preprocess", {})
    iw, ih = int(pp.get("infer_width", 640)), int(pp.get("infer_height", 360))
    sy = cfg.get("synth", {})
    fps = float(sy.get("fps", 60))
    preempt = float(cfg.get("beatmaps", {}).get("preempt_ms", 1200.0))
    box_scale = float(cfg.get("yolo", {}).get("box_scale", 1.15))
    spinner_box = float(cfg.get("yolo", {}).get("spinner_box", 200.0))
    shades = sy.get("bg_shades", [12, 18, 25])
    noise = float(sy.get("noise", 4.0))

    yolo_dir = cfg.get("yolo", {}).get("dir", "data/yolo")
    img_dir = os.path.join(yolo_dir, "images")
    lbl_dir = os.path.join(yolo_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    r = circle_radius_osu_px(bm)
    rend = BWRenderer(iw, ih, seed)
    objs = bm.objects
    if not objs:
        return {"frames": 0, "map": tag}
    t0 = min(o.time for o in objs) - preempt
    t1 = max(o.end_time if (o.is_slider or o.is_spinner) else o.time for o in objs) + 300
    step = 1000.0 / fps * stride

    kept, with_obj = 0, 0
    t = t0
    shade = shades[seed % len(shades)]
    while t <= t1 and kept < max_frames:
        img = rend.background(shade)
        acts = objects_at(bm, t, preempt=preempt)
        lines = []
        for o, cls, xo, yo in acts:
            box = spinner_box if cls == "spinner" else 2 * r * box_scale
            cx, cy, rx, ry = osu_to_px(xo, yo, box)
            # --- отрисовка ---
            if cls in ("circle", "slider_head"):
                prog = 1.0 - (o.time - t) / preempt
                rr = r / OSU_W * iw
                rend.circle(img, cx, cy, max(4, rr), min(max(prog, 0.0), 1.0))
            elif cls == "slider_ball":
                pts = [(o.x / OSU_W * iw, o.y / OSU_H * ih)] + \
                      [(px / OSU_W * iw, py / OSU_H * ih) for px, py in o.curve_points]
                rend.slider_track(img, pts, thick=max(4, int(r / OSU_W * iw * 0.9)))
                rr = r / OSU_W * iw
                rend.circle(img, cx, cy, max(4, rr), 1.0)
                tx, ty = slider_tail_pos(o)
                rend.circle(img, tx / OSU_W * iw, ty / OSU_H * ih, max(3, rr * 0.7), 1.0, dim=0.7)
            elif cls == "slider_tail":
                rr = r / OSU_W * iw
                rend.circle(img, cx, cy, max(3, rr * 0.7), 1.0, dim=0.7)
            elif cls == "spinner":
                R = spinner_box / OSU_W * iw / 2.0
                span = max(1.0, o.end_time - o.time)
                rend.spinner(img, cx, cy, R, (t - o.time) / span)
            # --- лейбл ---
            bw, bh = rx * 2 / iw, ry * 2 / ih
            if 0 <= cx / iw <= 1 and 0 <= cy / ih <= 1:
                lines.append(f"{CLASS_TO_ID[cls]} {cx/iw:.4f} {cy/ih:.4f} "
                             f"{min(max(bw,0.01),1.0):.4f} {min(max(bh,0.01),1.0):.4f}")
        # легкое зерно поверх (как компрессия/дим)
        if noise > 0:
            nz = np.random.normal(0, noise, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + nz, 0, 255).astype(np.uint8)
        name = f"synth_{tag}_{kept:05d}"
        cv2.imwrite(os.path.join(img_dir, name + ".jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        with open(os.path.join(lbl_dir, name + ".txt"), "w") as f:
            f.write("\n".join(lines))
        kept += 1
        with_obj += 1 if lines else 0
        t += step
    return {"frames": kept, "with_objects": with_obj, "map": tag,
            "beatmap": f"{bm.artist} - {bm.title} [{bm.version}]"}


def pick_maps(beatmaps_dir: str):
    return sorted(glob.glob(os.path.join(beatmaps_dir, "*.osz")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--osz", default="")
    ap.add_argument("--diff", default="")
    ap.add_argument("--osu", default="")
    ap.add_argument("--all", action="store_true", help="все .osz из beatmaps.dir")
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    sy = cfg.get("synth", {})
    stride = args.stride or int(sy.get("stride", 3))
    mf = args.max_frames or int(sy.get("max_frames_per_map", 1500))

    jobs = []  # (tag, [Beatmap])
    if args.osu:
        jobs.append(("single", [parse_osu_file(args.osu)]))
    elif args.osz:
        from training.autolabel import find_osu_in_osz
        if args.diff:
            jobs.append(("single", [parse_osu_file(find_osu_in_osz(args.osz, args.diff))]))
        else:
            jobs.append(("single", parse_osz(args.osz)))
    elif args.all:
        for osz in pick_maps(cfg.get("beatmaps", {}).get("dir", "C:/Test OSU")):
            jobs.append((os.path.splitext(os.path.basename(osz))[0][:24], parse_osz(osz)))
    else:
        print("Нужно --osu / --osz [--diff] / --all")
        return

    total = 0
    for tag, maps in jobs:
        for i, bm in enumerate(maps):
            safe = "".join(c if c.isalnum() else "_" for c in
                           (bm.version or f"diff{i}"))[:20]
            res = render_map_to_yolo(bm, cfg, f"{tag}_{safe}", stride, mf,
                                     seed=args.seed + total)
            total += res["frames"]
            print(f"  {res['beatmap']}: {res['frames']} кадров ({res['with_objects']} с объектами)")

    from training.autolabel import write_dataset_yaml
    yp = write_dataset_yaml(cfg)
    print(f"ВСЕГО: {total} кадров -> data/yolo. dataset.yaml: {yp}")


if __name__ == "__main__":
    main()
