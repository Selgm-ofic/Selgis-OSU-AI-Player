"""Автолейблинг: сессия записи + .osu -> YOLO-датасет.

Проекция: osu-координаты (512x384) -> infer-кадр (640x360):
  1. osu (x,y) -> пиксели кадра: osu_field rect из конфига (mapping.osu_field)
  2. пиксели кадра -> ROI: вычесть playfield.left/top (для OBS; для DXGI ROI-кадр уже кропнут)
  3. ROI -> infer 640x360: линейный масштаб (labels нормализованы 0..1)

Размер бокса: 2*radius (из CS карты) * box_scale; спиннер — квадрат spinner_box.
Классы — см. perception/schema.py (порядок фиксирован!).

Выход: data/yolo/{images,labels}/ + dataset.yaml. Пустые кадры (без объектов)
тоже сохраняем с пустым txt — это учит детектор фону/скину.
"""
import os
import sys
import json
import glob
import argparse
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from capture.manager import load_cfg
from perception.beatmap import (parse_osu_file, parse_osz, objects_at,
                                circle_radius_osu_px, OSU_W, OSU_H)
from perception.schema import CLASS_TO_ID

DIFF_HINT = ("укажи --osu-путь к конкретной сложности; "
             "список смотри в выводе парсера (Version)")


def find_osu_in_osz(osz_path: str, diff: str) -> str:
    """Найти .osu нужной сложности внутри .osz (распаковка во временную папку)."""
    import tempfile
    import zipfile
    tmp = tempfile.mkdtemp(prefix="osz_pick_")
    with zipfile.ZipFile(osz_path, "r") as z:
        z.extractall(tmp)
    best = None
    for root, _, files in os.walk(tmp):
        for fn in files:
            if not fn.lower().endswith(".osu"):
                continue
            full = os.path.join(root, fn)
            with open(full, "r", encoding="utf-8", errors="ignore") as f:
                head = f.read(4000)
            if diff.lower() in head.lower() or diff.lower() in fn.lower():
                return full
            if best is None:
                best = full
    if diff and best is None:
        raise FileNotFoundError(f"в {osz_path} нет .osu. {DIFF_HINT}")
    return best or ""


def osu_to_infer(xo, yo, box_osu, cfg, from_roi_frame: bool):
    """(xo,yo) osu-px + размер бокса -> YOLO (cx,cy,w,h) 0..1 в infer 640x360."""
    pp = cfg.get("preprocess", {})
    iw, ih = int(pp.get("infer_width", 640)), int(pp.get("infer_height", 360))
    of = cfg.get("mapping", {}).get("osu_field", [0, 0, 1920, 1080])
    ol, ot, orr, ob = of
    ow, oh = max(1, orr - ol), max(1, ob - ot)
    # osu -> пиксели кадра
    px = ol + (xo / OSU_W) * ow
    py = ot + (yo / OSU_H) * oh
    bw_px = (box_osu / OSU_W) * ow
    bh_px = (box_osu / OSU_H) * oh
    if from_roi_frame:
        # DXGI-кадр уже кропнут по playfield ROI
        pf = cfg.get("playfield", {})
        px -= float(pf.get("left", 0)); py -= float(pf.get("top", 0))
        roi_w = float(pf.get("right", 1920)) - float(pf.get("left", 0))
        roi_h = float(pf.get("bottom", 1080)) - float(pf.get("top", 0))
    else:
        pf = cfg.get("playfield", {})
        px -= float(pf.get("left", 0)); py -= float(pf.get("top", 0))
        roi_w = float(pf.get("right", 1920)) - float(pf.get("left", 0))
        roi_h = float(pf.get("bottom", 1080)) - float(pf.get("top", 0))
    cx = (px / max(roi_w, 1) * iw) / iw  # = px/roi_w, но явно через infer
    cy = (py / max(roi_h, 1) * ih) / ih
    w = (bw_px / max(roi_w, 1) * iw) / iw
    h = (bh_px / max(roi_h, 1) * ih) / ih
    return cx, cy, w, h


def autolabel_session(sess_dir: str, osu_path: str, cfg: dict, stride: int = 2,
                      max_frames: int = 3000) -> dict:
    from_roi = "dxgi" in json.load(open(os.path.join(sess_dir, "meta.json"),
                                        encoding="utf-8")).get("source", "")
    meta = json.load(open(os.path.join(sess_dir, "meta.json"), encoding="utf-8"))
    bm = parse_osu_file(osu_path)
    r = circle_radius_osu_px(bm)
    yc = cfg.get("yolo", {})
    box_scale = float(yc.get("box_scale", 1.15))
    spinner_box = float(yc.get("spinner_box", 200.0))
    preempt = float(meta.get("preempt_ms", 1200.0))
    offset = float(meta.get("delay_ms", 0.0))  # запись могла стартовать раньше карты
    yolo_dir = yc.get("dir", "data/yolo")
    img_dir = os.path.join(yolo_dir, "images")
    lbl_dir = os.path.join(yolo_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    cap = cv2.VideoCapture(os.path.join(sess_dir, "video.mp4"))
    stamps = np.load(os.path.join(sess_dir, "stamps.npy"))
    sess = os.path.basename(sess_dir.rstrip("/\\"))
    kept, with_obj, fi = 0, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok or kept >= max_frames:
            break
        t_rec = float(stamps[fi]) if fi < len(stamps) else fi * 16.6
        fi += 1
        if fi % stride != 0:
            continue
        t_map = t_rec - offset
        acts = objects_at(bm, t_map, preempt=preempt)
        h, w = frame.shape[:2]
        tiny = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LINEAR)
        lines = []
        for _, cls, xo, yo in acts:
            box = spinner_box if cls == "spinner" else 2 * r * box_scale
            cx, cy, bw, bh = osu_to_infer(xo, yo, box, cfg, from_roi)
            if not (0 <= cx <= 1 and 0 <= cy <= 1):
                continue  # объект вне ROI — пропускаем
            bw = min(max(bw, 0.01), 1.0); bh = min(max(bh, 0.01), 1.0)
            lines.append(f"{CLASS_TO_ID[cls]} {cx:.4f} {cy:.4f} {bw:.4f} {bh:.4f}")
        name = f"{sess}_{kept:05d}"
        cv2.imwrite(os.path.join(img_dir, name + ".jpg"), tiny,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        with open(os.path.join(lbl_dir, name + ".txt"), "w") as f:
            f.write("\n".join(lines))
        kept += 1
        with_obj += 1 if lines else 0
    cap.release()
    return {"frames": kept, "with_objects": with_obj, "empty": kept - with_obj,
            "radius_osu_px": r, "beatmap": f"{bm.artist} - {bm.title} [{bm.version}]"}


def write_dataset_yaml(cfg: dict):
    from perception.schema import CLASS_NAMES
    yolo_dir = cfg.get("yolo", {}).get("dir", "data/yolo")
    p = os.path.join(yolo_dir, "dataset.yaml")
    with open(p, "w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(yolo_dir)}\n")
        f.write("train: images\nval: images\n\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write(f"names: {CLASS_NAMES}\n")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--session", required=True, help="data/recordings/<name>")
    ap.add_argument("--osz", default="", help="путь к .osz (тогда нужен --diff)")
    ap.add_argument("--diff", default="")
    ap.add_argument("--osu", default="", help="прямой путь к .osu сложности")
    ap.add_argument("--stride", type=int, default=2, help="брать каждый N-й кадр")
    ap.add_argument("--max-frames", type=int, default=3000)
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    osu_path = args.osu
    if not osu_path and args.osz:
        osu_path = find_osu_in_osz(args.osz, args.diff)
        print(f"выбрана сложность: {osu_path}")
    if not osu_path:
        print(f"ОШИБКА: нужен --osu или --osz + --diff. {DIFF_HINT}")
        return
    res = autolabel_session(args.session, osu_path, cfg, args.stride, args.max_frames)
    yp = write_dataset_yaml(cfg)
    print(f"готово: {res['frames']} кадров ({res['with_objects']} с объектами, "
          f"{res['empty']} фон), r={res['radius_osu_px']:.1f}px")
    print(f"карта: {res['beatmap']}")
    print(f"dataset.yaml: {yp}")


if __name__ == "__main__":
    main()
