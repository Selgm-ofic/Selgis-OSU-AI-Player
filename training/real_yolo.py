"""Export click-safe real recordings as a YOLO fine-tuning dataset."""
import argparse
import json
import os
import shutil

import cv2
import numpy as np

from perception.beatmap import (OSU_H, OSU_W, circle_radius_osu_px,
                                objects_at, parse_osu_file)
from perception.schema import CLASS_TO_ID
from training.geometry_sync import estimate_geometry_sync, save_geometry_sync


def _yield_to_user():
    try:
        import ctypes
        kernel = ctypes.windll.kernel32
        kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000)  # BELOW_NORMAL
    except Exception:
        pass


def _clipped_yolo_box(x, y, width, height):
    cx, cy = x / OSU_W, y / OSU_H
    bw, bh = min(1.0, width / OSU_W), min(1.0, height / OSU_H)
    left, right = max(0.0, cx - bw/2), min(1.0, cx + bw/2)
    top, bottom = max(0.0, cy - bh/2), min(1.0, cy + bh/2)
    return ((left + right) / 2, (top + bottom) / 2,
            right - left, bottom - top)


def export_session(session, out, stride=2, split_override=None):
    meta_path = os.path.join(session, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as handle:
            recording_meta = json.load(handle)
        if recording_meta.get("vision_dropped", 0):
            raise RuntimeError(
                f"temporal stream dropped {recording_meta['vision_dropped']} frames; "
                "record this map again")
    if not os.path.exists(meta_path):
        raise RuntimeError("recording has no meta.json")
    beatmap = parse_osu_file(recording_meta["beatmap_osu"])
    vision_dir = os.path.join(session, "vision_frames")
    vision_stamps_path = os.path.join(session, "vision_stamps.npy")
    temporal_ready = os.path.isdir(vision_dir) and os.path.exists(vision_stamps_path)
    effective_stride = 1 if temporal_ready else stride
    stamps = np.load(vision_stamps_path if temporal_ready else
                     os.path.join(session, "stamps.npy"))
    sync = estimate_geometry_sync(
        session, beatmap, float(recording_meta.get("rate", 1.0)))
    save_geometry_sync(session, beatmap, sync)
    print(f"sync: method={sync.method} scale={sync.transform.scale:.6f} "
          f"offset={sync.transform.offset_ms:.1f}ms "
          f"uncertainty<={sync.uncertainty_ms:.1f}ms "
          f"coverage={sync.covered_objects}/{sync.total_objects}", flush=True)
    if not sync.is_safe():
        raise RuntimeError("geometry sync failed quality gates; session was not exported")
    capture = None if temporal_ready else cv2.VideoCapture(
        os.path.join(session, "video.mp4"))
    for split in ("train", "val"):
        os.makedirs(os.path.join(out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out, "labels", split), exist_ok=True)
    radius = circle_radius_osu_px(beatmap)
    spinner_size = 200.0
    session_tag = "".join(c if c.isalnum() else "_" for c in os.path.basename(session))[:40]
    prefix = f"real_{session_tag}_"
    # Re-exporting a shorter recording must not leave stale samples behind.
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            folder = os.path.join(out, kind, split)
            for old_name in os.listdir(folder):
                if old_name.startswith(prefix):
                    os.remove(os.path.join(folder, old_name))
    exported = 0
    for index, stamp in enumerate(stamps):
        if index % effective_stride:
            if capture is not None:
                capture.grab()
            continue
        if not (sync.record_start_ms <= float(stamp) <= sync.record_end_ms):
            if capture is not None:
                capture.grab()
            continue
        if temporal_ready:
            source_image = os.path.join(vision_dir, f"{index:06d}.png")
            if not os.path.exists(source_image):
                raise RuntimeError(f"missing temporal frame {index:06d}.png")
            frame = None
        else:
            ok, frame = capture.read()
            if not ok:
                break
        map_ms = sync.transform.to_map(float(stamp))
        active = objects_at(beatmap, map_ms, preempt=beatmap.preempt_ms)
        rows = []
        for _, cls, x, y in active:
            box = spinner_size if cls == "spinner" else 2.0 * radius * 1.15
            cx, cy, bw, bh = _clipped_yolo_box(x, y, box, box)
            if bw > 0 and bh > 0:
                rows.append(
                    f"{CLASS_TO_ID[cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        rows = list(dict.fromkeys(rows))
        name = f"real_{session_tag}_{index:06d}"
        split = split_override or ("val" if exported % 5 == 0 else "train")
        image_out = os.path.join(out, "images", split, name + ".png")
        if temporal_ready:
            if os.path.exists(image_out):
                os.remove(image_out)
            try:
                os.link(source_image, image_out)
            except OSError:
                shutil.copy2(source_image, image_out)
        else:
            cv2.imwrite(image_out, frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        with open(os.path.join(out, "labels", split, name + ".txt"), "w", encoding="ascii") as handle:
            handle.write("\n".join(rows))
        exported += 1
    if capture is not None:
        capture.release()
    yaml_path = os.path.join(out, "dataset.yaml")
    with open(yaml_path, "w", encoding="utf-8") as handle:
        handle.write(f"path: {os.path.abspath(out)}\ntrain: images/train\nval: images/val\n"
                     "names: [circle, slider_head, slider_ball, slider_tail, spinner, cursor]\n")
    print(f"exported={exported} split={split_override or 'frame-mixed'} "
          f"source={'temporal3' if temporal_ready else 'legacy-video'} -> {out}")
    return exported


def export_many(sessions_root, out, stride=1):
    sessions = []
    for name in sorted(os.listdir(sessions_root)):
        path = os.path.join(sessions_root, name)
        if (os.path.isdir(path) and
                os.path.exists(os.path.join(path, "meta.json")) and
                os.path.exists(os.path.join(path, "vision_stamps.npy"))):
            sessions.append(path)
    if len(sessions) < 2:
        raise RuntimeError("at least two temporal sessions are required for map-disjoint train/val")
    # One whole map is validation. With three recordings this gives two maps
    # for learning and an honestly unseen third map for metrics.
    val_count = max(1, int(round(len(sessions) * 0.20)))
    frame_counts = []
    for session in sessions:
        meta_file = os.path.join(session, "meta.json")
        with open(meta_file, encoding="utf-8") as handle:
            frame_counts.append(int(json.load(handle).get("vision_frames", 0)))
    if val_count == 1:
        target = 0.20 * sum(frame_counts)
        val_ids = {min(range(len(sessions)),
                       key=lambda index: abs(frame_counts[index] - target))}
    else:
        rng = np.random.default_rng(7)
        val_ids = set(int(i) for i in rng.permutation(len(sessions))[:val_count])
    total = 0
    for index, session in enumerate(sessions):
        split = "val" if index in val_ids else "train"
        print(f"=== {split.upper()} {os.path.basename(session)} ===", flush=True)
        total += export_session(session, out, stride, split_override=split)
    print(f"dataset complete: sessions={len(sessions)} val_sessions={val_count} "
          f"frames={total}", flush=True)
    return total


def main():
    _yield_to_user()
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="")
    parser.add_argument("--sessions", default="")
    parser.add_argument("--out", default="data/yolo_real")
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()
    if args.sessions:
        export_many(args.sessions, args.out, max(1, args.stride))
    elif args.session:
        export_session(args.session, args.out, max(1, args.stride))
    else:
        parser.error("provide --session or --sessions")


if __name__ == "__main__":
    main()
