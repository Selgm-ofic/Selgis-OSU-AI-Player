"""Dataset v8 builder with discrete snap targeting, zero-leakage, and pure osu! playfield coordinates.

Key features:
1. Discrete Snap Targeting: No smoothstep interpolation in mid-air. As soon as the previous object
   ends, target coordinates snap instantly to the center of the upcoming hit object.
   Smooth interpolation is strictly reserved for active slider balls and spinners.
2. Frame-Aligned Note Center Targets: Targets are mapped directly to visual note centers on the
   captured playfield frame [0, 1], eliminating the 35px coordinate misalignment.
3. Cursor Motion De-leakage: Auto-mod cursor motion in channel 1 (absdiff) is completely suppressed.
4. Clean Session Filtering: Only click-safe sessions with max_time_uncertainty <= 16.5ms are included.
"""

from bisect import bisect_right
import concurrent.futures
import json
import math
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np


def _nearest(source: np.ndarray, query: np.ndarray) -> np.ndarray:
    right = np.searchsorted(source, query).clip(0, len(source) - 1)
    left = np.maximum(0, right - 1)
    return np.where(np.abs(source[left] - query) <= np.abs(source[right] - query), left, right)


def _endpoint(obj) -> tuple[float, float]:
    if obj.is_slider:
        from perception.beatmap import slider_ball_pos
        return slider_ball_pos(obj, obj.end_time)
    return obj.x, obj.y


def object_end_time(obj, click_hold_ms: float = 36.0) -> float:
    if obj.is_slider or obj.is_spinner:
        return max(obj.time, obj.end_time)
    return obj.time + click_hold_ms


def cursor_target_snap(
    bm,
    map_ms: float,
    click_hold_ms: float = 36.0,
    spinner_radius: float = 70.0,
    spinner_rps: float = 5.0,
) -> tuple[float, float, int]:
    from perception.beatmap import OSU_H, OSU_W, slider_ball_pos

    objects = bm.objects
    if not objects:
        return 0.5, 0.5, -1

    starts = [o.time for o in objects]
    current = max(0, bisect_right(starts, map_ms) - 1)
    obj = objects[current]

    # Active slider: follow slider ball smoothly
    if obj.is_slider and obj.time <= map_ms <= obj.end_time:
        x, y = slider_ball_pos(obj, map_ms)
        return x / OSU_W, y / OSU_H, current

    # Active spinner: rotate smoothly
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
    if map_ms < object_end_time(prev, click_hold_ms=click_hold_ms):
        px, py = _endpoint(prev)
        return px / OSU_W, py / OSU_H, next_index - 1

    # Discrete step snap to the center of incoming next object:
    return nxt.x / OSU_W, nxt.y / OSU_H, next_index


def process_single_session_v8(
    session: Path,
    output_root: Path,
    width: int = 96,
    height: int = 72,
    cursor_mask_r_640: int = 22,
    cursor_mask_r_small: int = 4,
) -> dict | None:
    manifest_path = session / "policy_manifest.json"
    video_path = session / "video.mp4"
    stamps_path = session / "stamps.npy"

    if not (manifest_path.exists() and video_path.exists() and stamps_path.exists()):
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Skipping {session.name}: invalid manifest ({exc})", flush=True)
        return None

    clock = manifest.get("clock", {})
    click_safe = clock.get("click_safe", False)
    uncertainty = float(clock.get("max_time_uncertainty_ms", 999.0))
    if (not click_safe) or uncertainty > 16.5:
        print(f"Skipping session {session.name} (click_safe={click_safe}, uncertainty={uncertainty:.2f}ms)", flush=True)
        return None

    out_npz = output_root / f"{session.name}.npz"
    if out_npz.exists():
        try:
            z = np.load(out_npz)
            if "version" in z and int(z["version"]) == 8 and "targets" in z and z["targets"].shape[1] == 6:
                n = len(z["images"])
                print(f"Cached {session.name}: {n} frames", flush=True)
                return {"name": session.name, "path": str(out_npz), "frames": n}
        except Exception:
            pass

    beatmap_path = manifest.get("beatmap", {}).get("path")
    if not beatmap_path or not Path(beatmap_path).exists():
        print(f"Skipping {session.name}: beatmap file missing ({beatmap_path})", flush=True)
        return None

    for p in [r"C:\Gamma\Osu_module", r"C:\Selgis OSU AI Player"]:
        if p not in sys.path:
            sys.path.insert(0, p)
    from perception.beatmap import parse_osu_file
    from training.action_timeline import TimeTransform, cursor_target_at

    beatmap = parse_osu_file(beatmap_path)
    stamps = np.load(stamps_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Skipping {session.name}: cannot open video {video_path}", flush=True)
        return None

    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(len(stamps), video_frame_count)
    stamps = stamps[:n]

    transform = TimeTransform(float(clock["offset_ms"]), float(clock["scale"]))
    starts = [o.time for o in beatmap.objects]
    preempt_ms = float(beatmap.preempt_ms)

    # 1. Target and click computations
    clicks = np.zeros(n, dtype=np.float32)
    key_events_file = session / "key_events.json"
    if key_events_file.exists():
        try:
            events = json.loads(key_events_file.read_text(encoding="utf-8"))
            downs = np.array([float(e["record_ms"]) for e in events if e.get("kind") == "down"], dtype=np.float64)
            if len(downs) > 0:
                idx = _nearest(stamps, downs)
                for down_t, peak_idx in zip(downs, idx):
                    dt = np.abs(stamps - down_t)
                    mask = dt <= 14.0
                    if np.any(mask):
                        soft = np.exp(-0.5 * (dt[mask] / 6.0) ** 2).astype(np.float32)
                        soft = soft / max(1e-5, soft.max())
                        clicks[mask] = np.maximum(clicks[mask], soft)
                    if abs(stamps[peak_idx] - down_t) <= 12.0:
                        clicks[peak_idx] = 1.0
        except Exception as exc:
            print(f"Warning: error parsing key_events for {session.name} ({exc}), falling back to beatmap", flush=True)
            key_events_file = None

    if not key_events_file or not key_events_file.exists():
        for obj in beatmap.objects:
            if obj.is_spinner:
                continue
            record_t = (obj.time - transform.offset_ms) / max(1e-5, transform.scale)
            if stamps[0] - 20.0 <= record_t <= stamps[-1] + 20.0:
                peak_idx = int(_nearest(stamps, np.array([record_t]))[0])
                dt = np.abs(stamps - record_t)
                mask = dt <= 14.0
                if np.any(mask):
                    soft = np.exp(-0.5 * (dt[mask] / 6.0) ** 2).astype(np.float32)
                    soft = soft / max(1e-5, soft.max())
                    clicks[mask] = np.maximum(clicks[mask], soft)
                if abs(stamps[peak_idx] - record_t) <= 12.0:
                    clicks[peak_idx] = 1.0

    targets = np.empty((n, 6), dtype=np.float32)
    valid = np.zeros(n, dtype=np.float32)

    for i in range(n):
        rec_t = float(stamps[i])
        map_t = transform.to_map(rec_t)

        tgt_cx, tgt_cy, tgt_owner = cursor_target_snap(beatmap, map_t)

        norm_x = np.clip(tgt_cx * 0.865111 + 0.0652055, 0.0, 1.0)
        norm_y = np.clip(tgt_cy * 0.814956 + 0.0926181, 0.0, 1.0)

        next_idx = bisect_right(starts, map_t)
        curr_idx = next_idx - 1
        is_slider_active = False

        if 0 <= curr_idx < len(beatmap.objects):
            curr_obj = beatmap.objects[curr_idx]
            if curr_obj.is_slider and (curr_obj.time <= map_t <= curr_obj.end_time):
                is_slider_active = True

        if next_idx < len(starts):
            tth = starts[next_idx] - map_t
        else:
            tth = 9999.0

        if is_slider_active:
            slider_rem = curr_obj.end_time - map_t
            tth = min(tth, slider_rem)

        tth = float(np.clip(tth, 0.0, 9999.0))
        hold = 1.0 if is_slider_active else 0.0

        targets[i, 0] = norm_x
        targets[i, 1] = norm_y
        targets[i, 2] = tth
        targets[i, 3] = clicks[i]
        targets[i, 4] = hold
        targets[i, 5] = float(tgt_owner)

        is_next_spinner = (next_idx < len(beatmap.objects) and beatmap.objects[next_idx].is_spinner)
        is_incoming_visible = (0.0 <= tth <= preempt_ms) and (not is_next_spinner)
        if is_slider_active or is_incoming_visible:
            valid[i] = 1.0

    # 2. Image decoding and cursor motion de-leakage with exact ROI cropping
    images = np.empty((n, 2, height, width), dtype=np.uint8)
    prev_gray_640 = None
    prev_cx = None
    prev_cy = None

    t0 = time.perf_counter()
    for i in range(n):
        ret, frame = cap.read()
        if not ret or frame is None:
            images = images[:i]
            targets = targets[:i]
            valid = valid[:i]
            stamps = stamps[:i]
            n = i
            break

        # Crop playfield ROI [top=35, bottom=1075, left=295, right=1625] if Full HD 1080p
        if frame.shape[0] == 1080 and frame.shape[1] == 1920:
            gray_roi = frame[35:1075, 295:1625, 0]
            gray_640 = cv2.resize(gray_roi, (640, 360), interpolation=cv2.INTER_AREA)
        elif frame.shape[0] == 360 and frame.shape[1] == 640:
            gray_640 = frame[:, :, 0]
        else:
            gray_640 = cv2.resize(frame[:, :, 0], (640, 360), interpolation=cv2.INTER_AREA)

        g_small = cv2.resize(gray_640, (width, height), interpolation=cv2.INTER_AREA)

        rec_t = float(stamps[i])
        map_t = transform.to_map(rec_t)
        curr_cx, curr_cy, _ = cursor_target_at(beatmap, map_t)

        if prev_gray_640 is None:
            d_small = np.zeros((height, width), dtype=np.uint8)
        else:
            diff_640 = cv2.convertScaleAbs(cv2.absdiff(gray_640, prev_gray_640), alpha=3.0)

            x1_640 = int(np.clip((curr_cx * 0.865111 + 0.0652055) * 640, 0, 639))
            y1_640 = int(np.clip((curr_cy * 0.814956 + 0.0926181) * 360, 0, 359))
            x0_640 = int(np.clip((prev_cx * 0.865111 + 0.0652055) * 640, 0, 639))
            y0_640 = int(np.clip((prev_cy * 0.814956 + 0.0926181) * 360, 0, 359))

            cv2.circle(diff_640, (x1_640, y1_640), cursor_mask_r_640, 0, -1)
            cv2.circle(diff_640, (x0_640, y0_640), cursor_mask_r_640, 0, -1)
            cv2.line(diff_640, (x0_640, y0_640), (x1_640, y1_640), 0, thickness=2 * cursor_mask_r_640)

            d_small = cv2.resize(diff_640, (width, height), interpolation=cv2.INTER_AREA)

            x1_small = int(np.clip((curr_cx * 0.865111 + 0.0652055) * width, 0, width - 1))
            y1_small = int(np.clip((curr_cy * 0.814956 + 0.0926181) * height, 0, height - 1))
            x0_small = int(np.clip((prev_cx * 0.865111 + 0.0652055) * width, 0, width - 1))
            y0_small = int(np.clip((prev_cy * 0.814956 + 0.0926181) * height, 0, height - 1))

            cv2.circle(d_small, (x1_small, y1_small), cursor_mask_r_small, 0, -1)
            cv2.circle(d_small, (x0_small, y0_small), cursor_mask_r_small, 0, -1)
            cv2.line(d_small, (x0_small, y0_small), (x1_small, y1_small), 0, thickness=2 * cursor_mask_r_small)

        images[i, 0] = g_small
        images[i, 1] = d_small

        prev_gray_640 = gray_640
        prev_cx = curr_cx
        prev_cy = curr_cy

    cap.release()
    elapsed = time.perf_counter() - t0
    fps_proc = n / max(1e-4, elapsed)

    np.savez_compressed(
        out_npz,
        images=images,
        targets=targets,
        valid=valid,
        stamps=stamps.astype(np.float32),
        version=8,
    )

    n_valid = int(np.sum(valid > 0.5))
    n_clicks = int(np.sum(clicks > 0.5))
    n_holds = int(np.sum(targets[:, 4] > 0.5))

    print(
        f"Built {session.name}: {n} frames ({fps_proc:.1f} fps) | "
        f"valid={n_valid} ({n_valid/n*100:.1f}%), clicks={n_clicks}, holds={n_holds}",
        flush=True,
    )

    return {"name": session.name, "path": str(out_npz), "frames": n}


def build_dataset_v8(
    source_root: str = r"C:\Gamma\Osu_module\data\recordings",
    output_root: str = r"C:\Selgis OSU AI Player\data\dataset_v8",
    width: int = 96,
    height: int = 72,
    cursor_mask_r_640: int = 22,
    cursor_mask_r_small: int = 4,
    max_workers: int = 4,
):
    source_root, output_root = Path(source_root), Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    sessions_meta = []
    total_frames_built = 0

    candidates = sorted([d for d in source_root.iterdir() if d.is_dir()])
    print(f"Discovered {len(candidates)} candidate session directories.", flush=True)

    workers = min(max_workers, max(1, os.cpu_count() or 4))
    if workers > 1 and len(candidates) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    process_single_session_v8,
                    session,
                    output_root,
                    width,
                    height,
                    cursor_mask_r_640,
                    cursor_mask_r_small,
                )
                for session in candidates
            ]
            for fut in concurrent.futures.as_completed(futures):
                res = fut.result()
                if res:
                    sessions_meta.append(res)
                    total_frames_built += res["frames"]
    else:
        for session in candidates:
            res = process_single_session_v8(
                session,
                output_root,
                width,
                height,
                cursor_mask_r_640,
                cursor_mask_r_small,
            )
            if res:
                sessions_meta.append(res)
                total_frames_built += res["frames"]

    sessions_meta.sort(key=lambda s: s["name"])
    index_data = {
        "version": 8,
        "name": "dataset_v8_snap",
        "description": "Discrete snap targeting on note centers, pure [0, 1] osu! coords, zero auto-cursor leakage.",
        "shape": [2, height, width],
        "channels": ["gray", "absdiff_deleaked"],
        "sessions": sessions_meta,
        "total_frames": total_frames_built,
    }
    index_path = output_root / "index.json"
    index_path.write_text(json.dumps(index_data, indent=2), encoding="utf-8")
    print(f"\nCompleted dataset_v8: {len(sessions_meta)} sessions, {total_frames_built} total frames.", flush=True)
    return index_data


if __name__ == "__main__":
    build_dataset_v8()
