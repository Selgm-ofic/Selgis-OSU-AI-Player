"""Dataset v7.1 builder with cursor motion de-leakage, 30ms lookahead and strict validity masks.

Eliminates the Auto-mod cursor motion leakage present in dataset_v7_clean by masking out
the cursor sprite trajectory in channel 1 (absdiff). Implements a 30ms lookahead target
to compensate for hardware grab and model inference latency, and synchronizes TTH and valid masks.
"""

from bisect import bisect_right
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np


def _nearest(source: np.ndarray, query: np.ndarray) -> np.ndarray:
    right = np.searchsorted(source, query).clip(0, len(source) - 1)
    left = np.maximum(0, right - 1)
    return np.where(np.abs(source[left] - query) <= np.abs(source[right] - query), left, right)


def build_dataset_v7_1(
    source_root: str = r"C:\Gamma\Osu_module\data\recordings",
    output_root: str = r"C:\Selgis OSU AI Player\data\dataset_v7_1",
    width: int = 96,
    height: int = 72,
    lookahead_ms: float = 30.0,
    cursor_mask_r_640: int = 22,
    cursor_mask_r_small: int = 4,
):
    source_root, output_root = Path(source_root), Path(output_root)
    legacy_root = source_root.parents[1]
    if str(legacy_root) not in sys.path:
        sys.path.insert(0, str(legacy_root))
    from perception.beatmap import parse_osu_file
    from training.action_timeline import TimeTransform, cursor_target_at

    output_root.mkdir(parents=True, exist_ok=True)
    sessions_meta = []
    total_frames_built = 0

    candidates = sorted([d for d in source_root.iterdir() if d.is_dir()])
    print(f"Discovered {len(candidates)} candidate session directories.", flush=True)

    for session in candidates:
        manifest_path = session / "policy_manifest.json"
        video_path = session / "video.mp4"
        stamps_path = session / "stamps.npy"

        if not (manifest_path.exists() and video_path.exists() and stamps_path.exists()):
            continue

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Skipping {session.name}: invalid manifest ({exc})", flush=True)
            continue

        clock = manifest.get("clock", {})
        click_safe = clock.get("click_safe", False)
        uncertainty = float(clock.get("max_time_uncertainty_ms", 999.0))
        if (not click_safe) or uncertainty > 16.5:
            print(f"Skipping session {session.name} (click_safe={click_safe}, uncertainty={uncertainty:.2f}ms)", flush=True)
            continue

        out_npz = output_root / f"{session.name}.npz"
        if out_npz.exists():
            z = np.load(out_npz)
            if "version" in z and int(z["version"]) == 7 and "targets" in z and z["targets"].shape[1] == 6:
                n = len(z["images"])
                sessions_meta.append({"name": session.name, "path": str(out_npz), "frames": n})
                total_frames_built += n
                print(f"Cached {session.name}: {n} frames", flush=True)
                continue

        beatmap_path = manifest.get("beatmap", {}).get("path")
        if not beatmap_path or not Path(beatmap_path).exists():
            print(f"Skipping {session.name}: beatmap file missing ({beatmap_path})", flush=True)
            continue

        beatmap = parse_osu_file(beatmap_path)
        stamps = np.load(stamps_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Skipping {session.name}: cannot open video {video_path}", flush=True)
            continue

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

        # Precompute target coordinates, TTH and valid masks for all frames
        for i in range(n):
            rec_t = float(stamps[i])
            map_t = transform.to_map(rec_t)

            # Lookahead target coordinates
            target_map_t = transform.to_map(rec_t + lookahead_ms)
            tgt_cx, tgt_cy, tgt_owner = cursor_target_at(beatmap, target_map_t)

            norm_x = np.clip(tgt_cx * 0.865111 + 0.0652055, 0.0, 1.0)
            norm_y = np.clip(tgt_cy * 0.814956 + 0.0926181, 0.0, 1.0)

            # TTH and active slider relative to current frame map_t
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

            # Hold flag: active slider tracking
            is_tgt_slider = False
            if 0 <= tgt_owner < len(beatmap.objects):
                tgt_obj = beatmap.objects[tgt_owner]
                if tgt_obj.is_slider and (tgt_obj.time <= target_map_t <= tgt_obj.end_time):
                    is_tgt_slider = True

            hold = 1.0 if (is_tgt_slider or is_slider_active) else 0.0

            targets[i, 0] = norm_x
            targets[i, 1] = norm_y
            targets[i, 2] = tth
            targets[i, 3] = clicks[i]
            targets[i, 4] = hold
            targets[i, 5] = float(tgt_owner)

            # Synchronize valid mask: visible incoming hit object or active slider ball
            is_next_spinner = (next_idx < len(beatmap.objects) and beatmap.objects[next_idx].is_spinner)
            is_incoming_visible = (0.0 <= tth <= preempt_ms) and (not is_next_spinner)
            if is_slider_active or is_incoming_visible:
                valid[i] = 1.0

        # 2. Image decoding and cursor motion de-leakage
        images = np.empty((n, 2, height, width), dtype=np.uint8)
        prev_gray_640 = None
        prev_cx = None
        prev_cy = None

        t0 = time.perf_counter()
        for i in range(n):
            ret, frame = cap.read()
            if not ret or frame is None:
                # Truncate if video ends earlier
                images = images[:i]
                targets = targets[:i]
                valid = valid[:i]
                stamps = stamps[:i]
                n = i
                break

            gray_640 = frame[:, :, 0]
            g_small = cv2.resize(gray_640, (width, height), interpolation=cv2.INTER_AREA)

            rec_t = float(stamps[i])
            map_t = transform.to_map(rec_t)
            curr_cx, curr_cy, _ = cursor_target_at(beatmap, map_t)

            if prev_gray_640 is None:
                d_small = np.zeros((height, width), dtype=np.uint8)
            else:
                diff_640 = cv2.convertScaleAbs(cv2.absdiff(gray_640, prev_gray_640), alpha=3.0)

                # Map cursor coordinates in 640x360
                x1_640 = int(np.clip((curr_cx * 0.865111 + 0.0652055) * 640, 0, 639))
                y1_640 = int(np.clip((curr_cy * 0.814956 + 0.0926181) * 360, 0, 359))
                x0_640 = int(np.clip((prev_cx * 0.865111 + 0.0652055) * 640, 0, 639))
                y0_640 = int(np.clip((prev_cy * 0.814956 + 0.0926181) * 360, 0, 359))

                # Mask out auto-cursor sprite in 640x360
                cv2.circle(diff_640, (x1_640, y1_640), cursor_mask_r_640, 0, -1)
                cv2.circle(diff_640, (x0_640, y0_640), cursor_mask_r_640, 0, -1)
                cv2.line(diff_640, (x0_640, y0_640), (x1_640, y1_640), 0, thickness=2 * cursor_mask_r_640)

                d_small = cv2.resize(diff_640, (width, height), interpolation=cv2.INTER_AREA)

                # Secondary suppression in 96x72 to guarantee zero boundary artifacts
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
            version=7,
        )

        n_valid = int(np.sum(valid > 0.5))
        n_clicks = int(np.sum(clicks > 0.5))
        n_holds = int(np.sum(targets[:, 4] > 0.5))
        total_frames_built += n

        sessions_meta.append({"name": session.name, "path": str(out_npz), "frames": n})
        print(
            f"Built {session.name}: {n} frames in {elapsed:.1f}s ({fps_proc:.0f} FPS) "
            f"[valid={n_valid} ({n_valid/n*100:.1f}%), clicks={n_clicks}, hold={n_holds}]",
            flush=True,
        )

    index = {
        "version": 2,
        "dataset_version": "v7_1",
        "shape": [2, height, width],
        "channels": ["gray", "absdiff_16ms_masked"],
        "lookahead_ms": lookahead_ms,
        "fps_target": 60,
        "total_frames": total_frames_built,
        "sessions": sessions_meta,
    }
    index_path = output_root / "index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Done. Built {len(sessions_meta)} sessions ({total_frames_built} frames) -> {index_path}", flush=True)
    return index


if __name__ == "__main__":
    build_dataset_v7_1()
