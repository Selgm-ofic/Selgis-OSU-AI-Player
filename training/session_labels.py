"""Build synchronized policy targets from a recorded osu! session.

The detector and the policy are deliberately trained separately.  YOLO is
used offline here only to align a video to its beatmap.  Policy labels then
come from the exact beatmap clock, not from a nearest video-frame click bit.
"""
from dataclasses import asdict
import argparse
import json
import os
import time
from typing import Iterable, Sequence

import cv2
import numpy as np

from capture.manager import load_cfg
from perception.beatmap import Beatmap, parse_osu_file, parse_osz
from perception.detector import load_detector
from perception.hybrid import augment_detections
from perception.approach import annotate_approach
from perception.schema import Det
from perception.tracker import Tracker
from policy.features import build_features_v3 as build_features
from training.action_timeline import TimeTransform, build_action_targets, build_key_schedule
from training.autolabel import find_osu_in_osz
from training.time_sync import (
    FrameObservation,
    ObservedObject,
    SyncResult,
    estimate_auto_anchor_sync,
    estimate_sync,
    identify_beatmap,
)

OBSERVATION_CACHE_VERSION = 4


def _observation_cache_path(session_dir):
    return os.path.join(session_dir, "vision_observations_v4.json")


def load_observation_cache(session_dir, stamps):
    path = _observation_cache_path(session_dir)
    video = os.path.join(session_dir, "video.mp4")
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if (payload.get("version") != OBSERVATION_CACHE_VERSION or
                payload.get("video_size") != os.path.getsize(video) or
                len(payload.get("frames", [])) != len(stamps)):
            return None
        return [FrameObservation(
            record_ms=float(row["record_ms"]),
            objects=tuple(ObservedObject(**item) for item in row["objects"]),
        ) for row in payload["frames"]]
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def save_observation_cache(session_dir, observations):
    video = os.path.join(session_dir, "video.mp4")
    payload = {
        "version": OBSERVATION_CACHE_VERSION,
        "video_size": os.path.getsize(video),
        "frames": [{
            "record_ms": frame.record_ms,
            "objects": [asdict(item) for item in frame.objects],
        } for frame in observations],
    }
    with open(_observation_cache_path(session_dir), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))


def collect_observations(
    session_dir: str,
    stamps: Sequence[float],
    detector,
    infer_size: tuple[int, int] = (640, 360),
) -> list[FrameObservation]:
    """Run the detector over every timestamped frame for frame-level sync."""
    temporal = (getattr(detector, "input_mode", "") == "temporal3" and
                os.path.isdir(os.path.join(session_dir, "vision_frames")))
    cap = None if temporal else cv2.VideoCapture(os.path.join(session_dir, "video.mp4"))
    observations = []
    frame_index = 0
    started = time.perf_counter()
    try:
        while frame_index < len(stamps):
            if temporal:
                path = os.path.join(session_dir, "vision_frames", f"{frame_index:06d}.png")
                frame = cv2.imread(path, cv2.IMREAD_COLOR)
                if frame is None:
                    break
                dets = detector.predict_preprocessed(frame)
                gray_current = frame[:, :, 0]
                approach_frame = cv2.cvtColor(gray_current, cv2.COLOR_GRAY2BGR)
            else:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.resize(frame, infer_size, interpolation=cv2.INTER_LINEAR)
                dets = detector.predict(frame)
                dets = augment_detections(frame, dets, prefer_geometry=True)
                approach_frame = frame
            annotate_approach(approach_frame, dets)
            dets = [d for d in dets if d.cls != "cursor"]
            observations.append(FrameObservation(
                record_ms=float(stamps[frame_index]),
                objects=tuple(
                    ObservedObject(d.cls, d.x, d.y, d.conf, d.w, d.h,
                                   approach_progress=d.urgency)
                    for d in dets
                ),
            ))
            frame_index += 1
            if frame_index % 100 == 0 or frame_index == len(stamps):
                elapsed = max(1e-6, time.perf_counter() - started)
                speed = frame_index / elapsed
                eta = (len(stamps) - frame_index) / max(speed, 1e-6)
                print(f"detect {frame_index}/{len(stamps)} "
                      f"({100.0*frame_index/len(stamps):.1f}%) "
                      f"{speed:.1f} fps ETA {eta:.0f}s", flush=True)
    finally:
        if cap is not None:
            cap.release()
    if not observations:
        raise ValueError("recording contains no readable frames")
    return observations


def load_geometry_sync(session_dir: str, bm: Beatmap):
    """Load the click-safe temporal-geometry clock produced during YOLO export."""
    path = os.path.join(session_dir, "detector_sync.json")
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        clock = payload["clock"]
        recorded_map = os.path.abspath(payload["beatmap"]["path"])
        if os.path.normcase(recorded_map) != os.path.normcase(os.path.abspath(bm.path)):
            return None
        if not bool(clock.get("safe", False)):
            return None
        quality = payload.get("quality", {})
        result = SyncResult(
            transform=TimeTransform(float(clock["offset_ms"]), float(clock["scale"])),
            score=float(quality.get("score", 0.0)),
            runner_up_score=float(quality.get("runner_up_score", 0.0)),
            observations_used=int(quality.get("anchors", 0)),
            max_time_uncertainty_ms=float(clock["uncertainty_ms"]),
            method=str(clock.get("method", "temporal_hit_geometry_v1")),
        )
        bounds = payload.get("usable_recording_ms", [float("-inf"), float("inf")])
        return result, (float(bounds[0]), float(bounds[1]))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def write_policy_labels(
    session_dir: str,
    bm: Beatmap,
    stamps: Sequence[float],
    sync: SyncResult,
    observations: Sequence[FrameObservation] | None = None,
) -> dict:
    """Write continuous state targets plus exact, between-frame key events."""
    targets = build_action_targets(bm, stamps, sync.transform)
    columns = [
        "record_ms", "map_ms", "cursor_x", "cursor_y",
        "time_to_next_hit_ms", "hold", "object_index",
    ]
    rows = np.asarray([
        [t.record_ms, t.map_ms, t.cursor_x, t.cursor_y,
         t.next_hit_ms, float(t.hold), float(t.object_index)]
        for t in targets
    ], dtype=np.float64)
    np.savez_compressed(
        os.path.join(session_dir, "policy_targets.npz"),
        targets=rows,
        columns=np.asarray(columns),
    )

    if observations is not None:
        tracker = Tracker()
        feature_rows = []
        previous_ms = None
        previous_cursor = None
        for frame, target in zip(observations, targets):
            now_s = frame.record_ms / 1000.0
            dt = 1.0 / 60.0 if previous_ms is None else max(
                0.001, (frame.record_ms - previous_ms) / 1000.0
            )
            dets = [
                Det(o.cls, o.x, o.y, o.w, o.h, o.confidence,
                    urgency=o.approach_progress)
                for o in frame.objects
            ]
            tracks = tracker.update(dets, dt, now_s)
            cursor = np.asarray([target.cursor_x, target.cursor_y], dtype=np.float64)
            velocity = np.zeros(2) if previous_cursor is None else (
                cursor - previous_cursor
            ) / dt
            clicking = target.z_down or target.x_down
            feature_rows.append(build_features(
                tracks,
                (cursor[0], cursor[1], velocity[0], velocity[1], clicking),
            ))
            previous_cursor = cursor
            previous_ms = frame.record_ms
        np.save(os.path.join(session_dir, "policy_features.npy"),
                np.asarray(feature_rows, dtype=np.float32))

    # These are scheduled on a 250--500 Hz control clock.  They are not
    # rounded to video frames and are not direct GRU classification targets.
    events = []
    for event in build_key_schedule(bm):
        events.append({
            "record_ms": sync.transform.to_record(event.time_ms),
            "map_ms": event.time_ms,
            "kind": "down" if event.down else "up",
            "key": event.key,
            "object_index": event.object_index,
        })
    with open(os.path.join(session_dir, "key_events.json"), "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2)

    manifest = {
        "beatmap": {
            "path": bm.path,
            "artist": bm.artist,
            "title": bm.title,
            "version": bm.version,
        },
        "clock": {
            "formula": "beatmap_ms = scale * recording_ms + offset_ms",
            "offset_ms": sync.transform.offset_ms,
            "scale": sync.transform.scale,
            "score": sync.score,
            "runner_up_score": sync.runner_up_score,
            "max_time_uncertainty_ms": sync.max_time_uncertainty_ms,
            "method": sync.method,
            "click_safe": sync.is_click_safe(),
        },
        "frames": len(targets),
        "features": len(observations) if observations is not None else 0,
        "events": len(events),
        "policy_target_columns": columns,
        "split_group": os.path.abspath(bm.path),
    }
    with open(os.path.join(session_dir, "policy_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def _lazer_maps(title_query: str) -> dict[str, Beatmap]:
    root = os.path.join(os.environ.get("APPDATA", ""), "osu", "files")
    if not os.path.isdir(root):
        raise FileNotFoundError(f"lazer file store not found: {root}")
    query = title_query.casefold()
    found = {}
    for folder, _, names in os.walk(root):
        for name in names:
            path = os.path.join(folder, name)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                    head = handle.read(65536)
                if query not in head.casefold() or "Mode: 0" not in head:
                    continue
                bm = parse_osu_file(path)
                if not bm.objects:
                    continue
                key = f"{bm.title} [{bm.version}] ({name[:8]})"
                found[key] = bm
            except (OSError, UnicodeError):
                continue
    if not found:
        raise FileNotFoundError(f"no osu!standard lazer maps match: {title_query}")
    return found


def _candidate_maps(osu: str, osz: str, diff: str,
                    lazer_title: str = "") -> dict[str, Beatmap]:
    if lazer_title:
        return _lazer_maps(lazer_title)
    if osu:
        bm = parse_osu_file(osu)
        return {bm.version or os.path.basename(osu): bm}
    if not osz:
        raise ValueError("provide --osu or --osz")
    if diff:
        bm = parse_osu_file(find_osu_in_osz(osz, diff))
        return {bm.version or diff: bm}
    maps = parse_osz(osz)
    return {(bm.version or f"difficulty_{i}"): bm for i, bm in enumerate(maps)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--session", required=True)
    ap.add_argument("--osu", default="")
    ap.add_argument("--osz", default="")
    ap.add_argument("--diff", default="")
    ap.add_argument("--lazer-title", default="")
    ap.add_argument("--rate", type=float, default=1.0,
                    help="map clock / real clock: 1 normal, 1.5 DT, 0.75 HT")
    ap.add_argument("--allow-unsafe", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    candidates = _candidate_maps(args.osu, args.osz, args.diff, args.lazer_title)
    print(f"map candidates: {len(candidates)}", flush=True)
    temporal_stamps = os.path.join(args.session, "vision_stamps.npy")
    use_temporal = (str(cfg.get("detector", {}).get("input_mode", "")) == "temporal3" and
                    os.path.exists(temporal_stamps))
    stamps = np.load(temporal_stamps if use_temporal else
                     os.path.join(args.session, "stamps.npy"))
    print(f"policy vision source: {'lossless temporal3 @ 30 FPS' if use_temporal else 'legacy video'}",
          flush=True)
    detector = load_detector(cfg)
    preprocess = cfg.get("preprocess", {})
    infer_size = (int(preprocess.get("infer_width", 640)),
                  int(preprocess.get("infer_height", 360)))
    observations = load_observation_cache(args.session, stamps)
    if observations is None:
        observations = collect_observations(args.session, stamps, detector, infer_size)
        save_observation_cache(args.session, observations)
        print("vision cache: saved", flush=True)
    else:
        print(f"vision cache: reused ({len(observations)} frames)", flush=True)
    sync_args = dict(
        offset_range_ms=(-15000.0, 15000.0),
        coarse_step_ms=50.0,
        fine_step_ms=1.0,
        scales=(args.rate,),
        preempt_ms=float(cfg.get("beatmaps", {}).get("preempt_ms", 1200.0)),
    )
    usable_bounds = (float("-inf"), float("inf"))
    saved_sync = None
    if len(candidates) == 1:
        map_name, bm = next(iter(candidates.items()))
        saved_sync = load_geometry_sync(args.session, bm)
        if saved_sync is not None:
            sync, usable_bounds = saved_sync
            print("sync: reused click-safe temporal geometry clock", flush=True)
        else:
            sync = estimate_auto_anchor_sync(observations, bm, scale=args.rate)
            if sync is None or not sync.is_click_safe():
                sync = estimate_sync(observations, bm, **sync_args)
    else:
        map_name, sync = identify_beatmap(observations, candidates, **sync_args)
        bm = candidates[map_name]

    print(f"map: {map_name}")
    print(f"sync: method={sync.method}, scale={sync.transform.scale:.6f}, "
          f"offset={sync.transform.offset_ms:.1f} ms, "
          f"uncertainty<={sync.max_time_uncertainty_ms:.1f} ms")
    if not sync.is_click_safe() and not args.allow_unsafe:
        raise RuntimeError(
            "sync is not click-safe; keep this session out of policy training "
            "or rerun with --allow-unsafe for diagnostics"
        )
    pairs = [
        (stamp, observation)
        for stamp, observation in zip(stamps, observations)
        if usable_bounds[0] <= float(stamp) <= usable_bounds[1]
    ]
    if not pairs:
        raise RuntimeError("no policy frames remain inside the synchronized map window")
    used_stamps = np.asarray([row[0] for row in pairs], dtype=np.float64)
    used_observations = [row[1] for row in pairs]
    print(f"usable policy frames: {len(used_stamps)}/{min(len(stamps), len(observations))}",
          flush=True)
    manifest = write_policy_labels(
        args.session, bm, used_stamps, sync, observations=used_observations
    )
    print(f"wrote {manifest['frames']} frame targets and "
          f"{manifest['events']} exact key events")


if __name__ == "__main__":
    main()
