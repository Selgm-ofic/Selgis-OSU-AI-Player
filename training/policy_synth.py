"""Generate noisy tracker-feature policy sessions directly from beatmaps."""
import argparse
import glob
import os
import random

import numpy as np

from capture.manager import load_cfg
from perception.beatmap import objects_at, parse_osu_file, parse_osz
from training.action_timeline import TimeTransform
from training.session_labels import write_policy_labels
from training.time_sync import FrameObservation, ObservedObject, SyncResult


def safe_name(text):
    value = "".join(c if c.isalnum() or c in "-_" else "_" for c in text)
    return value[:80] or "map"


def generate_session(bm, output_dir, fps=60.0, max_frames=12000,
                     preempt_ms=1200.0, seed=7):
    rng = random.Random(seed)
    if not bm.objects:
        return 0
    preempt_ms = bm.preempt_ms
    start = max(0.0, bm.objects[0].time - preempt_ms)
    end = max((o.end_time if (o.is_slider or o.is_spinner) else o.time)
              for o in bm.objects) + 300.0
    timestamps = []
    observations = []
    step = 1000.0 / fps
    record_ms = 0.0
    map_ms = start
    while map_ms <= end and len(timestamps) < max_frames:
        detected = []
        for obj, cls, x, y in objects_at(bm, map_ms, preempt_ms):
            # Auto hits circles on time; the playable object disappears then.
            # Keeping the parser's 150 ms post-hit labeling grace period here
            # makes the already-consumed circle look like the next target.
            if cls == "circle" and map_ms >= obj.time:
                continue
            if rng.random() < 0.035:
                continue
            detected.append(ObservedObject(
                cls=cls,
                x=min(1.0, max(0.0, x / 512.0 + rng.gauss(0.0, 0.003))),
                y=min(1.0, max(0.0, y / 384.0 + rng.gauss(0.0, 0.003))),
                confidence=rng.uniform(0.78, 0.99),
                w=max(0.025, 0.08 + rng.gauss(0.0, 0.004)),
                h=max(0.025, 0.08 + rng.gauss(0.0, 0.004)),
                approach_progress=(
                    float(np.clip(1.0 - (obj.time - map_ms) / preempt_ms, 0.0, 1.0))
                    if cls in ("circle", "slider_head") else None
                ),
            ))
        timestamps.append(record_ms)
        observations.append(FrameObservation(record_ms, tuple(detected)))
        record_ms += step
        map_ms += step
    os.makedirs(output_dir, exist_ok=True)
    sync = SyncResult(TimeTransform(offset_ms=start), 1.0, 0.0,
                      len(observations), 0.0)
    write_policy_labels(output_dir, bm, timestamps, sync, observations)
    return len(timestamps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--osu", default="")
    parser.add_argument("--osz", default="")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--out", default="data/policy_synth")
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--max-frames", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    jobs = []
    if args.osu:
        jobs = [parse_osu_file(args.osu)]
    elif args.osz:
        jobs = parse_osz(args.osz)
    elif args.all:
        for path in sorted(glob.glob(os.path.join(
                cfg.get("beatmaps", {}).get("dir", "C:/Test OSU"), "*.osz"))):
            jobs.extend(parse_osz(path))
    else:
        raise ValueError("provide --osu, --osz or --all")
    total = 0
    for index, bm in enumerate(jobs):
        label = safe_name(f"{bm.artist}_{bm.title}_{bm.version}_{index}")
        count = generate_session(
            bm, os.path.join(args.out, label), args.fps, args.max_frames,
            float(cfg.get("beatmaps", {}).get("preempt_ms", 1200.0)),
            args.seed + index,
        )
        total += count
        print(f"{label}: {count} frames")
    print(f"generated {len(jobs)} maps / {total} frames -> {args.out}")


if __name__ == "__main__":
    main()
