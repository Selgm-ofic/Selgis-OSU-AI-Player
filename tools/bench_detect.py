"""Замер латентности детектора на синт-кадрах (без камеры).

  python tools/bench_detect.py --frames 100
"""
import argparse
import os
import sys
import glob
import time
import statistics
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from capture.manager import load_cfg
from perception.detector import load_detector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--weights", default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    if args.weights:
        cfg["detector"]["weights"] = args.weights
    det = load_detector(cfg)
    print("warmup...")
    det.warmup()

    imgs = sorted(glob.glob("data/yolo/images/*.jpg"))
    if not imgs:
        print("нет кадров в data/yolo/images")
        return
    det.reset_stats()
    counts = []
    for i in range(args.frames):
        im = cv2.imread(imgs[i % len(imgs)])
        t0 = time.perf_counter()
        dets = det.predict(im)
        counts.append(len(dets))
        if i == 0:
            for d in dets[:8]:
                print(f"  {d.cls:12s} cx={d.x:.2f} cy={d.y:.2f} conf={d.conf:.2f}")
    print(f"frames={args.frames} avg_det={det.avg_ms:.2f}ms "
          f"avg_obj={sum(counts)/len(counts):.1f}")
    print(f"бюджет: capture~16 + detect {det.avg_ms:.1f} + policy ~1-2мс -> "
          f"{'OK (<25мс)' if 16 + det.avg_ms + 2 < 25 else 'ПЛОТНО, нужен async-пайплайн'}")


if __name__ == "__main__":
    main()
