"""Phase 0 realtime budget check: capture + crop + resize.

  python tools/bench.py --config configs/default.yaml --source auto
"""
import argparse
import os
import sys
import time
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from capture.manager import CaptureManager, load_cfg


def check_providers():
    info = {}
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.cuda.is_available()
        if info["torch_cuda"]:
            info["torch_gpu"] = torch.cuda.get_device_name(0)
    except Exception as e:
        info["torch_error"] = str(e)
    try:
        import onnxruntime as ort
        info["onnx"] = ort.__version__
        info["onnx_providers"] = ort.get_available_providers()
    except Exception as e:
        info["onnx_error"] = str(e)
    return info


def stat(a):
    if not a:
        return "n/a"
    s = sorted(a)
    p95 = s[min(len(s) - 1, int(len(s) * 0.95))]
    return f"avg={statistics.mean(a):.2f}ms p95={p95:.2f}ms max={max(a):.2f}ms"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--source", default=None, choices=["obs", "dxgi", "auto"])
    ap.add_argument("--frames", type=int, default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    if args.source:
        cfg["source"] = args.source
    n = args.frames or int(cfg.get("runtime", {}).get("bench_frames", 200))

    print("== providers ==")
    for k, v in check_providers().items():
        print(f"  {k}: {v}")

    man = CaptureManager(cfg)
    if not man.open():
        print("FAIL: no capture source. Enable OBS Virtual Camera.")
        return
    print(f"OK source={man.mode}, target: {n} frames")

    t_read, t_crop, t_pre = [], [], []
    got = 0
    for _ in range(n):
        t0 = time.perf_counter()
        frame, _ = man.read()
        t1 = time.perf_counter()
        if frame is None:
            continue
        roi = man.get_playfield(frame)
        t2 = time.perf_counter()
        tiny, _ = man.preprocess(roi)
        t3 = time.perf_counter()
        t_read.append((t1 - t0) * 1000)
        t_crop.append((t2 - t1) * 1000)
        t_pre.append((t3 - t2) * 1000)
        got += 1

    mode, fps = man.mode, man.fps
    man.close()

    print(f"frames: {got}/{n}, source={mode}, fps~{fps:.0f}")
    print(f"  read (capture):      {stat(t_read)}")
    print(f"  crop (ROI):          {stat(t_crop)}")
    print(f"  resize (640x360):    {stat(t_pre)}")
    total = (sum(t_read) + sum(t_crop) + sum(t_pre)) / max(got, 1)
    print(f"  TOTAL preprocess:    avg={total:.2f}ms")
    print(f"  headroom for YOLO-nano + policy: ~{16.6 - total:.1f}ms of 16.6ms frame")


if __name__ == "__main__":
    main()
