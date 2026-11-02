"""Обучение детектора YOLO-nano на data/yolo (ultralytics, CUDA).

  python main.py train
  python main.py train --epochs 80 --batch 16

Веса: data/runs/detect/osu_nano/weights/best.pt
Устройство: GTX 1660 Ti (device=0), imgsz=640, pretrained yolov8n.pt.
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from capture.manager import load_cfg


def _yield_to_user():
    """Дневной режим: отдать приоритет интерактиву (браузер/видео).
    GPU-декод видео (NVDEC) отдельный блок — с BELOW_NORMAL обучение
    ему почти не мешает."""
    try:
        os.environ.setdefault("OMP_NUM_THREADS", "4")
        os.environ.setdefault("MKL_NUM_THREADS", "4")
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x4000)  # BELOW_NORMAL
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--name", default="osu_nano")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--weights", default="",
                    help="initial checkpoint for fine-tuning")
    ap.add_argument("--nice", action="store_true",
                    help="дневной режим: низкий приоритет процесса")
    ap.add_argument("--data", default=None,
                    help="переопределить датасет (напр. data/yolo2k)")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    dc = cfg.get("detector", {})
    if args.nice:
        _yield_to_user()
    from ultralytics import YOLO

    yolo_dir = args.data or cfg.get("yolo", {}).get("dir", "data/yolo")
    data_yaml = os.path.join(yolo_dir, "dataset.yaml")
    runs = os.path.abspath(os.path.join("data", "runs", "detect"))

    model_path = args.weights or "yolov8n.pt"
    last = os.path.join(runs, args.name, "weights", "last.pt")
    resuming = args.resume and os.path.exists(last)
    if resuming:
        model_path = last
        print(f"resume: {last}")

    model = YOLO(model_path)
    if args.nice:
        import time
        # WDDM cannot hard-cap compute utilization per process.  Frequent
        # short yields keep video playback and the desktop responsive while
        # retaining most of the training throughput.
        model.add_callback("on_train_batch_end", lambda trainer: time.sleep(0.008))
        model.add_callback("on_val_batch_end", lambda validator: time.sleep(0.004))
    res = model.train(
        data=os.path.abspath(data_yaml),
        resume=resuming,
        epochs=args.epochs or int(dc.get("epochs", 60)),
        batch=args.batch or int(dc.get("batch", 16)),
        imgsz=args.imgsz or int(dc.get("imgsz", 640)),
        device=int(dc.get("device", 0)),
        project=runs,
        name=args.name,
        exist_ok=True,
        patience=int(dc.get("patience", 15)),
        workers=args.workers if args.workers is not None else int(dc.get("workers", 4)),
        seed=7,
        # These are semantic channels, not RGB. Hue/saturation/value jitter
        # would mix or distort gray/motion/edge meaning between train and live.
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        verbose=True,
    )
    print(f"done. best: {os.path.join(runs, args.name, 'weights', 'best.pt')}")
    print(f"metrics: mAP50={res.results_dict.get('metrics/mAP50(B)', '?')} "
          f"mAP50-95={res.results_dict.get('metrics/mAP50-95(B)', '?')}")


if __name__ == "__main__":
    main()
