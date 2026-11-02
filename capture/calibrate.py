"""Калибровка: показывает кадр + ROI, помогает подогнать playfield и оценить lag.

Использование:
  python -m tools.calibrate --config configs/default.yaml
  Q - выход, S - сохранить текущий ROI обратно в yaml (вручную).
"""
import argparse
import cv2
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from capture.manager import CaptureManager, load_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--source", default=None, choices=["obs", "dxgi", "auto"])
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    if args.source:
        cfg["source"] = args.source
    man = CaptureManager(cfg)
    if not man.open():
        print("FAIL: нет источника. Включи OBS Virtual Camera или проверь монитор.")
        return
    print(f"OK source={man.mode}. Q=выход.")
    while True:
        frame, ts = man.read()
        if frame is None:
            continue
        roi = man.get_playfield(frame)
        tiny, _ = man.preprocess(roi)
        pf = cfg["playfield"]
        vis = frame.copy()
        cv2.rectangle(vis, (pf["left"], pf["top"]), (pf["right"], pf["bottom"]), (0, 255, 0), 2)
        cv2.putText(vis, f"{man.mode} fps={man.fps:.0f} frame={frame.shape[1]}x{frame.shape[0]}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("calibrate: full frame (green=playfield)", vis)
        cv2.imshow("calibrate: infer input 640x360", tiny)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    man.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
