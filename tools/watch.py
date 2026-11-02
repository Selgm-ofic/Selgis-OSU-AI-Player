"""Watch: живой предпросмотр захвата + ROI. Q = выход."""
import argparse
import os
import sys
import cv2

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
        print("FAIL: нет захвата")
        return
    print(f"Watch source={man.mode}, Q=выход")
    while True:
        frame, _ = man.read()
        if frame is None:
            continue
        roi = man.get_playfield(frame)
        pf = cfg["playfield"]
        cv2.rectangle(frame, (pf["left"], pf["top"]), (pf["right"], pf["bottom"]), (0, 255, 0), 2)
        cv2.putText(frame, f"{man.mode} fps={man.fps:.0f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("osu_module watch", frame)
        cv2.imshow("playfield 640x360", man.preprocess(roi)[0])
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    man.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
