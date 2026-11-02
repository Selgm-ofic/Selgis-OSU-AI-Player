"""Newest-frame perception and policy runtime; dry-run unless --live is set."""
import argparse
import ctypes
import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from capture.manager import CaptureManager, load_cfg
from control.controller import Controller
from control.hit_scheduler import HitScheduler
from control.approach_clock import ApproachClock
from perception.detector import load_detector
from perception.approach import annotate_approach
from perception.tracker import Tracker


class NullController:
    """Scheduler sink for dry-run; records state without calling SendInput."""
    def __init__(self):
        self.keys = {}

    def set_key(self, key, down):
        self.keys[key] = bool(down)

    def release_all(self):
        self.keys.clear()


class Player:
    def __init__(self, cfg, policy_path=None):
        self.cfg = cfg
        self.man = CaptureManager(cfg)
        self.det = load_detector(cfg)
        self.tracker = Tracker(preempt_s=float(
            cfg.get("beatmaps", {}).get("preempt_ms", 1200.0)) / 1000.0)
        self.ctl = Controller()
        self.policy = None
        self.scheduler = None
        self.approach_clock = ApproachClock()
        self._hidden = None
        if policy_path and os.path.exists(policy_path):
            from policy.model import PolicyGRU
            self.policy = PolicyGRU.load(policy_path)
            print(f"policy: {policy_path}")
        self._frame = None
        self._frame_ts = 0.0
        self._lock = threading.Lock()
        self._run = False
        self._alternate = False
        self.stats = {"frames": 0, "det_ms": 0.0, "policy_ms": 0.0,
                      "miss": 0, "t0": 0.0}

    def _capture_loop(self):
        while self._run:
            frame, timestamp = self.man.read()
            if frame is None:
                self.stats["miss"] += 1
                time.sleep(0.002)
                continue
            with self._lock:
                self._frame = frame
                self._frame_ts = timestamp

    def _to_screen(self, nx, ny):
        field = self.cfg["playfield"]
        width = float(field["right"]) - float(field["left"])
        height = float(field["bottom"]) - float(field["top"])
        return float(field["left"]) + nx * width, float(field["top"]) + ny * height

    def _heuristic(self, tracks):
        if not tracks:
            return None, False
        target = tracks[0]
        hold = any(t.cls == "slider_ball" and t.missed == 0 for t in tracks)
        return self._to_screen(target.x, target.y), hold or target.urgency > 0.85

    def run(self, debug=False, live=False, max_frames=None):
        if live and self.policy is None:
            raise ValueError("--live requires a trained --policy")
        if not self.man.open():
            print("FAIL: no capture")
            return
        print(f"play source={self.man.mode} mode={'LIVE' if live else 'DRY-RUN'}")
        if live:
            print("LIVE control starts in 3 seconds. Hold F10 for emergency stop.")
            time.sleep(3.0)
        self.det.warmup(5)
        self._run = True
        self.stats["t0"] = time.perf_counter()
        threading.Thread(target=self._capture_loop, daemon=True).start()
        last_loop = time.perf_counter()
        last_frame_ts = None
        cursor = np.array([0.5, 0.5], dtype=float)
        previous_cursor = cursor.copy()
        tth_ms, hold, timing_source = float("inf"), False, "policy"
        if self.policy is not None:
            runtime = self.cfg.get("runtime", {})
            self.scheduler = HitScheduler(
                self.ctl if live else NullController(),
                latency_ms=float(runtime.get("click_latency_ms", 8.0)),
                tick_hz=float(runtime.get("control_hz", 500.0)),
            )
            self.scheduler.start()
        try:
            while True:
                if live and (ctypes.windll.user32.GetAsyncKeyState(0x79) & 0x8000):
                    print("F10 emergency stop")
                    break
                with self._lock:
                    frame, frame_ts = self._frame, self._frame_ts
                if frame is None or frame_ts == last_frame_ts:
                    time.sleep(0.001)
                    continue
                last_frame_ts = frame_ts
                now = time.perf_counter()
                dt = max(0.001, now - last_loop)
                last_loop = now
                roi = self.man.get_playfield(frame)
                tiny, _ = self.man.preprocess(roi)
                started = time.perf_counter()
                detections = self.det.predict(tiny)
                annotate_approach(tiny, detections,
                                  float(self.cfg.get("yolo", {}).get("box_scale", 1.15)))
                tracks = self.tracker.update(detections, dt, now)
                self.stats["det_ms"] += (time.perf_counter() - started) * 1000.0
                self.stats["frames"] += 1

                if self.policy is not None:
                    from policy.features import (build_features, build_features_v3,
                                                 FEATURE_DIM, FEATURE_DIM_V3)
                    from policy.model import (CURSOR_SPEED_SCALE, MAX_CURSOR_STEP,
                                              TTH_SCALE_MS)
                    if live:
                        sx_actual, sy_actual = self.ctl.position()
                        field = self.cfg["playfield"]
                        current_cursor = np.asarray([
                            (sx_actual - field["left"]) /
                            (field["right"] - field["left"]),
                            (sy_actual - field["top"]) /
                            (field["bottom"] - field["top"]),
                        ], dtype=float)
                        current_cursor = np.clip(current_cursor, 0.0, 1.0)
                    else:
                        current_cursor = cursor.copy()
                    velocity = (current_cursor - previous_cursor) / dt
                    previous_cursor = current_cursor.copy()
                    clicking = bool(self.scheduler and self.scheduler.key_down)
                    # dispatch to matching feature layout for the loaded policy
                    if getattr(self.policy, "input_dim", FEATURE_DIM) == FEATURE_DIM_V3:
                        feature = build_features_v3(
                            tracks, (current_cursor[0], current_cursor[1],
                                     velocity[0], velocity[1], clicking)
                        )
                    else:
                        feature = build_features(
                            tracks, (current_cursor[0], current_cursor[1],
                                     velocity[0], velocity[1], clicking)
                        )
                    started = time.perf_counter()
                    target_slot = None
                    if getattr(self.policy, "is_v3", False):
                        output, self._hidden, target_slot = self.policy.act_with_target(
                            feature, self._hidden)
                    else:
                        output, self._hidden = self.policy.act(feature, self._hidden)
                    self.stats["policy_ms"] += (time.perf_counter() - started) * 1000.0
                    tth_ms = float(output[2]) * TTH_SCALE_MS
                    timing_source = "policy"
                    phase_hit = self.approach_clock.update(tracks, now)
                    if phase_hit is not None:
                        tth_ms = phase_hit[0]
                        timing_source = "ring"
                    aim_target = None
                    if phase_hit is not None:
                        aim_target = np.asarray(phase_hit[2:4], dtype=float)
                    elif target_slot is not None and target_slot < len(tracks):
                        selected = tracks[target_slot]
                        if selected.missed <= 2 and selected.cls != "spinner":
                            aim_target = np.asarray([selected.x, selected.y], dtype=float)
                    if aim_target is not None:
                        delta = aim_target - current_cursor
                        delta_norm = float(np.linalg.norm(delta))
                        if delta_norm > MAX_CURSOR_STEP:
                            delta *= MAX_CURSOR_STEP / delta_norm
                        cursor = np.clip(current_cursor + delta, 0.0, 1.0)
                    else:
                        cursor = current_cursor.copy()
                    hold = bool(output[3] >= 0.5)
                    sx, sy = self._to_screen(cursor[0], cursor[1])
                    if self.scheduler is not None:
                        click_targets = [t for t in tracks
                                         if t.cls in ("circle", "slider_head", "spinner")
                                         and t.missed == 0 and t.age >= 2]
                        if phase_hit is not None:
                            self.scheduler.update(tth_ms, hold,
                                                  (phase_hit[2], phase_hit[3]), now)
                        elif click_targets:
                            target = max(click_targets, key=lambda t: t.urgency)
                            self.scheduler.update(tth_ms, hold,
                                                  (target.x, target.y), now)
                        else:
                            self.scheduler.invalidate(now)
                else:
                    position, press = self._heuristic(tracks)
                    if position is None:
                        continue
                    sx, sy = position
                    field = self.cfg["playfield"]
                    cursor = np.asarray([
                        (sx - field["left"]) / (field["right"] - field["left"]),
                        (sy - field["top"]) / (field["bottom"] - field["top"]),
                    ])
                    if live:
                        self._alternate = not self._alternate if press else self._alternate
                        self.ctl.set_key("z", press and self._alternate)
                        self.ctl.set_key("x", press and not self._alternate)

                if live:
                    self.ctl.move(sx, sy)
                if debug and self.stats["frames"] % 30 == 0:
                    count = self.stats["frames"]
                    elapsed = now - self.stats["t0"]
                    policy_avg = self.stats["policy_ms"] / count
                    print(f"fps={count/max(elapsed, 1e-6):.1f} "
                          f"det={self.stats['det_ms']/count:.1f}ms "
                          f"policy={policy_avg:.2f}ms tracks={len(tracks)} "
                          f"tth={tth_ms:.0f}ms timing={timing_source} hold={int(hold)} "
                          f"hits={self.scheduler.hits if self.scheduler else 0}")
                if max_frames is not None and self.stats["frames"] >= max_frames:
                    break
        except KeyboardInterrupt:
            print("stop")
        finally:
            self._run = False
            if self.scheduler is not None:
                self.scheduler.close()
            else:
                self.ctl.release_all()
            self.man.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--source", default=None, choices=["obs", "dxgi", "auto"])
    parser.add_argument("--policy", default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--live", action="store_true",
                        help="move mouse and press keys; default is dry-run")
    parser.add_argument("--frames", type=int, default=None)
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    if args.source:
        cfg["source"] = args.source
    Player(cfg, args.policy).run(debug=args.debug, live=args.live,
                                 max_frames=args.frames)


if __name__ == "__main__":
    main()
