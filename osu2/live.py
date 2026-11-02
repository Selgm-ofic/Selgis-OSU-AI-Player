from __future__ import annotations

import base64
from collections import deque
import ctypes
import math
import queue
import threading
import time

import cv2
import numpy as np
import torch

from .train import load_policy


class LowPassFilter:
    def __init__(self):
        self.y = None

    def filter(self, val, alpha):
        if self.y is None:
            self.y = val
        else:
            self.y = alpha * val + (1.0 - alpha) * self.y
        return self.y


class OneEuroFilter:
    def __init__(self, min_cutoff=1.5, beta=0.04, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_time = None

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / max(1e-5, dt))

    def filter(self, val, timestamp):
        if self.last_time is None:
            self.last_time = timestamp
            return self.x_filter.filter(val, 1.0)
        dt = max(1e-4, timestamp - self.last_time)
        self.last_time = timestamp
        prev_x = self.x_filter.y if self.x_filter.y is not None else val
        dx = (val - prev_x) / dt
        edx = self.dx_filter.filter(dx, self._alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self.x_filter.filter(val, self._alpha(cutoff, dt))

    def reset(self):
        self.x_filter.y = None
        self.dx_filter.y = None
        self.last_time = None


class RelaxLive:
    def __init__(self, config_path, weights, source, messages, frames, enabled=False, clicks_enabled=True, beatmap=None, confidence_thresh=0.60, target_fps=60.0, playfield=None):
        self.config_path, self.weights, self.source = config_path, weights, source
        self.messages, self.frames = messages, frames
        self.clicks_enabled = clicks_enabled
        self.beatmap = beatmap
        self.confidence_thresh = float(confidence_thresh)
        self.target_fps = float(target_fps)
        self.playfield_override = dict(playfield) if playfield else None
        self._fps_lock = threading.Lock()
        self._reset_timing = False
        self.control_enabled = threading.Event()
        if enabled: self.control_enabled.set()
        self.hotkey = None
        self.stop_event = threading.Event()
        self.thread = None
        self.preview_queue = queue.Queue(maxsize=1)
        self.preview_thread = None

    def set_target_fps(self, fps: float):
        with self._fps_lock:
            self.target_fps = max(10.0, min(360.0, float(fps)))
            self._reset_timing = True

    def set_playfield(self, playfield: dict):
        self.playfield_override = dict(playfield)

    @property
    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self):
        if self.running:
            return
        self.stop_event.clear()
        import sys
        from pathlib import Path
        legacy_root = Path(r"C:\Selgis OSU AI Player")
        if str(legacy_root) not in sys.path: sys.path.insert(0, str(legacy_root))
        from training.recorder import HotkeyArm
        self.hotkey = HotkeyArm("F2")
        self.hotkey.start()
        threading.Thread(target=self._hotkey_loop, daemon=True).start()
        self.preview_thread = threading.Thread(target=self._preview_worker, daemon=True)
        self.preview_thread.start()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _hotkey_loop(self):
        while not self.stop_event.is_set() and self.hotkey is not None:
            if self.hotkey.wait_press(timeout=.2):
                if self.control_enabled.is_set():
                    self.control_enabled.clear()
                    state = "paused"
                else:
                    self.control_enabled.set()
                    state = "active"
                self.messages.put(("live_state", {"state": state, "message": "F2 toggle"}))

    def stop(self):
        self.stop_event.set()
        if self.hotkey is not None:
            self.hotkey.close()
            self.hotkey = None
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=2.0)
        self.thread = None

    def _preview_worker(self):
        while not self.stop_event.is_set():
            try:
                frame = self.preview_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                continue
            payload = base64.b64encode(encoded).decode("ascii")
            try:
                self.frames.put_nowait(payload)
            except queue.Full:
                try: self.frames.get_nowait()
                except queue.Empty: pass
                try: self.frames.put_nowait(payload)
                except queue.Full: pass

    def _publish(self, frame):
        try:
            self.preview_queue.put_nowait(frame)
        except queue.Full:
            try: self.preview_queue.get_nowait()
            except queue.Empty: pass
            try: self.preview_queue.put_nowait(frame)
            except queue.Full: pass

    def _run(self):
        import sys
        from pathlib import Path
        legacy_root = Path(r"C:\Selgis OSU AI Player")
        if str(legacy_root) not in sys.path: sys.path.insert(0, str(legacy_root))
        from capture.manager import CaptureManager, load_cfg
        from control.controller import Controller
        from control.hit_scheduler import HitScheduler
        manager = None
        scheduler = None
        timer_set = False
        try:
            try:
                ctypes.windll.winmm.timeBeginPeriod(1)
                timer_set = True
            except Exception:
                pass
            cfg = load_cfg(self.config_path)
            cfg["source"] = self.source
            manager = CaptureManager(cfg)
            if not manager.open(): raise RuntimeError("DXGI/OBS capture did not open")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, _ = load_policy(self.weights, device)
            with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                model(torch.zeros((1, 6, 2, 72, 96), device=device))
            field = self.playfield_override if self.playfield_override else cfg.get("playfield", {})
            controller = Controller(smooth=.0)
            scheduler = HitScheduler(controller, latency_ms=18.0, tap_ms=32.0, lookahead_ms=180.0, tick_hz=500.0)
            scheduler.start()
            filter_x = OneEuroFilter(min_cutoff=2.0, beta=0.6, d_cutoff=1.0)
            filter_y = OneEuroFilter(min_cutoff=2.0, beta=0.6, d_cutoff=1.0)
            last_target_xy = None
            history = deque(maxlen=6)
            past_grays = deque(maxlen=20)
            current_target_fps = self.target_fps
            interval = 1.0 / max(1.0, current_target_fps)
            count = 0
            start_loop = time.perf_counter()
            started_all = start_loop
            last_stats = start_loop
            last_preview = 0.0
            self.messages.put(("live_state", {"state": "paused", "mode": "mouse+keys" if self.clicks_enabled else "mouse-only", "message": "F2 activates controls"}))
            while not self.stop_event.is_set():
                now = time.perf_counter()
                if ctypes.windll.user32.GetAsyncKeyState(0x79) & 0x8000:
                    self.stop_event.set()
                    break

                with self._fps_lock:
                    if self._reset_timing:
                        current_target_fps = self.target_fps
                        interval = 1.0 / max(1.0, current_target_fps)
                        start_loop = time.perf_counter()
                        started_all = start_loop
                        count = 0
                        self._reset_timing = False

                if self.playfield_override:
                    field = self.playfield_override

                frame, _ = manager.read()
                roi = manager.get_playfield(frame)
                if roi is None:
                    time.sleep(0.001)
                    continue
                gray_full = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray_full, (640, 360), interpolation=cv2.INTER_AREA)

                lag_frames = max(1, round(current_target_fps / 60.0))
                past_grays.append(gray)
                if len(past_grays) > lag_frames:
                    past_gray = past_grays[-1 - lag_frames]
                    diff = cv2.convertScaleAbs(cv2.absdiff(gray, past_gray), alpha=3.0)
                else:
                    diff = np.zeros_like(gray)

                s_gray = cv2.resize(gray, (96, 72), interpolation=cv2.INTER_AREA)
                s_diff = cv2.resize(diff, (96, 72), interpolation=cv2.INTER_AREA)
                small = np.stack((s_gray, s_diff), axis=0)
                history.append(small)
                if len(history) < 6:
                    if now - last_preview >= .1:
                        last_preview = now
                        self._publish(roi.copy())
                    continue
                x = torch.from_numpy(np.stack(history).astype(np.float32) / 255.0).unsqueeze(0).to(device)
                started_model = time.perf_counter()
                with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    output = model(x, temperature=0.5)
                model_ms = (time.perf_counter() - started_model) * 1000.0
                px, py = output["xy"][0].detach().cpu().numpy()
                click_val = float(output["click"][0].detach().cpu()) if "click" in output else 0.0
                hold_val = float(output["hold"][0].detach().cpu()) if "hold" in output else 0.0
                tth_val = float(output["tth"][0].detach().cpu())
                present_val = float(output["present"][0].detach().cpu()) if "present" in output else 1.0

                # Calibrated confidence calculation (0.0 to 1.0)
                if "present" in output and present_val > 0.05:
                    confidence = present_val
                else:
                    peak_logit = float(output["heatmap"][0].max().detach().cpu())
                    confidence = float(1.0 / (1.0 + np.exp(-peak_logit)))

                confidence_thresh = getattr(self, "confidence_thresh", 0.60)

                if self.control_enabled.is_set():
                    target_x, target_y = float(px), float(py)
                    last_target_xy = (target_x, target_y)
                    controller.move(field["left"] + target_x * (field["right"] - field["left"]),
                                    field["top"] + target_y * (field["bottom"] - field["top"]))

                    should_hold = (hold_val >= 0.40)
                    if self.clicks_enabled:
                        scheduler.update(tth_val, should_hold, (target_x, target_y), now=now, click_prob=click_val)
                    else:
                        scheduler.invalidate(now)
                else:
                    last_target_xy = None
                    filter_x.reset()
                    filter_y.reset()
                    scheduler.invalidate(now)

                if now - last_preview >= .1:
                    last_preview = now
                    shown = roi.copy()
                    draw_x = last_target_xy[0] if last_target_xy is not None else float(px)
                    draw_y = last_target_xy[1] if last_target_xy is not None else float(py)
                    cv2.drawMarker(shown, (round(draw_x * (shown.shape[1]-1)), round(draw_y * (shown.shape[0]-1))), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 20, 2)
                    self._publish(shown)
                count += 1
                if now - last_stats >= .5:
                    last_stats = now
                    self.messages.put(("live_stats", {"fps": count / max(.001, now - started_all), "model_ms": model_ms, "tth_ms": float(output["tth"][0])}))
                target_time = start_loop + count * interval
                now_after = time.perf_counter()
                rem = target_time - now_after
                if rem > 0.0015:
                    time.sleep(rem - 0.001)
                while time.perf_counter() < target_time:
                    pass
                if now_after > target_time + 0.05:
                    start_loop = time.perf_counter()
                    count = 0
                    started_all = start_loop
        except Exception as exc:
            self.messages.put(("live_state", {"state": "error", "message": str(exc)}))
        finally:
            if timer_set:
                try:
                    ctypes.windll.winmm.timeEndPeriod(1)
                except Exception:
                    pass
            if scheduler is not None:
                scheduler.close()
            if manager is not None:
                manager.close()
            controller.release_all()
            self.messages.put(("live_state", {"state": "stopped"}))

