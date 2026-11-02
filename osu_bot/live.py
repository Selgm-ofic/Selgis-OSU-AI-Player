import ctypes
from collections import deque
import math
import queue
import threading
import time
from typing import Optional

import cv2
import numpy as np
import torch

from .model import load_policy
from .controller import Controller
from .capture import CaptureManager


class HitScheduler:
    """High-frequency dual-key alternating hit scheduler for osu!.
    Provides independent state tracking for 'z' and 'x', enabling overlapping key presses
    on high-BPM streams while preventing double-tap chattering on single circles.
    """
    def __init__(
        self,
        controller: Controller,
        latency_ms: float = 8.0,
        tap_ms: float = 28.0,
        lookahead_ms: float = 180.0,
        tick_hz: float = 500.0,
        confidence_thresh: float = 0.60,
    ):
        self.controller = controller
        self.latency_ms = float(latency_ms)
        self.tap_ms = float(tap_ms)
        self.lookahead_ms = float(lookahead_ms)
        self.period = 1.0 / max(1.0, float(tick_hz))
        self.confidence_thresh = float(confidence_thresh)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._deadline = None
        self._active_keys = {}
        self._alternate = False
        self._fired = False
        self._last_tth = None
        self._last_target = None
        self._last_fire_time = None
        self._prev_click_prob = 0.0
        self._hold = False
        self.hits = 0
        self._thread: Optional[threading.Thread] = None

    @property
    def key_down(self) -> bool:
        with self._lock:
            return bool(self._active_keys)

    def start(self):
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def _trigger_key_locked(self, now: float, hold: bool):
        self._alternate = not self._alternate
        key = "x" if self._alternate else "z"
        other = "z" if key == "x" else "x"
        if key in self._active_keys and other not in self._active_keys:
            key = other
        elif key in self._active_keys:
            self.controller.set_key(key, False)
        self.controller.set_key(key, True)
        self._active_keys[key] = None if hold else now + (self.tap_ms / 1000.0)
        self._fired = True
        self._last_fire_time = now
        self._deadline = None
        self.hits += 1

    def update(self, time_to_hit_ms: float, hold: bool, target: tuple, now: Optional[float] = None, click_prob: float = 0.0):
        now = time.perf_counter() if now is None else now
        tth = float(time_to_hit_ms)
        with self._lock:
            changed_target = (self._last_target is not None and math.dist(target, self._last_target) > 0.035)
            jumped_to_next = (self._last_tth is not None and tth - self._last_tth > 28.0)
            time_since_fire = (now - self._last_fire_time) if self._last_fire_time is not None else 1.0
            rising_edge = (click_prob >= self.confidence_thresh and click_prob > self._prev_click_prob + 0.15)

            if self._fired:
                if tth > 10.0 or changed_target or jumped_to_next:
                    self._fired = False
                    self._deadline = None
                elif click_prob < (self.confidence_thresh * 0.5) and time_since_fire >= 0.012:
                    self._fired = False
                    self._deadline = None
                elif rising_edge and time_since_fire >= 0.016:
                    self._fired = False
                    self._deadline = None

            self._prev_click_prob = click_prob
            self._hold = bool(hold)
            if not self._hold:
                for key, release_at in list(self._active_keys.items()):
                    if release_at is None:
                        self._active_keys[key] = now + (self.tap_ms / 1000.0)

            zero_crossing = (-16.0 <= tth <= 0.0)
            should_fire = (
                (click_prob >= self.confidence_thresh) or
                (click_prob >= (self.confidence_thresh * 0.6) and -25.0 <= tth <= 20.0) or
                (zero_crossing and (click_prob >= (self.confidence_thresh * 0.4) or self._deadline is not None))
            )
            if should_fire and not self._fired:
                self._trigger_key_locked(now, self._hold)
            elif not self._fired and 0.0 <= tth <= self.lookahead_ms:
                self._deadline = now + max(0.0, tth - self.latency_ms) / 1000.0

            self._last_tth = tth
            self._last_target = tuple(target)

    def invalidate(self, now: Optional[float] = None):
        now = time.perf_counter() if now is None else now
        with self._lock:
            self._deadline = None
            self._fired = False
            self._last_tth = None
            self._last_target = None
            self._hold = False
            for key in list(self._active_keys):
                self._active_keys[key] = now

    def _loop(self):
        while not self._stop.is_set():
            now = time.perf_counter()
            with self._lock:
                if self._deadline is not None and now >= self._deadline and not self._fired:
                    self._trigger_key_locked(now, self._hold)
                for key, release_at in list(self._active_keys.items()):
                    if release_at is not None and now >= release_at:
                        self.controller.set_key(key, False)
                        del self._active_keys[key]
            time.sleep(self.period)

    def close(self):
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.2)
        self._thread = None
        self.controller.release_all()


class RelaxLive:
    def __init__(
        self,
        config_dict: dict,
        weights_path: str,
        clicks_enabled: bool = True,
        smooth: float = 0.0,
        show_preview: bool = False,
        messages_queue: Optional[queue.Queue] = None,
        frames_queue: Optional[queue.Queue] = None,
        confidence_thresh: float = 0.60,
    ):
        self.cfg = config_dict
        self.weights_path = str(weights_path)
        self.clicks_enabled = bool(clicks_enabled)
        self.smooth = float(smooth)
        self.show_preview = bool(show_preview)
        self.messages_queue = messages_queue
        self.frames_queue = frames_queue
        self.confidence_thresh = float(confidence_thresh)

        self.running = False
        self.active = False
        self._thread: Optional[threading.Thread] = None
        self.manager: Optional[CaptureManager] = None
        self.controller: Optional[Controller] = None
        self.scheduler: Optional[HitScheduler] = None

        # Playfield bounds
        field = self.cfg.get("playfield", {})
        self.fl = float(field.get("left", 295))
        self.ft = float(field.get("top", 35))
        self.fr = float(field.get("right", 1625))
        self.fb = float(field.get("bottom", 1075))
        self.fw = max(10.0, self.fr - self.fl)
        self.fh = max(10.0, self.fb - self.ft)

    @property
    def capture(self):
        return self.manager

    def start(self):
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._thread_entry, daemon=True)
        self._thread.start()

    def _thread_entry(self):
        try:
            self.run()
        except Exception as exc:
            if self.messages_queue is not None:
                self.messages_queue.put(("error", str(exc)))
            print(f"[ERROR] Live runner thread exception: {exc}")
        finally:
            self.running = False

    def stop(self):
        self.running = False
        if self.scheduler is not None:
            self.scheduler.close()
            self.scheduler = None
        if self.controller is not None:
            self.controller.release_all()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._thread = None

    def is_active(self) -> bool:
        return self.active

    def set_clicks(self, enabled: bool):
        self.clicks_enabled = bool(enabled)
        if not self.clicks_enabled:
            if self.scheduler is not None:
                self.scheduler.invalidate()
            if self.controller is not None:
                self.controller.release_all()

    def set_confidence(self, conf: float):
        self.confidence_thresh = float(conf)
        if self.scheduler is not None:
            self.scheduler.confidence_thresh = float(conf)

    def set_smooth(self, smooth: float):
        self.smooth = float(smooth)
        if self.controller is not None:
            self.controller.smooth = float(smooth)

    def update_roi(self, left: int, top: int, right: int, bottom: int):
        self.fl = float(left)
        self.ft = float(top)
        self.fr = float(right)
        self.fb = float(bottom)
        self.fw = max(10.0, self.fr - self.fl)
        self.fh = max(10.0, self.fb - self.ft)
        if self.manager is not None:
            self.manager.set_playfield(int(left), int(top), int(right), int(bottom))

    def run(self):
        timer_set = False
        try:
            ctypes.windll.winmm.timeBeginPeriod(1)
            timer_set = True
        except Exception:
            pass

        self.manager = CaptureManager(self.cfg)
        if not self.manager.open():
            raise RuntimeError("Capture initialization failed. Please ensure your display is accessible.")

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        print(f"Loading policy: {self.weights_path}")
        try:
            model = load_policy(self.weights_path, device=device)
            model.eval()
            dummy = torch.zeros((1, 6, 2, 72, 96), device=device)
            with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                model(dummy, temperature=0.5)
            print(f"Device: {device}")
        except Exception as exc:
            if device.type == "cuda":
                print(f"[!] Warning: CUDA execution failed ({exc}). Falling back to CPU...")
                device = torch.device("cpu")
                model = load_policy(self.weights_path, device=device)
                model.eval()
                print(f"Device: {device} (Fallback active)")
            else:
                raise exc

        # Re-sync playfield bounds
        field = self.cfg.get("playfield", {})
        self.fl = float(field.get("left", self.fl))
        self.ft = float(field.get("top", self.ft))
        self.fr = float(field.get("right", self.fr))
        self.fb = float(field.get("bottom", self.fb))
        self.fw = max(10.0, self.fr - self.fl)
        self.fh = max(10.0, self.fb - self.ft)
        print(f"Playfield bounds: Left={self.fl:.0f}, Top={self.ft:.0f}, Width={self.fw:.0f}, Height={self.fh:.0f}")

        target_fps = float(self.cfg.get("runtime", {}).get("target_fps", 60))
        interval = 1.0 / max(1.0, target_fps)

        self.controller = Controller(smooth=self.smooth)
        self.scheduler = HitScheduler(
            self.controller,
            latency_ms=8.0,
            tap_ms=28.0,
            lookahead_ms=180.0,
            tick_hz=500.0,
            confidence_thresh=self.confidence_thresh,
        )
        self.scheduler.start()

        last_target_xy = None
        history = deque(maxlen=6)
        previous_gray = None

        loop_start = time.perf_counter()
        count = 0
        started_all = loop_start
        last_stats = loop_start
        f2_was_pressed = False

        print("=" * 60)
        print("Semga Live is ready.")
        print("Hotkeys:")
        print("  [F2]  - Toggle Bot ON / OFF (starts PAUSED)")
        print("  [F10] - Exit Bot")
        print("Status: PAUSED (press F2 in-game to activate)")
        print("=" * 60)

        # Notify initial status if GUI message queue is present
        if self.messages_queue is not None:
            self.messages_queue.put(("status", self.active))

        self.running = True
        try:
            while self.running:
                now = time.perf_counter()

                # Hotkey F10: Exit
                if ctypes.windll.user32.GetAsyncKeyState(0x79) & 0x8000:
                    print("\n[F10] Exit requested. Stopping...")
                    break

                # Hotkey F2: Toggle Active / Paused
                f2_state = bool(ctypes.windll.user32.GetAsyncKeyState(0x71) & 0x8000)
                if f2_state and not f2_was_pressed:
                    self.active = not self.active
                    status_text = "ACTIVE" if self.active else "PAUSED"
                    print(f"\n>>> BOT STATUS: {status_text} <<<")
                    if not self.active and self.scheduler is not None:
                        self.scheduler.invalidate(now)
                    if self.messages_queue is not None:
                        self.messages_queue.put(("status", self.active))
                f2_was_pressed = f2_state

                frame, _ = self.manager.read()
                roi = self.manager.get_playfield(frame)
                if roi is None:
                    time.sleep(0.001)
                    continue

                gray_full = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray_full, (640, 360), interpolation=cv2.INTER_AREA)

                if previous_gray is None or previous_gray.shape != gray.shape:
                    diff = np.zeros_like(gray)
                else:
                    diff = cv2.convertScaleAbs(cv2.absdiff(gray, previous_gray), alpha=3.0)
                previous_gray = gray

                s_gray = cv2.resize(gray, (96, 72), interpolation=cv2.INTER_AREA)
                s_diff = cv2.resize(diff, (96, 72), interpolation=cv2.INTER_AREA)
                small = np.stack((s_gray, s_diff), axis=0)
                history.append(small)

                if len(history) < 6:
                    time.sleep(0.001)
                    continue

                x = torch.from_numpy(np.stack(history).astype(np.float32) / 255.0).unsqueeze(0).to(device)

                infer_t0 = time.perf_counter()
                with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    output = model(x, temperature=0.5)
                infer_ms = (time.perf_counter() - infer_t0) * 1000.0

                px, py = output["xy"][0].detach().cpu().numpy()
                click_val = float(output["click"][0].detach().cpu())
                hold_val = float(output["hold"][0].detach().cpu())
                tth_val = float(output["tth"][0].detach().cpu())
                present_val = float(output["present"][0].detach().cpu())

                if self.active:
                    target_x, target_y = float(px), float(py)
                    last_target_xy = (target_x, target_y)
                    self.controller.move(self.fl + target_x * self.fw, self.ft + target_y * self.fh)

                    if self.clicks_enabled:
                        should_hold = (hold_val >= (self.confidence_thresh * 0.7))
                        self.scheduler.update(
                            time_to_hit_ms=tth_val,
                            hold=should_hold,
                            target=(target_x, target_y),
                            now=now,
                            click_prob=click_val
                        )
                else:
                    last_target_xy = None
                    if self.scheduler is not None:
                        self.scheduler.invalidate(now)

                # Queue frame for GUI live preview
                if self.frames_queue is not None:
                    try:
                        preview_img = cv2.resize(roi, (640, 360))
                        draw_x = last_target_xy[0] if last_target_xy is not None else float(px)
                        draw_y = last_target_xy[1] if last_target_xy is not None else float(py)
                        cx = int(round(draw_x * 639))
                        cy = int(round(draw_y * 359))
                        color = (0, 255, 0) if self.active else (0, 165, 255)
                        cv2.drawMarker(preview_img, (cx, cy), color, cv2.MARKER_CROSS, 20, 2)
                        if self.frames_queue.full():
                            try:
                                self.frames_queue.get_nowait()
                            except Exception:
                                pass
                        self.frames_queue.put_nowait(preview_img)
                    except Exception:
                        pass

                # Standalone OpenCV window preview (CLI runner)
                if self.show_preview and self.frames_queue is None:
                    preview_img = cv2.resize(roi, (640, 360))
                    draw_x = last_target_xy[0] if last_target_xy is not None else float(px)
                    draw_y = last_target_xy[1] if last_target_xy is not None else float(py)
                    cx = int(round(draw_x * 639))
                    cy = int(round(draw_y * 359))
                    color = (0, 255, 0) if self.active else (0, 0, 255)
                    cv2.drawMarker(preview_img, (cx, cy), color, cv2.MARKER_TILTED_CROSS, 16, 2)
                    cv2.imshow("Semga Live Preview", preview_img)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

                count += 1
                if now - last_stats >= 0.5:
                    fps = count / max(1e-4, now - started_all)
                    cap_fps = getattr(self.manager.dxgi, "fps", fps)
                    if self.messages_queue is not None:
                        self.messages_queue.put((
                            "metrics",
                            {
                                "inf_fps": fps,
                                "cap_fps": cap_fps,
                                "inf_ms": infer_ms,
                                "tth_ms": tth_val,
                            }
                        ))
                    else:
                        st = "ACTIVE" if self.active else "PAUSED"
                        print(f"[{st}] FPS: {fps:.1f} | Model: {infer_ms:.2f}ms | TTH: {tth_val:.1f}ms", end="\r", flush=True)
                    last_stats = now
                    count = 0
                    started_all = now

                # FPS throttle
                target_time = loop_start + count * interval
                rem = target_time - time.perf_counter()
                if rem > 0.0015:
                    time.sleep(rem - 0.001)
                while time.perf_counter() < target_time:
                    pass
                if time.perf_counter() > target_time + 0.05:
                    loop_start = time.perf_counter()
                    count = 0
                    started_all = loop_start

        finally:
            self.running = False
            if self.scheduler is not None:
                self.scheduler.close()
                self.scheduler = None
            if self.show_preview and self.frames_queue is None:
                cv2.destroyAllWindows()
            if timer_set:
                try:
                    ctypes.windll.winmm.timeEndPeriod(1)
                except Exception:
                    pass
            if self.manager is not None:
                self.manager.close()
            if self.controller is not None:
                self.controller.release_all()
            print("\nBot stopped safely.")
