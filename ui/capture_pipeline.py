"""Single-owner capture pipeline for preview, recording and live diagnostics."""
import base64
import json
import os
import queue
import threading
import time

import cv2
import numpy as np

from capture.manager import CaptureManager, load_cfg
from control.approach_clock import ApproachClock
from control.controller import Controller
from control.hit_scheduler import HitScheduler
from perception.approach import annotate_approach
from perception.detector import load_detector
from perception.hybrid import augment_detections
from perception.tracker import Tracker
from perception.temporal import CHANNELS, temporal_frame
from policy.features import build_features, build_features_v3, FEATURE_DIM, FEATURE_DIM_V3
from policy.model import CURSOR_SPEED_SCALE, MAX_CURSOR_STEP, PolicyGRU, TTH_SCALE_MS
from training.recorder import HotkeyArm, beep


class NullController:
    def __init__(self):
        self.keys = {}

    def set_key(self, key, down):
        self.keys[key] = bool(down)

    def release_all(self):
        self.keys.clear()


class CapturePipeline:
    """Capture once, fan out to 60 FPS recording and newest-frame inference."""
    def __init__(self, messages, frames, config_path, source="dxgi", enable_inference=True):
        self.messages = messages
        self.frames = frames
        self.config_path = config_path
        self.source = source
        self.enable_inference = bool(enable_inference)
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.inference_ready = threading.Event()
        self.inference_error = None
        self.frame_lock = threading.Lock()
        self.record_lock = threading.RLock()
        self.control_lock = threading.RLock()
        self.aim_lock = threading.Lock()
        self.latest_roi = None
        self.latest_annotated = None
        self.latest_seq = 0
        self.manager = None
        self.detector = None
        self.capture_thread = None
        self.infer_thread = None
        self.hotkey = None
        self.recording = False
        self.record_armed = False
        self.session_dir = None
        self.record_meta = {}
        self.raw_writer = None
        self.detect_writer = None
        self.record_thread = None
        self.vision_thread = None
        self.vision_queue = None
        self.record_stop = threading.Event()
        self.stamps = []
        self.record_started = 0.0
        self.last_record_toggle_at = 0.0
        self.record_toggle_debounce_s = 0.70
        self.detected_frames = 0
        self.vision_frames = 0
        self.vision_stamps = []
        self.vision_dropped = 0
        self.record_size = [640, 360]
        self.record_fps_target = 60.0
        self.capture_count = 0
        self.infer_count = 0
        self.capture_started = 0.0
        self.infer_started = 0.0
        self.last_stats = 0.0
        self.last_preview_at = 0.0
        self.control_mode = "off"
        self.policy = None
        self.hidden = None
        self.controller = None
        self.scheduler = None
        self.aim_thread = None
        self.aim_stop = threading.Event()
        self.aim_target = None
        self.aim_updated_at = 0.0
        self.aim_target_id = None
        self.aim_start = None
        self.aim_started_at = 0.0
        self.aim_duration = .05
        self.aim_proposal_id = None
        self.aim_proposal_count = 0
        self.aim_speed = 12.0
        self.aim_min_s = .02
        self.aim_max_s = .14
        self.aim_early_ms = 35.0
        self.aim_ema = .25
        self.cursor = np.asarray([0.5, 0.5], dtype=float)
        self.previous_cursor = self.cursor.copy()
        self.last_policy_at = None
        self.last_detection_at = 0.0
        self.last_publish_at = 0.0

    @property
    def running(self):
        return (self.capture_thread is not None and
                self.capture_thread.is_alive() and not self.stop_event.is_set())

    def start(self):
        if self.running:
            return
        self.stop_event.clear()
        self.ready.clear()
        self.inference_ready.clear()
        self.inference_error = None
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        if self.enable_inference:
            self.infer_thread = threading.Thread(target=self._infer_loop, daemon=True)
            self.infer_thread.start()
        else:
            # Recording does not need realtime inference: actionlabel performs
            # the detector pass offline, after all lossless frames are saved.
            self.inference_ready.set()
        self.hotkey = HotkeyArm("F1")
        self.hotkey.start()
        threading.Thread(target=self._hotkey_loop, daemon=True).start()
        self.messages.put(("log", "F1 monitor started (hook + polling).\n"))

    def stop(self):
        self.stop_event.set()
        if self.hotkey is not None:
            self.hotkey.close()
        if self.recording:
            self.stop_recording()
        self.stop_control()
        for thread in (self.capture_thread, self.infer_thread):
            if thread is not None:
                thread.join(timeout=3.0)
        if self.manager is not None:
            self.manager.close()
        self.messages.put(("pipeline_stopped", None))

    def arm_recording(self, session_dir, metadata):
        self.session_dir = os.path.abspath(session_dir)
        self.record_meta = dict(metadata)
        self.record_armed = True
        os.makedirs(os.path.dirname(self.session_dir), exist_ok=True)
        self.messages.put(("record_state", {
            "state": "armed", "path": self.session_dir,
            "message": "ARMED: press F1 or the GUI toggle button",
        }))

    def _hotkey_loop(self):
        while not self.stop_event.is_set():
            if self.hotkey.wait_press(timeout=.2):
                self.messages.put(("log", "F1 detected by GUI capture pipeline.\n"))
                self.toggle_recording()

    def toggle_recording(self):
        # The low-level hook and its polling fallback can both observe one F1
        # edge around focus changes.  Without a short debounce, the second
        # notification immediately stops a session that the first just began.
        now = time.perf_counter()
        if now - self.last_record_toggle_at < self.record_toggle_debounce_s:
            self.messages.put(("log", "Ignored duplicate F1 toggle.\n"))
            return False
        self.last_record_toggle_at = now
        if self.recording:
            self.stop_recording()
            return True
        if not self.record_armed or not self.session_dir:
            self.messages.put(("record_state", {
                "state": "error", "message": "Choose a map/session and arm recording first",
            }))
            return False
        return self.start_recording()

    def start_recording(self):
        if not self.ready.wait(5.0):
            self.messages.put(("record_state", {
                "state": "error", "message": "Capture is not ready",
            }))
            return False
        if self.enable_inference and not self.inference_ready.is_set():
            self.messages.put(("record_state", {
                "state": "loading", "message": "F1 принят; жду загрузку и прогрев YOLO...",
            }))
        if (self.enable_inference and
                (not self.inference_ready.wait(30.0) or self.inference_error)):
            self.messages.put(("record_state", {
                "state": "error",
                "message": self.inference_error or "YOLO не загрузился за 30 секунд",
            }))
            return False
        with self.record_lock:
            os.makedirs(self.session_dir, exist_ok=True)
            self._clear_previous_session_outputs()
            self.stamps = []
            self.detected_frames = 0
            self.vision_frames = 0
            self.vision_stamps = []
            self.vision_dropped = 0
            self.record_started = time.perf_counter()
            self.raw_writer = None
            self.detect_writer = None
            self.recording = True
            self.record_stop.clear()
            self.vision_queue = queue.Queue(maxsize=96)
            self.vision_thread = threading.Thread(
                target=self._vision_writer_loop, daemon=True)
            self.vision_thread.start()
            self.record_thread = threading.Thread(target=self._record_loop, daemon=True)
            self.record_thread.start()
        beep(880, 120)
        self.messages.put(("record_state", {
            "state": "recording", "path": self.session_dir,
            "message": "RECORDING",
        }))
        return True

    def _clear_previous_session_outputs(self):
        """Remove only files generated by an earlier run of this same session."""
        generated = (
            "video.mp4", "detected.mp4", "stamps.npy", "vision_stamps.npy",
            "meta.json", "vision_observations_v3.json", "policy_targets.npz",
            "policy_features.npy", "key_events.json", "policy_manifest.json",
        )
        for name in generated:
            path = os.path.join(self.session_dir, name)
            if os.path.isfile(path):
                os.remove(path)
        vision_dir = os.path.join(self.session_dir, "vision_frames")
        if os.path.isdir(vision_dir):
            for name in os.listdir(vision_dir):
                path = os.path.join(vision_dir, name)
                if name.lower().endswith(".png") and os.path.isfile(path):
                    os.remove(path)

    def stop_recording(self):
        with self.record_lock:
            if not self.recording:
                return
            self.recording = False
            self.record_stop.set()
        if self.record_thread is not None and self.record_thread is not threading.current_thread():
            self.record_thread.join(timeout=3.0)
        self.record_thread = None
        if self.vision_queue is not None:
            self.vision_queue.put(None)
        if self.vision_thread is not None and self.vision_thread is not threading.current_thread():
            self.vision_thread.join(timeout=10.0)
        self.vision_thread = None
        with self.record_lock:
            self.record_armed = False
            if self.raw_writer is not None:
                self.raw_writer.release()
                self.raw_writer = None
            if self.detect_writer is not None:
                self.detect_writer.release()
                self.detect_writer = None
            stamps = np.asarray(self.stamps, dtype=np.float64)
            duration = float(stamps[-1]) if len(stamps) else 0.0
            np.save(os.path.join(self.session_dir, "stamps.npy"), stamps)
            np.save(os.path.join(self.session_dir, "vision_stamps.npy"),
                    np.asarray(self.vision_stamps, dtype=np.float64))
            meta = dict(self.record_meta)
            meta.update({
                "session": os.path.basename(self.session_dir),
                "source": self.manager.mode if self.manager else self.source,
                "frames": len(stamps), "duration_ms": duration,
                "record_fps": (1000.0 * max(0, len(stamps) - 1) / duration
                               if duration > 0 else 0.0),
                "detected_frames": self.detected_frames,
                "vision_frames": self.vision_frames,
                "vision_dropped": self.vision_dropped,
                "vision_complete": self.vision_frames > 0 and self.vision_dropped == 0,
                "vision_format": "png-lossless-temporal3",
                "vision_channels": list(CHANNELS),
                "vision_fps_target": self.record_fps_target,
                "video_size": self.record_size,
                "video_fps_target": self.record_fps_target,
                "detected_fps_target": self.record_fps_target,
                "hotkey": "F1", "pipeline": "gui-unified-temporal-v3",
            })
            with open(os.path.join(self.session_dir, "meta.json"), "w",
                      encoding="utf-8") as handle:
                json.dump(meta, handle, indent=2, ensure_ascii=False)
        beep(440, 140)
        self.messages.put(("record_state", {
            "state": "saved", "path": self.session_dir,
            "frames": len(stamps), "detected_frames": self.detected_frames,
            "vision_frames": self.vision_frames,
            "vision_dropped": self.vision_dropped,
            "vision_complete": self.vision_frames > 0 and self.vision_dropped == 0,
            "duration_ms": duration,
            "message": "SAVED",
        }))

    def start_control(self, policy_path, live=False):
        if not os.path.exists(policy_path):
            raise FileNotFoundError(policy_path)
        with self.control_lock:
            self.stop_control()
            self.policy = PolicyGRU.load(policy_path)
            self.hidden = None
            self.cursor[:] = .5
            self.previous_cursor[:] = .5
            self.last_policy_at = None
            self.controller = Controller() if live else NullController()
            cfg = load_cfg(self.config_path)
            runtime = cfg.get("runtime", {})
            self.aim_speed = max(1.0, float(runtime.get(
                "aim_speed_fields_per_s", 12.0)))
            self.aim_min_s = float(runtime.get("aim_min_travel_ms", 20.0)) / 1000.0
            self.aim_max_s = float(runtime.get("aim_max_travel_ms", 140.0)) / 1000.0
            self.aim_early_ms = float(runtime.get("aim_arrive_early_ms", 35.0))
            self.aim_ema = float(np.clip(runtime.get("aim_target_ema", .25), .05, 1.0))
            self.scheduler = HitScheduler(
                self.controller,
                latency_ms=float(runtime.get("click_latency_ms", 8.0)),
                tick_hz=float(runtime.get("control_hz", 500.0)),
            )
            self.scheduler.start()
            self.control_mode = "live" if live else "dry"
            self.aim_target = None
            self.aim_updated_at = 0.0
            self.aim_target_id = None
            self.aim_start = None
            self.aim_proposal_id = None
            self.aim_proposal_count = 0
            self.aim_stop.clear()
            if live:
                self.aim_thread = threading.Thread(target=self._aim_loop, daemon=True)
                self.aim_thread.start()
        self.messages.put(("control_state", {
            "mode": self.control_mode,
            "message": "LIVE armed; F10 stops control" if live else "DRY-RUN active",
        }))

    def stop_control(self):
        self.aim_stop.set()
        if (self.aim_thread is not None and
                self.aim_thread is not threading.current_thread()):
            self.aim_thread.join(timeout=.5)
        self.aim_thread = None
        with self.control_lock:
            if self.scheduler is not None:
                self.scheduler.close()
            self.scheduler = None
            self.controller = None
            self.policy = None
            self.hidden = None
            if self.control_mode != "off":
                self.messages.put(("control_state", {"mode": "off", "message": "Control stopped"}))
            self.control_mode = "off"

    def _aim_loop(self):
        """Move toward the latest vision target independently of YOLO latency."""
        cfg = load_cfg(self.config_path)
        runtime = cfg.get("runtime", {})
        hz = max(60.0, float(runtime.get("aim_hz", 240.0)))
        ttl = max(.03, float(runtime.get("aim_target_ttl_ms", 150.0)) / 1000.0)
        period = 1.0 / hz
        field = cfg["playfield"]
        width = field["right"] - field["left"]
        height = field["bottom"] - field["top"]
        last = time.perf_counter()
        while not self.stop_event.is_set() and not self.aim_stop.is_set():
            now = time.perf_counter()
            dt = min(.02, max(.001, now - last))
            last = now
            with self.aim_lock:
                target = None if self.aim_target is None else self.aim_target.copy()
                start = None if self.aim_start is None else self.aim_start.copy()
                started_at = self.aim_started_at
                duration = self.aim_duration
                updated_at = self.aim_updated_at
            controller = self.controller
            if (controller is not None and target is not None and start is not None
                    and now - updated_at <= ttl):
                phase = float(np.clip((now - started_at) / max(duration, .001), 0.0, 1.0))
                # Smoothstep has finite travel time: unlike EMA it reaches 1.0
                # exactly, so smoothing cannot make us miss the hit forever.
                eased = phase * phase * (3.0 - 2.0 * phase)
                cursor = np.clip(start + (target - start) * eased, 0.0, 1.0)
                controller.move(field["left"] + cursor[0] * width,
                                field["top"] + cursor[1] * height)
                self.cursor = cursor
            remaining = period - (time.perf_counter() - now)
            if remaining > 0:
                time.sleep(remaining)

    def _update_aim_target(self, target, target_id, tth_ms, current, now):
        """Debounce target switches and create a finite-time smooth trajectory."""
        target = np.asarray(target, dtype=float)
        target_id = int(target_id)
        with self.aim_lock:
            changing = self.aim_target_id is not None and target_id != self.aim_target_id
            if changing:
                if self.aim_proposal_id == target_id:
                    self.aim_proposal_count += 1
                else:
                    self.aim_proposal_id = target_id
                    self.aim_proposal_count = 1
                stale = now - self.aim_updated_at > .08
                if self.aim_proposal_count < 2 and not stale:
                    return
            if self.aim_target_id == target_id and self.aim_target is not None:
                self.aim_target = (self.aim_target * (1.0 - self.aim_ema) +
                                   target * self.aim_ema)
                self.aim_updated_at = now
                self.aim_proposal_id = None
                self.aim_proposal_count = 0
                return
            distance = float(np.linalg.norm(target - current))
            duration = float(np.clip(distance / self.aim_speed,
                                     self.aim_min_s, self.aim_max_s))
            if (tth_ms is not None and np.isfinite(tth_ms) and
                    tth_ms > self.aim_early_ms):
                duration = min(duration, max(
                    .012, (tth_ms - self.aim_early_ms) / 1000.0))
            self.aim_target_id = target_id
            self.aim_target = target
            self.aim_start = np.asarray(current, dtype=float).copy()
            self.aim_started_at = now
            self.aim_duration = duration
            self.aim_updated_at = now
            self.aim_proposal_id = None
            self.aim_proposal_count = 0

    def _capture_loop(self):
        try:
            cfg = load_cfg(self.config_path)
            cfg["source"] = self.source
            self.manager = CaptureManager(cfg)
            if not self.manager.open():
                raise RuntimeError("DXGI/OBS capture did not open")
            target_fps = float(cfg.get("recording", {}).get("fps", 60.0))
            self.capture_started = time.perf_counter()
            self.ready.set()
            self.messages.put(("capture_ready", {"source": self.manager.mode}))
            while not self.stop_event.is_set():
                frame, _ = self.manager.read()
                if frame is None:
                    time.sleep(.002)
                    continue
                roi = self.manager.get_playfield(frame)
                now = time.perf_counter()
                self.capture_count += 1
                with self.frame_lock:
                    self.latest_roi = roi
                    self.latest_seq += 1
                # In record-only mode the GUI previews the raw playfield; no
                # detector is loaded merely to draw boxes over it.
                if not self.enable_inference and now - self.last_preview_at >= 1 / 30:
                    self.last_preview_at = now
                    # dxcam may reuse its backing buffer on the next grab;
                    # encode an owned image for Tk rather than a moving view.
                    self._publish_frame(roi.copy())
                if not self.enable_inference and now - self.last_stats >= .5:
                    self.last_stats = now
                    elapsed = max(.001, now - self.capture_started)
                    self.messages.put(("pipeline_stats", {
                        "capture_fps": self.capture_count / elapsed,
                        "inference_fps": 0.0, "detect_ms": 0.0, "policy_ms": 0.0,
                        "detections": 0, "tracks": 0, "ring_tth_ms": None,
                        "recording": self.recording, "record_frames": len(self.stamps),
                        "record_elapsed": now - self.record_started if self.recording else 0.0,
                    }))
                time.sleep(.001)
        except Exception as exc:
            self.messages.put(("pipeline_error", str(exc)))
            self.ready.set()
        finally:
            if self.manager is not None:
                self.manager.close()

    def _record_loop(self):
        """Encode at fixed rates without blocking capture or inference."""
        cfg = load_cfg(self.config_path)
        fps = float(cfg.get("recording", {}).get("fps", 60.0))
        codec = cfg.get("recording", {}).get("video_codec", "mp4v")
        record_width = int(cfg.get("preprocess", {}).get("infer_width", 640))
        record_height = int(cfg.get("preprocess", {}).get("infer_height", 360))
        self.record_size = [record_width, record_height]
        raw_interval = 1.0 / max(1.0, fps)
        detect_interval = 1.0 / max(1.0, fps)
        next_raw = time.perf_counter()
        next_detect = next_raw
        try:
            while not self.record_stop.is_set() and not self.stop_event.is_set():
                now = time.perf_counter()
                if now < next_raw:
                    time.sleep(min(.004, next_raw - now))
                    continue
                with self.frame_lock:
                    roi = None if self.latest_roi is None else self.latest_roi.copy()
                    annotated = (None if self.latest_annotated is None
                                 else self.latest_annotated.copy())
                if roi is None:
                    time.sleep(.002)
                    continue
                if roi.shape[1] != record_width or roi.shape[0] != record_height:
                    roi = cv2.resize(roi, (record_width, record_height),
                                     interpolation=cv2.INTER_AREA)
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi = cv2.cvtColor(gray_roi, cv2.COLOR_GRAY2BGR)
                with self.record_lock:
                    if not self.recording:
                        break
                    if self.raw_writer is None:
                        height, width = roi.shape[:2]
                        self.raw_writer = cv2.VideoWriter(
                            os.path.join(self.session_dir, "video.mp4"),
                            cv2.VideoWriter_fourcc(*codec), fps, (width, height))
                    self.raw_writer.write(roi)
                    self.stamps.append((now - self.record_started) * 1000.0)
                    if now >= next_detect:
                        if annotated is not None and self.detect_writer is None:
                            height, width = annotated.shape[:2]
                            self.detect_writer = cv2.VideoWriter(
                                os.path.join(self.session_dir, "detected.mp4"),
                                cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                        if self.detect_writer is not None:
                            self.detect_writer.write(annotated)
                            self.detected_frames += 1
                        try:
                            self.vision_queue.put_nowait(
                                (roi.copy(), (now - self.record_started) * 1000.0))
                        except queue.Full:
                            self.vision_dropped += 1
                        next_detect += detect_interval
                next_raw += raw_interval
                if next_raw < now - raw_interval:
                    next_raw = now + raw_interval
                if next_detect < now - detect_interval:
                    next_detect = now + detect_interval
        except Exception as exc:
            self.messages.put(("pipeline_error", f"record writer: {exc}"))

    def _vision_writer_loop(self):
        vision_dir = os.path.join(self.session_dir, "vision_frames")
        os.makedirs(vision_dir, exist_ok=True)
        previous_gray = None
        try:
            while True:
                item = self.vision_queue.get()
                if item is None:
                    break
                frame, stamp = item
                packed, previous_gray = temporal_frame(frame, previous_gray)
                index = self.vision_frames
                path = os.path.join(vision_dir, f"{index:06d}.png")
                if not cv2.imwrite(path, packed, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                    raise RuntimeError(f"failed to write {path}")
                self.vision_stamps.append(stamp)
                self.vision_frames += 1
        except Exception as exc:
            self.messages.put(("pipeline_error", f"temporal writer: {exc}"))

    def _infer_loop(self):
        if not self.ready.wait(10.0) or self.stop_event.is_set():
            return
        try:
            cfg = load_cfg(self.config_path)
            weights = cfg.get("detector", {}).get("weights")
            if weights and not os.path.isabs(weights):
                cfg["detector"]["weights"] = os.path.abspath(
                    os.path.join(os.path.dirname(os.path.dirname(self.config_path)), weights))
            self.detector = load_detector(cfg)
            self.detector.warmup(3)
            self.detector.reset_sequence()
            self.inference_ready.set()
            self.messages.put(("pipeline_ready", {"source": self.manager.mode}))
            tracker = Tracker()
            clock = ApproachClock()
            box_scale = float(cfg.get("yolo", {}).get("box_scale", 1.15))
            detector_cfg = cfg.get("detector", {})
            input_mode = str(detector_cfg.get("input_mode", "gray_bgr"))
            use_hybrid = bool(detector_cfg.get(
                "hybrid_fallback", input_mode != "temporal3"))
            target_infer_fps = max(1.0, float(detector_cfg.get("realtime_fps", 30.0)))
            infer_interval = 1.0 / target_infer_fps
            last_seq = -1
            last = time.perf_counter()
            next_infer = last
            self.infer_started = last
            while not self.stop_event.is_set():
                loop_now = time.perf_counter()
                if loop_now < next_infer:
                    time.sleep(min(.002, next_infer - loop_now))
                    continue
                with self.frame_lock:
                    seq = self.latest_seq
                    roi = self.latest_roi
                if roi is None or seq == last_seq:
                    time.sleep(.001)
                    continue
                last_seq = seq
                next_infer += infer_interval
                if loop_now - next_infer > infer_interval:
                    next_infer = loop_now
                tiny = cv2.resize(roi, (int(cfg.get("preprocess", {}).get("infer_width", 640)),
                                        int(cfg.get("preprocess", {}).get("infer_height", 360))),
                                  interpolation=cv2.INTER_LINEAR)
                now = time.perf_counter()
                detect_started = now
                detections = self.detector.predict(tiny)
                if use_hybrid:
                    detections = augment_detections(tiny, detections)
                annotate_approach(tiny, detections, box_scale)
                tracks = tracker.update(detections, max(.001, now - last), now)
                detect_ms = (time.perf_counter() - detect_started) * 1000.0
                last = now
                self.infer_count += 1
                if detections:
                    self.last_detection_at = now
                ring_hit = clock.update(tracks, now)
                timing_hit = ring_hit
                timing_source = "ring" if ring_hit is not None else None
                if timing_hit is None:
                    candidates = [track for track in tracks
                                  if track.cls in ("circle", "slider_head")
                                  and track.missed == 0 and track.age >= 2]
                    if candidates:
                        target = max(candidates, key=lambda track: track.urgency)
                        preempt_ms = float(cfg.get("beatmaps", {}).get("preempt_ms", 1200.0))
                        fallback_tth = (1.0 - target.urgency) * preempt_ms
                        if -25.0 <= fallback_tth <= 500.0:
                            timing_hit = (fallback_tth, target.tid, target.x, target.y)
                            timing_source = "track fallback"
                shown = self._draw(tiny, detections, timing_hit, timing_source)
                policy_ms = 0.0
                policy_tth = None
                with self.control_lock:
                    if self.policy is not None:
                        dt = max(.001, now - (self.last_policy_at or now - 1 / 60))
                        if self.control_mode == "live":
                            sx, sy = self.controller.position()
                            field = cfg["playfield"]
                            current_cursor = np.asarray([
                                (sx - field["left"]) / (field["right"] - field["left"]),
                                (sy - field["top"]) / (field["bottom"] - field["top"]),
                            ], dtype=float)
                            current_cursor = np.clip(current_cursor, 0.0, 1.0)
                        else:
                            current_cursor = self.cursor.copy()
                        velocity = (current_cursor - self.previous_cursor) / dt
                        self.previous_cursor = current_cursor.copy()
                        cursor_state = (current_cursor[0], current_cursor[1],
                                        velocity[0], velocity[1],
                                        bool(self.scheduler and self.scheduler.key_down))
                        if getattr(self.policy, "input_dim", FEATURE_DIM) == FEATURE_DIM_V3:
                            feature = build_features_v3(tracks, cursor_state)
                        else:
                            feature = build_features(tracks, cursor_state)
                        policy_started = time.perf_counter()
                        target_slot = None
                        if getattr(self.policy, "is_v3", False):
                            output, self.hidden, target_slot = self.policy.act_with_target(
                                feature, self.hidden)
                        else:
                            output, self.hidden = self.policy.act(feature, self.hidden)
                        policy_ms = (time.perf_counter() - policy_started) * 1000.0
                        self.last_policy_at = now
                        policy_tth = float(output[2]) * TTH_SCALE_MS
                        aim_target = None
                        aim_target_id = None
                        aim_tth = None
                        if timing_hit is not None:
                            aim_target = np.asarray(timing_hit[2:4], dtype=float)
                            aim_target_id = timing_hit[1]
                            aim_tth = timing_hit[0]
                        elif target_slot is not None and target_slot < len(tracks):
                            selected = tracks[target_slot]
                            if selected.missed <= 2 and selected.cls != "spinner":
                                aim_target = np.asarray([selected.x, selected.y], dtype=float)
                                aim_target_id = selected.tid
                                aim_tth = policy_tth
                        if aim_target is not None:
                            if self.control_mode == "live":
                                self._update_aim_target(
                                    aim_target, aim_target_id, aim_tth,
                                    current_cursor, now)
                                self.cursor = current_cursor.copy()
                            else:
                                delta = aim_target - current_cursor
                                delta_norm = float(np.linalg.norm(delta))
                                if delta_norm > MAX_CURSOR_STEP:
                                    delta *= MAX_CURSOR_STEP / delta_norm
                                self.cursor = np.clip(current_cursor + delta, 0.0, 1.0)
                        else:
                            self.cursor = current_cursor.copy()
                        hold = bool(output[3] >= .5)
                        if self.scheduler is not None:
                            click_targets = [t for t in tracks
                                             if t.cls in ("circle", "slider_head", "spinner")
                                             and t.missed == 0 and t.age >= 2]
                            if timing_hit is not None:
                                # Stable object coordinates, not the moving policy cursor.
                                scheduler_target = (timing_hit[2], timing_hit[3])
                                self.scheduler.update(timing_hit[0], hold,
                                                      scheduler_target, now)
                            elif click_targets:
                                target = max(click_targets, key=lambda t: t.urgency)
                                self.scheduler.update(policy_tth, hold,
                                                      (target.x, target.y), now)
                            else:
                                self.scheduler.invalidate(now)
                        px = int(self.cursor[0] * shown.shape[1])
                        py = int(self.cursor[1] * shown.shape[0])
                        cv2.drawMarker(shown, (px, py), (255, 255, 0),
                                       cv2.MARKER_CROSS, 22, 2)
                if self.control_mode == "live":
                    import ctypes
                    if ctypes.windll.user32.GetAsyncKeyState(0x79) & 0x8000:
                        self.stop_control()
                with self.frame_lock:
                    self.latest_annotated = shown
                self._publish_frame(shown)
                if now - self.last_stats >= .5:
                    self.last_stats = now
                    elapsed = max(.001, now - self.infer_started)
                    record_elapsed = now - self.record_started if self.recording else 0.0
                    self.messages.put(("pipeline_stats", {
                        "capture_fps": self.capture_count / max(.001, now - self.capture_started),
                        "inference_fps": self.infer_count / elapsed,
                        "detect_ms": detect_ms, "policy_ms": policy_ms,
                        "detections": len(detections), "tracks": len(tracks),
                        "ring_tth_ms": timing_hit[0] if timing_hit else None,
                        "timing_source": timing_source,
                        "policy_tth_ms": policy_tth,
                        "recording": self.recording, "record_frames": len(self.stamps),
                        "record_elapsed": record_elapsed,
                        "detected_frames": self.detected_frames,
                        "vision_frames": self.vision_frames,
                        "vision_dropped": self.vision_dropped,
                        "control": self.control_mode,
                        "hits": self.scheduler.hits if self.scheduler else 0,
                        "live_ready": bool(tracks and timing_hit and
                                           self.infer_count / elapsed >= 25.0),
                    }))
        except Exception as exc:
            self.inference_error = f"inference: {exc}"
            self.inference_ready.set()
            self.messages.put(("pipeline_error", self.inference_error))

    def _draw(self, image, detections, timing_hit, timing_source):
        shown = image.copy()
        height, width = shown.shape[:2]
        for det in detections:
            x1 = round((det.x - det.w / 2) * width)
            y1 = round((det.y - det.h / 2) * height)
            x2 = round((det.x + det.w / 2) * width)
            y2 = round((det.y + det.h / 2) * height)
            color = (40, 220, 80) if det.urgency is not None else (0, 180, 255)
            cv2.rectangle(shown, (x1, y1), (x2, y2), color, 2)
            phase = "" if det.urgency is None else f" p={det.urgency:.2f}"
            cv2.putText(shown, f"{det.cls} {det.conf:.2f}{phase}",
                        (x1, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                        .42, color, 1, cv2.LINE_AA)
        timing = (f"{timing_source} {timing_hit[0]:.0f}ms"
                  if timing_hit else "timing unavailable")
        cv2.putText(shown, timing, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    .52, (80, 255, 255), 1, cv2.LINE_AA)
        if self.recording:
            cv2.circle(shown, (width - 18, 18), 8, (0, 0, 255), -1)
            cv2.putText(shown, "REC", (width - 72, 23), cv2.FONT_HERSHEY_SIMPLEX,
                        .52, (0, 0, 255), 2, cv2.LINE_AA)
        return shown

    def _publish_frame(self, shown):
        now = time.perf_counter()
        if now - self.last_publish_at < 1.0 / 30.0:
            return
        self.last_publish_at = now
        ok, encoded = cv2.imencode(".png", shown)
        if not ok:
            return
        payload = base64.b64encode(encoded).decode("ascii")
        try:
            self.frames.put_nowait(payload)
        except queue.Full:
            try:
                self.frames.get_nowait()
            except queue.Empty:
                pass
            self.frames.put_nowait(payload)
