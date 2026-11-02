"""High-frequency key scheduler driven by continuous time-to-hit estimates."""
import math
import threading
import time


class HitScheduler:
    def __init__(self, controller, latency_ms=18.0, tap_ms=32.0,
                 lookahead_ms=180.0, tick_hz=500.0, min_stream_interval_ms=36.0):
        self.controller = controller
        self.latency_ms = latency_ms
        self.tap_ms = tap_ms
        self.lookahead_ms = lookahead_ms
        self.min_stream_interval = min_stream_interval_ms / 1000.0
        self.period = 1.0 / tick_hz
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._deadline = None
        self._active_keys = {}
        self._alternate = False
        self._fired = False
        self._last_tth = None
        self._min_tth_since_fire = 999.0
        self._last_target = None
        self._last_fire_time = None
        self._hold = False
        self.hits = 0
        self._thread = None

    @property
    def key_down(self):
        with self._lock:
            return bool(self._active_keys)

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def _trigger_key_locked(self, now, hold, tth=0.0):
        self._alternate = not self._alternate
        key = "x" if self._alternate else "z"
        other = "z" if key == "x" else "x"
        if key in self._active_keys and other not in self._active_keys:
            key = other
        elif key in self._active_keys:
            self.controller.set_key(key, False)
        self.controller.set_key(key, True)
        self._active_keys[key] = None if hold else now + self.tap_ms / 1000.0
        self._fired = True
        self._last_fire_time = now
        self._min_tth_since_fire = tth
        self._deadline = None
        self.hits += 1

    def update(self, time_to_hit_ms, hold, target, now=None, click_prob=0.0):
        now = time.perf_counter() if now is None else now
        tth = float(time_to_hit_ms)
        eff_tth = tth - self.latency_ms
        with self._lock:
            changed_target = (self._last_target is not None and
                              math.dist(target, self._last_target) > 0.020)
            jumped_to_next = (self._last_tth is not None and
                              tth - self._last_tth > 16.0)
            time_since_fire = (now - self._last_fire_time) if self._last_fire_time is not None else 1.0
            rising_edge = (click_prob >= 0.40 and
                           click_prob > getattr(self, "_prev_click_prob", 0.0) + 0.10)
            if self._fired:
                self._min_tth_since_fire = min(self._min_tth_since_fire, tth)
                tth_rebounded = (tth - self._min_tth_since_fire >= 14.0)
                if time_since_fire >= self.min_stream_interval:
                    if (tth > 30.0 or tth_rebounded or changed_target or jumped_to_next):
                        self._fired = False
                        self._deadline = None
                    elif click_prob < 0.25:
                        self._fired = False
                        self._deadline = None
                    elif rising_edge and time_since_fire >= self.min_stream_interval:
                        self._fired = False
                        self._deadline = None
            self._prev_click_prob = click_prob
            self._hold = bool(hold)
            if not self._hold:
                for key, release_at in list(self._active_keys.items()):
                    if release_at is None:
                        self._active_keys[key] = now
            in_hit_window = (-20.0 <= eff_tth <= 14.0)
            should_fire = (
                (in_hit_window and click_prob >= 0.30) or
                (-12.0 <= eff_tth <= 8.0 and click_prob >= 0.20) or
                (rising_edge and -30.0 <= eff_tth <= 25.0)
            )
            if should_fire and not self._fired:
                self._trigger_key_locked(now, self._hold, tth=tth)
            elif not self._fired and 0.0 <= eff_tth <= self.lookahead_ms:
                self._deadline = now + eff_tth / 1000.0
            self._last_tth = tth
            self._last_target = tuple(target)

    def invalidate(self, now=None):
        """Cancel stale timing when perception has no click-capable target."""
        now = time.perf_counter() if now is None else now
        with self._lock:
            self._deadline = None
            self._fired = False
            self._last_tth = None
            self._min_tth_since_fire = 999.0
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
        if self._thread is not None:
            self._thread.join(timeout=0.2)
        self.controller.release_all()
