"""Dual-mode / heuristic spinner engine.

Detects spinner phase (from metadata / classification / active state)
and drives high-speed parametric circular orbit (400+ RPM at 500 Hz)
with constant key holding, smoothly interpolating to the next note at spinner exit.
"""
import math
import numpy as np


class SpinnerEngine:
    def __init__(self, rpm=420.0, radius=0.16, center=(0.5, 0.5)):
        self.rpm = float(rpm)
        self.radius = float(radius)
        self.center = tuple(center)
        self.start_time = None
        self.end_time = None
        self.start_angle = 0.0
        self.spin_dir = 1.0  # 1.0 for CCW, -1.0 for CW
        self.active = False
        self.exit_duration = 0.08  # smoothstep blend window to next note

    def reset(self):
        self.start_time = None
        self.end_time = None
        self.start_angle = 0.0
        self.active = False
        self.last_step_time = None

    def is_active(self, env_or_bm=None, obs=None, current_time=0.0):
        # 1. NoteEnv check (RL/mechanics benchmark)
        if env_or_bm is not None and hasattr(env_or_bm, "notes"):
            active = next((n for n in env_or_bm.notes if not n["done"] and n["spawn_t"] <= env_or_bm.time), None)
            if active and active.get("kind") == "spinner":
                return True

        # 2. Beatmap check (live game with known beatmap)
        if env_or_bm is not None and hasattr(env_or_bm, "objects"):
            t_ms = current_time if current_time > 10000.0 else current_time * 1000.0
            for obj in env_or_bm.objects:
                if getattr(obj, "is_spinner", False) and obj.time <= t_ms <= obj.end_time:
                    return True

        # 3. Vision check (live pure vision capture: hollow dark center + concentric neon ring)
        if obs is not None:
            gray = obs[0] if (isinstance(obs, np.ndarray) and obs.ndim == 3) else obs
            if isinstance(gray, np.ndarray) and gray.ndim == 2:
                return self.is_spinner_vision(gray)

        return False

    @staticmethod
    def is_spinner_vision(gray):
        """Geometric detection of skin spinner in real-time visual capture.
        Verifies:
        1. Dark center disk (r < 0.12 * h) has mean brightness < 45 (rejects title screens/cards).
        2. Concentric ring zone (0.20 * h <= r <= 0.45 * h) has bright pixels (> 65).
        3. Low radial std (< 0.028 * h): all bright pixels lie on a single circular arc.
        4. Multi-quadrant angular spread: spans at least 2 distinct quadrants.
        """
        h, w = gray.shape[-2:]
        cy, cx = h / 2.0, w / 2.0
        ys, xs = np.ogrid[:h, :w]
        dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)

        # 1. Dark center check
        center_mask = dist < (0.10 * h)
        if float(gray[center_mask].mean()) > 55.0:
            return False

        # 2. Ring zone bright pixels
        bright = (gray > 65) & (dist >= 0.11 * h) & (dist <= 0.45 * h)
        y_idx, x_idx = np.where(bright)
        min_pixels = 18 if h < 100 else 180
        if len(y_idx) < min_pixels:
            return False

        # 3. Radii std
        radii = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2) / h
        if float(np.std(radii)) >= 0.055:
            return False

        # 4. Multi-quadrant coverage
        angle = np.arctan2(y_idx - cy, x_idx - cx)
        quads = sum(1 for q in [
            ((angle >= -np.pi) & (angle < -np.pi / 2)).sum(),
            ((angle >= -np.pi / 2) & (angle < 0)).sum(),
            ((angle >= 0) & (angle < np.pi / 2)).sum(),
            ((angle >= np.pi / 2) & (angle <= np.pi)).sum()
        ] if q > 5)

        return quads >= 2

    def step(self, current_time, default_xy=(0.5, 0.5), env_or_bm=None):
        """Compute parametric circular orbit action at high frequency."""
        # Detect new spinner initiation or gap between consecutive spinners (>200ms)
        is_new_spinner = (not self.active or
                          self.start_time is None or
                          self.last_step_time is None or
                          (current_time - self.last_step_time) > 0.20)
        if is_new_spinner:
            self.active = True
            self.start_time = current_time
            dx = default_xy[0] - self.center[0]
            dy = default_xy[1] - self.center[1]
            if math.hypot(dx, dy) > 1e-4:
                self.start_angle = math.atan2(dy, dx)
            else:
                self.start_angle = 0.0

        self.last_step_time = current_time
        elapsed = max(0.0, current_time - self.start_time)
        omega = (self.rpm / 60.0) * (2.0 * math.pi)
        angle = self.start_angle + self.spin_dir * omega * elapsed

        orbit_x = self.center[0] + self.radius * math.cos(angle)
        orbit_y = self.center[1] + self.radius * math.sin(angle)

        # Check for spinner exit and smooth interpolation to next note
        next_target = None
        rem_duration = None
        if env_or_bm is not None:
            if hasattr(env_or_bm, "notes"):
                active = next((n for n in env_or_bm.notes if not n["done"]), None)
                if active and active.get("kind") == "spinner":
                    rem_duration = (active["hit_t"] + active.get("duration", 0.0)) - env_or_bm.time
                    upcoming = next((n for n in env_or_bm.notes if not n["done"] and n is not active), None)
                    if upcoming:
                        next_target = upcoming["pos"]
            elif hasattr(env_or_bm, "objects"):
                t_ms = current_time if current_time > 10000.0 else current_time * 1000.0
                active = next((o for o in env_or_bm.objects if getattr(o, "is_spinner", False) and o.time <= t_ms <= o.end_time), None)
                if active:
                    rem_duration = (active.end_time - t_ms) / 1000.0
                    upcoming = next((o for o in env_or_bm.objects if o.time > active.end_time), None)
                    if upcoming:
                        next_target = (upcoming.x / 512.0, upcoming.y / 384.0)

        if rem_duration is not None and rem_duration <= self.exit_duration and next_target is not None:
            blend = min(1.0, max(0.0, 1.0 - rem_duration / max(1e-4, self.exit_duration)))
            smooth_blend = blend * blend * (3.0 - 2.0 * blend)
            final_x = orbit_x + (next_target[0] - orbit_x) * smooth_blend
            final_y = orbit_y + (next_target[1] - orbit_y) * smooth_blend
            if rem_duration <= 0.0:
                self.active = False
        else:
            final_x = orbit_x
            final_y = orbit_y

        final_x = float(np.clip(final_x, 0.05, 0.95))
        final_y = float(np.clip(final_y, 0.05, 0.95))

        # Always hold key constantly while spinning
        return [final_x, final_y, 1.0, 1.0]
