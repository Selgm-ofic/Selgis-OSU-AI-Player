from __future__ import annotations

import math, queue, random, threading, time
from datetime import datetime
from collections import deque
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli

from .model import TinyOsuPolicy, UNetTinyOsuPolicy, target_heatmap

ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = Path(r"C:\Selgis OSU AI Player")


class SkinRenderer:
    def __init__(self):
        root = ROOT / "data" / "skin_voidveil"
        self.assets = {}
        self._cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
        for name in (
            "hitcircle.png", "hitcircleoverlay.png", "approachcircle.png",
            "sliderb0.png", "sliderendcircle.png", "sliderfollowcircle.png",
            "spinner-background.png", "spinner-circle.png",
            "spinner-approachcircle.png", "spinner-top.png", "cursor.png",
        ):
            image = cv2.imread(str(root / name), cv2.IMREAD_UNCHANGED)
            if image is not None:
                self.assets[name] = image

    def _get_cached(self, name: str, size: int):
        key = (name, size)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        sprite = self.assets.get(name)
        if sprite is None:
            return None, None
        resized = cv2.resize(sprite, (size, size), interpolation=cv2.INTER_AREA)
        if resized.ndim == 3 and resized.shape[2] == 4:
            rgb = resized[:, :, :3]
            alpha = resized[:, :, 3].astype(np.float32) / 255.0
        elif resized.ndim == 3:
            rgb = resized
            alpha = np.ones(resized.shape[:2], np.float32)
        else:
            rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            alpha = np.ones(resized.shape[:2], np.float32)
        self._cache[key] = (rgb, alpha)
        return rgb, alpha

    def paste(self, canvas, name, xy, size, alpha_scale=1.0):
        rgb, alpha = self._get_cached(name, size)
        if rgb is None:
            return
        x, y = int(xy[0] - size / 2), int(xy[1] - size / 2)
        h, w = canvas.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + size), min(h, y + size)
        if x0 >= x1 or y0 >= y1:
            return
        src_rgb = rgb[y0 - y:y1 - y, x0 - x:x1 - x]
        a = alpha[y0 - y:y1 - y, x0 - x:x1 - x] * alpha_scale
        dst = canvas[y0:y1, x0:x1]
        canvas[y0:y1, x0:x1] = (src_rgb * a[:, :, None] + dst * (1.0 - a[:, :, None])).astype(np.uint8)


class NoteEnv:
    W, H = 512.0, 384.0
    RW, RH = 96, 72
    PREEMPT = 0.45

    def __init__(self, seed=7, pattern=None, kind=None):
        self.rng = random.Random(seed)
        self.pattern_override = kind if kind is not None else pattern
        self.skin = SkinRenderer()
        self._canvas_buf = np.zeros((self.RH, self.RW, 3), dtype=np.uint8)
        self._prev_gray = np.zeros((self.RH, self.RW), dtype=np.uint8)
        self._out_buf = np.zeros((2, self.RH, self.RW), dtype=np.float32)
        self.reset()

    def reset(self):
        self.cursor = np.array([0.5, 0.5], np.float32)
        self.notes = []
        self.time = 0.0
        self.combo = 0
        self.max_combo = 0
        self.score = 0.0
        self.n_300 = 0
        self.n_100 = 0
        self.n_50 = 0
        self.n_miss = 0
        self.recent_judgments = deque(maxlen=100)
        self._prev_gray.fill(0)
        self._refill_queue()
        return self.obs()

    def _refill_queue(self):
        while len(self.notes) < 6:
            self._spawn_pattern()

    def _spawn_pattern(self):
        base_t = self.notes[-1]["hit_t"] if self.notes else self.time
        choice = self.pattern_override if self.pattern_override is not None else self.rng.choice(("jumps", "stream", "slider", "spinner"))
        if choice == "stream":
            start = np.array([self.rng.uniform(0.25, 0.75), self.rng.uniform(0.25, 0.75)], np.float32)
            angle = self.rng.uniform(0, 2 * math.pi)
            curve = self.rng.uniform(-0.25, 0.25)
            length = self.rng.randint(4, 6)
            t = max(base_t + 0.35, self.time + 0.35)
            r = 0.08
            for i in range(length):
                pos = start + np.array([math.cos(angle + i * curve), math.sin(angle + i * curve)], np.float32) * (0.045 * (i + 1))
                pos = np.clip(pos, 0.15, 0.85)
                self.notes.append({
                    "kind": "circle", "pos": pos, "spawn_t": t - self.PREEMPT, "hit_t": t,
                    "radius": r, "done": False, "pattern": "stream"
                })
                t += 0.14
        elif choice == "jumps":
            count = self.rng.randint(3, 5)
            t = max(base_t + 0.40, self.time + 0.40)
            for _ in range(count):
                pos = np.array([self.rng.uniform(0.20, 0.80), self.rng.uniform(0.20, 0.80)], np.float32)
                r = self.rng.uniform(0.075, 0.095)
                self.notes.append({
                    "kind": "circle", "pos": pos, "spawn_t": t - self.PREEMPT, "hit_t": t,
                    "radius": r, "done": False, "pattern": "jump"
                })
                t += 0.38
        elif choice == "slider":
            pos = np.array([self.rng.uniform(0.22, 0.78), self.rng.uniform(0.22, 0.78)], np.float32)
            end = np.clip(pos + np.array([self.rng.uniform(-0.25, 0.25), self.rng.uniform(-0.20, 0.20)]), 0.15, 0.85)
            t = max(base_t + 0.45, self.time + 0.45)
            dur = self.rng.uniform(0.40, 0.65)
            r = 0.085
            self.notes.append({
                "kind": "slider", "pos": pos, "end": end, "spawn_t": t - self.PREEMPT,
                "hit_t": t, "duration": dur, "radius": r, "progress": 0.0,
                "done": False, "pattern": "slider"
            })
        else:
            t = max(base_t + 0.20, self.time + 0.20)
            dur = 0.75
            self.notes.append({
                "kind": "spinner", "pos": np.array([0.5, 0.5], np.float32),
                "spawn_t": t, "hit_t": t, "duration": dur, "radius": 0.20,
                "revolutions": 0.0, "done": False, "pattern": "spinner"
            })

    @property
    def circle_radius(self):
        active = next((n for n in self.notes if not n["done"]), None)
        return active["radius"] if active else 0.08

    @property
    def kind(self):
        active = next((n for n in self.notes if not n["done"]), None)
        return active["kind"] if active else "circle"

    def current_target(self):
        active = next((n for n in self.notes if not n["done"] and n["spawn_t"] <= self.time), None)
        if active is None:
            upcoming = next((n for n in self.notes if not n["done"]), None)
            if upcoming is not None:
                return upcoming["pos"].copy()
            return np.array([0.5, 0.5], np.float32)
        if active["kind"] == "circle":
            return active["pos"].copy()
        elif active["kind"] == "slider":
            if self.time < active["hit_t"]:
                return active["pos"].copy()
            prog = min(1.0, (self.time - active["hit_t"]) / max(0.01, active["duration"]))
            return (active["pos"] + (active["end"] - active["pos"]) * prog).astype(np.float32)
        else:
            if self.time < active["hit_t"]:
                return np.array([0.5, 0.5], np.float32)
            angle = 14.0 * (self.time - active["hit_t"])
            return np.array([0.5 + 0.16 * math.cos(angle), 0.5 + 0.16 * math.sin(angle)], np.float32)

    def obs(self):
        t_center = self.current_target()
        active = next((n for n in self.notes if not n["done"]), None)
        kind_val = 0.0
        rem_time = 0.0
        if active:
            kind_val = {"circle": 0.0, "slider": 0.5, "spinner": 1.0}[active["kind"]]
            rem_time = (active["hit_t"] - self.time) / 0.4
        return np.array([*self.cursor, *t_center, kind_val, rem_time, self.combo / 50.0], np.float32)

    def fast_obs(self):
        canvas = self._canvas_buf
        canvas.fill(0)
        visible = [n for n in self.notes if (n["spawn_t"] <= self.time <= n["hit_t"] + n.get("duration", 0.0) + 0.10) and not n["done"]]
        visible.sort(key=lambda n: n["hit_t"], reverse=True)

        for n in visible:
            p = (int(n["pos"][0] * (self.RW - 1)), int(n["pos"][1] * (self.RH - 1)))
            r_px = max(6, int(n["radius"] * self.RW))
            if n["kind"] == "circle":
                approach = max(0.0, (n["hit_t"] - self.time) / self.PREEMPT)
                app_size = max(r_px, int(r_px * (1.0 + 1.8 * approach)))
                self.skin.paste(canvas, "approachcircle.png", p, app_size, 0.85)
                self.skin.paste(canvas, "hitcircle.png", p, r_px)
                self.skin.paste(canvas, "hitcircleoverlay.png", p, r_px)
            elif n["kind"] == "slider":
                end = (int(n["end"][0] * (self.RW - 1)), int(n["end"][1] * (self.RH - 1)))
                cv2.line(canvas, p, end, (140, 140, 140), max(3, r_px))
                self.skin.paste(canvas, "sliderendcircle.png", end, r_px)
                if self.time < n["hit_t"]:
                    approach = max(0.0, (n["hit_t"] - self.time) / self.PREEMPT)
                    app_size = max(r_px, int(r_px * (1.0 + 1.8 * approach)))
                    self.skin.paste(canvas, "approachcircle.png", p, app_size, 0.85)
                self.skin.paste(canvas, "hitcircle.png", p, r_px)
                self.skin.paste(canvas, "hitcircleoverlay.png", p, r_px)
                if self.time >= n["hit_t"]:
                    prog = min(1.0, (self.time - n["hit_t"]) / max(0.01, n["duration"]))
                    bx = int(p[0] + (end[0] - p[0]) * prog)
                    by = int(p[1] + (end[1] - p[1]) * prog)
                    self.skin.paste(canvas, "sliderb0.png", (bx, by), r_px - 1)
            elif n["kind"] == "spinner":
                approach = max(0.0, (n["hit_t"] + n["duration"] - self.time) / max(0.01, n["duration"]))
                self.skin.paste(canvas, "spinner-circle.png", (48, 36), 38)
                self.skin.paste(canvas, "spinner-approachcircle.png", (48, 36), int(38 + 26 * approach))

        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        diff = cv2.convertScaleAbs(cv2.absdiff(gray, self._prev_gray), alpha=3.0)
        self._prev_gray = gray.copy()
        self._out_buf[0] = gray.astype(np.float32) / 255.0
        self._out_buf[1] = diff.astype(np.float32) / 255.0
        return self._out_buf.copy()

    def render(self):
        img = self._canvas_buf.copy()
        cx = int(self.cursor[0] * (self.RW - 1))
        cy = int(self.cursor[1] * (self.RH - 1))
        self.skin.paste(img, "cursor.png", (cx, cy), 14)
        return img

    def step(self, action):
        dt = 1.0 / 60.0
        self.time += dt
        prev_cursor = self.cursor.copy()
        target_xy = np.clip(np.asarray(action[:2], np.float32), 0.0, 1.0)
        self.cursor = np.clip(self.cursor + (target_xy - self.cursor) * 0.85, 0.0, 1.0)
        click = bool(action[2] > 0.5)
        hold = bool(action[3] > 0.5) if len(action) >= 4 else click

        t_center = self.current_target()
        prev_dist = float(np.linalg.norm(prev_cursor - t_center))
        curr_dist = float(np.linalg.norm(self.cursor - t_center))

        reward = 3.0 * (prev_dist - curr_dist)

        active_target = next((n for n in self.notes if not n["done"] and (n["spawn_t"] <= self.time)), None)
        if active_target is not None:
            r_trg = active_target.get("radius", 0.08)
            d_trg = float(np.linalg.norm(self.cursor - active_target["pos"]))
            if d_trg > r_trg:
                reward -= 0.15 * min(2.5, d_trg / r_trg)

        if click:
            on_any = any(float(np.linalg.norm(self.cursor - n["pos"])) <= n["radius"] * 1.3 for n in self.notes if not n["done"])
            if not on_any:
                reward -= 2.5
                self.score = max(0.0, self.score - 100.0)

        for active in [n for n in self.notes if not n["done"]]:
            r_hit = active["radius"]
            dist = float(np.linalg.norm(self.cursor - active["pos"]))
            if active["kind"] == "circle":
                hit_by_click = click and (active["hit_t"] - 0.12 <= self.time <= active["hit_t"] + 0.12)
                if dist <= r_hit:
                    if hit_by_click:
                        active["done"] = True
                        timing_err = abs(self.time - active["hit_t"])
                        center_ratio = dist / r_hit
                        center_bonus = max(0.0, 1.0 - center_ratio)
                        if timing_err <= 0.055 and center_ratio <= 0.85:
                            reward += 14.0 + 8.0 * center_bonus
                            self.n_300 += 1
                            self.combo += 1
                            self.score += 300.0 * (1.0 + self.combo * 0.05)
                            self.recent_judgments.append(300)
                        elif timing_err <= 0.095:
                            reward += 5.0 + 4.0 * center_bonus
                            self.n_100 += 1
                            self.combo += 1
                            self.score += 100.0 * (1.0 + self.combo * 0.02)
                            self.recent_judgments.append(100)
                        else:
                            reward += 1.0 + 1.0 * center_bonus
                            self.n_50 += 1
                            self.combo += 1
                            self.score += 50.0
                            self.recent_judgments.append(50)
                        self.max_combo = max(self.max_combo, self.combo)
                        break
                elif self.time > active["hit_t"] + 0.12:
                    active["done"] = True
                    reward -= 30.0
                    self.n_miss += 1
                    self.combo = 0
                    self.score = max(0.0, self.score * 0.60 - 2000.0)
                    self.recent_judgments.append(0)
                    break
            elif active["kind"] == "slider":
                if self.time >= active["hit_t"]:
                    prog = min(1.0, (self.time - active["hit_t"]) / max(0.01, active["duration"]))
                    ball_pos = active["pos"] + (active["end"] - active["pos"]) * prog
                    b_dist = float(np.linalg.norm(self.cursor - ball_pos))
                    is_tracking = (b_dist <= r_hit * 1.5) and (hold or click)
                    if is_tracking:
                        active["progress"] += dt
                        reward += 0.30 * (1.0 - b_dist / (r_hit * 1.5))
                    else:
                        reward -= 0.35
                    if self.time >= active["hit_t"] + active["duration"]:
                        active["done"] = True
                        good = active["progress"] >= active["duration"] * 0.70
                        if good:
                            reward += 14.0
                            self.n_300 += 1
                            self.combo += 1
                            self.score += 300.0 * (1.0 + self.combo * 0.05)
                            self.max_combo = max(self.max_combo, self.combo)
                            self.recent_judgments.append(300)
                        else:
                            reward -= 25.0
                            self.n_miss += 1
                            self.combo = 0
                            self.score = max(0.0, self.score * 0.60 - 1500.0)
                            self.recent_judgments.append(0)
                        break
            elif active["kind"] == "spinner":
                prev_ang = math.atan2(prev_cursor[1] - 0.5, prev_cursor[0] - 0.5)
                curr_ang = math.atan2(self.cursor[1] - 0.5, self.cursor[0] - 0.5)
                signed_da = (curr_ang - prev_ang + math.pi) % (2 * math.pi) - math.pi
                dist_c = float(np.linalg.norm(self.cursor - np.array([0.5, 0.5])))
                is_spinning = (0.07 <= dist_c <= 0.38) and (abs(signed_da) > 0.03) and (hold or click)
                if is_spinning:
                    if "spin_dir" not in active:
                        active["spin_dir"] = 1 if signed_da > 0 else -1
                        active["net_angle"] = 0.0
                    elif active["spin_dir"] == 0 and abs(signed_da) > 0.03:
                        active["spin_dir"] = 1 if signed_da > 0 else -1
                    progress = signed_da * active.get("spin_dir", 1)
                    if progress > 0:
                        active["net_angle"] = active.get("net_angle", 0.0) + progress
                        active["revolutions"] = active["net_angle"] / (2 * math.pi)
                        reward += progress * 2.5
                    else:
                        active["net_angle"] = max(0.0, active.get("net_angle", 0.0) + progress * 0.5)
                        active["revolutions"] = active["net_angle"] / (2 * math.pi)
                        reward -= 0.10
                else:
                    reward -= 0.20
                if self.time >= active["hit_t"] + active["duration"]:
                    active["done"] = True
                    good = active["revolutions"] >= 1.0
                    if good:
                        reward += 16.0
                        self.n_300 += 1
                        self.combo += 1
                        self.score += 500.0 * (1.0 + self.combo * 0.05)
                        self.max_combo = max(self.max_combo, self.combo)
                        self.recent_judgments.append(300)
                    else:
                        reward -= 25.0
                        self.n_miss += 1
                        self.combo = 0
                        self.score = max(0.0, self.score * 0.60 - 1500.0)
                        self.recent_judgments.append(0)
                    break

        self.notes = [n for n in self.notes if not n["done"] or (self.time <= n["hit_t"] + n.get("duration", 0.0) + 0.15)]
        self._refill_queue()

        if len(self.recent_judgments) > 0:
            acc = sum(self.recent_judgments) / (300.0 * len(self.recent_judgments)) * 100.0
        else:
            acc = 100.0

        active_now = next((n for n in self.notes if not n["done"] and n["spawn_t"] <= self.time), None)
        if active_now is None:
            active_now = next((n for n in self.notes if not n["done"]), None)
        info = {
            "hits": self.n_300 + self.n_100 + self.n_50,
            "misses": self.n_miss,
            "300": self.n_300, "100": self.n_100, "50": self.n_50,
            "combo": self.combo, "max_combo": self.max_combo,
            "score": self.score, "accuracy": acc,
            "pattern": active_now["pattern"] if active_now else "circle",
            "kind": active_now["kind"] if active_now else "circle",
            "dist": curr_dist
        }
        return self.obs(), reward, False, info


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 96), nn.Tanh(),
            nn.Linear(96, 96), nn.Tanh(),
        )
        self.mu = nn.Linear(96, 2)
        self.click = nn.Linear(96, 1)
        self.hold = nn.Linear(96, 1)
        self.value = nn.Linear(96, 1)
        self.logstd = nn.Parameter(torch.full((2,), -3.5))

        nn.init.orthogonal_(self.mu.weight, gain=0.01)
        nn.init.constant_(self.mu.bias, 0.5)
        nn.init.orthogonal_(self.click.weight, gain=0.01)
        nn.init.constant_(self.click.bias, -0.5)
        nn.init.orthogonal_(self.hold.weight, gain=0.01)
        nn.init.constant_(self.hold.bias, -0.5)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.constant_(self.value.bias, 0.0)

    def forward(self, x):
        h = self.net(x)
        xy = torch.sigmoid(self.mu(h))
        return {
            "xy": xy,
            "click_logit": self.click(h).squeeze(-1),
            "hold_logit": self.hold(h).squeeze(-1),
            "value": self.value(h)
        }


class TinyOsuActorCritic(nn.Module):
    def __init__(self, spatial_channels=80, hidden=160, heatmap_hw=(18, 24), use_unet=False):
        super().__init__()
        self.spatial_channels = spatial_channels
        self.hidden = hidden
        if use_unet:
            self.policy = UNetTinyOsuPolicy(hidden=hidden, spatial_channels=spatial_channels, heatmap_hw=heatmap_hw)
        else:
            self.policy = TinyOsuPolicy(hidden=hidden, spatial_channels=spatial_channels, heatmap_hw=heatmap_hw)
        self.value = nn.Linear(hidden, 1)
        self.logstd = nn.Parameter(torch.full((2,), -3.8))
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.constant_(self.value.bias, 0.0)

    def forward(self, x, state=None):
        out = self.policy(x, state=state)
        out["value"] = self.value(out["last"])
        return out


def load_model_from_checkpoint(path_or_kind: str, device: torch.device):
    if path_or_kind == "new_unet_vision_gru":
        return TinyOsuActorCritic(spatial_channels=80, hidden=160, use_unet=True).to(device), "vision_gru", "Новая U-Net CNN-GRU (GroupNorm, Skip, 80c, 160h)"
    if path_or_kind == "new_vision_gru":
        return TinyOsuActorCritic(spatial_channels=80, hidden=160, use_unet=False).to(device), "vision_gru", "Новая CNN-GRU-Heatmap (80c, 160h)"
    if path_or_kind == "new_mlp_state":
        return ActorCritic().to(device), "mlp_state", "Новый MLP (state)"

    p = Path(path_or_kind)
    if not p.exists():
        raise FileNotFoundError(f"Файл модели не найден: {p}")

    ck = torch.load(str(p), map_location=device, weights_only=False)
    state = ck["state"] if (isinstance(ck, dict) and "state" in ck) else ck

    if any(k.startswith("vision") for k in state.keys()):
        is_unet = (isinstance(ck, dict) and ck.get("arch") == "unet") or any(k.startswith("vision.enc") for k in state.keys())
        spatial = 80 if is_unet else 64
        hidden = 160 if is_unet else 128
        if isinstance(ck, dict) and "spatial_channels" in ck:
            spatial = int(ck["spatial_channels"])
        elif "vision.6.weight" in state:
            spatial = int(state["vision.6.weight"].shape[0])
        elif "vision.4.weight" in state:
            spatial = int(state["vision.4.weight"].shape[0])

        if isinstance(ck, dict) and "hidden" in ck:
            hidden = int(ck["hidden"])
        elif "gru.weight_ih_l0" in state:
            hidden = int(state["gru.weight_ih_l0"].shape[0]) // 3

        model = TinyOsuActorCritic(spatial_channels=spatial, hidden=hidden, use_unet=is_unet).to(device)
        missing, unexpected = model.policy.load_state_dict(state, strict=False)
        arch_title = "U-Net" if is_unet else "CNN-GRU"
        desc = f"{arch_title} [{p.name}] (spatial={spatial}, hidden={hidden}, перенесено {len(state)-len(missing)} слоев)"
        return model, "vision_gru", desc
    else:
        model = ActorCritic().to(device)
        missing, _ = model.load_state_dict(state, strict=False)
        desc = f"MLP [{p.name}] (перенесено {len(state)-len(missing)} слоев)"
        return model, "mlp_state", desc


ROLLOUT_LEN = 64


class RLLab(tk.Tk):
    def __init__(self, episodes=2500):
        super().__init__()
        self.title("osu! v5 — RL lab: Обучение и тонкая настройка моделей")
        self.geometry("860x740")
        self.episodes = episodes
        self.q = queue.Queue(maxsize=2)
        self.stop = threading.Event()

        self.models_catalog: dict[str, str] = {}
        self.model_var = tk.StringVar()
        self.model_desc_var = tk.StringVar(value="")
        self.speed_var = tk.DoubleVar(value=120)
        self.turbo_var = tk.BooleanVar(value=False)
        self.render_var = tk.BooleanVar(value=True)
        self.render_enabled = True
        self.turbo_enabled = False
        self.train_speed = 120.0

        self.render_var.trace_add("write", lambda *_: setattr(self, "render_enabled", bool(self.render_var.get())))
        self.turbo_var.trace_add("write", lambda *_: setattr(self, "turbo_enabled", bool(self.turbo_var.get())))

        self._refresh_models()

        controls = ttk.LabelFrame(self, text="Выбор модели и параметры симуляции", padding=9)
        controls.pack(fill=tk.X, padx=10, pady=6)

        row0 = ttk.Frame(controls)
        row0.pack(fill=tk.X, pady=3)
        ttk.Label(row0, text="Обучать модель:").pack(side=tk.LEFT)
        self.model_combo = ttk.Combobox(
            row0, textvariable=self.model_var,
            values=list(self.models_catalog.keys()),
            state="readonly", width=42,
        )
        self.model_combo.pack(side=tk.LEFT, padx=6)
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_selected)

        ttk.Button(row0, text="Обзор…", command=self._browse_model).pack(side=tk.LEFT, padx=3)
        ttk.Button(row0, text="⟳", width=3, command=self._refresh_models).pack(side=tk.LEFT, padx=2)

        ttk.Label(controls, textvariable=self.model_desc_var, font=("Segoe UI", 8), foreground="#455a64").pack(anchor=tk.W, pady=(2, 4))

        row1 = ttk.Frame(controls)
        row1.pack(fill=tk.X, pady=3)
        ttk.Label(row1, text="Скорость:").pack(side=tk.LEFT)
        self.speed_scale = ttk.Scale(row1, from_=1, to=500, variable=self.speed_var,
                                     orient=tk.HORIZONTAL, length=180)
        self.speed_scale.pack(side=tk.LEFT, padx=4)
        self.speed_label = ttk.Label(row1, text="120x", width=6)
        self.speed_label.pack(side=tk.LEFT)
        self.speed_var.trace_add("write", lambda *_: self._update_speed_label())

        ttk.Checkbutton(row1, text="Турбо (без задержки)", variable=self.turbo_var).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(row1, text="Отрисовка превью", variable=self.render_var).pack(side=tk.LEFT, padx=4)

        row2 = ttk.Frame(controls)
        row2.pack(fill=tk.X, pady=(6, 2))
        self.start_btn = ttk.Button(row2, text="▶ Начать обучение", command=self.start)
        self.start_btn.pack(side=tk.LEFT, padx=4)
        self.stop_btn = ttk.Button(row2, text="⏹ Остановить", command=self.stop.set)
        self.stop_btn.pack(side=tk.LEFT, padx=4)

        dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        self.device_label = ttk.Label(
            row2,
            text=f"Устройство: {'CUDA (' + dev_name + ')' if torch.cuda.is_available() else 'CPU'}",
            foreground="#2e7d32" if torch.cuda.is_available() else "#d32f2f",
            font=("Segoe UI", 9, "bold"),
        )
        self.device_label.pack(side=tk.RIGHT, padx=6)

        self.canvas = tk.Canvas(self, width=640, height=480, bg="#101318", highlightthickness=0)
        self.canvas.pack(padx=10, pady=6)

        self.status = tk.StringVar(value="Выбери модель выше и нажми 'Начать обучение'.")
        ttk.Label(self, textvariable=self.status, font=("Segoe UI", 10, "bold")).pack(pady=(0, 6))

        self.protocol("WM_DELETE_WINDOW", self._close)
        self.after(30, self._pump)

    def _scan_all_models(self) -> dict[str, str]:
        cat: dict[str, str] = {
            "[Новая] U-Net CNN-GRU (GroupNorm, Skip, 6x2x72x96)": "new_unet_vision_gru",
            "[Новая] CNN-GRU-Heatmap (vision 6x2x72x96)": "new_vision_gru",
            "[Новая] MLP (state 9d)": "new_mlp_state",
        }
        folders = [
            ("Osu_2", ROOT / "data" / "models"),
            ("Legacy", LEGACY_ROOT / "data" / "policy"),
            ("LegacyModels", LEGACY_ROOT / "data" / "models"),
        ]
        for tag, folder in folders:
            if folder.is_dir():
                for p in sorted(folder.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True):
                    sz = p.stat().st_size / (1024 * 1024)
                    label = f"[{tag}] {p.name} ({sz:.1f} MB)"
                    cat[label] = str(p.resolve())
        return cat

    def _refresh_models(self):
        curr = self.model_var.get()
        custom_target = None
        if curr in self.models_catalog and self.models_catalog[curr] not in ("new_vision_gru", "new_mlp_state"):
            custom_target = self.models_catalog[curr]

        self.models_catalog = self._scan_all_models()
        if custom_target and Path(custom_target).exists():
            cp = Path(custom_target)
            sz = cp.stat().st_size / (1024 * 1024)
            self.models_catalog[f"[Custom] {cp.name} ({sz:.1f} MB)"] = str(cp.resolve())

        keys = list(self.models_catalog.keys())
        if hasattr(self, "model_combo"):
            self.model_combo.configure(values=keys)

        if curr in self.models_catalog:
            self.model_var.set(curr)
        elif keys:
            for k in keys:
                if "tiny_policy_v6" in k:
                    self.model_var.set(k)
                    break
                if "tiny_policy_v5" in k:
                    self.model_var.set(k)
                    break
            else:
                self.model_var.set(keys[0])

        self._on_model_selected()

    def _browse_model(self):
        chosen = filedialog.askopenfilename(
            title="Выбрать веса модели для обучения",
            filetypes=[("PyTorch Weights", "*.pt *.pth"), ("All files", "*.*")],
            initialdir=str(ROOT / "data" / "models") if (ROOT / "data" / "models").exists() else str(ROOT),
        )
        if chosen:
            p = Path(chosen)
            sz = p.stat().st_size / (1024 * 1024)
            label = f"[Custom] {p.name} ({sz:.1f} MB)"
            self.models_catalog[label] = str(p.resolve())
            self.model_combo.configure(values=list(self.models_catalog.keys()))
            self.model_var.set(label)
            self._on_model_selected()

    def _on_model_selected(self, *args):
        key = self.model_var.get()
        target = self.models_catalog.get(key, "new_vision_gru")
        if target.startswith("new_"):
            self.model_desc_var.set("Случайная инициализация новой архитектуры с нуля")
        else:
            p = Path(target)
            if p.exists():
                dt = datetime.fromtimestamp(p.stat().st_mtime).strftime("%d.%m.%Y %H:%M")
                self.model_desc_var.set(f"Файл: {p.resolve()} | Изменен: {dt}")
            else:
                self.model_desc_var.set(target)

    def _update_speed_label(self):
        try:
            val = int(self.speed_var.get())
            self.train_speed = float(val)
            self.speed_label.configure(text=f"{val}x")
        except Exception:
            pass

    def start(self):
        self.start_btn.configure(state=tk.DISABLED)
        self.stop.clear()
        target = self.models_catalog.get(self.model_var.get(), "new_vision_gru")
        threading.Thread(target=self._train, args=(target,), daemon=True).start()

    def _train(self, target_spec: str):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model, kind, desc = load_model_from_checkpoint(target_spec, device)
        except Exception as e:
            self.status.set(f"Ошибка загрузки модели: {e}")
            self.start_btn.configure(state=tk.NORMAL)
            return

        use_vision = (kind == "vision_gru")
        if use_vision:
            is_pretrained = (target_spec not in ("new_vision_gru", "new_unet_vision_gru"))
            if is_pretrained:
                for p in model.policy.vision.parameters():
                    p.requires_grad = False
                for p in model.policy.embed.parameters():
                    p.requires_grad = False
                for m in model.policy.vision.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
                trainable_params = [
                    {"params": model.policy.gru.parameters(), "lr": 3e-5},
                    {"params": model.policy.context.parameters(), "lr": 6e-5},
                    {"params": model.policy.heatmap.parameters(), "lr": 6e-5},
                    {"params": model.policy.action.parameters(), "lr": 6e-5},
                    {"params": model.value.parameters(), "lr": 2e-4},
                    {"params": [model.logstd], "lr": 1e-4},
                ]
            else:
                for p in model.parameters():
                    p.requires_grad = True
                trainable_params = [
                    {"params": model.policy.vision.parameters(), "lr": 1e-4},
                    {"params": model.policy.embed.parameters(), "lr": 1e-4},
                    {"params": model.policy.gru.parameters(), "lr": 1e-4},
                    {"params": model.policy.context.parameters(), "lr": 1e-4},
                    {"params": model.policy.heatmap.parameters(), "lr": 1e-4},
                    {"params": model.policy.action.parameters(), "lr": 1e-4},
                    {"params": model.value.parameters(), "lr": 2e-4},
                    {"params": [model.logstd], "lr": 2e-4},
                ]
            opt = torch.optim.AdamW(trainable_params, weight_decay=1e-4)
        else:
            for p in model.parameters():
                p.requires_grad = True
            opt = torch.optim.Adam(model.parameters(), lr=2e-4)

        env = NoteEnv()
        obs = env.reset()
        total_steps = self.episodes * 96
        reward_ema = 0.0

        history = deque(maxlen=6)
        if use_vision:
            for _ in range(6):
                history.append(env.fast_obs())

        last_gui_time = time.perf_counter()
        step = 0
        batch_idx = 0
        lambda_bc = 1.0

        obs_list = []
        actions_move = []
        actions_click = []
        actions_hold = []
        actions_logp = []
        rewards = []
        values = []
        targets_list = []

        while step < total_steps and not self.stop.is_set():
            obs_list.clear()
            actions_move.clear()
            actions_click.clear()
            actions_hold.clear()
            actions_logp.clear()
            rewards.clear()
            values.clear()
            targets_list.clear()

            with torch.inference_mode():
                for _ in range(ROLLOUT_LEN):
                    if self.stop.is_set():
                        break

                    if use_vision:
                        seq_np = np.stack(history)
                        curr_tensor = torch.from_numpy(seq_np).unsqueeze(0).to(device)
                    else:
                        curr_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)

                    out = model(curr_tensor)
                    mu = out["xy"]
                    std = model.logstd.exp()
                    normal = Normal(mu, std)
                    move = torch.clamp(normal.sample(), 0.0, 1.0)
                    click_logit = out.get("click_logit", out["hold_logit"])
                    hold_logit = out["hold_logit"]
                    click_dist = Bernoulli(logits=click_logit)
                    hold_dist = Bernoulli(logits=hold_logit)
                    click = click_dist.sample()
                    hold = hold_dist.sample()
                    logp = normal.log_prob(move).sum(1) + click_dist.log_prob(click) + hold_dist.log_prob(hold)
                    val = out["value"]

                    if use_vision:
                        t_center = env.current_target()
                        targets_list.append(torch.from_numpy(t_center).unsqueeze(0).to(device))

                    action = np.r_[move.cpu().numpy()[0], click.cpu().item(), hold.cpu().item()]
                    nxt, r, _, info = env.step(action)

                    obs_list.append(curr_tensor)
                    actions_move.append(move)
                    actions_click.append(click)
                    actions_hold.append(hold)
                    actions_logp.append(logp)
                    rewards.append(r)
                    values.append(val.squeeze(-1))

                    obs = nxt
                    if use_vision:
                        history.append(env.fast_obs())

                    reward_ema = 0.995 * reward_ema + 0.005 * r
                    step += 1

                    now = time.perf_counter()
                    if now - last_gui_time >= 0.12:
                        last_gui_time = now
                        frame = env.render() if getattr(self, "render_enabled", True) else None
                        try:
                            self.q.put_nowait((frame, info, reward_ema, step, env.circle_radius, env.kind, lambda_bc))
                        except queue.Full:
                            pass

            if self.stop.is_set() or len(obs_list) == 0:
                break

            with torch.inference_mode():
                if use_vision:
                    seq_np = np.stack(history)
                    next_tensor = torch.from_numpy(seq_np).unsqueeze(0).to(device)
                else:
                    next_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
                next_out = model(next_tensor)
                next_val = next_out["value"].squeeze(-1)

            values_tensor = torch.stack(values)
            n_steps = len(obs_list)
            advantages = torch.zeros(n_steps, 1, device=device)
            lastgaelam = 0.0
            for t in reversed(range(n_steps)):
                nextvalues = next_val.squeeze() if t == n_steps - 1 else values_tensor[t + 1]
                delta = rewards[t] + 0.99 * nextvalues - values_tensor[t]
                advantages[t] = lastgaelam = delta + 0.99 * 0.95 * lastgaelam
            returns = advantages + values_tensor.unsqueeze(1)

            adv_norm = advantages
            if adv_norm.numel() > 1 and adv_norm.std() > 1e-6:
                adv_norm = (adv_norm - adv_norm.mean()) / (adv_norm.std() + 1e-8)

            batch_x = torch.cat(obs_list, dim=0)
            batch_moves = torch.cat(actions_move, dim=0)
            batch_clicks = torch.cat(actions_click, dim=0)
            batch_holds = torch.cat(actions_hold, dim=0)
            old_logp = torch.cat(actions_logp, dim=0).unsqueeze(1).detach()
            batch_targets = torch.cat(targets_list, dim=0) if len(targets_list) > 0 else None

            lambda_bc = 1.0
            batch_idx += 1

            for _ in range(2):
                out = model(batch_x)
                mu = out["xy"]
                std = model.logstd.exp()
                normal = Normal(mu, std)
                click_logit = out.get("click_logit", out["hold_logit"])
                hold_logit = out["hold_logit"]
                click_dist = Bernoulli(logits=click_logit)
                hold_dist = Bernoulli(logits=hold_logit)
                pred_values = out["value"]

                new_logp = normal.log_prob(batch_moves).sum(1, keepdim=True) + click_dist.log_prob(batch_clicks) + hold_dist.log_prob(batch_holds)
                ratio = torch.exp(new_logp - old_logp)
                surr1 = ratio * adv_norm.detach()
                surr2 = torch.clamp(ratio, 0.8, 1.2) * adv_norm.detach()
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * (pred_values - returns.detach()).square().mean()
                entropy_loss = -0.005 * (normal.entropy().sum(1, keepdim=True) + click_dist.entropy() + hold_dist.entropy()).mean()

                loss = policy_loss + value_loss + entropy_loss

                if use_vision and batch_targets is not None:
                    expected_hm = target_heatmap(batch_targets)
                    hm_anchor_loss = -(expected_hm * F.log_softmax(out["heatmap"], 1)).sum(1).mean()
                    xy_anchor_loss = F.smooth_l1_loss(out["xy"], batch_targets)
                    loss = loss + lambda_bc * (hm_anchor_loss + 2.0 * xy_anchor_loss)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            if not getattr(self, "turbo_enabled", False):
                speed = max(1.0, float(getattr(self, "train_speed", 120.0)))
                if speed < 300:
                    time.sleep(max(0.0001, ROLLOUT_LEN / (60.0 * speed)))

        save_dir = ROOT / "data" / "models"
        save_dir.mkdir(parents=True, exist_ok=True)
        if use_vision:
            save_payload = {
                "state": model.policy.state_dict(),
                "steps": step,
                "version": 2,
                "input": [2, 72, 96],
                "heatmap": [18, 24],
                "spatial_channels": getattr(model, "spatial_channels", 80),
                "hidden": getattr(model, "hidden", 160),
            }
        else:
            save_payload = {
                "state": model.state_dict(),
                "steps": step,
                "model": kind,
                "input": "state[9]",
            }

        torch.save(save_payload, str(save_dir / "synthetic_rl_policy.pt"))

        try:
            self.q.put_nowait((None, None, reward_ema, step, 0.0, "", 0.0))
        except queue.Full:
            pass

    def _pump(self):
        try:
            latest = None
            while True:
                latest = self.q.get_nowait()
        except queue.Empty:
            pass

        if latest is not None:
            frame, info, ema, step, radius, kind, bc_w = latest
            if info is None:
                self.status.set(f"Готово: {step} шагов · reward EMA {ema:.3f}")
                self.start_btn.configure(state=tk.NORMAL)
            else:
                if frame is not None:
                    self._draw(frame)
                total = info["hits"] + info["misses"]
                rate = float(info.get("accuracy", 100 * info["hits"] / max(1, total)))
                score = float(info.get("score", 0.0))
                combo = int(info.get("combo", 0))
                max_combo = int(info.get("max_combo", 0))
                n300 = int(info.get("300", 0))
                n100 = int(info.get("100", 0))
                n50 = int(info.get("50", 0))
                n_miss = int(info.get("misses", 0))
                pat = str(info.get("pattern", kind))
                self.status.set(
                    f"Шаг {step} · {pat} · Точн: {rate:.1f}% · 300/100/50/x: {n300}/{n100}/{n50}/{n_miss} · Комбо: {combo}x (макс: {max_combo}x) · Счёт: {int(score)}"
                )

        if self.winfo_exists():
            self.after(30, self._pump)

    def _draw(self, frame_bgr):
        c = self.canvas
        c.delete("all")
        rgb = cv2.cvtColor(cv2.resize(frame_bgr, (640, 480), interpolation=cv2.INTER_NEAREST), cv2.COLOR_BGR2RGB)
        self.preview = ImageTk.PhotoImage(Image.fromarray(rgb))
        c.create_image(0, 0, anchor=tk.NW, image=self.preview)

    def _close(self):
        self.stop.set()
        self.destroy()


def run(episodes=2500):
    RLLab(episodes).mainloop()


