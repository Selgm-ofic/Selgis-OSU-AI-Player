"""
================================================================================
 Semga inference app (Normis Edition)
 Features:
   - 3D Rotating Semga Logo + 45° Spinning 'S' Fall (60 FPS)
   - 100% Transparent UI Components (Full Wallpaper Visibility)
   - Zero Extra Background Overlays
   - Independent Header Wallpaper & Window Wallpaper
   - Custom Transparent Controls & Sliders
   - Full Theming & Persistent Configuration
================================================================================
"""
from __future__ import annotations

import ctypes
import json
import math
import os
import queue
import random
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml
from PIL import Image, ImageTk, ImageEnhance
import tkinter as tk
from tkinter import filedialog, messagebox

# Ensure osu_bot module can be imported
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from osu_bot import load_policy, RelaxLive

# Win32 integration
try:
    import win32gui
    import win32con
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False


# ==============================================================================
# Theme Palettes Configuration (Default: Obsidian & Platinum)
# ==============================================================================
THEMES = {
    "neutral_gray": {
        "name": "Нейтральный темно-серый",
        "bg": "#1c1c1f",
        "card_inner": "#28282c",
        "card_hover": "#36363c",
        "border": "#44444a",
        "border_bright": "#767680",
        "accent": "#d1d5db",
        "accent_glow": "#f3f4f6",
        "text_primary": "#f3f4f6",
        "text_muted": "#9ca3af",
        "btn_primary": "#3b3b42",
        "btn_primary_hover": "#4b4b54",
        "btn_stop": "#52525b",
        "btn_stop_hover": "#71717a",
        "badge_bg": "#28282c",
        "badge_fg": "#e5e7eb",
        "s_colors": ["#44444a", "#5e5e66", "#767680", "#9ca3af", "#d1d5db"],
        "logo_depth": ["#28282c", "#36363c", "#44444a", "#5e5e66", "#767680"],
        "logo_front": "#f3f4f6",
    },
    "obsidian": {
        "name": "Обсидиан & Платина",
        "bg": "#08080a",
        "card_inner": "#121216",
        "card_hover": "#1e1e24",
        "border": "#4a4a59",
        "border_bright": "#d4d4dc",
        "accent": "#e0e0e8",
        "accent_glow": "#ffffff",
        "text_primary": "#ffffff",
        "text_muted": "#8a8a9a",
        "btn_primary": "#2d3748",
        "btn_primary_hover": "#4a5568",
        "btn_stop": "#742a2a",
        "btn_stop_hover": "#9b2c2c",
        "badge_bg": "#1a202c",
        "badge_fg": "#e2e8f0",
        "s_colors": ["#4a4a59", "#71717a", "#a1a1aa", "#d4d4dc", "#ffffff"],
        "logo_depth": ["#27272a", "#3f3f46", "#52525b", "#71717a", "#a1a1aa"],
        "logo_front": "#f4f4f5",
    },
    "burgundy_gold": {
        "name": "Бордо & Золото",
        "bg": "#12030d",
        "card_inner": "#220819",
        "card_hover": "#320c25",
        "border": "#805b10",
        "border_bright": "#ffd700",
        "accent": "#ffd700",
        "accent_glow": "#fff3a8",
        "text_primary": "#ffffff",
        "text_muted": "#c09bb5",
        "btn_primary": "#2e7d32",
        "btn_primary_hover": "#388e3c",
        "btn_stop": "#b71c1c",
        "btn_stop_hover": "#d32f2f",
        "badge_bg": "#332200",
        "badge_fg": "#ffd700",
        "s_colors": ["#d4af37", "#b8860b", "#e6ca65", "#ffd700", "#fff3a8"],
        "logo_depth": ["#664608", "#8b6508", "#b8860b", "#d4af37", "#e6ca65"],
        "logo_front": "#fff2a3",
    },
    "cyberpunk": {
        "name": "Киберпанк Неон",
        "bg": "#050814",
        "card_inner": "#0c132c",
        "card_hover": "#142048",
        "border": "#0088aa",
        "border_bright": "#00f0ff",
        "accent": "#00f0ff",
        "accent_glow": "#70ffff",
        "text_primary": "#ffffff",
        "text_muted": "#8098c0",
        "btn_primary": "#00838f",
        "btn_primary_hover": "#00acc1",
        "btn_stop": "#c2185b",
        "btn_stop_hover": "#e91e63",
        "badge_bg": "#002b36",
        "badge_fg": "#00f0ff",
        "s_colors": ["#005577", "#0088aa", "#00c4cc", "#00f0ff", "#ffffff"],
        "logo_depth": ["#003344", "#005566", "#007788", "#00aacc", "#00f0ff"],
        "logo_front": "#e0ffff",
    },
    "emerald": {
        "name": "Изумрудный Бархат",
        "bg": "#041009",
        "card_inner": "#0a2215",
        "card_hover": "#123622",
        "border": "#1b663e",
        "border_bright": "#00e676",
        "accent": "#00e676",
        "accent_glow": "#b9f6ca",
        "text_primary": "#ffffff",
        "text_muted": "#88b89d",
        "btn_primary": "#1b5e20",
        "btn_primary_hover": "#2e7d32",
        "btn_stop": "#b71c1c",
        "btn_stop_hover": "#d32f2f",
        "badge_bg": "#003314",
        "badge_fg": "#00e676",
        "s_colors": ["#0f3d24", "#1b663e", "#2e8b57", "#00e676", "#b9f6ca"],
        "logo_depth": ["#0b2918", "#114227", "#195c37", "#237d4b", "#00e676"],
        "logo_front": "#d4fcdb",
    },
}


# ==============================================================================
# Custom Dark Dropdown (Compact, Transparent Integration)
# ==============================================================================
class DarkDropdown(tk.Frame):
    def __init__(self, parent, current_val: str, on_select, theme: dict = None, width: int = 240, **kwargs):
        self.theme = theme or THEMES["neutral_gray"]
        super().__init__(parent, bg=self.theme["card_inner"], highlightthickness=1,
                         highlightbackground=self.theme["border"], **kwargs)
        self.on_select = on_select
        self.current_val = tk.StringVar(value=current_val)
        self.values: list[str] = []
        self.popup = None

        self.btn = tk.Button(
            self, textvariable=self.current_val,
            bg=self.theme["card_inner"], fg=self.theme["text_primary"],
            activebackground=self.theme["card_hover"], activeforeground=self.theme["accent"],
            font=("Segoe UI", 9, "bold"), relief=tk.FLAT, bd=0, padx=8, pady=2,
            anchor=tk.W, cursor="hand2", command=self._toggle_popup
        )
        self.btn.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.arrow = tk.Label(
            self, text="▼", bg=self.theme["card_inner"], fg=self.theme["accent"],
            font=("Segoe UI", 8), padx=6, cursor="hand2"
        )
        self.arrow.pack(side=tk.RIGHT)
        self.arrow.bind("<Button-1>", lambda e: self._toggle_popup())

    def apply_theme(self, theme: dict):
        self.theme = theme
        self.configure(bg=self.theme["card_inner"], highlightbackground=self.theme["border"])
        self.btn.configure(bg=self.theme["card_inner"], fg=self.theme["text_primary"],
                           activebackground=self.theme["card_hover"], activeforeground=self.theme["accent"])
        self.arrow.configure(bg=self.theme["card_inner"], fg=self.theme["accent"])

    def set_values(self, values: list[str]):
        self.values = values
        if not self.current_val.get() and self.values:
            self.current_val.set(self.values[0])
            if self.on_select:
                self.on_select(self.values[0])
        elif self.values and self.current_val.get() not in self.values:
            # Check partial match by filename
            matched = False
            cur = self.current_val.get()
            for v in self.values:
                if cur and (v.split()[0] in cur or cur.split()[0] in v):
                    self.current_val.set(v)
                    matched = True
                    break
            if not matched and self.values:
                self.current_val.set(self.values[0])
                if self.on_select:
                    self.on_select(self.values[0])

    def set_value(self, val: str):
        self.current_val.set(val)

    def get_value(self) -> str:
        return self.current_val.get()

    def _toggle_popup(self):
        if self.popup and self.popup.winfo_exists():
            self._close_popup()
            return
        if not self.values:
            return

        x = self.winfo_rootx()
        y = self.winfo_rooty() + self.winfo_height() + 2
        w = max(self.winfo_width(), 280)

        self.popup = tk.Toplevel(self)
        self.popup.wm_overrideredirect(True)
        self.popup.wm_geometry(f"{w}x{min(240, len(self.values) * 30 + 8)}+{x}+{y}")
        self.popup.configure(bg=self.theme["card_inner"], highlightthickness=1,
                              highlightbackground=self.theme["border_bright"])
        self.popup.attributes("-topmost", True)

        list_frame = tk.Frame(self.popup, bg=self.theme["card_inner"])
        list_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        for val in self.values:
            is_active = (val == self.current_val.get())
            fg = self.theme["accent"] if is_active else self.theme["text_primary"]
            bg = self.theme["card_hover"] if is_active else self.theme["card_inner"]

            row = tk.Label(
                list_frame, text=val, bg=bg, fg=fg,
                font=("Segoe UI", 9, "bold" if is_active else "normal"),
                anchor=tk.W, padx=8, pady=4, cursor="hand2"
            )
            row.pack(fill=tk.X)

            def make_handler(v=val):
                return lambda e: self._select_val(v)

            row.bind("<Button-1>", make_handler(val))
            row.bind("<Enter>", lambda e, r=row: r.configure(bg=self.theme["card_hover"]))
            row.bind("<Leave>", lambda e, r=row, active=is_active: r.configure(
                bg=self.theme["card_hover"] if active else self.theme["card_inner"]
            ))

        self.bind_all("<Button-1>", self._on_click_outside, add="+")

    def _select_val(self, val: str):
        self.current_val.set(val)
        self._close_popup()
        if self.on_select:
            self.on_select(val)

    def _on_click_outside(self, event):
        if self.popup and self.popup.winfo_exists():
            x, y = event.x_root, event.y_root
            px, py = self.popup.winfo_rootx(), self.popup.winfo_rooty()
            pw, ph = self.popup.winfo_width(), self.popup.winfo_height()
            if not (px <= x <= px + pw and py <= y <= py + ph):
                self._close_popup()

    def _close_popup(self):
        if self.popup and self.popup.winfo_exists():
            self.popup.destroy()
            self.popup = None
        try:
            self.unbind_all("<Button-1>")
        except Exception:
            pass


# ==============================================================================
# Header Canvas: 3D Rotating Semga Logo + 45° Spinning 'S' Fall (60 FPS)
# Supports Independent Dedicated Header Wallpaper or Main Wallpaper Flow
# ==============================================================================
class SemgaBannerCanvas(tk.Canvas):
    def __init__(self, parent, width: int = 1240, height: int = 70, theme: dict = None, **kwargs):
        self.theme = theme or THEMES["neutral_gray"]
        super().__init__(parent, width=width, height=height, bg=self.theme["bg"],
                         highlightthickness=1, highlightbackground=self.theme["border"], **kwargs)
        self.w = width
        self.h = height

        # Header Wallpaper state
        self.wallpaper_path: str = ""
        self.wallpaper_image: Optional[Image.Image] = None
        self.wallpaper_photo: Optional[ImageTk.PhotoImage] = None

        # 1. Spinning 45-degree Falling 'S' particles setup
        self.particles: list[dict] = []
        self.p_items: list[int] = []
        self._init_particles()

        # 2. 3D Rotating Semga setup
        self.semga_center_x = 125
        self.semga_center_y = 35
        self.semga_yaw = 0.0
        self.semga_depth_slices = 5
        self.semga_letters = ["S", "E", "M", "G", "A"]
        self.letter_spacing = 22

        # 3D canvas items: [slice_index][letter_index]
        self.depth_items: list[list[int]] = []
        self.front_items: list[int] = []
        self.glint_items: list[int] = []
        self.glint_active = False
        self.glint_timer = 0
        self.next_glint_time = time.time() + 2.0

        self._init_3d_semga()

        # Start 60 FPS loop (16 ms)
        self.after(16, self._tick_60fps)

    def set_wallpaper(self, path: str):
        self.wallpaper_path = path
        if not path:
            self.wallpaper_image = None
            self.wallpaper_photo = None
            self.delete("banner_bg")
            self.configure(bg=self.theme["bg"])
            return

        p = Path(path)
        if not p.exists():
            return

        try:
            self.wallpaper_image = Image.open(p).convert("RGB")
            self._render_wallpaper()
        except Exception:
            pass

    def set_main_wallpaper_slice(self, main_image: Optional[Image.Image]):
        """When no dedicated header wallpaper is set, flows main wallpaper through header seamlessly."""
        if self.wallpaper_path:
            return  # Dedicated header wallpaper takes precedence
        if main_image is None:
            self.delete("banner_bg")
            self.wallpaper_photo = None
            self.configure(bg=self.theme["bg"])
            return

        try:
            w = max(200, self.winfo_width() or self.w)
            h = max(40, self.winfo_height() or self.h)
            img_w, img_h = main_image.size
            slice_w = min(w, img_w)
            slice_h = min(h, img_h)
            crop = main_image.crop((0, 0, slice_w, slice_h)).resize((w, h), Image.Resampling.LANCZOS)
            self.wallpaper_photo = ImageTk.PhotoImage(crop)

            if self.find_withtag("banner_bg"):
                self.itemconfig("banner_bg", image=self.wallpaper_photo)
            else:
                self.create_image(0, 0, image=self.wallpaper_photo, anchor=tk.NW, tags="banner_bg")
                self.tag_lower("banner_bg")
        except Exception:
            pass

    def _render_wallpaper(self):
        if self.wallpaper_image is None:
            return
        w = max(200, self.winfo_width() or self.w)
        h = max(40, self.winfo_height() or self.h)
        resized = self.wallpaper_image.resize((w, h), Image.Resampling.LANCZOS)
        self.wallpaper_photo = ImageTk.PhotoImage(resized)

        if self.find_withtag("banner_bg"):
            self.itemconfig("banner_bg", image=self.wallpaper_photo)
        else:
            self.create_image(0, 0, image=self.wallpaper_photo, anchor=tk.NW, tags="banner_bg")
            self.tag_lower("banner_bg")

    def apply_theme(self, theme: dict):
        self.theme = theme
        self.configure(bg=self.theme["bg"], highlightbackground=self.theme["border"])
        s_pal = self.theme["s_colors"]
        for p, item in zip(self.particles, self.p_items):
            p["base_color"] = random.choice(s_pal)
            self.itemconfig(item, fill=p["base_color"])

        for d_idx, slice_list in enumerate(self.depth_items):
            d_col = self.theme["logo_depth"][min(d_idx, len(self.theme["logo_depth"]) - 1)]
            for item in slice_list:
                self.itemconfig(item, fill=d_col)
        for item in self.front_items:
            self.itemconfig(item, fill=self.theme["logo_front"])

    def _init_particles(self):
        random.seed(42)
        self.particles.clear()
        self.p_items.clear()
        s_pal = self.theme["s_colors"]

        for _ in range(48):
            p = {
                "x": random.uniform(-60, self.w + 60),
                "y": random.uniform(-40, self.h + 20),
                "speed": random.uniform(0.9, 2.2),
                "angle": random.uniform(0, 360),
                "rot_speed": random.uniform(1.8, 4.5) * random.choice([-1, 1]),
                "size": random.choice([9, 11, 13, 15, 17]),
                "base_color": random.choice(s_pal),
                "gleam_frames": 0
            }
            self.particles.append(p)
            item = self.create_text(
                p["x"], p["y"], text="S", angle=int(p["angle"]),
                font=("Segoe UI", p["size"], "bold"), fill=p["base_color"]
            )
            self.p_items.append(item)

    def _init_3d_semga(self):
        self.depth_items.clear()
        self.front_items.clear()
        self.glint_items.clear()

        for d_idx in range(self.semga_depth_slices, 0, -1):
            slice_list = []
            d_col = self.theme["logo_depth"][min(d_idx, len(self.theme["logo_depth"]) - 1)]
            for _ in self.semga_letters:
                it = self.create_text(
                    self.semga_center_x, self.semga_center_y, text="",
                    font=("Segoe UI Black", 21, "bold"), fill=d_col
                )
                slice_list.append(it)
            self.depth_items.append(slice_list)

        for _ in self.semga_letters:
            it = self.create_text(
                self.semga_center_x, self.semga_center_y, text="",
                font=("Segoe UI Black", 21, "bold"), fill=self.theme["logo_front"]
            )
            self.front_items.append(it)

        for _ in range(3):
            git = self.create_text(-100, -100, text="✦", font=("Segoe UI", 16, "bold"), fill="#ffffff")
            self.glint_items.append(git)

    def _tick_60fps(self):
        w_cur = self.winfo_width() or self.w
        s_pal = self.theme["s_colors"]
        for p, item in zip(self.particles, self.p_items):
            p["x"] += p["speed"] * 1.0
            p["y"] += p["speed"] * 0.95
            p["angle"] = (p["angle"] + p["rot_speed"]) % 360

            if p["y"] > self.h + 25 or p["x"] > w_cur + 25:
                p["y"] = random.uniform(-35, -5)
                p["x"] = random.uniform(-50, w_cur - 50)
                p["speed"] = random.uniform(0.9, 2.2)

            self.coords(item, p["x"], p["y"])
            self.itemconfig(item, angle=int(p["angle"]))

            if p["gleam_frames"] > 0:
                p["gleam_frames"] -= 1
                if p["gleam_frames"] > 8:
                    self.itemconfig(item, fill="#ffffff")
                elif p["gleam_frames"] > 4:
                    self.itemconfig(item, fill=self.theme["accent_glow"])
                else:
                    self.itemconfig(item, fill=self.theme["accent"])
            else:
                self.itemconfig(item, fill=p["base_color"])

        if random.random() < 0.08:
            lucky = random.choice(self.particles)
            lucky["gleam_frames"] = 16

        self.semga_yaw = (self.semga_yaw + 0.038) % (2 * math.pi)
        D = 280.0
        now = time.time()

        if not self.glint_active and now >= self.next_glint_time:
            if math.cos(self.semga_yaw) > 0.65:
                self.glint_active = True
                self.glint_timer = 20
                self.next_glint_time = now + random.uniform(2.5, 4.0)

        cos_y = math.cos(self.semga_yaw)
        sin_y = math.sin(self.semga_yaw)

        for l_idx, char in enumerate(self.semga_letters):
            base_x = (l_idx - 2) * self.letter_spacing
            rot_x = base_x * cos_y
            rot_z = base_x * sin_y

            scale = D / (D + rot_z)
            proj_x = self.semga_center_x + rot_x * scale
            proj_y = self.semga_center_y + math.sin(self.semga_yaw + l_idx * 0.4) * 2.5

            font_size = max(14, int(21 * scale))
            cur_font = ("Segoe UI Black", font_size, "bold")

            for d_idx, slice_list in enumerate(self.depth_items):
                depth_step = (d_idx + 1) * 1.4
                dx = depth_step * sin_y * 1.5
                dy = depth_step * 0.8
                it = slice_list[l_idx]
                self.coords(it, proj_x + dx, proj_y + dy)
                self.itemconfig(it, text=char, font=cur_font)

            fit = self.front_items[l_idx]
            self.coords(fit, proj_x, proj_y)

            if cos_y > 0.2:
                face_col = "#ffffff" if (self.glint_active and self.glint_timer > 10 and l_idx in (0, 2, 4)) else self.theme["logo_front"]
            else:
                face_col = self.theme["logo_depth"][0]

            self.itemconfig(fit, text=char, font=cur_font, fill=face_col)

        if self.glint_active:
            self.glint_timer -= 1
            if self.glint_timer <= 0:
                self.glint_active = False
                for git in self.glint_items:
                    self.coords(git, -100, -100)
            else:
                lead_x = self.semga_center_x + (10 - self.glint_timer) * 8 - 30
                lead_y = self.semga_center_y - 12
                self.coords(self.glint_items[0], lead_x, lead_y)
                self.itemconfig(self.glint_items[0], fill=self.theme["accent_glow"])

                self.coords(self.glint_items[1], lead_x + 18, lead_y + 14)
                self.itemconfig(self.glint_items[1], fill="#ffffff")

        self.after(16, self._tick_60fps)


# ==============================================================================
# Fullscreen Drag & Select Overlay for Interactive ROI
# ==============================================================================
class DragSelectOverlay:
    def __init__(self, master: tk.Tk, on_complete, theme: dict):
        self.master = master
        self.on_complete = on_complete
        self.theme = theme

        self.top = tk.Toplevel(master)
        self.top.attributes("-fullscreen", True)
        self.top.attributes("-alpha", 0.35)
        self.top.attributes("-topmost", True)
        self.top.configure(bg="black", cursor="crosshair")

        self.canvas = tk.Canvas(self.top, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.tip = self.canvas.create_text(
            self.top.winfo_screenwidth() // 2, 60,
            text="Выделите рамкой игровую область osu! (Playfield ROI)\n[ESC] — отмена",
            font=("Segoe UI", 16, "bold"), fill="#ffffff", justify=tk.CENTER
        )

        self.start_x = None
        self.start_y = None
        self.rect_id = None

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.top.bind("<Escape>", lambda e: self.top.destroy())

    def _on_press(self, event):
        self.start_x = event.x_root
        self.start_y = event.y_root
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline=self.theme["border_bright"], fill=self.theme["accent"],
            stipple="gray25", width=2
        )

    def _on_drag(self, event):
        if self.rect_id:
            cur_x = event.x
            cur_y = event.y
            self.canvas.coords(self.rect_id, self.start_x, self.start_y, cur_x, cur_y)

    def _on_release(self, event):
        end_x = event.x_root
        end_y = event.y_root

        left = min(self.start_x, end_x)
        top = min(self.start_y, end_y)
        right = max(self.start_x, end_x)
        bottom = max(self.start_y, end_y)

        self.top.destroy()
        if (right - left) > 50 and (bottom - top) > 50:
            self.on_complete(left, top, right, bottom)


# ==============================================================================
# Main Application Window: Semga inference app
# ==============================================================================
class NormisApp(tk.Tk):
    def __init__(self):
        super().__init__()
        # Official Window Title
        self.title("osu! AI Player")
        self.geometry("1240x820")
        self.minsize(1050, 720)

        # Enable Windows 11 / 10 Dark Titlebar
        self._enable_dark_titlebar()

        self.config_path = APP_DIR / "config.yaml"
        self.cfg = self._load_config()

        # Theme & Transparency State
        theme_cfg = self.cfg.get("theme", {})
        self.theme_key = theme_cfg.get("palette", "neutral_gray")
        if self.theme_key not in THEMES:
            self.theme_key = "neutral_gray"
        self.theme = THEMES[self.theme_key]
        self.window_opacity = float(theme_cfg.get("window_opacity", 1.0))
        self.wallpaper_brightness = float(theme_cfg.get("wallpaper_brightness", 0.90))
        self.wallpaper_path = theme_cfg.get("wallpaper", "")
        self.header_wallpaper_path = theme_cfg.get("header_wallpaper", "")

        self.attributes("-alpha", self.window_opacity)

        # Wallpaper Image buffers
        self.wallpaper_image: Optional[Image.Image] = None
        self.wallpaper_photo: Optional[ImageTk.PhotoImage] = None

        # ROI Coordinates State
        pf = self.cfg.get("playfield", {})
        self.roi_left = tk.IntVar(value=pf.get("left", 0))
        self.roi_top = tk.IntVar(value=pf.get("top", 0))
        self.roi_right = tk.IntVar(value=pf.get("right", 1920))
        self.roi_bottom = tk.IntVar(value=pf.get("bottom", 1080))
        self.roi_dims_var = tk.StringVar()
        self._update_roi_dims_label()

        # Model & Inference State
        sett = self.cfg.get("settings", {})
        self.model_var = tk.StringVar(value=sett.get("selected_model", ""))
        self.clicks_var = tk.BooleanVar(value=sett.get("clicks_enabled", True))
        self.conf_var = tk.DoubleVar(value=sett.get("confidence", 0.60))
        self.smooth_var = tk.DoubleVar(value=sett.get("smooth", 0.0))

        # Status & Telemetry
        self.status_badge_var = tk.StringVar(value="● ОЖИДАНИЕ")
        self.metric_fps = tk.StringVar(value="0.0 FPS")
        self.metric_cap_fps = tk.StringVar(value="0.0 FPS")
        self.metric_latency = tk.StringVar(value="0.0 мс")
        self.metric_tth = tk.StringVar(value="-- мс")

        # Worker / Inference Threading
        self.live: Optional[RelaxLive] = None
        self.messages_queue = queue.Queue()
        self.frames_queue = queue.Queue(maxsize=2)
        self.preview_photo = None

        # Build Interface directly on master canvas
        self._build_ui()
        self._refresh_models()
        self._load_wallpaper()
        if self.header_wallpaper_path:
            self.banner_canvas.set_wallpaper(self.header_wallpaper_path)

        # Bindings & Event loop pump
        self.bind("<Configure>", self._on_window_resize)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(30, self._pump)

    def _enable_dark_titlebar(self):
        try:
            self.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(self.winfo_id())
            if hwnd:
                v = ctypes.c_int(1)
                ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(v), 4)
                ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 19, ctypes.byref(v), 4)
        except Exception:
            pass

    def _load_config(self) -> dict:
        if self.config_path.exists():
            try:
                return yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
            except Exception:
                pass
        return {
            "dxgi": {"monitor": 1},
            "playfield": {"left": 0, "top": 0, "right": 1920, "bottom": 1080},
            "runtime": {"target_fps": 60},
            "theme": {
                "palette": "obsidian",
                "window_opacity": 1.0,
                "wallpaper_brightness": 0.90,
                "wallpaper": "",
                "header_wallpaper": ""
            },
            "settings": {"selected_model": "", "clicks_enabled": True, "confidence": 0.60, "smooth": 0.0}
        }

    def _save_config(self):
        self.cfg["playfield"] = {
            "left": int(self.roi_left.get()),
            "top": int(self.roi_top.get()),
            "right": int(self.roi_right.get()),
            "bottom": int(self.roi_bottom.get()),
        }
        self.cfg["theme"] = {
            "palette": self.theme_key,
            "window_opacity": round(float(self.window_opacity), 2),
            "wallpaper_brightness": round(float(self.wallpaper_brightness), 2),
            "wallpaper": str(self.wallpaper_path or ""),
            "header_wallpaper": str(self.header_wallpaper_path or "")
        }
        self.cfg["settings"] = {
            "selected_model": self.model_dropdown.get_value() if hasattr(self, "model_dropdown") else "",
            "clicks_enabled": bool(self.clicks_var.get()),
            "confidence": round(float(self.conf_var.get()), 2),
            "smooth": round(float(self.smooth_var.get()), 2)
        }
        try:
            self.config_path.write_text(yaml.safe_dump(self.cfg, allow_unicode=True), encoding="utf-8")
        except Exception:
            pass

    # --------------------------------------------------------------------------
    # UI Construction (100% Transparent Components on Master Canvas)
    # --------------------------------------------------------------------------
    def _build_ui(self):
        # 1. Master Canvas covering the entire window
        self.bg_canvas = tk.Canvas(self, bg=self.theme["bg"], highlightthickness=0)
        self.bg_canvas.place(x=0, y=0, relwidth=1, relheight=1)

        # 2. Top Header Bar: 3D Semga + 60 FPS Spinning 'S' Fall
        self.header_frame = tk.Frame(self, bg=self.theme["bg"], height=70)
        self.header_frame.pack(fill=tk.X, padx=0, pady=0)
        self.header_frame.pack_propagate(False)

        self.banner_canvas = SemgaBannerCanvas(self.header_frame, width=1280, height=70, theme=self.theme)
        self.banner_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Header Right Tools
        tools_frame = tk.Frame(self.header_frame, bg=self.theme["card_inner"], highlightthickness=1,
                               highlightbackground=self.theme["border"])
        tools_frame.place(relx=1.0, rely=0.5, anchor=tk.E, x=-14)

        self.btn_theme = tk.Button(
            tools_frame, text=f"🎨 {self.theme['name']}",
            bg=self.theme["card_inner"], fg=self.theme["accent"],
            activebackground=self.theme["card_hover"], activeforeground="#ffffff",
            font=("Segoe UI", 9, "bold"), relief=tk.FLAT, bd=0,
            padx=10, pady=4, cursor="hand2", command=self._open_theme_menu
        )
        self.btn_theme.pack(side=tk.LEFT, padx=2)

        self.btn_wall = tk.Button(
            tools_frame, text="🖼 Фон",
            bg=self.theme["card_inner"], fg=self.theme["text_primary"],
            activebackground=self.theme["card_hover"], activeforeground="#ffffff",
            font=("Segoe UI", 9), relief=tk.FLAT, bd=0,
            padx=10, pady=4, cursor="hand2", command=self._open_wallpaper_dialog
        )
        self.btn_wall.pack(side=tk.LEFT, padx=2)

        self.btn_transp = tk.Button(
            tools_frame, text="👁 Прозрачность",
            bg=self.theme["card_inner"], fg=self.theme["text_primary"],
            activebackground=self.theme["card_hover"], activeforeground="#ffffff",
            font=("Segoe UI", 9), relief=tk.FLAT, bd=0,
            padx=10, pady=4, cursor="hand2", command=self._open_transparency_dialog
        )
        self.btn_transp.pack(side=tk.LEFT, padx=2)

        # 3. Interactive Widgets placed directly on bg_canvas
        # Card 1: Models
        self.model_dropdown = DarkDropdown(
            self.bg_canvas, current_val=self.model_var.get(), on_select=self._on_model_selected,
            theme=self.theme, width=285
        )
        self.win_dropdown = self.bg_canvas.create_window(0, 0, window=self.model_dropdown, anchor=tk.NW)

        self.btn_browse = tk.Button(
            self.bg_canvas, text="Обзор…", bg=self.theme["card_inner"], fg=self.theme["text_primary"],
            activebackground=self.theme["card_hover"], activeforeground="#ffffff",
            font=("Segoe UI", 9), relief=tk.SOLID, bd=1, padx=6, pady=1,
            command=self._browse_model, cursor="hand2"
        )
        self.win_browse = self.bg_canvas.create_window(0, 0, window=self.btn_browse, anchor=tk.NW)

        self.btn_refresh = tk.Button(
            self.bg_canvas, text="⟳", bg=self.theme["card_inner"], fg=self.theme["accent"],
            activebackground=self.theme["card_hover"], activeforeground="#ffffff",
            font=("Segoe UI", 10, "bold"), relief=tk.SOLID, bd=1, padx=4, pady=1,
            command=self._refresh_models, cursor="hand2"
        )
        self.win_refresh = self.bg_canvas.create_window(0, 0, window=self.btn_refresh, anchor=tk.NW)

        # Card 2: Inference
        self.btn_start = tk.Button(
            self.bg_canvas, text="▶ Запустить инференс",
            bg=self.theme["btn_primary"], fg="#ffffff",
            activebackground=self.theme["btn_primary_hover"], activeforeground="#ffffff",
            font=("Segoe UI", 10, "bold"), relief=tk.SOLID, bd=1, padx=10, pady=5,
            command=self.start_inference, cursor="hand2"
        )
        self.win_start = self.bg_canvas.create_window(0, 0, window=self.btn_start, anchor=tk.NW)

        self.btn_stop = tk.Button(
            self.bg_canvas, text="⏹ Остановить",
            bg=self.theme["btn_stop"], fg="#ffffff",
            activebackground=self.theme["btn_stop_hover"], activeforeground="#ffffff",
            font=("Segoe UI", 10, "bold"), relief=tk.SOLID, bd=1, padx=10, pady=5,
            command=self.stop_inference, cursor="hand2"
        )
        self.win_stop = self.bg_canvas.create_window(0, 0, window=self.btn_stop, anchor=tk.NW)

        self.lbl_badge = tk.Label(
            self.bg_canvas, textvariable=self.status_badge_var,
            bg=self.theme["badge_bg"], fg=self.theme["badge_fg"],
            font=("Segoe UI", 9, "bold"), padx=10, pady=2, relief=tk.SOLID, bd=1
        )
        self.win_badge = self.bg_canvas.create_window(0, 0, window=self.lbl_badge, anchor=tk.NW)

        # Card 3: ROI
        self.btn_drag = tk.Button(
            self.bg_canvas, text="🎯 Выбрать на экране (Drag & Select)",
            bg=self.theme["card_inner"], fg=self.theme["accent"],
            activebackground=self.theme["card_hover"], activeforeground="#ffffff",
            font=("Segoe UI", 9, "bold"), relief=tk.SOLID, bd=1, padx=6, pady=3,
            command=self._launch_drag_select, cursor="hand2"
        )
        self.win_drag = self.bg_canvas.create_window(0, 0, window=self.btn_drag, anchor=tk.NW)

        self.btn_find = tk.Button(
            self.bg_canvas, text="🔍 Найти окно osu!",
            bg=self.theme["card_inner"], fg=self.theme["text_primary"],
            activebackground=self.theme["card_hover"], activeforeground="#ffffff",
            font=("Segoe UI", 9, "bold"), relief=tk.SOLID, bd=1, padx=6, pady=3,
            command=self._auto_find_osu_window, cursor="hand2"
        )
        self.win_find = self.bg_canvas.create_window(0, 0, window=self.btn_find, anchor=tk.NW)

        self.preset_btns = []
        presets = [
            ("1080p FHD", 0, 0, 1920, 1080),
            ("1440p 2K", 0, 0, 2560, 1440),
            ("720p HD", 0, 0, 1280, 720),
            ("4:3 Center", 240, 0, 1680, 1080),
        ]
        for name, l, t, r, b in presets:
            b_p = tk.Button(
                self.bg_canvas, text=name, bg=self.theme["card_inner"], fg=self.theme["text_primary"],
                activebackground=self.theme["card_hover"], activeforeground="#ffffff",
                font=("Segoe UI", 8), relief=tk.SOLID, bd=1, padx=4, pady=1,
                command=lambda left=l, top=t, right=r, bottom=b: self._apply_roi_coords(left, top, right, bottom),
                cursor="hand2"
            )
            win_id = self.bg_canvas.create_window(0, 0, window=b_p, anchor=tk.NW)
            self.preset_btns.append((b_p, win_id))

        self.spinboxes = []
        for idx, (label, var) in enumerate([
            ("Left:", self.roi_left), ("Top:", self.roi_top),
            ("Right:", self.roi_right), ("Bottom:", self.roi_bottom)
        ]):
            sb = tk.Spinbox(
                self.bg_canvas, from_=0, to=7680, increment=10, textvariable=var,
                bg=self.theme["card_inner"], fg=self.theme["accent"],
                font=("Segoe UI", 9, "bold"), width=6, relief=tk.SOLID, bd=1,
                command=self._on_spin_change
            )
            sb.bind("<KeyRelease>", lambda e: self._on_spin_change())
            win_id = self.bg_canvas.create_window(0, 0, window=sb, anchor=tk.NW)
            self.spinboxes.append((label, sb, win_id))

        self.btn_save_roi = tk.Button(
            self.bg_canvas, text="💾 Сохранить область в config.yaml",
            bg=self.theme["card_inner"], fg=self.theme["accent"],
            activebackground=self.theme["card_hover"], activeforeground="#ffffff",
            font=("Segoe UI", 9, "bold"), relief=tk.SOLID, bd=1, padx=8, pady=3,
            command=self._save_roi_clicked, cursor="hand2"
        )
        self.win_save_roi = self.bg_canvas.create_window(0, 0, window=self.btn_save_roi, anchor=tk.NW)

        # Bind transparent slider interactions on canvas
        self.bg_canvas.bind("<Button-1>", self._on_canvas_click, add="+")
        self.bg_canvas.bind("<B1-Motion>", self._on_canvas_drag, add="+")

        # Initial layout computation
        self.update_idletasks()
        self._update_layout()

    # --------------------------------------------------------------------------
    # Layout Manager (Draws transparent cards directly on master canvas)
    # --------------------------------------------------------------------------
    def _update_layout(self):
        W = max(1050, self.winfo_width() or 1240)
        H = max(720, self.winfo_height() or 820)

        # Clear existing card outlines and canvas text
        self.bg_canvas.delete("card_elem")

        pad = 12
        gap = 8
        col_left_w = 490
        col_right_x = col_left_w + pad * 2
        col_right_w = max(400, W - col_right_x - pad)

        y_start = 76
        avail_right_h = H - y_start - pad
        h_telem = 120
        h_preview = max(240, avail_right_h - h_telem - gap)

        b_color = self.theme["border"]
        accent = self.theme["accent"]
        text_p = self.theme["text_primary"]
        text_m = self.theme["text_muted"]

        # ======================================================================
        # CARD 1: Models (Outline only, zero extra background!)
        # ======================================================================
        c1_y = y_start
        c1_h = 76
        self.bg_canvas.create_rectangle(pad, c1_y, pad + col_left_w, c1_y + c1_h,
                                        outline=b_color, width=1, tags="card_elem")
        self.bg_canvas.create_text(pad + 14, c1_y + 16, text="📁 Каталог моделей",
                                   anchor=tk.W, fill=accent, font=("Segoe UI", 10, "bold"), tags="card_elem")
        self.bg_canvas.create_text(pad + 14, c1_y + 46, text="Модель:",
                                   anchor=tk.W, fill=text_p, font=("Segoe UI", 9, "bold"), tags="card_elem")

        self.bg_canvas.coords(self.win_dropdown, pad + 75, c1_y + 34)
        self.bg_canvas.itemconfigure(self.win_dropdown, width=285, height=26)

        self.bg_canvas.coords(self.win_browse, pad + 370, c1_y + 34)
        self.bg_canvas.itemconfigure(self.win_browse, width=65, height=26)

        self.bg_canvas.coords(self.win_refresh, pad + 442, c1_y + 34)
        self.bg_canvas.itemconfigure(self.win_refresh, width=32, height=26)

        # ======================================================================
        # CARD 2: Inference (100% Transparent Controls)
        # ======================================================================
        c2_y = c1_y + c1_h + gap
        c2_h = 224
        self.bg_canvas.create_rectangle(pad, c2_y, pad + col_left_w, c2_y + c2_h,
                                        outline=b_color, width=1, tags="card_elem")
        self.bg_canvas.create_text(pad + 14, c2_y + 16, text="⚡ Управление инференсом",
                                   anchor=tk.W, fill=accent, font=("Segoe UI", 10, "bold"), tags="card_elem")

        self.bg_canvas.coords(self.win_start, pad + 14, c2_y + 36)
        self.bg_canvas.itemconfigure(self.win_start, width=225, height=36)

        self.bg_canvas.coords(self.win_stop, pad + 249, c2_y + 36)
        self.bg_canvas.itemconfigure(self.win_stop, width=225, height=36)

        self.bg_canvas.coords(self.win_badge, pad + 14, c2_y + 82)
        self.bg_canvas.create_text(pad + 150, c2_y + 94, text="[F2] Вкл/Пауза  ·  [F10] Стоп",
                                   anchor=tk.W, fill=accent, font=("Segoe UI", 9), tags="card_elem")

        # Custom Transparent Checkbox for Z/X clicks
        chk_y = c2_y + 120
        self.chk_box_rect = (pad + 14, chk_y - 8, pad + 30, chk_y + 8)
        self._draw_clicks_checkbox()

        # Custom Transparent Confidence Slider
        self.conf_slider_coords = (pad + 150, pad + 410, c2_y + 158)
        self.bg_canvas.create_text(pad + 14, c2_y + 158, text="Порог уверенности:",
                                   anchor=tk.W, fill=text_p, font=("Segoe UI", 9), tags="card_elem")
        self._draw_conf_slider()

        # Custom Transparent Mouse Smoothing Slider
        self.smooth_slider_coords = (pad + 150, pad + 410, c2_y + 192)
        self.bg_canvas.create_text(pad + 14, c2_y + 192, text="Сглаживание мыши:",
                                   anchor=tk.W, fill=text_p, font=("Segoe UI", 9), tags="card_elem")
        self._draw_smooth_slider()

        # ======================================================================
        # CARD 3: ROI Editor (100% Transparent Controls)
        # ======================================================================
        c3_y = c2_y + c2_h + gap
        c3_h = max(260, H - c3_y - pad)
        self.bg_canvas.create_rectangle(pad, c3_y, pad + col_left_w, c3_y + c3_h,
                                        outline=b_color, width=1, tags="card_elem")
        self.bg_canvas.create_text(pad + 14, c3_y + 16, text="🎯 Редактор области захвата (ROI)",
                                   anchor=tk.W, fill=accent, font=("Segoe UI", 10, "bold"), tags="card_elem")
        self.bg_canvas.create_text(pad + 14, c3_y + 38, text=self.roi_dims_var.get(),
                                   anchor=tk.W, fill=text_m, font=("Segoe UI", 8), tags=("card_elem", "roi_dims_tag"))

        self.bg_canvas.coords(self.win_drag, pad + 14, c3_y + 56)
        self.bg_canvas.itemconfigure(self.win_drag, width=225, height=28)

        self.bg_canvas.coords(self.win_find, pad + 249, c3_y + 56)
        self.bg_canvas.itemconfigure(self.win_find, width=225, height=28)

        self.bg_canvas.create_text(pad + 14, c3_y + 102, text="Пресеты:",
                                   anchor=tk.W, fill=text_m, font=("Segoe UI", 8), tags="card_elem")
        for i, (btn, win_id) in enumerate(self.preset_btns):
            self.bg_canvas.coords(win_id, pad + 75 + i * 100, c3_y + 92)
            self.bg_canvas.itemconfigure(win_id, width=95, height=24)

        for idx, (lbl, sb, win_id) in enumerate(self.spinboxes):
            x_col = pad + 14 + idx * 118
            self.bg_canvas.create_text(x_col, c3_y + 138, text=lbl, anchor=tk.W, fill=text_p, font=("Segoe UI", 8), tags="card_elem")
            self.bg_canvas.coords(win_id, x_col + 50, c3_y + 126)
            self.bg_canvas.itemconfigure(win_id, width=62, height=24)

        self.bg_canvas.coords(self.win_save_roi, pad + 14, c3_y + 164)
        self.bg_canvas.itemconfigure(self.win_save_roi, width=460, height=28)

        # ======================================================================
        # CARD 4: Live Preview & Crosshair (Transparent when idle!)
        # ======================================================================
        c4_y = y_start
        self.bg_canvas.create_rectangle(col_right_x, c4_y, col_right_x + col_right_w, c4_y + h_preview,
                                        outline=b_color, width=1, tags="card_elem")
        self.bg_canvas.create_text(col_right_x + 14, c4_y + 16, text="🖥 Live Preview & Прицел",
                                   anchor=tk.W, fill=accent, font=("Segoe UI", 10, "bold"), tags="card_elem")
        self.bg_canvas.create_text(col_right_x + col_right_w - 14, c4_y + 16, text="DirectX DXGI · 60 FPS",
                                   anchor=tk.E, fill=text_m, font=("Segoe UI", 8), tags="card_elem")

        # Preview outline box (Inside Card 4)
        self.prev_x = col_right_x + 12
        self.prev_y = c4_y + 36
        self.prev_w = col_right_w - 24
        self.prev_h = h_preview - 48
        self.bg_canvas.create_rectangle(self.prev_x, self.prev_y, self.prev_x + self.prev_w, self.prev_y + self.prev_h,
                                        outline=b_color, width=1, tags="card_elem")

        if self.live is None:
            self.bg_canvas.create_text(
                self.prev_x + self.prev_w // 2, self.prev_y + self.prev_h // 2,
                text="Запустите инференс для отображения живого видеопотока.",
                fill=text_m, font=("Segoe UI", 11), justify=tk.CENTER, tags=("card_elem", "preview_idle_tag")
            )

        # ======================================================================
        # CARD 5: Telemetry (100% Transparent, No Solid Black Boxes!)
        # ======================================================================
        c5_y = c4_y + h_preview + gap
        self.bg_canvas.create_rectangle(col_right_x, c5_y, col_right_x + col_right_w, c5_y + h_telem,
                                        outline=b_color, width=1, tags="card_elem")
        self.bg_canvas.create_text(col_right_x + 14, c5_y + 16, text="📊 Телеметрия инференса",
                                   anchor=tk.W, fill=accent, font=("Segoe UI", 10, "bold"), tags="card_elem")

        cell_gap = 8
        cell_w = max(80, (col_right_w - 24 - 3 * cell_gap) // 4)
        telem_titles = ["Inference FPS", "Capture FPS", "Model Latency", "TTH (До ноты)"]
        telem_vars = [self.metric_fps, self.metric_cap_fps, self.metric_latency, self.metric_tth]

        for i, (title, var) in enumerate(zip(telem_titles, telem_vars)):
            cx = col_right_x + 12 + i * (cell_w + cell_gap)
            cy = c5_y + 34
            cw = cell_w
            ch = h_telem - 44

            # Transparent Cell Outline (NO solid background fill!)
            self.bg_canvas.create_rectangle(cx, cy, cx + cw, cy + ch, outline=b_color, width=1, tags="card_elem")
            self.bg_canvas.create_text(cx + 10, cy + 16, text=title, anchor=tk.W, fill=text_m, font=("Segoe UI", 8), tags="card_elem")
            self.bg_canvas.create_text(cx + 10, cy + 42, text=var.get(), anchor=tk.W, fill=accent, font=("Segoe UI", 12, "bold"),
                                       tags=("card_elem", f"telem_val_{i}"))

    # --------------------------------------------------------------------------
    # Custom Transparent Canvas Controls (Checkbox & Sliders)
    # --------------------------------------------------------------------------
    def _draw_clicks_checkbox(self):
        self.bg_canvas.delete("chk_tag")
        x1, y1, x2, y2 = self.chk_box_rect
        # Checkbox box outline (Transparent inside!)
        self.bg_canvas.create_rectangle(x1, y1, x2, y2, outline=self.theme["border_bright"], width=1, tags=("card_elem", "chk_tag"))
        if self.clicks_var.get():
            self.bg_canvas.create_text((x1 + x2) // 2, (y1 + y2) // 2, text="✓",
                                       fill=self.theme["accent"], font=("Segoe UI", 10, "bold"), tags=("card_elem", "chk_tag"))
        self.bg_canvas.create_text(x2 + 8, (y1 + y2) // 2, text="Клики Z/X (автонажатие в ритм)",
                                   anchor=tk.W, fill=self.theme["text_primary"], font=("Segoe UI", 9), tags=("card_elem", "chk_tag"))

    def _draw_conf_slider(self):
        self.bg_canvas.delete("conf_slider_tag")
        x_min, x_max, y = self.conf_slider_coords
        val = self.conf_var.get()
        # Track line
        self.bg_canvas.create_line(x_min, y, x_max, y, fill="#4a4a59", width=4, tags=("card_elem", "conf_slider_tag"))
        # Active fill line
        cur_x = x_min + (x_max - x_min) * (val - 0.10) / (1.00 - 0.10)
        self.bg_canvas.create_line(x_min, y, cur_x, y, fill=self.theme["accent"], width=4, tags=("card_elem", "conf_slider_tag"))
        # Thumb circle
        self.bg_canvas.create_oval(cur_x - 6, y - 6, cur_x + 6, y + 6, fill="#ffffff", outline=self.theme["accent"],
                                   width=2, tags=("card_elem", "conf_slider_tag"))
        # Numeric value readout
        self.bg_canvas.create_text(x_max + 20, y, text=f"{val:.2f}", fill=self.theme["accent"],
                                   font=("Segoe UI", 9, "bold"), anchor=tk.W, tags=("card_elem", "conf_slider_tag"))

    def _draw_smooth_slider(self):
        self.bg_canvas.delete("smooth_slider_tag")
        x_min, x_max, y = self.smooth_slider_coords
        val = self.smooth_var.get()
        # Track line
        self.bg_canvas.create_line(x_min, y, x_max, y, fill="#4a4a59", width=4, tags=("card_elem", "smooth_slider_tag"))
        # Active fill line
        cur_x = x_min + (x_max - x_min) * (val - 0.00) / (0.50 - 0.00)
        self.bg_canvas.create_line(x_min, y, cur_x, y, fill=self.theme["accent"], width=4, tags=("card_elem", "smooth_slider_tag"))
        # Thumb circle
        self.bg_canvas.create_oval(cur_x - 6, y - 6, cur_x + 6, y + 6, fill="#ffffff", outline=self.theme["accent"],
                                   width=2, tags=("card_elem", "smooth_slider_tag"))
        # Numeric value readout
        self.bg_canvas.create_text(x_max + 20, y, text=f"{val:.2f}", fill=self.theme["accent"],
                                   font=("Segoe UI", 9, "bold"), anchor=tk.W, tags=("card_elem", "smooth_slider_tag"))

    def _on_canvas_click(self, event):
        # 1. Checkbox click detection
        x1, y1, x2, y2 = self.chk_box_rect
        if x1 - 4 <= event.x <= x2 + 220 and y1 - 4 <= event.y <= y2 + 4:
            self.clicks_var.set(not self.clicks_var.get())
            self._on_clicks_toggle()
            self._draw_clicks_checkbox()
            return

        # 2. Confidence Slider click
        cx1, cx2, cy = self.conf_slider_coords
        if cx1 - 10 <= event.x <= cx2 + 10 and cy - 12 <= event.y <= cy + 12:
            self._handle_conf_slider(event.x)
            return

        # 3. Smoothness Slider click
        sx1, sx2, sy = self.smooth_slider_coords
        if sx1 - 10 <= event.x <= sx2 + 10 and sy - 12 <= event.y <= sy + 12:
            self._handle_smooth_slider(event.x)
            return

    def _on_canvas_drag(self, event):
        cx1, cx2, cy = self.conf_slider_coords
        if cy - 16 <= event.y <= cy + 16 and cx1 - 20 <= event.x <= cx2 + 20:
            self._handle_conf_slider(event.x)
            return

        sx1, sx2, sy = self.smooth_slider_coords
        if sy - 16 <= event.y <= sy + 16 and sx1 - 20 <= event.x <= sx2 + 20:
            self._handle_smooth_slider(event.x)
            return

    def _handle_conf_slider(self, mouse_x: int):
        x1, x2, _ = self.conf_slider_coords
        clamped = max(x1, min(x2, mouse_x))
        fraction = (clamped - x1) / (x2 - x1)
        val = round(0.10 + fraction * 0.90, 2)
        self.conf_var.set(val)
        self._on_conf_changed(val)
        self._draw_conf_slider()

    def _handle_smooth_slider(self, mouse_x: int):
        x1, x2, _ = self.smooth_slider_coords
        clamped = max(x1, min(x2, mouse_x))
        fraction = (clamped - x1) / (x2 - x1)
        val = round(0.00 + fraction * 0.50, 2)
        self.smooth_var.set(val)
        self._on_smooth_changed(val)
        self._draw_smooth_slider()

    # --------------------------------------------------------------------------
    # Model Catalog Methods
    # --------------------------------------------------------------------------
    def _scan_models(self) -> dict[str, Path]:
        res: dict[str, Path] = {}
        for candidate_dir in [APP_DIR / "data" / "models", APP_DIR / "models"]:
            if candidate_dir.is_dir():
                for p in sorted(candidate_dir.glob("*.pt"), key=lambda x: x.name):
                    if p.name not in res:
                        res[p.name] = p
        return res

    def _refresh_models(self):
        models = self._scan_models()
        keys = list(models.keys())
        self.model_dropdown.set_values(keys)

    def _on_model_selected(self, label: str):
        self.model_var.set(label)
        self._save_config()

    def _browse_model(self):
        path = filedialog.askopenfilename(
            title="Выбрать модель PyTorch (.pt)",
            filetypes=[("PyTorch Weights", "*.pt *.pth"), ("Все файлы", "*.*")]
        )
        if path:
            p = Path(path)
            label = p.name
            self.model_dropdown.set_values([label] + [v for v in self.model_dropdown.values if v != label])
            self.model_dropdown.set_value(label)
            self._save_config()


    def _get_selected_model_path(self) -> Optional[Path]:
        selected_label = self.model_dropdown.get_value() if hasattr(self, "model_dropdown") else ""
        models = self._scan_models()
        if not models:
            return None
        if selected_label in models:
            return models[selected_label]
        for label, path in models.items():
            if path.name in selected_label or label.split()[0] in selected_label:
                return path
        first_label = next(iter(models))
        if hasattr(self, "model_dropdown"):
            self.model_dropdown.set_value(first_label)
        return models[first_label]

    # --------------------------------------------------------------------------
    # Inference Callbacks
    # --------------------------------------------------------------------------
    def _on_clicks_toggle(self):
        if self.live is not None:
            self.live.set_clicks(self.clicks_var.get())
        self._save_config()

    def _on_conf_changed(self, val):
        if self.live is not None:
            self.live.set_confidence(float(val))
        self._save_config()

    def _on_smooth_changed(self, val):
        if self.live is not None:
            self.live.set_smooth(float(val))
        self._save_config()

    # --------------------------------------------------------------------------
    # ROI Methods
    # --------------------------------------------------------------------------
    def _update_roi_dims_label(self):
        w = max(0, self.roi_right.get() - self.roi_left.get())
        h = max(0, self.roi_bottom.get() - self.roi_top.get())
        dims_str = f"Размер области: {w} × {h} px"
        self.roi_dims_var.set(dims_str)
        if hasattr(self, "bg_canvas"):
            self.bg_canvas.itemconfig("roi_dims_tag", text=dims_str)

    def _on_spin_change(self):
        self._update_roi_dims_label()
        if self.live is not None and self.live.capture is not None:
            self.live.update_roi(
                self.roi_left.get(), self.roi_top.get(),
                self.roi_right.get(), self.roi_bottom.get()
            )
        self._save_config()

    def _apply_roi_coords(self, l, t, r, b):
        self.roi_left.set(int(l))
        self.roi_top.set(int(t))
        self.roi_right.set(int(r))
        self.roi_bottom.set(int(b))
        self._on_spin_change()

    def _launch_drag_select(self):
        DragSelectOverlay(self, self._apply_roi_coords, self.theme)

    def _auto_find_osu_window(self):
        if not HAS_WIN32:
            messagebox.showwarning("Win32", "Библиотека pywin32 не найдена.")
            return

        candidates = []

        def enum_cb(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if "osu!" in title:
                    candidates.append((hwnd, title))
            return True

        win32gui.EnumWindows(enum_cb, None)
        if not candidates:
            messagebox.showinfo("Поиск osu!", "Окно osu! не найдено. Запустите игру или выберите область вручную.")
            return

        target_hwnd = candidates[0][0]
        client_rect = win32gui.GetClientRect(target_hwnd)
        p1 = win32gui.ClientToScreen(target_hwnd, (client_rect[0], client_rect[1]))
        p2 = win32gui.ClientToScreen(target_hwnd, (client_rect[2], client_rect[3]))
        l, t, r, b = p1[0], p1[1], p2[0], p2[1]
        self._apply_roi_coords(l, t, r, b)
        messagebox.showinfo("Окно найдено", f"Область osu! определена:\nLeft={l}, Top={t}, Right={r}, Bottom={b}")

    def _save_roi_clicked(self):
        self._save_config()
        if self.live is not None:
            self.live.update_roi(
                self.roi_left.get(), self.roi_top.get(),
                self.roi_right.get(), self.roi_bottom.get()
            )
        messagebox.showinfo("Сохранено", "Область захвата (ROI) успешно сохранена в config.yaml!")

    # --------------------------------------------------------------------------
    # Themes & Wallpaper Management
    # --------------------------------------------------------------------------
    def _open_theme_menu(self):
        m = tk.Menu(self, tearoff=0, bg=self.theme["card_inner"], fg=self.theme["text_primary"],
                    activebackground=self.theme["card_hover"], activeforeground=self.theme["accent"])
        for k, v in THEMES.items():
            mark = "✓ " if k == self.theme_key else "   "
            m.add_command(label=f"{mark}{v['name']}", command=lambda key=k: self.set_theme(key))
        x = self.btn_theme.winfo_rootx()
        y = self.btn_theme.winfo_rooty() + self.btn_theme.winfo_height() + 2
        m.post(x, y)

    def set_theme(self, key: str):
        if key not in THEMES:
            return
        self.theme_key = key
        self.theme = THEMES[key]
        self._save_config()

        self.btn_theme.config(text=f"🎨 {self.theme['name']}")
        self.banner_canvas.apply_theme(self.theme)
        self.model_dropdown.apply_theme(self.theme)
        self._update_layout()

    def _open_wallpaper_dialog(self):
        m = tk.Menu(self, tearoff=0, bg=self.theme["card_inner"], fg=self.theme["text_primary"],
                    activebackground=self.theme["card_hover"], activeforeground=self.theme["accent"])
        m.add_command(label="🖼 Выбрать основной фон (Окно)...", command=self._choose_main_wallpaper)
        if self.wallpaper_path:
            m.add_command(label="✕ Сбросить основной фон", command=self._clear_main_wallpaper)
        m.add_separator()
        m.add_command(label="👑 Выбрать фон для шапки (SEMGA & S)...", command=self._choose_header_wallpaper)
        if self.wallpaper_path:
            m.add_command(label="🔄 Использовать основной фон в шапке", command=self._duplicate_main_to_header)
        if self.header_wallpaper_path:
            m.add_command(label="✕ Сбросить фон шапки", command=self._clear_header_wallpaper)
        m.add_separator()
        m.add_command(label="👁 Настроить прозрачность и яркость...", command=self._open_transparency_dialog)

        x = self.btn_wall.winfo_rootx()
        y = self.btn_wall.winfo_rooty() + self.btn_wall.winfo_height() + 2
        m.post(x, y)

    def _choose_main_wallpaper(self):
        chosen = filedialog.askopenfilename(
            title="Выбрать основное фоновое изображение",
            filetypes=[("Изображения", "*.png *.jpg *.jpeg *.webp *.bmp"), ("Все файлы", "*.*")]
        )
        if chosen:
            self.wallpaper_path = chosen
            self._save_config()
            self._load_wallpaper()

    def _clear_main_wallpaper(self):
        self.wallpaper_path = ""
        self.wallpaper_image = None
        self.wallpaper_photo = None
        self.bg_canvas.delete("bg_wall")
        self.bg_canvas.configure(bg=self.theme["bg"])
        self.banner_canvas.set_main_wallpaper_slice(None)
        self._save_config()

    def _choose_header_wallpaper(self):
        chosen = filedialog.askopenfilename(
            title="Выбрать фоновое изображение для шапки (SEMGA)",
            filetypes=[("Изображения", "*.png *.jpg *.jpeg *.webp *.bmp"), ("Все файлы", "*.*")]
        )
        if chosen:
            self.header_wallpaper_path = chosen
            self.banner_canvas.set_wallpaper(chosen)
            self._save_config()

    def _duplicate_main_to_header(self):
        if self.wallpaper_path:
            self.header_wallpaper_path = self.wallpaper_path
            self.banner_canvas.set_wallpaper(self.header_wallpaper_path)
            self._save_config()

    def _clear_header_wallpaper(self):
        self.header_wallpaper_path = ""
        self.banner_canvas.set_wallpaper("")
        if self.wallpaper_image:
            self.banner_canvas.set_main_wallpaper_slice(self.wallpaper_image)
        self._save_config()

    def _load_wallpaper(self):
        if not self.wallpaper_path:
            self.bg_canvas.delete("bg_wall")
            self.bg_canvas.configure(bg=self.theme["bg"])
            self.banner_canvas.set_main_wallpaper_slice(None)
            return
        p = Path(self.wallpaper_path)
        if not p.exists():
            return
        try:
            self.wallpaper_image = Image.open(p).convert("RGB")
            self._render_wallpaper()
            # If no dedicated header wallpaper, flow main wallpaper into banner
            if not self.header_wallpaper_path:
                self.banner_canvas.set_main_wallpaper_slice(self.wallpaper_image)
        except Exception:
            pass

    def _render_wallpaper(self):
        if self.wallpaper_image is None:
            return
        w = max(200, self.winfo_width() or 1240)
        h = max(200, self.winfo_height() or 820)
        resized = self.wallpaper_image.resize((w, h), Image.Resampling.LANCZOS)
        enhancer = ImageEnhance.Brightness(resized)
        darkened = enhancer.enhance(self.wallpaper_brightness)
        self.wallpaper_photo = ImageTk.PhotoImage(darkened)

        if self.bg_canvas.find_withtag("bg_wall"):
            self.bg_canvas.itemconfig("bg_wall", image=self.wallpaper_photo)
        else:
            self.bg_canvas.create_image(0, 0, image=self.wallpaper_photo, anchor=tk.NW, tags="bg_wall")
            self.bg_canvas.tag_lower("bg_wall")

    def _open_transparency_dialog(self):
        dlg = tk.Toplevel(self)
        dlg.title("Настройка прозрачности и яркости")
        dlg.geometry("400x200")
        dlg.resizable(False, False)
        dlg.configure(bg=self.theme["card_inner"], padx=16, pady=14)
        dlg.attributes("-topmost", True)

        # 1. Window Alpha
        tk.Label(dlg, text="1. Прозрачность окна (Window Alpha):", bg=self.theme["card_inner"],
                 fg=self.theme["text_primary"], font=("Segoe UI", 9, "bold")).pack(anchor=tk.W)

        val_w = tk.DoubleVar(value=self.window_opacity)
        lbl_w = tk.Label(dlg, text=f"{int(self.window_opacity * 100)}%", bg=self.theme["card_inner"],
                         fg=self.theme["accent"], font=("Segoe UI", 9, "bold"))
        lbl_w.pack(anchor=tk.E)

        def on_window_alpha(v):
            val = float(v)
            lbl_w.config(text=f"{int(val * 100)}%")
            self.window_opacity = val
            self.attributes("-alpha", val)
            self._save_config()

        sc_w = tk.Scale(
            dlg, from_=0.60, to=1.00, resolution=0.05, orient=tk.HORIZONTAL,
            variable=val_w, showvalue=0, bg=self.theme["card_inner"], fg=self.theme["accent"],
            troughcolor=self.theme["bg"], highlightthickness=0, command=on_window_alpha
        )
        sc_w.pack(fill=tk.X, pady=(0, 10))

        # 2. Wallpaper Brightness
        tk.Label(dlg, text="2. Яркость фонового изображения:", bg=self.theme["card_inner"],
                 fg=self.theme["text_primary"], font=("Segoe UI", 9, "bold")).pack(anchor=tk.W)

        val_b = tk.DoubleVar(value=self.wallpaper_brightness)
        lbl_b = tk.Label(dlg, text=f"{int(self.wallpaper_brightness * 100)}%", bg=self.theme["card_inner"],
                         fg=self.theme["accent"], font=("Segoe UI", 9, "bold"))
        lbl_b.pack(anchor=tk.E)

        def on_wall_bright(v):
            val = float(v)
            lbl_b.config(text=f"{int(val * 100)}%")
            self.wallpaper_brightness = val
            self._render_wallpaper()
            self._save_config()

        sc_b = tk.Scale(
            dlg, from_=0.30, to=1.00, resolution=0.05, orient=tk.HORIZONTAL,
            variable=val_b, showvalue=0, bg=self.theme["card_inner"], fg=self.theme["accent"],
            troughcolor=self.theme["bg"], highlightthickness=0, command=on_wall_bright
        )
        sc_b.pack(fill=tk.X, pady=(0, 12))

        tk.Button(
            dlg, text="Готово", bg=self.theme["btn_primary"], fg="#ffffff",
            activebackground=self.theme["btn_primary_hover"], activeforeground="#ffffff",
            font=("Segoe UI", 9, "bold"), relief=tk.SOLID, bd=1, padx=14, pady=3,
            command=dlg.destroy, cursor="hand2"
        ).pack(anchor=tk.E)

    def _on_window_resize(self, event):
        if event.widget == self:
            self._render_wallpaper()
            self._update_layout()
            if not self.header_wallpaper_path and self.wallpaper_image:
                self.banner_canvas.set_main_wallpaper_slice(self.wallpaper_image)

    # --------------------------------------------------------------------------
    # Inference Lifecycle
    # --------------------------------------------------------------------------
    def start_inference(self):
        if self.live is not None and self.live.running:
            return

        model_path = self._get_selected_model_path()
        if not model_path or not model_path.exists():
            messagebox.showerror("Ошибка модели", "В папке models/ не найдено ни одного файла модели (.pt).")
            return

        cfg_runtime = {
            "dxgi": self.cfg.get("dxgi", {"monitor": 1}),
            "playfield": {
                "left": int(self.roi_left.get()),
                "top": int(self.roi_top.get()),
                "right": int(self.roi_right.get()),
                "bottom": int(self.roi_bottom.get()),
            },
            "runtime": {"target_fps": 60}
        }

        try:
            self.live = RelaxLive(
                config_dict=cfg_runtime,
                weights_path=str(model_path),
                messages_queue=self.messages_queue,
                frames_queue=self.frames_queue,
                clicks_enabled=self.clicks_var.get(),
                smooth=self.smooth_var.get(),
                confidence_thresh=self.conf_var.get()
            )
            self.live.start()
            self.status_badge_var.set("● АКТИВЕН")
            self.lbl_badge.configure(bg="#1b5e20", fg="#00e676")
            self.btn_start.configure(state=tk.DISABLED)
            self.btn_stop.configure(state=tk.NORMAL)
            self.bg_canvas.delete("preview_idle_tag")
        except Exception as e:
            messagebox.showerror("Ошибка запуска", f"Не удалось инициализировать RelaxLive:\n{e}")
            self.live = None

    def stop_inference(self):
        if self.live is not None:
            try:
                self.live.stop()
            except Exception:
                pass
            self.live = None

        self.status_badge_var.set("● ОЖИДАНИЕ")
        self.lbl_badge.configure(bg=self.theme["badge_bg"], fg=self.theme["badge_fg"])
        self.btn_start.configure(state=tk.NORMAL)
        self.btn_stop.configure(state=tk.DISABLED)

        # Clear video stream image from canvas and restore idle text
        self.bg_canvas.delete("preview_stream")
        self.preview_photo = None
        self.bg_canvas.create_text(
            self.prev_x + self.prev_w // 2, self.prev_y + self.prev_h // 2,
            text="Запустите инференс для отображения живого видеопотока.",
            fill=self.theme["text_muted"], font=("Segoe UI", 11), justify=tk.CENTER,
            tags=("card_elem", "preview_idle_tag")
        )

        self.metric_fps.set("0.0 FPS")
        self.metric_cap_fps.set("0.0 FPS")
        self.metric_latency.set("0.0 мс")
        self.metric_tth.set("-- мс")
        self.bg_canvas.itemconfig("telem_val_0", text="0.0 FPS")
        self.bg_canvas.itemconfig("telem_val_1", text="0.0 FPS")
        self.bg_canvas.itemconfig("telem_val_2", text="0.0 мс")
        self.bg_canvas.itemconfig("telem_val_3", text="-- мс")

    def _pump(self):
        # Process telemetry messages
        while True:
            try:
                msg = self.messages_queue.get_nowait()
                m_type = msg.get("type")
                if m_type == "telemetry":
                    fps_str = f"{msg.get('fps', 0.0):.1f} FPS"
                    cap_str = f"{msg.get('capture_fps', 0.0):.1f} FPS"
                    lat_str = f"{msg.get('latency_ms', 0.0):.1f} мс"
                    tth = msg.get('tth_ms', -1)
                    tth_str = f"{tth:.0f} мс" if tth >= 0 else "-- мс"

                    self.metric_fps.set(fps_str)
                    self.metric_cap_fps.set(cap_str)
                    self.metric_latency.set(lat_str)
                    self.metric_tth.set(tth_str)

                    self.bg_canvas.itemconfig("telem_val_0", text=fps_str)
                    self.bg_canvas.itemconfig("telem_val_1", text=cap_str)
                    self.bg_canvas.itemconfig("telem_val_2", text=lat_str)
                    self.bg_canvas.itemconfig("telem_val_3", text=tth_str)
                elif m_type == "state":
                    is_active = msg.get("active", False)
                    if is_active:
                        self.status_badge_var.set("● АКТИВЕН")
                        self.lbl_badge.configure(bg="#1b5e20", fg="#00e676")
                    else:
                        self.status_badge_var.set("❚❚ ПАУЗА")
                        self.lbl_badge.configure(bg="#4a3b00", fg="#ffd700")
            except queue.Empty:
                break
            except Exception:
                break

        # Process preview video frame
        latest_frame = None
        while True:
            try:
                latest_frame = self.frames_queue.get_nowait()
            except queue.Empty:
                break

        if latest_frame is not None:
            try:
                cv_w = self.prev_w
                cv_h = self.prev_h
                if cv_w > 50 and cv_h > 50:
                    img_h, img_w = latest_frame.shape[:2]
                    scale = min(cv_w / img_w, cv_h / img_h)
                    nw = int(img_w * scale)
                    nh = int(img_h * scale)

                    resized = cv2.resize(latest_frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
                    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb)
                    self.preview_photo = ImageTk.PhotoImage(pil_img)

                    ox = self.prev_x + (cv_w - nw) // 2
                    oy = self.prev_y + (cv_h - nh) // 2

                    if self.bg_canvas.find_withtag("preview_stream"):
                        self.bg_canvas.itemconfig("preview_stream", image=self.preview_photo)
                        self.bg_canvas.coords("preview_stream", ox, oy)
                    else:
                        self.bg_canvas.delete("preview_idle_tag")
                        self.bg_canvas.create_image(ox, oy, image=self.preview_photo, anchor=tk.NW, tags="preview_stream")
            except Exception:
                pass

        self.after(30, self._pump)

    def on_close(self):
        self.stop_inference()
        self._save_config()
        self.destroy()


# ==============================================================================
# Entry Point
# ==============================================================================
if __name__ == "__main__":
    app = NormisApp()
    app.mainloop()
