from __future__ import annotations

import base64
import ctypes
from datetime import datetime
from io import BytesIO
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageTk
import yaml

ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = Path(r"C:\Selgis OSU AI Player")
if str(LEGACY_ROOT) not in sys.path:
    sys.path.insert(0, str(LEGACY_ROOT))

from ui.capture_pipeline import CapturePipeline
from ui.map_catalog import filter_maps, scan_lazer_maps
from osu2.live import RelaxLive
from osu2.danser_engine import DanserQueueManager, RenderItem, scan_danser_maps


# ==============================================================================
# Fullscreen Interactive Screen ROI Calibrator
# ==============================================================================
class ScreenRoiSelector:
    """Full-screen interactive calibrator for osu! playfield with desktop freeze and loupe."""

    def __init__(self, master: tk.Tk, current_roi: tuple[int, int, int, int], on_complete, monitor_idx: int = 1):
        self.master = master
        self.on_complete = on_complete
        self.monitor_idx = int(monitor_idx)

        self.screenshot: Image.Image | None = None
        self.dimmed_photo: ImageTk.PhotoImage | None = None
        self.bright_photo: ImageTk.PhotoImage | None = None
        self.loupe_photo: ImageTk.PhotoImage | None = None

        self.mon_left = 0
        self.mon_top = 0
        self.screen_w = master.winfo_screenwidth()
        self.screen_h = master.winfo_screenheight()

        self._capture_desktop()

        self.top = tk.Toplevel(master)
        self.top.attributes("-fullscreen", True)
        self.top.attributes("-topmost", True)
        self.top.geometry(f"{self.screen_w}x{self.screen_h}+{self.mon_left}+{self.mon_top}")
        self.top.configure(bg="#18181b", cursor="crosshair")

        if self.screenshot is None:
            self.top.attributes("-alpha", 0.45)

        self.canvas = tk.Canvas(
            self.top, width=self.screen_w, height=self.screen_h,
            highlightthickness=0, bg="#18181b", bd=0
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        if self.dimmed_photo is not None:
            self.canvas.create_image(0, 0, image=self.dimmed_photo, anchor=tk.NW, tags="bg_dimmed")

        l, t, r, b = current_roi
        self.left = max(0, min(self.screen_w - 60, int(l)))
        self.top_coord = max(0, min(self.screen_h - 60, int(t)))
        self.right = max(self.left + 50, min(self.screen_w, int(r)))
        self.bottom = max(self.top_coord + 50, min(self.screen_h, int(b)))

        self.drag_mode: str | None = None
        self.start_mouse = (0, 0)
        self.orig_roi = (self.left, self.top_coord, self.right, self.bottom)
        self.hover_handle: str | None = None

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Motion>", self._on_motion)

        self.top.bind("<Escape>", lambda e: self.top.destroy())
        self.top.bind("<Return>", lambda e: self._confirm())
        self.top.bind("<KP_Enter>", lambda e: self._confirm())
        self.top.bind("<Tab>", lambda e: self._snap_4_3())
        self.top.bind("<c>", lambda e: self._center_roi())
        self.top.bind("<C>", lambda e: self._center_roi())

        self.top.bind("<Left>", lambda e: self._nudge(-1, 0, e.state))
        self.top.bind("<Right>", lambda e: self._nudge(1, 0, e.state))
        self.top.bind("<Up>", lambda e: self._nudge(0, -1, e.state))
        self.top.bind("<Down>", lambda e: self._nudge(0, 1, e.state))

        self._redraw()

    def _capture_desktop(self):
        try:
            import mss
            try:
                hdesk = ctypes.windll.user32.OpenInputDesktop(0, False, 0x01FF)
                if hdesk:
                    ctypes.windll.user32.SetThreadDesktop(hdesk)
                    ctypes.windll.user32.CloseDesktop(hdesk)
            except Exception:
                pass

            with mss.MSS() as sct:
                idx = min(len(sct.monitors) - 1, max(1, self.monitor_idx))
                mon = sct.monitors[idx]
                self.mon_left = mon["left"]
                self.mon_top = mon["top"]
                self.screen_w = mon["width"]
                self.screen_h = mon["height"]
                raw = sct.grab(mon)
                img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
                self.screenshot = img
                enhancer = ImageEnhance.Brightness(img)
                dimmed = enhancer.enhance(0.38)
                self.dimmed_photo = ImageTk.PhotoImage(dimmed)
        except Exception:
            self.screenshot = None
            self.dimmed_photo = None

    def _get_handles(self) -> dict[str, tuple[int, int]]:
        l, t, r, b = self.left, self.top_coord, self.right, self.bottom
        mx = (l + r) // 2
        my = (t + b) // 2
        return {
            "nw": (l, t), "n": (mx, t), "ne": (r, t),
            "e": (r, my), "se": (r, b), "s": (mx, b),
            "sw": (l, b), "w": (l, my)
        }

    def _hit_handle(self, x: int, y: int) -> str | None:
        radius = 9
        for name, (hx, hy) in self._get_handles().items():
            if abs(x - hx) <= radius and abs(y - hy) <= radius:
                return name
        return None

    def _on_motion(self, event):
        x, y = event.x, event.y
        h = self._hit_handle(x, y)
        self.hover_handle = h

        cursor_map = {
            "nw": "size_nw_se", "se": "size_nw_se",
            "ne": "size_ne_sw", "sw": "size_ne_sw",
            "n": "size_ns", "s": "size_ns",
            "e": "size_we", "w": "size_we"
        }
        if h in cursor_map:
            self.canvas.config(cursor=cursor_map[h])
        elif self.left <= x <= self.right and self.top_coord <= y <= self.bottom:
            self.canvas.config(cursor="fleur")
        else:
            self.canvas.config(cursor="crosshair")

        self._draw_loupe(x, y)

    def _on_press(self, event):
        x, y = event.x, event.y
        self.start_mouse = (x, y)
        self.orig_roi = (self.left, self.top_coord, self.right, self.bottom)
        h = self._hit_handle(x, y)
        if h:
            self.drag_mode = h
        elif self.left <= x <= self.right and self.top_coord <= y <= self.bottom:
            self.drag_mode = "move"
        else:
            self.drag_mode = "draw_new"
            self.left, self.top_coord = x, y
            self.right, self.bottom = x, y
            self._redraw()

    def _on_drag(self, event):
        if not self.drag_mode:
            return
        x, y = event.x, event.y
        dx = x - self.start_mouse[0]
        dy = y - self.start_mouse[1]
        ol, ot, oright, ob = self.orig_roi

        if self.drag_mode == "move":
            w = oright - ol
            h = ob - ot
            nl = max(0, min(self.screen_w - w, ol + dx))
            nt = max(0, min(self.screen_h - h, ot + dy))
            self.left, self.right = nl, nl + w
            self.top_coord, self.bottom = nt, nt + h
        elif self.drag_mode == "draw_new":
            self.right = max(0, min(self.screen_w, x))
            self.bottom = max(0, min(self.screen_h, y))
        else:
            if "w" in self.drag_mode:
                self.left = max(0, min(oright - 40, ol + dx))
            if "e" in self.drag_mode:
                self.right = max(ol + 40, min(self.screen_w, oright + dx))
            if "n" in self.drag_mode:
                self.top_coord = max(0, min(ob - 40, ot + dy))
            if "s" in self.drag_mode:
                self.bottom = max(ot + 40, min(self.screen_h, ob + dy))

        self._redraw()
        self._draw_loupe(x, y)

    def _on_release(self, event):
        if self.left > self.right:
            self.left, self.right = self.right, self.left
        if self.top_coord > self.bottom:
            self.top_coord, self.bottom = self.bottom, self.top_coord
        if self.right - self.left < 40:
            self.right = self.left + 40
        if self.bottom - self.top_coord < 40:
            self.bottom = self.top_coord + 40
        self.drag_mode = None
        self._redraw()

    def _snap_4_3(self):
        cur_h = self.bottom - self.top_coord
        cx = (self.left + self.right) // 2
        cy = (self.top_coord + self.bottom) // 2
        new_w = round(cur_h * (4.0 / 3.0))
        if new_w > self.screen_w:
            new_w = self.screen_w
            cur_h = round(new_w * (3.0 / 4.0))
        self.left = max(0, cx - new_w // 2)
        self.right = min(self.screen_w, self.left + new_w)
        self.top_coord = max(0, cy - cur_h // 2)
        self.bottom = min(self.screen_h, self.top_coord + cur_h)
        self._redraw()

    def _center_roi(self):
        w = self.right - self.left
        h = self.bottom - self.top_coord
        self.left = max(0, (self.screen_w - w) // 2)
        self.right = self.left + w
        self.top_coord = max(0, (self.screen_h - h) // 2)
        self.bottom = self.top_coord + h
        self._redraw()

    def _nudge(self, dx: int, dy: int, state: int):
        step = 10 if (state & 0x0001) else 1
        w = self.right - self.left
        h = self.bottom - self.top_coord
        nl = max(0, min(self.screen_w - w, self.left + dx * step))
        nt = max(0, min(self.screen_h - h, self.top_coord + dy * step))
        self.left = nl
        self.right = nl + w
        self.top_coord = nt
        self.bottom = nt + h
        self._redraw()

    def _confirm(self):
        l = int(min(self.left, self.right))
        t = int(min(self.top_coord, self.bottom))
        r = int(max(self.left, self.right))
        b = int(max(self.top_coord, self.bottom))
        self.top.destroy()
        if self.on_complete:
            self.on_complete((l, t, r, b))

    def _redraw(self):
        self.canvas.delete("roi_layer")
        l, t = min(self.left, self.right), min(self.top_coord, self.bottom)
        r, b = max(self.left, self.right), max(self.top_coord, self.bottom)
        w, h = r - l, b - t

        # Render bright crop inside ROI
        if self.screenshot is not None and w > 0 and h > 0:
            try:
                crop = self.screenshot.crop((l, t, r, b))
                self.bright_photo = ImageTk.PhotoImage(crop)
                self.canvas.create_image(l, t, image=self.bright_photo, anchor=tk.NW, tags="roi_layer")
            except Exception:
                pass

        # Outer border and grid
        accent = "#d1d5db"
        self.canvas.create_rectangle(l, t, r, b, outline=accent, width=2, tags="roi_layer")
        self.canvas.create_line(l, (t + b) // 2, r, (t + b) // 2, fill="#0ea5e9", dash=(3, 5), tags="roi_layer")
        self.canvas.create_line((l + r) // 2, t, (l + r) // 2, b, fill="#0ea5e9", dash=(3, 5), tags="roi_layer")

        # 8 Resize handles
        for hx, hy in self._get_handles().values():
            self.canvas.create_rectangle(hx - 5, hy - 5, hx + 5, hy + 5, fill="#ffffff", outline=accent, width=2, tags="roi_layer")

        # Floating Bottom HUD
        self._draw_hud(w, h, l, t, r, b)

    def _draw_hud(self, w: int, h: int, l: int, t: int, r: int, b: int):
        ratio = (w / max(1, h))
        is_43 = abs(ratio - (4.0 / 3.0)) <= 0.02
        ratio_tag = "osu! 4:3 OK" if is_43 else f"{ratio:.3f}"

        hud_w, hud_h = 740, 52
        hx0 = (self.screen_w - hud_w) // 2
        hy0 = self.screen_h - 70
        hx1 = hx0 + hud_w
        hy1 = hy0 + hud_h

        self.canvas.create_rectangle(hx0, hy0, hx1, hy1, fill="#18181b", outline="#3f3f46", width=2, tags="roi_layer")
        self.canvas.create_rectangle(hx0 + 1, hy0 + 1, hx1 - 1, hy0 + 3, fill="#d1d5db", outline="", tags="roi_layer")

        coords_str = f"L: {l}   T: {t}   R: {r}   B: {b}   |   {w}×{h} px   |   Пропорция: {ratio_tag}"
        self.canvas.create_text(hx0 + 20, hy0 + 17, text=coords_str, anchor=tk.W, fill="#f8fafc", font=("Consolas", 10, "bold"), tags="roi_layer")

        # Action button badges
        buttons = [
            ("Применить (Enter)", "#059669", self._confirm),
            ("Выровнять 4:3 (Tab)", "#0284c7", self._snap_4_3),
            ("Центр (C)", "#475569", self._center_roi),
            ("Отмена (Esc)", "#dc2626", lambda: self.top.destroy()),
        ]
        bx = hx0 + 20
        by = hy0 + 36
        for title, col, cmd in buttons:
            tag_name = f"btn_{title[:3]}"
            tid = self.canvas.create_text(bx, by, text=f"[{title}]", anchor=tk.W, fill=col, font=("Segoe UI", 9, "bold"), tags=("roi_layer", tag_name))
            self.canvas.tag_bind(tag_name, "<Button-1>", lambda e, c=cmd: c())
            bbox = self.canvas.bbox(tid)
            bx = bbox[2] + 16 if bbox else bx + 120

    def _draw_loupe(self, x: int, y: int):
        self.canvas.delete("loupe_layer")
        if self.screenshot is None:
            return

        lx = x + 24
        ly = y + 24
        if lx + 120 > self.screen_w:
            lx = x - 144
        if ly + 120 > self.screen_h:
            ly = y - 144

        r = 10
        x0 = max(0, min(self.screen_w - 2 * r, x - r))
        y0 = max(0, min(self.screen_h - 2 * r, y - r))
        try:
            chip = self.screenshot.crop((x0, y0, x0 + 2 * r, y0 + 2 * r))
            chip = chip.resize((120, 120), Image.Resampling.NEAREST)
            self.loupe_photo = ImageTk.PhotoImage(chip)
            self.canvas.create_image(lx, ly, image=self.loupe_photo, anchor=tk.NW, tags="loupe_layer")
            self.canvas.create_rectangle(lx, ly, lx + 120, ly + 120, outline="#d1d5db", width=2, tags="loupe_layer")
            self.canvas.create_line(lx + 60, ly + 50, lx + 60, ly + 70, fill="#f43f5e", width=2, tags="loupe_layer")
            self.canvas.create_line(lx + 50, ly + 60, lx + 70, ly + 60, fill="#f43f5e", width=2, tags="loupe_layer")

            px = self.screenshot.getpixel((min(self.screen_w - 1, max(0, x)), min(self.screen_h - 1, max(0, y))))
            col_hex = f"#{px[0]:02x}{px[1]:02x}{px[2]:02x}"
            info = f"X:{x} Y:{y} {col_hex}"
            self.canvas.create_rectangle(lx, ly + 122, lx + 120, ly + 140, fill="#0b0f19", outline="", tags="loupe_layer")
            self.canvas.create_text(lx + 60, ly + 131, text=info, fill="#ffffff", font=("Consolas", 8), tags="loupe_layer")
        except Exception:
            pass


# ==============================================================================
# Main Recording and Live App
# ==============================================================================
class RecordingApp(tk.Tk):
    # Neutral Dark-Gray Palette
    THEME_BG = "#1c1c1f"          # Neutral Dark Gray 900
    THEME_CARD = "#28282c"        # Neutral Dark Card 800
    THEME_CARD_BORDER = "#3f3f46" # Neutral Border 700
    THEME_FG = "#f3f4f6"          # Neutral Light Text 50
    THEME_FG_MUTED = "#9ca3af"    # Neutral Muted Text 400
    THEME_ACCENT = "#d1d5db"      # Neutral Accent 300
    THEME_ACCENT_HOVER = "#e5e7eb"# Neutral Accent Hover 200
    THEME_ACTIVE = "#71717a"      # Neutral Active Gray 500
    THEME_SUCCESS = "#22c55e"     # Green 500
    THEME_ERROR = "#ef4444"       # Red 500

    def __init__(self):
        super().__init__()
        self.title("osu! policy — запись карт и управление")
        self.geometry("1180x800")
        self.minsize(940, 640)
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.configure(bg=self.THEME_BG)

        self.config_path = ROOT / "config.yaml"
        self.legacy_config_path = LEGACY_ROOT / "configs" / "default.yaml"

        self.messages = queue.Queue()
        self.frames = queue.Queue(maxsize=1)
        self.pipeline = None
        self.live: RelaxLive | None = None
        self.catalog, self.visible_maps, self.selected_map = [], [], None

        self.output_root = tk.StringVar(value=str(LEGACY_ROOT / "data" / "recordings"))
        self.session_name = tk.StringVar(value=self._default_session_name())
        self.source_var = tk.StringVar(value="dxgi")
        self.search_var = tk.StringVar()
        self.models_catalog: dict[str, Path] = {}
        self.model_var = tk.StringVar()
        self.model_info_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Выбери карту слева, затем подготовь запись.")
        self.clicks_var = tk.BooleanVar(value=True)
        self.conf_var = tk.DoubleVar(value=0.60)

        # ROI state
        self.roi_left = 295
        self.roi_top = 35
        self.roi_right = 1625
        self.roi_bottom = 1075
        self.roi_display_var = tk.StringVar()

        # FPS regulation state
        self.target_fps_var = tk.IntVar(value=60)
        self.fps_label_var = tk.StringVar(value="60 FPS")

        self.metric_vars = {key: tk.StringVar(value="—") for key in (
            "capture", "inference", "detect", "policy", "reaction", "objects", "timing", "record")}

        # Dynamic scalable preview state
        self.preview_image = None
        self._latest_raw_image: Image.Image | None = None
        self._resize_timer = None
        self._preview_container_w = 640
        self._preview_container_h = 360

        # Danser Synthetic Pipeline State
        self.danser_songs_dir = r"C:\Users\Game PC\AppData\Local\osu!\Songs"
        self.danser_output_dir_var = tk.StringVar(value=str(ROOT / "data" / "synthetic_renders"))
        self.danser_source_var = tk.StringVar(value="Все источники")
        self.danser_search_var = tk.StringVar()
        self.danser_fps_var = tk.StringVar(value="60")
        self.danser_encoder_var = tk.StringVar(value="libx264 ultrafast (~180 FPS)")
        self.danser_auto_dataset_var = tk.BooleanVar(value=True)
        self.danser_catalog: list[dict] = []
        self.danser_visible_maps: list[dict] = []
        self.selected_danser_map: dict | None = None

        # Queue Progress & Telemetry variables
        self.render_map_progress_var = tk.DoubleVar(value=0.0)
        self.render_map_label_var = tk.StringVar(value="Очередь ожидает запуска")
        self.render_queue_progress_var = tk.DoubleVar(value=0.0)
        self.render_queue_label_var = tk.StringVar(value="0 карт в очереди")
        self.render_telemetry_var = tk.StringVar(value="Скорость: — | Прошло: 00:00 | Примерно осталось: —")

        self.danser_manager = DanserQueueManager(
            danser_dir=r"C:\Users\Game PC\Desktop\Danser OSU",
            output_dir=self.danser_output_dir_var.get(),
            on_progress=lambda stats: self.messages.put(("render_progress", stats)),
            on_queue_update=lambda queue_items: self.messages.put(("render_queue", queue_items)),
            on_log=lambda msg: self.messages.put(("render_log", msg)),
        )

        self._apply_dark_theme()
        self._load_config()
        self._update_roi_display()
        self._refresh_models()
        self._build()

        self.bind_all("<F4>", lambda e: self.open_roi_calibrator())
        self.after(50, self._pump)
        self.after(100, self.refresh_maps)
        self.after(200, self.refresh_danser_maps)

    def _apply_dark_theme(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        # Global Defaults
        style.configure(".", background=self.THEME_BG, foreground=self.THEME_FG, font=("Segoe UI", 9))
        style.configure("TFrame", background=self.THEME_BG)
        style.configure("Card.TFrame", background=self.THEME_CARD)

        # LabelFrame
        style.configure(
            "TLabelframe",
            background=self.THEME_CARD,
            foreground=self.THEME_ACCENT,
            relief="solid",
            borderwidth=1,
            bordercolor=self.THEME_CARD_BORDER
        )
        style.configure(
            "TLabelframe.Label",
            background=self.THEME_CARD,
            foreground=self.THEME_ACCENT,
            font=("Segoe UI", 9, "bold")
        )

        # Labels
        style.configure("TLabel", background=self.THEME_CARD, foreground=self.THEME_FG)
        style.configure("Muted.TLabel", background=self.THEME_CARD, foreground=self.THEME_FG_MUTED, font=("Segoe UI", 8))
        style.configure("Status.TLabel", background=self.THEME_BG, foreground=self.THEME_ACCENT, font=("Segoe UI", 10, "bold"))
        style.configure("Header.TLabel", background=self.THEME_CARD, foreground=self.THEME_FG, font=("Segoe UI", 11, "bold"))

        # Buttons
        style.configure(
            "TButton",
            background=self.THEME_CARD_BORDER,
            foreground=self.THEME_FG,
            relief="flat",
            borderwidth=0,
            padding=5,
            font=("Segoe UI", 9)
        )
        style.map(
            "TButton",
            background=[("pressed", self.THEME_CARD), ("active", "#475569")],
            foreground=[("pressed", self.THEME_ACCENT), ("active", "#ffffff")]
        )
        style.configure(
            "Accent.TButton",
            background=self.THEME_ACCENT_HOVER,
            foreground="#ffffff",
            font=("Segoe UI", 9, "bold")
        )
        style.map(
            "Accent.TButton",
            background=[("pressed", "#0369a1"), ("active", self.THEME_ACCENT)]
        )

        # Entries and Comboboxes
        style.configure(
            "TEntry",
            fieldbackground=self.THEME_BG,
            foreground=self.THEME_FG,
            insertcolor=self.THEME_ACCENT,
            bordercolor=self.THEME_CARD_BORDER,
            lightcolor=self.THEME_CARD_BORDER,
            darkcolor=self.THEME_CARD_BORDER,
            padding=4
        )
        style.configure(
            "TCombobox",
            fieldbackground=self.THEME_BG,
            foreground=self.THEME_FG,
            selectbackground=self.THEME_ACCENT_HOVER,
            selectforeground="#ffffff",
            arrowcolor=self.THEME_ACCENT,
            bordercolor=self.THEME_CARD_BORDER,
            padding=4
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", self.THEME_BG)],
            foreground=[("readonly", self.THEME_FG)]
        )

        # Spinbox
        style.configure(
            "TSpinbox",
            fieldbackground=self.THEME_BG,
            foreground=self.THEME_FG,
            arrowcolor=self.THEME_ACCENT,
            bordercolor=self.THEME_CARD_BORDER
        )

        # Checkbutton
        style.configure(
            "TCheckbutton",
            background=self.THEME_CARD,
            foreground=self.THEME_FG,
            focuscolor=""
        )

        # Scale
        style.configure(
            "TScale",
            background=self.THEME_CARD,
            troughcolor=self.THEME_BG
        )

        # Treeview (Map Catalog)
        style.configure(
            "Treeview",
            background=self.THEME_BG,
            fieldbackground=self.THEME_BG,
            foreground=self.THEME_FG,
            rowheight=25,
            borderwidth=0
        )
        style.map(
            "Treeview",
            background=[("selected", self.THEME_ACCENT_HOVER)],
            foreground=[("selected", "#ffffff")]
        )
        style.configure(
            "Treeview.Heading",
            background=self.THEME_CARD,
            foreground=self.THEME_ACCENT,
            relief="flat",
            font=("Segoe UI", 9, "bold")
        )
        style.map(
            "Treeview.Heading",
            background=[("active", self.THEME_CARD_BORDER)]
        )

        # Scrollbar
        style.configure(
            "Vertical.TScrollbar",
            background=self.THEME_CARD,
            troughcolor=self.THEME_BG,
            arrowcolor=self.THEME_ACCENT,
            relief="flat",
            borderwidth=0
        )

        # Panedwindow
        style.configure("Panedwindow", background=self.THEME_BG)

        # Notebook & Tabs
        style.configure("TNotebook", background=self.THEME_BG, borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            background=self.THEME_CARD,
            foreground=self.THEME_FG,
            padding=[14, 7],
            font=("Segoe UI", 9, "bold")
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", self.THEME_CARD_BORDER), ("active", "#334155")],
            foreground=[("selected", self.THEME_ACCENT), ("active", "#ffffff")]
        )

        # Progressbar
        style.configure(
            "Horizontal.TProgressbar",
            background=self.THEME_ACCENT,
            troughcolor=self.THEME_BG,
            bordercolor=self.THEME_CARD_BORDER,
            lightcolor=self.THEME_ACCENT_HOVER,
            darkcolor=self.THEME_ACCENT_HOVER
        )

    @staticmethod
    def _default_session_name():
        return datetime.now().strftime("session_%Y%m%d_%H%M%S")

    def _update_roi_display(self):
        w = self.roi_right - self.roi_left
        h = self.roi_bottom - self.roi_top
        ratio = w / max(1, h)
        self.roi_display_var.set(
            f"ROI: [{self.roi_left}, {self.roi_top}, {self.roi_right}, {self.roi_bottom}]  ({w}×{h} px, {ratio:.2f})"
        )

    def _load_config(self):
        cfg = {}
        if self.config_path.exists():
            try:
                cfg = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
            except Exception:
                cfg = {}

        legacy_cfg = {}
        if self.legacy_config_path.exists():
            try:
                legacy_cfg = yaml.safe_load(self.legacy_config_path.read_text(encoding="utf-8")) or {}
            except Exception:
                legacy_cfg = {}

        # 1. Playfield
        pf = cfg.get("playfield", legacy_cfg.get("playfield", {}))
        if pf:
            self.roi_left = int(pf.get("left", self.roi_left))
            self.roi_top = int(pf.get("top", self.roi_top))
            self.roi_right = int(pf.get("right", self.roi_right))
            self.roi_bottom = int(pf.get("bottom", self.roi_bottom))

        # 2. Target FPS
        runtime = cfg.get("runtime", {})
        target_fps = int(runtime.get("target_fps", runtime.get("inference_fps", 60)))
        self.target_fps_var.set(max(10, min(360, target_fps)))
        self.fps_label_var.set(f"{self.target_fps_var.get()} FPS")

        # 3. GUI settings
        gui_cfg = cfg.get("gui", {})
        if "clicks_enabled" in gui_cfg:
            self.clicks_var.set(bool(gui_cfg["clicks_enabled"]))
        if "confidence" in gui_cfg:
            self.conf_var.set(float(gui_cfg["confidence"]))
        if "source" in gui_cfg:
            self.source_var.set(str(gui_cfg["source"]))
        if "selected_model" in gui_cfg and gui_cfg["selected_model"]:
            self.model_var.set(str(gui_cfg["selected_model"]))

    def _save_config(self):
        try:
            cfg = {}
            if self.config_path.exists():
                try:
                    cfg = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
                except Exception:
                    cfg = {}

            cfg["playfield"] = {
                "left": int(self.roi_left),
                "top": int(self.roi_top),
                "right": int(self.roi_right),
                "bottom": int(self.roi_bottom)
            }
            if "runtime" not in cfg:
                cfg["runtime"] = {}
            cfg["runtime"]["target_fps"] = int(self.target_fps_var.get())
            cfg["gui"] = {
                "selected_model": self.model_var.get(),
                "clicks_enabled": bool(self.clicks_var.get()),
                "confidence": float(self.conf_var.get()),
                "source": self.source_var.get()
            }
            self.config_path.write_text(yaml.dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

            # Keep legacy default.yaml in sync for CaptureManager and older scripts
            if self.legacy_config_path.exists():
                try:
                    leg = yaml.safe_load(self.legacy_config_path.read_text(encoding="utf-8")) or {}
                    leg["playfield"] = dict(cfg["playfield"])
                    leg["mapping"]["osu_field"] = [
                        int(self.roi_left), int(self.roi_top),
                        int(self.roi_right), int(self.roi_bottom)
                    ]
                    self.legacy_config_path.write_text(yaml.dump(leg, sort_keys=False, allow_unicode=True), encoding="utf-8")
                except Exception:
                    pass
        except Exception as exc:
            print(f"Warning: could not save config: {exc}", file=sys.stderr)

    def open_roi_calibrator(self):
        cur_roi = (self.roi_left, self.roi_top, self.roi_right, self.roi_bottom)
        ScreenRoiSelector(self, cur_roi, self._on_roi_calibrated)

    def _on_roi_calibrated(self, roi: tuple[int, int, int, int]):
        self.roi_left, self.roi_top, self.roi_right, self.roi_bottom = roi
        self._update_roi_display()
        self._save_config()

        # Update running live session on the fly
        if self.live is not None and self.live.running:
            self.live.set_playfield({
                "left": self.roi_left, "top": self.roi_top,
                "right": self.roi_right, "bottom": self.roi_bottom
            })
        self.status_var.set(f"ROI откалиброван: [{self.roi_left}, {self.roi_top}, {self.roi_right}, {self.roi_bottom}]")

    def _on_fps_slide(self, val):
        fps = int(float(val))
        self.target_fps_var.set(fps)
        self.fps_label_var.set(f"{fps} FPS")
        if self.live is not None and self.live.running:
            self.live.set_target_fps(fps)
        self._save_config()

    def set_fps(self, fps: int):
        self.target_fps_var.set(fps)
        self.fps_label_var.set(f"{fps} FPS")
        if self.live is not None and self.live.running:
            self.live.set_target_fps(fps)
        self._save_config()

    def _scan_models(self) -> dict[str, Path]:
        res: dict[str, Path] = {}
        dirs = [
            ("Osu_2", ROOT / "data" / "models"),
            ("Legacy", LEGACY_ROOT / "data" / "policy"),
            ("LegacyModels", LEGACY_ROOT / "data" / "models"),
        ]
        for tag, folder in dirs:
            if folder.is_dir():
                for p in sorted(folder.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True):
                    size_mb = p.stat().st_size / (1024 * 1024)
                    label = f"[{tag}] {p.name} ({size_mb:.1f} MB)"
                    res[label] = p
        return res

    def _refresh_models(self):
        current = self.model_var.get()
        custom_paths = []
        if current in self.models_catalog:
            p = self.models_catalog[current]
            if not any(p == v for v in self._scan_models().values()):
                custom_paths.append(p)

        self.models_catalog = self._scan_models()
        for cp in custom_paths:
            if cp.exists():
                size_mb = cp.stat().st_size / (1024 * 1024)
                self.models_catalog[f"[Custom] {cp.name} ({size_mb:.1f} MB)"] = cp

        keys = list(self.models_catalog.keys())
        if hasattr(self, "model_combo"):
            self.model_combo.configure(values=keys)

        if keys:
            if current not in self.models_catalog:
                self.model_var.set(keys[0])
            self._update_model_info()
        else:
            self.model_var.set("")
            self.model_info_var.set("Модели не найдены")

    def _browse_model(self):
        chosen = filedialog.askopenfilename(
            title="Выбрать веса модели",
            filetypes=[("PyTorch Weights", "*.pt *.pth"), ("All files", "*.*")],
            initialdir=str(ROOT / "data" / "models") if (ROOT / "data" / "models").exists() else str(ROOT)
        )
        if chosen:
            p = Path(chosen)
            size_mb = p.stat().st_size / (1024 * 1024)
            label = f"[Custom] {p.name} ({size_mb:.1f} MB)"
            self.models_catalog[label] = p
            self.model_combo.configure(values=list(self.models_catalog.keys()))
            self.model_var.set(label)
            self._update_model_info()
            self._save_config()

    def _update_model_info(self, *args):
        key = self.model_var.get()
        p = self.models_catalog.get(key)
        if p and p.exists():
            dt = datetime.fromtimestamp(p.stat().st_mtime).strftime("%d.%m.%Y %H:%M")
            self.model_info_var.set(f"Путь: {p.resolve()} | Изменен: {dt}")
        else:
            self.model_info_var.set("")
        self._save_config()

    def _build(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Tab 1: Live & Manual Recording
        tab_live = ttk.Frame(self.notebook, padding=4)
        self.notebook.add(tab_live, text="  Запись и Live  ")

        # Tab 2: Synthetic Danser Render & Queue
        tab_render = ttk.Frame(self.notebook, padding=4)
        self.notebook.add(tab_render, text="  Синтетический рендер карт (Danser)  ")

        # Layout inside Tab 1
        pane = ttk.Panedwindow(tab_live, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True)
        maps, recording = ttk.Frame(pane, padding=6), ttk.Frame(pane, padding=6)
        pane.add(maps, weight=3)
        pane.add(recording, weight=4)
        self._build_maps(maps)
        self._build_recording(recording)

        # Layout inside Tab 2
        self._build_render_tab(tab_render)

    def _build_maps(self, parent):
        box = ttk.LabelFrame(parent, text="Карты osu!lazer", padding=8)
        box.pack(fill=tk.BOTH, expand=True)
        row = ttk.Frame(box)
        row.pack(fill=tk.X)
        search = ttk.Entry(row, textvariable=self.search_var)
        search.pack(side=tk.LEFT, fill=tk.X, expand=True)
        search.bind("<KeyRelease>", lambda _: self.apply_filter())
        ttk.Button(row, text="Обновить", command=self.refresh_maps).pack(side=tk.LEFT, padx=(6, 0))
        self.map_status = ttk.Label(box, text="Индексирую карты…")
        self.map_status.pack(anchor=tk.W, pady=6)
        columns = ("title", "diff", "ar", "objects")
        table = ttk.Frame(box)
        table.pack(fill=tk.BOTH, expand=True)
        self.tree = ttk.Treeview(table, columns=columns, show="headings", selectmode="browse")
        for key, label, width in (("title", "Карта", 235), ("diff", "Сложность", 155), ("ar", "AR", 45), ("objects", "Объекты", 68)):
            self.tree.heading(key, text=label)
            self.tree.column(key, width=width, stretch=key in ("title", "diff"))
        scroll = ttk.Scrollbar(table, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.bind("<<TreeviewSelect>>", self._select_map)

    def _build_recording(self, parent):
        tk.Label(
            parent, text="Запись обучающих карт и Live", bg=self.THEME_CARD, fg=self.THEME_FG,
            font=("Segoe UI", 13, "bold"), pady=8, relief="solid", borderwidth=1
        ).pack(fill=tk.X, pady=(0, 8))

        # Setup box
        setup = ttk.LabelFrame(parent, text="Параметры сессии и захвата", padding=9)
        setup.pack(fill=tk.X)
        self._path_row(setup, "Папка записей", self.output_root, self.choose_output_root, "Выбрать…")
        self._path_row(setup, "Имя сессии", self.session_name, self.new_session_name, "Новое имя")

        row = ttk.Frame(setup)
        row.pack(fill=tk.X, pady=4)
        ttk.Label(row, text="Захват:").pack(side=tk.LEFT)
        ttk.Combobox(row, textvariable=self.source_var, values=("dxgi", "obs"), width=9, state="readonly").pack(side=tk.LEFT, padx=7)

        # ROI Calibrator row
        roi_row = ttk.Frame(setup)
        roi_row.pack(fill=tk.X, pady=(4, 2))
        ttk.Button(roi_row, text="Калибровка ROI (F4)", command=self.open_roi_calibrator).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(roi_row, textvariable=self.roi_display_var, font=("Consolas", 9, "bold")).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Neural Model Selector
        model_box = ttk.LabelFrame(parent, text="Каталог нейросетевых моделей", padding=9)
        model_box.pack(fill=tk.X, pady=(6, 0))
        m_row = ttk.Frame(model_box)
        m_row.pack(fill=tk.X, pady=2)
        ttk.Label(m_row, text="Модель:").pack(side=tk.LEFT)
        self.model_combo = ttk.Combobox(
            m_row, textvariable=self.model_var,
            values=list(self.models_catalog.keys()),
            state="readonly", width=34
        )
        self.model_combo.pack(side=tk.LEFT, padx=6)
        self.model_combo.bind("<<ComboboxSelected>>", self._update_model_info)
        ttk.Button(m_row, text="Обзор…", command=self._browse_model).pack(side=tk.LEFT, padx=3)
        ttk.Button(m_row, text="⟳", width=3, command=self._refresh_models).pack(side=tk.LEFT, padx=2)
        ttk.Label(model_box, textvariable=self.model_info_var, font=("Segoe UI", 8), style="Muted.TLabel").pack(anchor=tk.W, pady=(4, 0))

        # Recording Controls
        controls = ttk.LabelFrame(parent, text="Управление записью", padding=9)
        controls.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(controls, text="1. Подготовить запись", command=self.arm_record).pack(side=tk.LEFT, padx=3)
        ttk.Button(controls, text="2. Старт / стоп (F1)", command=self.toggle_recording).pack(side=tk.LEFT, padx=3)
        ttk.Button(controls, text="Остановить захват", command=self.stop_pipeline).pack(side=tk.RIGHT, padx=3)

        # Live Controls with FPS Regulation
        live_box = ttk.LabelFrame(parent, text="Live режим (инференс, FPS и клики)", padding=9)
        live_box.pack(fill=tk.X, pady=(6, 0))

        live_row1 = ttk.Frame(live_box)
        live_row1.pack(fill=tk.X, pady=2)
        ttk.Button(live_row1, text="Запустить Live", style="Accent.TButton", command=self.start_live).pack(side=tk.LEFT, padx=3)
        ttk.Button(live_row1, text="Стоп Live (F10)", command=self.stop_live).pack(side=tk.LEFT, padx=3)
        ttk.Checkbutton(live_row1, text="Клики (Z/X)", variable=self.clicks_var, command=self._save_config).pack(side=tk.LEFT, padx=6)
        ttk.Label(live_row1, text="Уверенность:").pack(side=tk.LEFT, padx=(4, 2))
        conf_spin = ttk.Spinbox(live_row1, from_=0.10, to=1.00, increment=0.05, textvariable=self.conf_var, width=5, command=self._save_config)
        conf_spin.pack(side=tk.LEFT, padx=2)
        ttk.Label(live_row1, text="F2 — вкл/выкл · F10 — аварийный стоп", style="Muted.TLabel").pack(side=tk.LEFT, padx=8)

        # FPS Slider and quick buttons
        fps_frame = ttk.Frame(live_box)
        fps_frame.pack(fill=tk.X, pady=(8, 2))
        ttk.Label(fps_frame, text="Целевой FPS:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(fps_frame, textvariable=self.fps_label_var, font=("Segoe UI", 9, "bold"), width=7).pack(side=tk.LEFT)
        self.fps_slider = ttk.Scale(
            fps_frame, from_=30, to=240, variable=self.target_fps_var,
            command=self._on_fps_slide
        )
        self.fps_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        for preset in (30, 60, 120, 144, 240):
            ttk.Button(fps_frame, text=str(preset), width=4, command=lambda p=preset: self.set_fps(p)).pack(side=tk.LEFT, padx=1)

        # Status text
        ttk.Label(parent, textvariable=self.status_var, style="Status.TLabel", wraplength=610).pack(fill=tk.X, pady=(10, 4))

        # Telemetry metrics
        metrics = ttk.LabelFrame(parent, text="Телеметрия", padding=8)
        metrics.pack(fill=tk.X, pady=(6, 0))
        labels = (
            ("capture", "Capture FPS"), ("inference", "Inference FPS"),
            ("detect", "Detector"), ("policy", "Policy"),
            ("reaction", "Реакция"), ("objects", "Det / tracks"),
            ("timing", "До ноты"), ("record", "Запись")
        )
        for index, (key, title) in enumerate(labels):
            r, c = divmod(index, 4)
            cell = ttk.Frame(metrics)
            cell.grid(row=r, column=c, sticky="ew", padx=5, pady=2)
            ttk.Label(cell, text=title, style="Muted.TLabel").pack(anchor=tk.W)
            ttk.Label(cell, textvariable=self.metric_vars[key], font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)
        for c in range(4):
            metrics.columnconfigure(c, weight=1)

        # Preview frame with dynamic scaling container
        preview = ttk.LabelFrame(parent, text="Превью захвата (масштабируется)", padding=6)
        preview.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

        self.preview_container = tk.Frame(preview, bg="#050811")
        self.preview_container.pack(fill=tk.BOTH, expand=True)
        self.preview_container.bind("<Configure>", self._on_preview_container_resize)

        self.preview_label = tk.Label(
            self.preview_container,
            text="Запусти захват или Live, чтобы увидеть изображение.",
            bg="#050811",
            fg=self.THEME_FG_MUTED,
            font=("Segoe UI", 10),
            anchor=tk.CENTER
        )
        self.preview_label.pack(fill=tk.BOTH, expand=True)

    def _build_render_tab(self, parent):
        pane = ttk.Panedwindow(parent, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        maps_frame = ttk.Frame(pane, padding=6)
        queue_frame = ttk.Frame(pane, padding=6)
        pane.add(maps_frame, weight=3)
        pane.add(queue_frame, weight=4)

        self._build_render_maps(maps_frame)
        self._build_render_queue(queue_frame)

    def _build_render_maps(self, parent):
        box = ttk.LabelFrame(parent, text="Каталог карт osu! (Lazer + Stable)", padding=8)
        box.pack(fill=tk.BOTH, expand=True)

        row = ttk.Frame(box)
        row.pack(fill=tk.X, pady=(0, 4))
        
        ttk.Label(row, text="Источник:").pack(side=tk.LEFT, padx=(0, 3))
        src_combo = ttk.Combobox(
            row, textvariable=self.danser_source_var,
            values=("Все источники", "osu! Lazer", "osu! Stable"),
            width=13, state="readonly"
        )
        src_combo.pack(side=tk.LEFT, padx=(0, 6))
        src_combo.bind("<<ComboboxSelected>>", lambda _: self.apply_danser_filter())

        ttk.Label(row, text="Поиск:").pack(side=tk.LEFT, padx=(0, 3))
        search = ttk.Entry(row, textvariable=self.danser_search_var)
        search.pack(side=tk.LEFT, fill=tk.X, expand=True)
        search.bind("<KeyRelease>", lambda _: self.apply_danser_filter())
        ttk.Button(row, text="Обновить", command=self.refresh_danser_maps).pack(side=tk.LEFT, padx=(6, 0))

        btn_row = ttk.Frame(box)
        btn_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(btn_row, text="Добавить в очередь (+)", style="Accent.TButton", command=self.add_selected_to_render_queue).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(btn_row, text="Добавить все", command=self.add_all_to_render_queue).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row, text="Выбрать .osu файл…", command=self.browse_custom_osu).pack(side=tk.LEFT, padx=4)

        self.danser_map_status = ttk.Label(box, text="Индексирую карты osu! (Lazer + Stable)…")
        self.danser_map_status.pack(anchor=tk.W, pady=(0, 4))

        columns = ("source", "title", "diff", "ar", "cs", "dur", "objects")
        table = ttk.Frame(box)
        table.pack(fill=tk.BOTH, expand=True)

        self.danser_tree = ttk.Treeview(table, columns=columns, show="headings", selectmode="browse")
        headings = (
            ("source", "Клиент", 65),
            ("title", "Карта (Исполнитель — Название)", 200),
            ("diff", "Сложность", 120),
            ("ar", "AR", 40),
            ("cs", "CS", 40),
            ("dur", "Длина", 50),
            ("objects", "Ноты", 50),
        )
        for key, label, width in headings:
            self.danser_tree.heading(key, text=label)
            self.danser_tree.column(key, width=width, stretch=key in ("title", "diff"))

        scroll = ttk.Scrollbar(table, command=self.danser_tree.yview)
        self.danser_tree.configure(yscrollcommand=scroll.set)
        self.danser_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.danser_tree.bind("<<TreeviewSelect>>", self._on_danser_map_select)
        self.danser_tree.bind("<Double-1>", lambda e: self.add_selected_to_render_queue())

    def _build_render_queue(self, parent):
        tk.Label(
            parent, text="Синтетический рендер карт (Danser-go Headless)",
            bg=self.THEME_CARD, fg=self.THEME_FG,
            font=("Segoe UI", 12, "bold"), pady=6, relief="solid", borderwidth=1
        ).pack(fill=tk.X, pady=(0, 6))

        # 1. Queue Table Box
        q_box = ttk.LabelFrame(parent, text="Очередь рендера карт", padding=8)
        q_box.pack(fill=tk.BOTH, expand=True)

        q_table = ttk.Frame(q_box)
        q_table.pack(fill=tk.BOTH, expand=True)

        q_cols = ("num", "status", "source", "title", "diff", "fps", "dur", "progress")
        self.danser_queue_tree = ttk.Treeview(q_table, columns=q_cols, show="headings", selectmode="browse")
        q_headings = (
            ("num", "#", 35),
            ("status", "Статус", 95),
            ("source", "Клиент", 65),
            ("title", "Карта", 200),
            ("diff", "Сложность", 110),
            ("fps", "FPS", 45),
            ("dur", "Длина", 50),
            ("progress", "Прогресс", 70),
        )
        for key, label, width in q_headings:
            self.danser_queue_tree.heading(key, text=label)
            self.danser_queue_tree.column(key, width=width, stretch=key in ("title", "diff"))

        self.danser_queue_tree.tag_configure("Рендерится", foreground=self.THEME_ACCENT)
        self.danser_queue_tree.tag_configure("Готово", foreground=self.THEME_SUCCESS)
        self.danser_queue_tree.tag_configure("Ошибка", foreground=self.THEME_ERROR)
        self.danser_queue_tree.tag_configure("В очереди", foreground=self.THEME_FG)
        self.danser_queue_tree.tag_configure("Отменено", foreground=self.THEME_FG_MUTED)

        q_scroll = ttk.Scrollbar(q_table, command=self.danser_queue_tree.yview)
        self.danser_queue_tree.configure(yscrollcommand=q_scroll.set)
        self.danser_queue_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        q_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Queue Toolbar
        tb = ttk.Frame(q_box)
        tb.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(tb, text="Удалить выбранную", command=self.remove_selected_from_render_queue).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(tb, text="Очистить очередь", command=self.clear_render_queue).pack(side=tk.LEFT, padx=4)
        ttk.Button(tb, text="Вверх", width=6, command=self.move_render_queue_up).pack(side=tk.LEFT, padx=4)
        ttk.Button(tb, text="Вниз", width=6, command=self.move_render_queue_down).pack(side=tk.LEFT, padx=4)

        # 2. Render Settings Box
        cfg_box = ttk.LabelFrame(parent, text="Параметры рендера (Danser)", padding=8)
        cfg_box.pack(fill=tk.X, pady=(6, 0))

        row1 = ttk.Frame(cfg_box)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="Частота кадров (FPS):").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Combobox(
            row1, textvariable=self.danser_fps_var,
            values=("30", "60", "120", "144", "240"),
            width=7, state="readonly"
        ).pack(side=tk.LEFT, padx=(0, 14))

        ttk.Label(row1, text="Энкодер:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Combobox(
            row1, textvariable=self.danser_encoder_var,
            values=("libx264 ultrafast (~180 FPS)", "h264_nvenc (Hardware NVENC)"),
            width=32, state="readonly"
        ).pack(side=tk.LEFT, padx=(0, 4))

        row2 = ttk.Frame(cfg_box)
        row2.pack(fill=tk.X, pady=(4, 2))
        ttk.Label(row2, text="Папка вывода:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(row2, textvariable=self.danser_output_dir_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        ttk.Button(row2, text="Выбрать…", command=self.choose_danser_output_dir).pack(side=tk.LEFT)

        row3 = ttk.Frame(cfg_box)
        row3.pack(fill=tk.X, pady=(2, 0))
        ttk.Checkbutton(
            row3, text="Автоматически создавать сессию датасета (.npz, manifest, stamps) для обучения нейросети",
            variable=self.danser_auto_dataset_var
        ).pack(side=tk.LEFT)

        # 3. Progress & Telemetry Box
        prog_box = ttk.LabelFrame(parent, text="Прогресс рендера и расчет времени", padding=8)
        prog_box.pack(fill=tk.X, pady=(6, 0))

        # Current Map Progress
        m_row = ttk.Frame(prog_box)
        m_row.pack(fill=tk.X, pady=(2, 1))
        ttk.Label(m_row, textvariable=self.render_map_label_var, font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
        self.render_map_progress_bar = ttk.Progressbar(
            prog_box, orient=tk.HORIZONTAL, mode="determinate", variable=self.render_map_progress_var
        )
        self.render_map_progress_bar.pack(fill=tk.X, pady=(1, 6))

        # Queue Progress
        q_row = ttk.Frame(prog_box)
        q_row.pack(fill=tk.X, pady=(2, 1))
        ttk.Label(q_row, textvariable=self.render_queue_label_var, font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
        self.render_queue_progress_bar = ttk.Progressbar(
            prog_box, orient=tk.HORIZONTAL, mode="determinate", variable=self.render_queue_progress_var
        )
        self.render_queue_progress_bar.pack(fill=tk.X, pady=(1, 6))

        # Telemetry & ETA
        ttk.Label(
            prog_box, textvariable=self.render_telemetry_var,
            font=("Consolas", 9, "bold"), foreground=self.THEME_ACCENT
        ).pack(anchor=tk.W, pady=(2, 6))

        # Control Buttons
        c_row = ttk.Frame(prog_box)
        c_row.pack(fill=tk.X, pady=(2, 0))
        ttk.Button(c_row, text="Запустить рендер очереди", style="Accent.TButton", command=self.start_danser_render).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(c_row, text="Остановить рендер", command=self.stop_danser_render).pack(side=tk.LEFT, padx=6)
        ttk.Button(c_row, text="Открыть папку с видео", command=self.open_output_folder).pack(side=tk.RIGHT)

        # 4. Console Logs Box
        log_box = ttk.LabelFrame(parent, text="Консоль Danser", padding=6)
        log_box.pack(fill=tk.X, pady=(6, 0))
        log_frame = ttk.Frame(log_box)
        log_frame.pack(fill=tk.X)

        self.render_log_text = tk.Text(
            log_frame, height=4, bg=self.THEME_BG, fg=self.THEME_FG,
            font=("Consolas", 8), relief="flat", borderwidth=0, state="disabled"
        )
        log_scroll = ttk.Scrollbar(log_frame, command=self.render_log_text.yview)
        self.render_log_text.configure(yscrollcommand=log_scroll.set)
        self.render_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def refresh_danser_maps(self):
        if hasattr(self, "danser_map_status"):
            self.danser_map_status.configure(text="Синхронизирую osu! Lazer и Stable...")

        def worker():
            try:
                from osu2.lazer_linker import sync_lazer_hardlinks
                from osu2.danser_catalog import scan_all_catalogs
                sync_lazer_hardlinks(force_export=False)
                maps = scan_all_catalogs(include_lazer=True, include_stable=True)
                self.messages.put(("danser_maps", maps))
            except Exception as exc:
                self.messages.put(("error", f"Ошибка сканирования карт: {exc}"))

        threading.Thread(target=worker, daemon=True).start()

    def apply_danser_filter(self):
        query = self.danser_search_var.get().strip().casefold()
        src_filter = self.danser_source_var.get().strip().casefold()

        filtered = []
        for m in self.danser_catalog:
            m_src = m.get("source", "").casefold()
            if "lazer" in src_filter and "lazer" not in m_src:
                continue
            if "stable" in src_filter and "stable" not in m_src:
                continue

            if query:
                if query not in m["artist"].casefold() and query not in m["title"].casefold() and query not in m["version"].casefold():
                    continue
            filtered.append(m)

        self.danser_visible_maps = filtered
        self.danser_tree.delete(*self.danser_tree.get_children())
        for idx, m in enumerate(self.danser_visible_maps):
            title_text = f"{m['artist']} — {m['title']}"
            dur_text = f"{int(m['duration_s'] // 60)}:{int(m['duration_s'] % 60):02d}"
            src_tag = "Lazer" if "lazer" in m.get("source", "").lower() else "Stable"
            self.danser_tree.insert(
                "", tk.END, iid=str(idx),
                values=(src_tag, title_text, m["version"], f"{m['ar']:g}", f"{m['cs']:g}", dur_text, m["objects"])
            )
        if hasattr(self, "danser_map_status"):
            lazer_cnt = sum(1 for m in self.danser_visible_maps if "lazer" in m.get("source", "").lower())
            stable_cnt = len(self.danser_visible_maps) - lazer_cnt
            self.danser_map_status.configure(
                text=f"Показано {len(self.danser_visible_maps)} из {len(self.danser_catalog)} карт (Lazer: {lazer_cnt}, Stable: {stable_cnt})"
            )

    def _on_danser_map_select(self, _=None):
        sel = self.danser_tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        if 0 <= idx < len(self.danser_visible_maps):
            self.selected_danser_map = self.danser_visible_maps[idx]

    def add_selected_to_render_queue(self):
        sel = self.danser_tree.selection()
        if not sel:
            messagebox.showinfo("Очередь рендера", "Выбери карту в таблице слева для добавления в очередь.")
            return
        idx = int(sel[0])
        if 0 <= idx < len(self.danser_visible_maps):
            map_data = self.danser_visible_maps[idx]
            fps = int(self.danser_fps_var.get() or 60)
            enc = "libx264" if "libx264" in self.danser_encoder_var.get() else "h264_nvenc"
            self.danser_manager.add_item(map_data, fps=fps, encoder=enc)

    def add_all_to_render_queue(self):
        if not self.danser_visible_maps:
            messagebox.showinfo("Очередь рендера", "Нет отфильтрованных карт для добавления.")
            return
        fps = int(self.danser_fps_var.get() or 60)
        enc = "libx264" if "libx264" in self.danser_encoder_var.get() else "h264_nvenc"
        for map_data in self.danser_visible_maps:
            self.danser_manager.add_item(map_data, fps=fps, encoder=enc)

    def browse_custom_osu(self):
        chosen = filedialog.askopenfilename(
            title="Выбрать файл .osu",
            filetypes=[("osu! beatmap", "*.osu"), ("All files", "*.*")],
            initialdir=self.danser_songs_dir if Path(self.danser_songs_dir).exists() else str(ROOT)
        )
        if chosen:
            p = Path(chosen)
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
                meta = {}
                hits = []
                in_hitobjects = False
                for line in content.splitlines():
                    s = line.strip()
                    if s == "[HitObjects]":
                        in_hitobjects = True
                        continue
                    if not in_hitobjects and ":" in s:
                        k, v = s.split(":", 1)
                        meta[k.strip()] = v.strip()
                    elif in_hitobjects and s:
                        hits.append(s)

                dur_s = 60.0
                if hits:
                    f_parts = hits[0].split(",")
                    l_parts = hits[-1].split(",")
                    if len(f_parts) >= 3 and len(l_parts) >= 3 and f_parts[2].isdigit() and l_parts[2].isdigit():
                        dur_s = max(0.0, (int(l_parts[2]) - int(f_parts[2])) / 1000.0)

                map_data = {
                    "title": meta.get("Title", p.stem),
                    "artist": meta.get("Artist", "Unknown"),
                    "version": meta.get("Version", "Normal"),
                    "ar": float(meta.get("ApproachRate", 5.0)),
                    "cs": float(meta.get("CircleSize", 4.0)),
                    "objects": len(hits),
                    "duration_s": dur_s,
                    "path": str(p.resolve())
                }
                fps = int(self.danser_fps_var.get() or 60)
                enc = "libx264" if "libx264" in self.danser_encoder_var.get() else "h264_nvenc"
                self.danser_manager.add_item(map_data, fps=fps, encoder=enc)
            except Exception as exc:
                messagebox.showerror("Ошибка", f"Не удалось прочитать .osu файл: {exc}")

    def remove_selected_from_render_queue(self):
        sel = self.danser_queue_tree.selection()
        if not sel:
            return
        item_id = sel[0]
        self.danser_manager.remove_item(item_id)

    def clear_render_queue(self):
        self.danser_manager.clear_queue()

    def move_render_queue_up(self):
        sel = self.danser_queue_tree.selection()
        if not sel:
            return
        self.danser_manager.move_up(sel[0])

    def move_render_queue_down(self):
        sel = self.danser_queue_tree.selection()
        if not sel:
            return
        self.danser_manager.move_down(sel[0])

    def choose_danser_output_dir(self):
        chosen = filedialog.askdirectory(title="Выбрать папку для синтетических видео и сессий")
        if chosen:
            self.danser_output_dir_var.set(chosen)
            self.danser_manager.output_dir = Path(chosen)

    def open_output_folder(self):
        p = Path(self.danser_output_dir_var.get())
        p.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(p))
        except Exception:
            pass

    def start_danser_render(self):
        if not self.danser_manager.queue:
            messagebox.showinfo("Очередь рендера", "Очередь пуста. Добавь карты из списка слева.")
            return
        self.danser_manager.output_dir = Path(self.danser_output_dir_var.get())
        self.danser_manager.start()

    def stop_danser_render(self):
        self.danser_manager.stop()

    def _update_render_progress(self, stats: dict):
        self.render_map_progress_var.set(stats["current_map_progress"])
        self.render_map_label_var.set(
            f"Текущая карта: {stats['current_map_title']} ({stats['current_map_progress']:.0f}%)"
        )

        self.render_queue_progress_var.set(stats["queue_overall_progress"])
        self.render_queue_label_var.set(
            f"Общий прогресс: Карта {stats['queue_current_idx']} из {stats['queue_total_items']} ({stats['queue_overall_progress']:.1f}%)"
        )

        elapsed = stats.get("queue_elapsed_s", 0.0)
        em = int(elapsed // 60)
        es = int(elapsed % 60)

        total_eta = stats.get("queue_total_eta_s", 0.0)
        if total_eta > 0:
            rm = int(total_eta // 60)
            rs = int(total_eta % 60)
            eta_str = f"~{rm:02d}:{rs:02d}"
        else:
            eta_str = "завершается"

        spd = stats.get("current_map_speed", 0.0)
        fps_val = stats.get("current_map_fps", 60) * spd

        self.render_telemetry_var.set(
            f"Скорость: {spd:.2f}x ({fps_val:.0f} FPS) | Прошло: {em:02d}:{es:02d} | Примерно осталось: {eta_str}"
        )

    def _update_render_queue_tree(self, queue_items: list[RenderItem]):
        self.danser_queue_tree.delete(*self.danser_queue_tree.get_children())
        for idx, item in enumerate(queue_items, 1):
            dur_str = f"{int(item.duration_s // 60)}:{int(item.duration_s % 60):02d}"
            prog_str = f"{item.progress:.0f}%" if item.status == "Рендерится" else ("100%" if item.status == "Готово" else "—")
            src_tag = "Lazer" if "lazer" in getattr(item, "source", "").lower() else "Stable"
            self.danser_queue_tree.insert(
                "", tk.END, iid=item.id,
                values=(
                    idx,
                    item.status,
                    src_tag,
                    f"{item.artist} — {item.title}",
                    item.version,
                    f"{item.fps}",
                    dur_str,
                    prog_str
                ),
                tags=(item.status,)
            )

        if not self.danser_manager.is_running:
            completed = sum(1 for i in queue_items if i.status == "Готово")
            total = len(queue_items)
            self.render_queue_label_var.set(f"В очереди: {total} карт (готово: {completed})")
            if total > 0 and completed == total:
                self.render_telemetry_var.set("Очередь полностью обработана.")

    def _append_render_log(self, text: str):
        if hasattr(self, "render_log_text"):
            self.render_log_text.configure(state="normal")
            self.render_log_text.insert(tk.END, text + "\n")
            self.render_log_text.see(tk.END)
            self.render_log_text.configure(state="disabled")

    @staticmethod
    def _path_row(parent, label, variable, command, button_text):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text=label, width=15).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=variable).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text=button_text, command=command).pack(side=tk.LEFT, padx=(6, 0))

    def choose_output_root(self):
        selected = filedialog.askdirectory(initialdir=self.output_root.get())
        if selected:
            self.output_root.set(selected)

    def new_session_name(self):
        self.session_name.set(self._default_session_name())

    def refresh_maps(self):
        self.map_status.configure(text="Индексирую карты osu!lazer…")

        def worker():
            try:
                self.catalog = scan_lazer_maps()
                self.messages.put(("maps", None))
            except Exception as exc:
                self.messages.put(("error", f"Не удалось прочитать карты: {exc}"))

        threading.Thread(target=worker, daemon=True).start()

    def apply_filter(self):
        self.visible_maps = filter_maps(self.catalog, self.search_var.get())
        self.tree.delete(*self.tree.get_children())
        for index, entry in enumerate(self.visible_maps):
            self.tree.insert("", tk.END, iid=str(index), values=(f"{entry.artist} — {entry.title}", entry.version, f"{entry.approach_rate:g}", entry.objects))
        self.map_status.configure(text=f"Показано {len(self.visible_maps)} из {len(self.catalog)} карт")

    def _select_map(self, _=None):
        selected = self.tree.selection()
        if not selected:
            return
        self.selected_map = self.visible_maps[int(selected[0])]
        safe = "".join(c if c.isalnum() else "_" for c in f"{self.selected_map.title}_{self.selected_map.version}")[:55]
        self.session_name.set(f"{safe}_{datetime.now():%H%M%S}")
        self.status_var.set(f"Выбрано: {self.selected_map.artist} — {self.selected_map.title} [{self.selected_map.version}]")

    def _ensure_pipeline(self):
        if self.pipeline is not None and self.pipeline.running:
            return self.pipeline
        self.pipeline = CapturePipeline(
            self.messages, self.frames, str(self.legacy_config_path),
            self.source_var.get(), enable_inference=False
        )
        self.pipeline.start()
        self.status_var.set("Запускаю захват ROI без realtime YOLO…")
        return self.pipeline

    def session_path(self):
        name = self.session_name.get().strip()
        if not name:
            raise ValueError("Укажи имя сессии")
        return Path(self.output_root.get()).expanduser() / name

    def arm_record(self):
        if self.selected_map is None:
            messagebox.showwarning("Карта", "Выбери карту и сложность слева.")
            return
        try:
            path = self.session_path()
        except ValueError as exc:
            messagebox.showerror("Сессия", str(exc))
            return
        if (path / "video.mp4").exists() and not messagebox.askyesno("Перезапись", f"В папке уже есть запись:\n{path}\nПерезаписать её?"):
            return
        entry = self.selected_map
        self._ensure_pipeline().arm_recording(
            str(path),
            {"beatmap_osu": entry.path, "artist": entry.artist, "title": entry.title,
             "difficulty": entry.version, "approach_rate": entry.approach_rate,
             "circle_size": entry.circle_size, "objects": entry.objects, "rate": 1.0}
        )

    def toggle_recording(self):
        if self.pipeline is None or (not self.pipeline.record_armed and not self.pipeline.recording):
            self.arm_record()
        else:
            self.pipeline.toggle_recording()

    def stop_pipeline(self):
        if self.pipeline is not None:
            pipeline, self.pipeline = self.pipeline, None
            pipeline.stop()
        self.status_var.set("Захват остановлен.")

    def start_live(self):
        self.stop_pipeline()
        key = self.model_var.get()
        weights = self.models_catalog.get(key)
        if weights is None or not weights.exists():
            messagebox.showwarning("Модель", "Выбери корректную модель из каталога.")
            return
        self.stop_live()
        pf = {
            "left": self.roi_left, "top": self.roi_top,
            "right": self.roi_right, "bottom": self.roi_bottom
        }
        self.live = RelaxLive(
            str(self.legacy_config_path),
            str(weights),
            self.source_var.get(),
            self.messages,
            self.frames,
            enabled=False,
            clicks_enabled=self.clicks_var.get(),
            beatmap=self.selected_map,
            confidence_thresh=self.conf_var.get(),
            target_fps=self.target_fps_var.get(),
            playfield=pf,
        )
        self.live.start()
        self.status_var.set(f"Live [{weights.name}] готов ({self.target_fps_var.get()} FPS): F2 включает управление; F10 — аварийный стоп.")

    def stop_live(self):
        if self.live is not None:
            live, self.live = self.live, None
            live.stop()

    def _pump(self):
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == "maps":
                    self.apply_filter()
                elif kind == "record_state":
                    state = payload.get("state", "")
                    if state == "armed":
                        self.status_var.set("Готово: в osu! нажми F1, чтобы начать запись.")
                    elif state == "recording":
                        self.status_var.set("● ИДЁТ ЗАПИСЬ — F1 остановит и сохранит её.")
                    elif state == "saved":
                        self.status_var.set(f"✓ Сохранено: {payload.get('path', '')}")
                        self.new_session_name()
                    elif state == "error":
                        self.status_var.set(f"Ошибка: {payload.get('message', state)}")
                elif kind == "pipeline_stats":
                    self._update_metrics(payload)
                elif kind == "live_stats":
                    fps_val = float(payload.get("fps", 60.0))
                    frame_ms = 1000.0 / max(1.0, fps_val)
                    self.metric_vars["capture"].set(f"{fps_val:.1f}")
                    self.metric_vars["inference"].set(f"{fps_val:.1f}")
                    self.metric_vars["policy"].set(f"{float(payload.get('model_ms', 0)):.2f} ms")
                    self.metric_vars["reaction"].set(f"{float(payload.get('model_ms', 0)):.1f} ms + {frame_ms:.0f} ms кадр")
                    self.metric_vars["timing"].set(f"{float(payload.get('tth_ms', 0)):.0f} ms")
                elif kind == "live_state":
                    if payload.get("state") == "error":
                        self.status_var.set(f"Live ошибка: {payload.get('message')}")
                    elif payload.get("state") == "active":
                        self.status_var.set("● LIVE АКТИВЕН: мышь управляется моделью. F2 — пауза.")
                    elif payload.get("state") == "paused":
                        self.status_var.set("Live запущен, мышь на паузе. F2 — активация.")
                    elif payload.get("state") == "stopped" and self.live is not None:
                        self.status_var.set("Live остановлен.")
                elif kind == "frame":
                    self._show_frame(payload)
                elif kind == "danser_maps":
                    self.danser_catalog = payload
                    self.apply_danser_filter()
                elif kind == "render_progress":
                    self._update_render_progress(payload)
                elif kind == "render_queue":
                    self._update_render_queue_tree(payload)
                elif kind == "render_log":
                    self._append_render_log(payload)
                elif kind in ("pipeline_error", "error"):
                    self.status_var.set(f"Ошибка: {payload}")
        except queue.Empty:
            pass

        newest = None
        try:
            while True:
                newest = self.frames.get_nowait()
        except queue.Empty:
            pass
        if newest is not None:
            self._show_frame(newest)
        if self.winfo_exists():
            self.after(50, self._pump)

    def close(self):
        self._save_config()
        self.stop_live()
        self.stop_pipeline()
        if hasattr(self, "danser_manager") and self.danser_manager is not None:
            self.danser_manager.stop()
        self.destroy()

    def _update_metrics(self, stats):
        detect_ms = float(stats.get("detect_ms") or 0.0)
        policy_ms = float(stats.get("policy_ms") or 0.0)
        tth = stats.get("ring_tth_ms")
        self.metric_vars["capture"].set(f"{float(stats.get('capture_fps') or 0):.1f}")
        inference_fps = float(stats.get("inference_fps") or 0)
        self.metric_vars["inference"].set("выкл. (запись)") if inference_fps == 0 else self.metric_vars["inference"].set(f"{inference_fps:.1f}")
        self.metric_vars["detect"].set("выкл.") if detect_ms == 0 else self.metric_vars["detect"].set(f"{detect_ms:.1f} ms")
        self.metric_vars["policy"].set("выкл.") if policy_ms == 0 else self.metric_vars["policy"].set(f"{policy_ms:.2f} ms")
        self.metric_vars["reaction"].set("захват → запись") if detect_ms + policy_ms == 0 else self.metric_vars["reaction"].set(f"{detect_ms + policy_ms:.1f} ms")
        self.metric_vars["objects"].set(f"{stats.get('detections', 0)} / {stats.get('tracks', 0)}")
        self.metric_vars["timing"].set("—" if tth is None else f"{float(tth):.0f} ms")
        if stats.get("recording"):
            self.metric_vars["record"].set(f"{stats.get('record_elapsed', 0):.1f} s · {stats.get('record_frames', 0)} кадр. · target 60 FPS")
        else:
            self.metric_vars["record"].set("не идёт")

    def _on_preview_container_resize(self, event):
        if event.width > 20 and event.height > 20:
            if event.width != self._preview_container_w or event.height != self._preview_container_h:
                self._preview_container_w = event.width
                self._preview_container_h = event.height
                if self._latest_raw_image is not None:
                    if self._resize_timer is not None:
                        self.after_cancel(self._resize_timer)
                    self._resize_timer = self.after(30, self._render_preview)

    def _render_preview(self):
        if self._latest_raw_image is None or not self.winfo_exists():
            return
        try:
            avail_w = max(100, self._preview_container_w - 4)
            avail_h = max(100, self._preview_container_h - 4)
            img_w, img_h = self._latest_raw_image.size
            if img_w <= 0 or img_h <= 0:
                return

            scale = min(avail_w / img_w, avail_h / img_h)
            target_w = max(1, int(img_w * scale))
            target_h = max(1, int(img_h * scale))

            resized = self._latest_raw_image.resize((target_w, target_h), Image.Resampling.BILINEAR)
            self.preview_image = ImageTk.PhotoImage(resized)
            self.preview_label.configure(image=self.preview_image, text="")
        except Exception as exc:
            self.status_var.set(f"Ошибка рендера превью: {exc}")

    def _show_frame(self, encoded):
        try:
            raw_img = Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")
            self._latest_raw_image = raw_img
            self._render_preview()
        except Exception as exc:
            self.status_var.set(f"Не удалось показать превью: {exc}")


def run():
    RecordingApp().mainloop()
