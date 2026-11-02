"""Native control center: one capture owner for preview, recording and live play."""
import argparse
import base64
import os
import queue
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2

from ui.capture_pipeline import CapturePipeline
from ui.map_catalog import filter_maps, scan_lazer_maps

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ProcessRunner:
    """Runs offline tools while streaming their output to the GUI."""
    def __init__(self, messages):
        self.messages = messages
        self.process = None
        self.current_kind = None

    @property
    def running(self):
        return self.process is not None and self.process.poll() is None

    def start(self, args, done=None):
        if self.running:
            raise RuntimeError("другая offline-операция уже выполняется")
        command = [sys.executable, "-u", os.path.join(PROJECT, "main.py"), *args]
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.current_kind = args[0] if args else "process"
        process = subprocess.Popen(
            command, cwd=PROJECT, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            creationflags=flags,
        )
        self.process = process
        self.messages.put(("log", "> " + subprocess.list2cmdline(command) + "\n"))
        self.messages.put(("process_state", {"kind": self.current_kind, "state": "running"}))

        def read_output():
            for line in process.stdout:
                self.messages.put(("log", line))
                self._parse_progress(line)
            code = process.wait()
            self.messages.put(("log", f"\n[процесс завершён, код {code}]\n"))
            self.messages.put(("process_state", {
                "kind": self.current_kind, "state": "done", "code": code,
            }))
            self.messages.put(("done", done))
            if self.process is process:
                self.process = None
                self.current_kind = None

        threading.Thread(target=read_output, daemon=True).start()

    def _parse_progress(self, line):
        clean = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line)
        if self.current_kind == "train":
            yolo = re.search(r"^\s*(\d+)\s*/\s*(\d+)\s+", clean)
            if yolo:
                self.messages.put(("yolo_epoch", {
                    "epoch": int(yolo.group(1)), "epochs": int(yolo.group(2)),
                    "raw": clean.strip(),
                }))
        match = re.search(
            r"epoch\s+(\d+)(?:/(\d+))?.*?train[=: ]+([0-9.eE+-]+).*?val[=: ]+([0-9.eE+-]+)",
            line, re.IGNORECASE,
        )
        if match:
            self.messages.put(("train_metric", {
                "epoch": int(match.group(1)),
                "total": int(match.group(2)) if match.group(2) else None,
                "train": float(match.group(3)), "val": float(match.group(4)),
                "raw": line.strip(),
            }))
        batch = re.search(
            r"train\s+epoch\s+(\d+)/(\d+)\s+batch\s+(\d+)/(\d+)\s+loss=([0-9.eE+-]+)",
            line, re.IGNORECASE,
        )
        if batch:
            self.messages.put(("train_batch", {
                "epoch": int(batch.group(1)), "epochs": int(batch.group(2)),
                "batch": int(batch.group(3)), "batches": int(batch.group(4)),
                "loss": float(batch.group(5)),
            }))
        detect = re.search(r"detect\s+(\d+)\s*/\s*(\d+)", line, re.IGNORECASE)
        if detect:
            self.messages.put(("offline_progress", {
                "value": int(detect.group(1)), "maximum": int(detect.group(2)),
            }))

    def stop(self):
        if self.running:
            self.process.terminate()
            self.messages.put(("log", "\n[запрошена остановка offline-процесса]\n"))


class VideoPlayer:
    """Plays an annotated session video through the GUI's newest-frame queue."""
    def __init__(self, messages, frames):
        self.messages = messages
        self.frames = frames
        self.stop_event = threading.Event()
        self.thread = None

    @property
    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self, path):
        self.stop()
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, args=(path,), daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=2.0)
        self.thread = None

    def _run(self, path):
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            self.messages.put(("video_state", {"state": "error", "message": f"Не открылось: {path}"}))
            return
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.messages.put(("video_state", {"state": "playing", "path": path, "total": total, "fps": fps}))
        deadline = time.perf_counter()
        index = 0
        while not self.stop_event.is_set():
            ok, frame = capture.read()
            if not ok:
                break
            index += 1
            ok, encoded = cv2.imencode(".png", frame)
            if ok:
                payload = base64.b64encode(encoded).decode("ascii")
                try:
                    self.frames.put_nowait(payload)
                except queue.Full:
                    try:
                        self.frames.get_nowait()
                    except queue.Empty:
                        pass
                    self.frames.put_nowait(payload)
            if index % 10 == 0:
                self.messages.put(("video_progress", {"value": index, "maximum": total}))
            deadline += 1.0 / max(1.0, fps)
            time.sleep(max(0.0, deadline - time.perf_counter()))
        capture.release()
        self.messages.put(("video_state", {"state": "done", "frame": index, "total": total}))


class App(tk.Tk):
    def __init__(self, config_path):
        super().__init__()
        self.title("osu_module — control center")
        self.geometry("1380x900")
        self.minsize(1100, 740)
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.config_path = os.path.abspath(config_path)
        self.messages = queue.Queue()
        self.frames = queue.Queue(maxsize=1)
        self.runner = ProcessRunner(self.messages)
        self.pipeline = None
        self.player = VideoPlayer(self.messages, self.frames)
        self.catalog = []
        self.visible_maps = []
        self.selected_map = None
        self.preview_image = None
        self.last_stats = {}
        self.train_metrics = []
        self.live_countdown_token = 0
        self.closing = False
        self._build()
        self.after(40, self._pump)
        self.after(100, self.refresh_maps)

    def _build(self):
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        pane = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        left = ttk.Frame(pane, padding=6)
        right = ttk.Frame(pane, padding=6)
        pane.add(left, weight=2)
        pane.add(right, weight=5)
        self._build_maps(left)

        self.state_label = tk.Label(
            right, text="ГОТОВО", bg="#263238", fg="white",
            font=("Segoe UI", 13, "bold"), pady=7,
        )
        self.state_label.pack(fill=tk.X, pady=(0, 7))
        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        live_tab = ttk.Frame(self.notebook, padding=8)
        train_tab = ttk.Frame(self.notebook, padding=8)
        video_tab = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(live_tab, text="Запись / Live")
        self.notebook.add(train_tab, text="Разметка / Обучение")
        self.notebook.add(video_tab, text="Ролик с детектом")
        self._build_live(live_tab)
        self._build_training(train_tab)
        self._build_video(video_tab)

        log_box = ttk.LabelFrame(right, text="Журнал — здесь виден текущий этап и ошибки", padding=5)
        log_box.pack(fill=tk.BOTH, pady=(8, 0))
        self.log = tk.Text(log_box, height=8, wrap=tk.WORD, state=tk.DISABLED,
                           font=("Consolas", 9))
        self.log.pack(fill=tk.BOTH, expand=True)

    def _build_maps(self, parent):
        box = ttk.LabelFrame(parent, text="Карты osu!lazer", padding=8)
        box.pack(fill=tk.BOTH, expand=True)
        row = ttk.Frame(box)
        row.pack(fill=tk.X)
        self.search_var = tk.StringVar()
        search = ttk.Entry(row, textvariable=self.search_var)
        search.pack(side=tk.LEFT, fill=tk.X, expand=True)
        search.bind("<KeyRelease>", lambda _: self.apply_filter())
        ttk.Button(row, text="Обновить", command=self.refresh_maps).pack(side=tk.LEFT, padx=(6, 0))
        self.map_status = ttk.Label(box, text="Индексирую карты...")
        self.map_status.pack(fill=tk.X, pady=5)
        columns = ("title", "diff", "ar", "objects")
        frame = ttk.Frame(box)
        frame.pack(fill=tk.BOTH, expand=True)
        self.tree = ttk.Treeview(frame, columns=columns, show="headings", selectmode="browse")
        for key, title, width in (("title", "Название", 225), ("diff", "Сложность", 145),
                                  ("ar", "AR", 42), ("objects", "Объекты", 62)):
            self.tree.heading(key, text=title)
            self.tree.column(key, width=width, stretch=key in ("title", "diff"))
        scroll = ttk.Scrollbar(frame, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.bind("<<TreeviewSelect>>", self._select_map)

    def _build_live(self, parent):
        setup = ttk.LabelFrame(parent, text="Сессия", padding=7)
        setup.pack(fill=tk.X)
        self.session_root = tk.StringVar(value=os.path.join(PROJECT, "data", "recordings"))
        self.session_name = tk.StringVar(value=datetime.now().strftime("session_%Y%m%d_%H%M%S"))
        preferred_policy = os.path.join(PROJECT, "data", "policy", "policy_v4.pt")
        self.policy_var = tk.StringVar(value=preferred_policy)
        self.source_var = tk.StringVar(value="dxgi")
        self.rate_var = tk.StringVar(value="1.0")
        self._path_row(setup, "Папка записей", self.session_root,
                       lambda: self._choose_dir(self.session_root))
        self._path_row(setup, "Имя сессии", self.session_name,
                       self.choose_session, "Выбрать готовую")
        self._path_row(setup, "Policy weight", self.policy_var,
                       lambda: self._choose_file(self.policy_var, [("PyTorch", "*.pt")]))
        opts = ttk.Frame(setup)
        opts.pack(fill=tk.X, pady=4)
        ttk.Label(opts, text="Захват:").pack(side=tk.LEFT)
        ttk.Combobox(opts, textvariable=self.source_var, values=("dxgi", "obs"),
                     state="readonly", width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(opts, text="Скорость карты:").pack(side=tk.LEFT, padx=(15, 0))
        ttk.Combobox(opts, textvariable=self.rate_var,
                     values=("0.75", "1.0", "1.5"), width=7).pack(side=tk.LEFT, padx=5)
        ttk.Label(
            setup,
            text="YOLO dataset: 30 FPS lossless PNG · gray(t) + temporal difference + edges",
            foreground="#455a64",
        ).pack(fill=tk.X, pady=(2, 0))

        controls = ttk.LabelFrame(parent, text="Управление", padding=7)
        controls.pack(fill=tk.X, pady=(7, 0))
        row = ttk.Frame(controls)
        row.pack(fill=tk.X)
        self.preview_button = ttk.Button(row, text="▶ Включить превью", command=self.toggle_preview)
        self.preview_button.pack(side=tk.LEFT, padx=3)
        ttk.Button(row, text="1. Подготовить запись", command=self.arm_record).pack(side=tk.LEFT, padx=3)
        ttk.Button(row, text="2. Старт / стоп (F1)", command=self.toggle_record_now).pack(side=tk.LEFT, padx=3)
        ttk.Button(row, text="■ Всё остановить", command=self.stop_operation).pack(side=tk.RIGHT, padx=3)
        row2 = ttk.Frame(controls)
        row2.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(row2, text="Dry-run модели", command=self.dry_run).pack(side=tk.LEFT, padx=3)
        ttk.Button(row2, text="LIVE — мышь + Z/X", command=self.live_run).pack(side=tk.LEFT, padx=3)
        ttk.Button(row2, text="Стоп управления (F10)", command=self.stop_control).pack(side=tk.LEFT, padx=3)
        ttk.Label(row2, text="F1 работает глобально, пока включено превью. F10 аварийно выключает LIVE.").pack(side=tk.LEFT, padx=12)

        telemetry = ttk.LabelFrame(parent, text="Телеметрия", padding=7)
        telemetry.pack(fill=tk.X, pady=(7, 0))
        self.telemetry_vars = {}
        names = [
            ("source", "Источник"), ("capture", "Capture FPS"),
            ("infer", "Inference FPS"), ("detect", "YOLO, ms"),
            ("policy", "Policy, ms"), ("objects", "Det / tracks"),
            ("timing", "Ring / policy TTH"), ("record", "Запись"),
            ("detvideo", "Detected / temporal"), ("control", "Управление"),
            ("hits", "Запланировано кликов"), ("ready", "Live readiness"),
        ]
        for index, (key, title) in enumerate(names):
            row, col = divmod(index, 4)
            cell = ttk.Frame(telemetry)
            cell.grid(row=row, column=col, sticky="ew", padx=4, pady=2)
            ttk.Label(cell, text=title, foreground="#555").pack(anchor=tk.W)
            var = tk.StringVar(value="—")
            ttk.Label(cell, textvariable=var, font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)
            self.telemetry_vars[key] = var
        for col in range(4):
            telemetry.columnconfigure(col, weight=1)

        preview = ttk.LabelFrame(parent, text="Realtime детект — отображается и во время записи", padding=5)
        preview.pack(fill=tk.BOTH, expand=True, pady=(7, 0))
        self.image_label = ttk.Label(preview, anchor=tk.CENTER, text="Нажми «Включить превью»")
        self.image_label.pack(fill=tk.BOTH, expand=True)

    def _build_training(self, parent):
        setup = ttk.LabelFrame(parent, text="Данные и параметры", padding=7)
        setup.pack(fill=tk.X)
        self.train_sessions = tk.StringVar(value=os.path.join(PROJECT, "data"))
        self.epochs_var = tk.StringVar(value="30")
        self.batch_var = tk.StringVar(value="256")
        self.seq_var = tk.StringVar(value="32")
        self.stride_var = tk.StringVar(value="2")
        self.policy_nice_var = tk.BooleanVar(value=True)
        self.yolo_dataset_var = tk.StringVar(
            value=os.path.join(PROJECT, "data", "yolo_temporal_real"))
        old_detector = os.path.join(
            PROJECT, "data", "runs", "detect", "osu_nano", "weights", "best.pt")
        self.yolo_base_var = tk.StringVar(
            value=old_detector if os.path.exists(old_detector) else "yolov8n.pt")
        self._path_row(setup, "Корень датасетов", self.train_sessions,
                       lambda: self._choose_dir(self.train_sessions))
        self._path_row(setup, "Temporal YOLO dataset", self.yolo_dataset_var,
                       lambda: self._choose_dir(self.yolo_dataset_var))
        self._path_row(setup, "Base YOLO weights", self.yolo_base_var,
                       lambda: self._choose_file(self.yolo_base_var,
                                                 [("PyTorch", "*.pt")]))
        row = ttk.Frame(setup)
        row.pack(fill=tk.X, pady=4)
        for label, var, width in (("Эпох", self.epochs_var, 6), ("Batch", self.batch_var, 7),
                                  ("Seq len", self.seq_var, 7), ("Stride", self.stride_var, 7)):
            ttk.Label(row, text=label).pack(side=tk.LEFT, padx=(8, 2))
            ttk.Entry(row, textvariable=var, width=width).pack(side=tk.LEFT)
        ttk.Checkbutton(row, text="Low-impact training",
                        variable=self.policy_nice_var).pack(side=tk.LEFT, padx=10)
        actions = ttk.Frame(setup)
        actions.pack(fill=tk.X, pady=5)
        ttk.Button(actions, text="Разметить выбранную запись", command=self.label_session).pack(side=tk.LEFT, padx=3)
        ttk.Button(actions, text="Обучить policy", command=self.train_policy).pack(side=tk.LEFT, padx=3)
        ttk.Button(actions, text="Offline test", command=self.offline_test).pack(side=tk.LEFT, padx=3)
        ttk.Button(actions, text="Остановить", command=self.runner.stop).pack(side=tk.RIGHT, padx=3)

        progress_box = ttk.LabelFrame(parent, text="Ход операции", padding=7)
        yolo_actions = ttk.Frame(setup)
        yolo_actions.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(yolo_actions, text="1. Export all recordings to temporal YOLO",
                   command=self.export_temporal_yolo).pack(side=tk.LEFT, padx=3)
        ttk.Button(yolo_actions, text="2. Train temporal YOLO",
                   command=self.train_temporal_yolo).pack(side=tk.LEFT, padx=3)
        ttk.Button(yolo_actions, text="3. Activate temporal YOLO",
                   command=self.activate_temporal_yolo).pack(side=tk.LEFT, padx=3)
        progress_box.pack(fill=tk.X, pady=(7, 0))
        self.train_status = tk.StringVar(value="Операция не запущена")
        ttk.Label(progress_box, textvariable=self.train_status,
                  font=("Segoe UI", 10, "bold")).pack(fill=tk.X)
        self.train_progress = ttk.Progressbar(progress_box, mode="determinate")
        self.train_progress.pack(fill=tk.X, pady=5)
        self.train_detail = tk.StringVar(value="train loss: —    val loss: —")
        ttk.Label(progress_box, textvariable=self.train_detail).pack(fill=tk.X)

        chart_box = ttk.LabelFrame(parent, text="Loss по эпохам — train (синий), validation (оранжевый)", padding=5)
        chart_box.pack(fill=tk.BOTH, expand=True, pady=(7, 0))
        self.chart = tk.Canvas(chart_box, bg="#111820", highlightthickness=0)
        self.chart.pack(fill=tk.BOTH, expand=True)
        self.chart.bind("<Configure>", lambda _: self._draw_chart())

    def _build_video(self, parent):
        controls = ttk.Frame(parent)
        controls.pack(fill=tk.X)
        ttk.Button(controls, text="▶ Смотреть detected.mp4 выбранной сессии",
                   command=self.play_detected).pack(side=tk.LEFT, padx=3)
        ttk.Button(controls, text="■ Стоп", command=self.player.stop).pack(side=tk.LEFT, padx=3)
        self.video_status = tk.StringVar(value="После записи здесь можно проверить каждый детект.")
        ttk.Label(parent, textvariable=self.video_status).pack(fill=tk.X, pady=6)
        self.video_progress = ttk.Progressbar(parent, mode="determinate")
        self.video_progress.pack(fill=tk.X)
        box = ttk.LabelFrame(parent, text="Воспроизведение", padding=5)
        box.pack(fill=tk.BOTH, expand=True, pady=(7, 0))
        self.video_label = ttk.Label(box, anchor=tk.CENTER, text="detected.mp4 ещё не открыт")
        self.video_label.pack(fill=tk.BOTH, expand=True)

    def _path_row(self, parent, label, variable, command, button="Обзор"):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=18).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=variable).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text=button, command=command).pack(side=tk.LEFT, padx=(5, 0))

    def _choose_dir(self, variable):
        path = filedialog.askdirectory(initialdir=variable.get() or PROJECT)
        if path:
            variable.set(path)

    def _choose_file(self, variable, types):
        path = filedialog.askopenfilename(initialdir=os.path.dirname(variable.get()) or PROJECT,
                                          filetypes=types)
        if path:
            variable.set(path)

    def session_path(self):
        return os.path.join(self.session_root.get(), self.session_name.get().strip())

    def choose_session(self):
        path = filedialog.askdirectory(initialdir=self.session_root.get())
        if path:
            self.session_root.set(os.path.dirname(path))
            self.session_name.set(os.path.basename(path))

    def refresh_maps(self):
        self.map_status.configure(text="Индексирую хранилище lazer...")
        def scan():
            maps = scan_lazer_maps(progress=lambda checked, found:
                self.messages.put(("map_progress", (checked, found))))
            self.messages.put(("maps", maps))
        threading.Thread(target=scan, daemon=True).start()

    def apply_filter(self):
        self.visible_maps = filter_maps(self.catalog, self.search_var.get())
        self.tree.delete(*self.tree.get_children())
        for index, entry in enumerate(self.visible_maps):
            self.tree.insert("", tk.END, iid=str(index), values=(
                f"{entry.artist} — {entry.title}", entry.version,
                f"{entry.approach_rate:g}", entry.objects,
            ))
        self.map_status.configure(text=f"Показано {len(self.visible_maps)} из {len(self.catalog)}")

    def _select_map(self, _=None):
        selection = self.tree.selection()
        if not selection:
            return
        self.selected_map = self.visible_maps[int(selection[0])]
        safe = "".join(c if c.isalnum() else "_" for c in
                       f"{self.selected_map.title}_{self.selected_map.version}")[:55]
        self.session_name.set(safe + "_" + datetime.now().strftime("%H%M%S"))
        self._append_log(f"[карта] {self.selected_map.artist} — {self.selected_map.title} "
                         f"[{self.selected_map.version}]\n")

    def _ensure_pipeline(self):
        if self.runner.running:
            messagebox.showwarning("Занято", "Сначала останови offline-разметку или обучение")
            return None
        if self.player.running:
            self.player.stop()
        if self.pipeline is not None and self.pipeline.running:
            return self.pipeline
        self.pipeline = CapturePipeline(
            self.messages, self.frames, self.config_path, self.source_var.get())
        self.pipeline.start()
        self.preview_button.configure(text="■ Выключить превью")
        self._set_state("ЗАПУСК ЗАХВАТА И YOLO...", "busy")
        return self.pipeline

    def toggle_preview(self):
        if self.pipeline is not None and self.pipeline.running:
            pipeline = self.pipeline
            self.pipeline = None
            pipeline.stop()
            self.preview_button.configure(text="▶ Включить превью")
            self._set_state("ПРЕВЬЮ ОСТАНОВЛЕНО", "idle")
        else:
            self._ensure_pipeline()

    def arm_record(self):
        if self.selected_map is None:
            messagebox.showwarning("Карта", "Выбери карту и конкретную сложность слева")
            return
        path = self.session_path()
        if os.path.exists(os.path.join(path, "video.mp4")) and not messagebox.askyesno(
                "Перезапись", f"В папке уже есть запись:\n{path}\nПерезаписать её?"):
            return
        pipeline = self._ensure_pipeline()
        if pipeline is None:
            return
        metadata = {
            "beatmap_osu": self.selected_map.path,
            "artist": self.selected_map.artist,
            "title": self.selected_map.title,
            "difficulty": self.selected_map.version,
            "approach_rate": self.selected_map.approach_rate,
            "circle_size": self.selected_map.circle_size,
            "objects": self.selected_map.objects,
            "rate": float(self.rate_var.get()),
        }
        pipeline.arm_recording(path, metadata)

    def toggle_record_now(self):
        pipeline = self._ensure_pipeline()
        if pipeline is not None:
            pipeline.toggle_recording()

    def _stop_pipeline_for_offline(self):
        if self.pipeline is not None:
            pipeline = self.pipeline
            self.pipeline = None
            self.preview_button.configure(text="▶ Включить превью")
            pipeline.stop()

    def _start_offline(self, args, done=None):
        if self.runner.running:
            messagebox.showwarning("Занято", "Другой offline-процесс уже выполняется")
            return
        self.player.stop()
        self._stop_pipeline_for_offline()
        try:
            self.runner.start(args, done)
        except Exception as exc:
            messagebox.showerror("Ошибка запуска", str(exc))

    def label_session(self):
        if self.selected_map is None:
            messagebox.showwarning("Карта", "Выбери фактически сыгранную сложность")
            return
        if not os.path.exists(os.path.join(self.session_path(), "video.mp4")):
            messagebox.showwarning("Сессия", "В выбранной папке нет video.mp4")
            return
        self.notebook.select(1)
        self.train_status.set("Разметка: синхронизация карты и кадров...")
        self.train_progress.configure(value=0, maximum=100)
        self._start_offline(["actionlabel", "--session", self.session_path(),
                             "--osu", self.selected_map.path, "--rate", self.rate_var.get()])

    def train_policy(self):
        try:
            epochs = int(self.epochs_var.get())
            batch = int(self.batch_var.get())
            seq_len = int(self.seq_var.get())
            stride = int(self.stride_var.get())
            if min(epochs, batch, seq_len, stride) <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Параметры", "Эпохи, batch, seq len и stride должны быть > 0")
            return
        self.train_metrics.clear()
        self._draw_chart()
        self.train_progress.configure(value=0, maximum=epochs)
        self.train_status.set("Подготовка окон датасета...")
        args = ["policytrain", "--sessions", self.train_sessions.get(),
                "--out", self.policy_var.get(), "--epochs", str(epochs),
                "--batch", str(batch), "--seq-len", str(seq_len),
                "--stride", str(stride)]
        if self.policy_nice_var.get():
            args.append("--nice")
        self._start_offline(args)

    def offline_test(self):
        if not os.path.exists(os.path.join(self.session_path(), "policy_features.npy")):
            messagebox.showwarning("Сессия", "Сначала разметь запись: нет features.npy")
            return
        self._start_offline(["policytest", "--session", self.session_path(),
                             "--policy", self.policy_var.get()])

    def export_temporal_yolo(self):
        sessions_root = self.session_root.get()
        temporal_sessions = [
            entry.path for entry in os.scandir(sessions_root)
            if entry.is_dir() and
            os.path.isdir(os.path.join(entry.path, "vision_frames"))
        ] if os.path.isdir(sessions_root) else []
        if len(temporal_sessions) < 2:
            messagebox.showwarning(
                "Temporal recording",
                "At least two temporal recordings are required for map-disjoint train/validation.")
            return
        self.notebook.select(1)
        self.train_status.set("Exporting lossless temporal frames and beatmap labels...")
        self.train_progress.configure(value=0, maximum=100)
        self._start_offline([
            "realyolo", "--sessions", sessions_root,
            "--out", self.yolo_dataset_var.get(), "--stride", "1",
        ])

    def train_temporal_yolo(self):
        dataset = self.yolo_dataset_var.get()
        yaml_path = os.path.join(dataset, "dataset.yaml")
        train_dir = os.path.join(dataset, "images", "train")
        count = (len([name for name in os.listdir(train_dir)
                      if name.lower().endswith((".jpg", ".png"))])
                 if os.path.isdir(train_dir) else 0)
        if not os.path.exists(yaml_path) or count < 100:
            messagebox.showwarning(
                "Temporal YOLO dataset",
                f"Only {count} train frames found. Label and export recordings first (100 minimum; several maps recommended).")
            return
        try:
            epochs = int(self.epochs_var.get())
            batch = int(self.batch_var.get())
            if epochs <= 0 or batch <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Parameters", "Epochs and batch must be positive integers.")
            return
        args = [
            "train", "--data", dataset, "--name", "osu_temporal",
            "--epochs", str(epochs), "--batch", str(batch),
        ]
        base = self.yolo_base_var.get().strip()
        if base:
            args += ["--weights", base]
        self.train_status.set(
            f"Training temporal YOLO on {count} train frames; details are in the log below.")
        self.train_progress.configure(value=0, maximum=epochs)
        self._start_offline(args)

    def activate_temporal_yolo(self):
        weights = os.path.join(
            PROJECT, "data", "runs", "detect", "osu_temporal", "weights", "best.pt")
        if not os.path.exists(weights):
            messagebox.showwarning(
                "Temporal YOLO", "Train the detector first; osu_temporal/weights/best.pt was not found.")
            return
        if not messagebox.askyesno(
                "Activate temporal YOLO",
                "Switch realtime inference to the newly trained temporal detector?\n"
                "The capture preview will be restarted if it is running."):
            return
        self._stop_pipeline_for_offline()
        with open(self.config_path, "r", encoding="utf-8") as handle:
            config_text = handle.read()
        relative_weights = os.path.relpath(weights, PROJECT).replace("\\", "/")
        config_text, mode_changes = re.subn(
            r'(?m)^(\s*input_mode:\s*).+$', r'\1"temporal3"', config_text, count=1)
        config_text, weight_changes = re.subn(
            r'(?m)^(\s*weights:\s*).+$',
            lambda match: match.group(1) + f'"{relative_weights}"',
            config_text, count=1)
        if mode_changes != 1 or weight_changes != 1:
            messagebox.showerror(
                "Config", "Could not find detector.input_mode/weights in the config; nothing was changed.")
            return
        with open(self.config_path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(config_text)
        self._append_log(
            f"[detector] activated temporal3 weights: {relative_weights}\n")
        self.train_status.set("Temporal YOLO activated. Start preview and verify detections before LIVE.")

    def dry_run(self):
        pipeline = self._ensure_pipeline()
        if pipeline is None:
            return
        try:
            pipeline.start_control(self.policy_var.get(), live=False)
        except Exception as exc:
            messagebox.showerror("Dry-run", str(exc))

    def live_run(self):
        pipeline = self._ensure_pipeline()
        if pipeline is None:
            return
        if not messagebox.askyesno(
                "LIVE управление",
                "Модель получит управление мышью и Z/X.\n\n"
                "Перед стартом открой карту; на превью должны быть det/tracks и не меньше 25 inference FPS.\n"
                "F10 аварийно прекращает управление. Продолжить?"):
            return
        self.live_countdown_token += 1
        token = self.live_countdown_token
        self._append_log("[control] LIVE запустится через 3 секунды; верни фокус osu!\n")
        for delay, seconds in ((0, 3), (1000, 2), (2000, 1)):
            self.after(delay, lambda value=seconds, mark=token:
                       self._live_countdown(mark, value))
        self.after(3000, lambda mark=token, owner=pipeline:
                   self._activate_live(mark, owner))

    def _live_countdown(self, token, seconds):
        if token == self.live_countdown_token:
            self._set_state(f"LIVE ЧЕРЕЗ {seconds} · ВЕРНИ ФОКУС OSU!", "armed")

    def _activate_live(self, token, pipeline):
        if (token != self.live_countdown_token or self.pipeline is not pipeline or
                not pipeline.running):
            return
        try:
            pipeline.start_control(self.policy_var.get(), live=True)
        except Exception as exc:
            messagebox.showerror("LIVE", str(exc))

    def stop_control(self):
        self.live_countdown_token += 1
        if self.pipeline is not None:
            self.pipeline.stop_control()

    def play_detected(self):
        path = os.path.join(self.session_path(), "detected.mp4")
        if not os.path.exists(path):
            messagebox.showwarning("Ролик", "В выбранной сессии пока нет detected.mp4")
            return
        self._stop_pipeline_for_offline()
        self.notebook.select(2)
        self.player.start(path)

    def stop_operation(self):
        self.live_countdown_token += 1
        self.runner.stop()
        self.player.stop()
        if self.pipeline is not None:
            pipeline = self.pipeline
            self.pipeline = None
            pipeline.stop()
        if hasattr(self, "preview_button"):
            self.preview_button.configure(text="▶ Включить превью")

    def _set_state(self, text, state="idle"):
        colors = {
            "idle": "#263238", "busy": "#1565c0", "armed": "#ef6c00",
            "recording": "#c62828", "ok": "#2e7d32", "error": "#b71c1c",
            "live": "#6a1b9a",
        }
        self.state_label.configure(text=text, bg=colors.get(state, colors["idle"]))

    def _update_stats(self, stats):
        self.last_stats = stats
        val = self.telemetry_vars
        val["capture"].set(f"{stats['capture_fps']:.1f}")
        val["infer"].set(f"{stats['inference_fps']:.1f}")
        val["detect"].set(f"{stats['detect_ms']:.1f}")
        val["policy"].set(f"{stats['policy_ms']:.2f}" if stats["control"] != "off" else "выкл.")
        val["objects"].set(f"{stats['detections']} / {stats['tracks']}")
        source = stats.get("timing_source") or "—"
        ring = ("—" if stats["ring_tth_ms"] is None
                else f"{stats['ring_tth_ms']:.0f} ms ({source})")
        policy = "—" if stats["policy_tth_ms"] is None else f"{stats['policy_tth_ms']:.0f} ms"
        val["timing"].set(f"{ring} / {policy}")
        if stats["recording"]:
            val["record"].set(f"{stats['record_elapsed']:.1f} s · {stats['record_frames']} frames")
        else:
            val["record"].set("не идёт")
        val["detvideo"].set(
            f"{stats['detected_frames']} / {stats.get('vision_frames', 0)}"
            f" (drop {stats.get('vision_dropped', 0)})")
        val["control"].set(stats["control"].upper())
        val["hits"].set(str(stats["hits"]))
        if stats["inference_fps"] < 25:
            readiness = "НЕТ: inference < 25 FPS"
        elif not stats["tracks"]:
            readiness = "ЖДУ ОБЪЕКТЫ"
        elif stats["ring_tth_ms"] is None:
            readiness = "ЖДУ APPROACH RING"
        else:
            readiness = "ГОТОВО"
        val["ready"].set(readiness)

    def _append_log(self, text):
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, text)
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def _draw_chart(self):
        if not hasattr(self, "chart"):
            return
        canvas = self.chart
        canvas.delete("all")
        width = max(200, canvas.winfo_width())
        height = max(120, canvas.winfo_height())
        pad = 38
        canvas.create_line(pad, 10, pad, height - pad, fill="#607d8b")
        canvas.create_line(pad, height - pad, width - 10, height - pad, fill="#607d8b")
        if not self.train_metrics:
            canvas.create_text(width / 2, height / 2, text="График появится после первой эпохи",
                               fill="#90a4ae", font=("Segoe UI", 11))
            return
        values = [m[key] for m in self.train_metrics for key in ("train", "val")]
        maximum = max(values) if values else 1.0
        minimum = min(values) if values else 0.0
        span = max(1e-9, maximum - minimum)
        n = max(2, len(self.train_metrics))
        def point(index, value):
            x = pad + index * (width - pad - 15) / (n - 1)
            y = 10 + (maximum - value) * (height - pad - 15) / span
            return x, y
        for key, color in (("train", "#42a5f5"), ("val", "#ff9800")):
            points = [coord for i, metric in enumerate(self.train_metrics)
                      for coord in point(i, metric[key])]
            if len(points) >= 4:
                canvas.create_line(*points, fill=color, width=2, smooth=True)
        canvas.create_text(5, 11, text=f"{maximum:.4g}", fill="#90a4ae", anchor=tk.NW)
        canvas.create_text(5, height - pad, text=f"{minimum:.4g}", fill="#90a4ae", anchor=tk.SW)
        canvas.create_text(width - 12, height - pad + 5,
                           text=f"epoch {self.train_metrics[-1]['epoch']}",
                           fill="#90a4ae", anchor=tk.NE)

    def _handle_message(self, kind, payload):
        if kind == "log":
            self._append_log(payload)
        elif kind == "maps":
            self.catalog = payload
            self.apply_filter()
        elif kind == "map_progress":
            self.map_status.configure(text=f"Проверено {payload[0]}, найдено {payload[1]}")
        elif kind == "pipeline_ready":
            self.telemetry_vars["source"].set(payload["source"])
            if self.pipeline is not None and self.pipeline.record_armed:
                self._set_state("ЗАПИСЬ ПОДГОТОВЛЕНА · НАЖМИ F1 В ИГРЕ", "armed")
            else:
                self._set_state("ПРЕВЬЮ РАБОТАЕТ · F1 МОНИТОР АКТИВЕН", "ok")
            self._append_log(f"[pipeline] захват готов: {payload['source']}; F1 активен\n")
        elif kind == "capture_ready":
            self.telemetry_vars["source"].set(payload["source"])
            suffix = " · ЗАПИСЬ УЖЕ ПОДГОТОВЛЕНА" if (
                self.pipeline is not None and self.pipeline.record_armed) else ""
            self._set_state("ЭКРАН ЗАХВАЧЕН · ЗАГРУЖАЮ И ПРОГРЕВАЮ YOLO" + suffix, "busy")
            self._append_log(f"[pipeline] экран захвачен: {payload['source']}; загружается YOLO\n")
        elif kind == "pipeline_stats":
            self._update_stats(payload)
        elif kind == "pipeline_error":
            self._set_state(f"ОШИБКА PIPELINE: {payload}", "error")
            self._append_log(f"[pipeline error] {payload}\n")
        elif kind == "pipeline_stopped":
            self._append_log("[pipeline] остановлен\n")
        elif kind == "record_state":
            state = payload.get("state")
            if state == "armed":
                self._set_state("ЗАПИСЬ ПОДГОТОВЛЕНА · НАЖМИ F1 В ИГРЕ", "armed")
            elif state == "loading":
                self._set_state(payload.get("message", "ЗАГРУЖАЮ YOLO..."), "busy")
            elif state == "recording":
                self._set_state("● ИДЁТ ЗАПИСЬ · F1 ОСТАНОВИТ", "recording")
            elif state == "saved":
                dropped = payload.get("vision_dropped", 0)
                self._set_state(
                    f"СОХРАНЕНО: {payload['frames']} RAW / "
                    f"{payload['detected_frames']} DETECTED / "
                    f"{payload.get('vision_frames', 0)} TEMPORAL / "
                    f"{dropped} DROPPED", "error" if dropped else "ok")
                self._append_log(f"[saved] {payload['path']}\n")
            else:
                self._set_state(payload.get("message", "Ошибка записи"), "error")
            self._append_log(f"[record] {payload.get('message', state)}\n")
        elif kind == "control_state":
            mode = payload.get("mode", "off")
            self._set_state(payload.get("message", mode), "live" if mode == "live" else "busy")
            self._append_log(f"[control] {payload.get('message', mode)}\n")
        elif kind == "process_state":
            if payload["state"] == "running":
                self._set_state(f"OFFLINE: {payload['kind']} ВЫПОЛНЯЕТСЯ", "busy")
                self.train_status.set(f"Выполняется: {payload['kind']}")
            else:
                ok = payload.get("code") == 0
                self._set_state(f"{payload['kind']} ЗАВЕРШЁН, КОД {payload.get('code')}", "ok" if ok else "error")
                self.train_status.set(f"Завершено: код {payload.get('code')}")
        elif kind == "train_metric":
            self.train_metrics.append(payload)
            total = payload.get("total") or int(self.epochs_var.get() or 1)
            self.train_progress.configure(maximum=total, value=payload["epoch"])
            self.train_status.set(f"Обучение: эпоха {payload['epoch']} из {total}")
            self.train_detail.set(f"train loss: {payload['train']:.6g}    val loss: {payload['val']:.6g}")
            self._draw_chart()
        elif kind == "train_batch":
            fraction = (payload["epoch"] - 1) + payload["batch"] / max(1, payload["batches"])
            self.train_progress.configure(maximum=payload["epochs"], value=fraction)
            self.train_status.set(
                f"Эпоха {payload['epoch']}/{payload['epochs']} · "
                f"batch {payload['batch']}/{payload['batches']}")
            self.train_detail.set(f"текущий train loss: {payload['loss']:.6g} · validation после эпохи")
        elif kind == "yolo_epoch":
            self.train_progress.configure(
                maximum=payload["epochs"], value=payload["epoch"])
            self.train_status.set(
                f"Temporal YOLO: epoch {payload['epoch']} / {payload['epochs']}")
            self.train_detail.set(payload["raw"])
        elif kind == "offline_progress":
            self.train_progress.configure(maximum=payload["maximum"], value=payload["value"])
        elif kind == "video_state":
            if payload["state"] == "playing":
                self.video_progress.configure(maximum=max(1, payload["total"]), value=0)
                self.video_status.set(f"Воспроизведение: {payload['fps']:.1f} FPS, {payload['total']} кадров")
            elif payload["state"] == "done":
                self.video_status.set(f"Ролик завершён: {payload['frame']} кадров")
            else:
                self.video_status.set(payload.get("message", "Ошибка видео"))
        elif kind == "video_progress":
            self.video_progress.configure(maximum=max(1, payload["maximum"]), value=payload["value"])
        elif kind == "done" and callable(payload):
            payload()

    def _pump(self):
        if self.closing:
            return
        try:
            while True:
                self._handle_message(*self.messages.get_nowait())
        except queue.Empty:
            pass
        try:
            payload = self.frames.get_nowait()
            self.preview_image = tk.PhotoImage(data=payload)
            if self.player.running or self.notebook.index(self.notebook.select()) == 2:
                self.video_label.configure(image=self.preview_image, text="")
            else:
                self.image_label.configure(image=self.preview_image, text="")
        except queue.Empty:
            pass
        self.after(40, self._pump)

    def destroy(self):
        if self.closing:
            return
        self.closing = True
        self.stop_operation()
        super().destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(PROJECT, "configs", "default.yaml"))
    args = parser.parse_args()
    App(args.config).mainloop()


if __name__ == "__main__":
    main()
