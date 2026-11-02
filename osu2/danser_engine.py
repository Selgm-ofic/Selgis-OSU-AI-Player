from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from typing import Callable, Optional

import numpy as np


@dataclass
class RenderItem:
    id: str
    title: str
    artist: str
    version: str
    path: str
    ar: float
    cs: float
    objects: int
    duration_s: float
    fps: int = 60
    encoder: str = "libx264"
    source: str = "osu! stable"
    status: str = "В очереди"  # "В очереди", "Рендерится", "Готово", "Ошибка", "Отменено"
    progress: float = 0.0
    speed: float = 0.0
    eta_s: float = 0.0
    output_path: str = ""
    error: str = ""


def scan_danser_maps(songs_dir: Optional[str] = None) -> list[dict]:
    """Fast metadata scan of all .osu beatmaps in the osu! Songs directory."""
    if songs_dir is None:
        songs_dir = r"C:\Users\Game PC\AppData\Local\osu!\Songs"
    p = Path(songs_dir)
    if not p.is_dir():
        return []

    maps = []
    for f in p.glob("*/*.osu"):
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as fp:
                content = fp.read()
            parts = content.split("[HitObjects]")
            if len(parts) < 2:
                continue
            header = parts[0]
            hits = parts[1].strip().splitlines()
            if not hits:
                continue
            meta: dict[str, str] = {}
            for line in header.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()

            first_t = 0
            first_parts = hits[0].split(",")
            if len(first_parts) >= 3 and first_parts[2].isdigit():
                first_t = int(first_parts[2])

            last_t = first_t
            last_parts = hits[-1].split(",")
            if len(last_parts) >= 3 and last_parts[2].isdigit():
                last_t = int(last_parts[2])

            dur_s = max(0.0, float((last_t - first_t) / 1000.0))
            maps.append({
                "title": meta.get("Title", f.stem),
                "artist": meta.get("Artist", "Unknown"),
                "version": meta.get("Version", "Normal"),
                "ar": float(meta.get("ApproachRate", 5.0)),
                "cs": float(meta.get("CircleSize", 4.0)),
                "objects": len(hits),
                "duration_s": dur_s,
                "path": str(f.resolve()),
                "folder": str(f.parent.resolve()),
            })
        except Exception:
            continue

    maps.sort(key=lambda m: (m["artist"].casefold(), m["title"].casefold(), m["ar"]))
    return maps


class DanserQueueManager:
    """Multi-threaded background queue manager for danser headless rendering."""

    def __init__(
        self,
        danser_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        on_progress: Optional[Callable[[dict], None]] = None,
        on_queue_update: Optional[Callable[[list[RenderItem]], None]] = None,
        on_log: Optional[Callable[[str], None]] = None,
    ):
        self.danser_dir = Path(danser_dir or r"C:\Users\Game PC\Desktop\Danser OSU")
        self.danser_exe = self.danser_dir / "danser-cli.exe"
        self.output_dir = Path(output_dir or r"C:\Selgis OSU AI Player\data\synthetic_renders")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.queue: list[RenderItem] = []
        self.current_item: Optional[RenderItem] = None
        self.active_proc: Optional[subprocess.Popen] = None

        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.is_running = False

        self.on_progress = on_progress
        self.on_queue_update = on_queue_update
        self.on_log = on_log

        self.total_queue_start_time = 0.0
        self.current_map_start_time = 0.0

    def add_item(self, map_data: dict, fps: int = 60, encoder: str = "libx264") -> RenderItem:
        with self._lock:
            item_id = f"item_{int(time.time() * 1000)}_{len(self.queue)}"
            item = RenderItem(
                id=item_id,
                title=map_data.get("title", "Unknown"),
                artist=map_data.get("artist", "Unknown"),
                version=map_data.get("version", "Normal"),
                path=map_data.get("path", ""),
                ar=float(map_data.get("ar", 5.0)),
                cs=float(map_data.get("cs", 4.0)),
                objects=int(map_data.get("objects", 0)),
                duration_s=float(map_data.get("duration_s", 0.0)),
                fps=int(fps),
                encoder=encoder,
                source=str(map_data.get("source", "osu! stable")),
            )
            self.queue.append(item)

        if self.on_queue_update:
            self.on_queue_update(self.get_queue_snapshot())
        return item

    def remove_item(self, item_id: str):
        with self._lock:
            # Cannot remove currently rendering item via remove; stop must be used
            self.queue = [item for item in self.queue if not (item.id == item_id and item.status != "Рендерится")]
        if self.on_queue_update:
            self.on_queue_update(self.get_queue_snapshot())

    def clear_queue(self):
        with self._lock:
            self.queue = [item for item in self.queue if item.status == "Рендерится"]
        if self.on_queue_update:
            self.on_queue_update(self.get_queue_snapshot())

    def move_up(self, item_id: str):
        with self._lock:
            for i in range(1, len(self.queue)):
                if self.queue[i].id == item_id and self.queue[i].status == "В очереди" and self.queue[i - 1].status == "В очереди":
                    self.queue[i - 1], self.queue[i] = self.queue[i], self.queue[i - 1]
                    break
        if self.on_queue_update:
            self.on_queue_update(self.get_queue_snapshot())

    def move_down(self, item_id: str):
        with self._lock:
            for i in range(len(self.queue) - 2, -1, -1):
                if self.queue[i].id == item_id and self.queue[i].status == "В очереди" and self.queue[i + 1].status == "В очереди":
                    self.queue[i], self.queue[i + 1] = self.queue[i + 1], self.queue[i]
                    break
        if self.on_queue_update:
            self.on_queue_update(self.get_queue_snapshot())

    def get_queue_snapshot(self) -> list[RenderItem]:
        with self._lock:
            return list(self.queue)

    def start(self):
        if self.is_running:
            return
        self._stop_event.clear()
        self.is_running = True
        self.total_queue_start_time = time.perf_counter()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        with self._lock:
            if self.active_proc is not None:
                try:
                    self.active_proc.terminate()
                except Exception:
                    pass
            if self.current_item:
                self.current_item.status = "Отменено"
        self.is_running = False
        if self.on_queue_update:
            self.on_queue_update(self.get_queue_snapshot())

    def _prepare_danser_config(self, item: RenderItem) -> str:
        """Create or patch Danser settings profile for this item's specific FPS, encoder, and songs directory."""
        preset_name = "ml_synth_x264" if item.encoder == "libx264" else "ml_synth"
        src_preset = self.danser_dir / "settings" / f"{preset_name}.json"
        if not src_preset.exists():
            src_preset = self.danser_dir / "settings" / "ml_synth_shift.json"

        cfg_data = json.loads(src_preset.read_text(encoding="utf-8"))

        # Dynamically set OsuSongsDir to the exact root containing this map
        if item.path:
            songs_root = Path(item.path).parent.parent
            if songs_root.exists() and songs_root.is_dir():
                cfg_data.setdefault("General", {})["OsuSongsDir"] = str(songs_root.resolve())

        # Enforce ML synthesis requirements:
        cfg_data.setdefault("Playfield", {})["OsuShift"] = True
        cfg_data["Playfield"]["DrawCursors"] = False

        rec = cfg_data.setdefault("Recording", {})
        rec["FPS"] = int(item.fps)
        rec["FrameWidth"] = 1920
        rec["FrameHeight"] = 1080
        rec["Encoder"] = "libx264" if item.encoder == "libx264" else "h264_nvenc"

        # Apply profile name
        target_preset_name = f"ml_active_{item.fps}fps"
        target_path = self.danser_dir / "settings" / f"{target_preset_name}.json"
        target_path.write_text(json.dumps(cfg_data, indent=2), encoding="utf-8")
        return target_preset_name

    def _worker(self):
        self._log(f"Очередь рендера запущена ({len(self.queue)} элементов).")

        progress_regex = re.compile(r"Progress:\s*(\d+)%,\s*Speed:\s*([\d.]+)x,\s*ETA:\s*(\d+)s")

        while not self._stop_event.is_set():
            # Find next pending item
            next_item: Optional[RenderItem] = None
            with self._lock:
                for item in self.queue:
                    if item.status == "В очереди":
                        next_item = item
                        break

            if next_item is None:
                break

            self.current_item = next_item
            next_item.status = "Рендерится"
            next_item.progress = 0.0
            self.current_map_start_time = time.perf_counter()

            if self.on_queue_update:
                self.on_queue_update(self.get_queue_snapshot())

            self._log(f"Рендер карты: {next_item.artist} — {next_item.title} [{next_item.version}] ({next_item.fps} FPS, {next_item.encoder})")

            # Safe filename
            safe_title = "".join(c if c.isalnum() else "_" for c in f"{next_item.title}_{next_item.version}")[:50]
            out_stem = f"{safe_title}_{next_item.fps}fps_{datetime.now():%H%M%S}"

            try:
                preset_name = self._prepare_danser_config(next_item)
                cmd = [
                    str(self.danser_exe),
                    f"-settings={preset_name}",
                    f"-title={next_item.title}",
                    f"-difficulty={next_item.version}",
                    f"-out={out_stem}",
                    "-noupdatecheck",
                    "-nodbcheck",
                    "-preciseprogress"
                ]

                flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(self.danser_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    creationflags=flags
                )
                self.active_proc = proc

                last_speed = 2.5
                for line in proc.stdout:
                    line = line.strip()
                    if not line:
                        continue

                    m = progress_regex.search(line)
                    if m:
                        prog_pct = float(m.group(1))
                        spd = float(m.group(2))
                        eta = float(m.group(3))
                        last_speed = max(0.1, spd)

                        next_item.progress = prog_pct
                        next_item.speed = spd
                        next_item.eta_s = eta

                        # Compute total queue statistics
                        stats = self._compute_queue_stats(next_item, last_speed)
                        if self.on_progress:
                            self.on_progress(stats)
                    elif "Video is available at:" in line or "Finished!" in line or "Error" in line:
                        self._log(line)

                code = proc.wait()
                self.active_proc = None

                if self._stop_event.is_set():
                    next_item.status = "Отменено"
                    self._log(f"Рендер {next_item.title} отменен пользователем.")
                    break

                if code == 0:
                    danser_video = self.danser_dir / "videos" / f"{out_stem}.mp4"
                    final_video = self.output_dir / f"{out_stem}.mp4"

                    if danser_video.exists():
                        shutil.move(str(danser_video), str(final_video))
                        next_item.output_path = str(final_video)
                        next_item.status = "Готово"
                        next_item.progress = 100.0
                        self._log(f"Карта успешно отрендерена: {final_video.name}")

                        # Auto-create dataset manifest and stamps for immediate neural training
                        self._create_dataset_metadata(next_item, final_video)
                    else:
                        next_item.status = "Ошибка"
                        next_item.error = "Файл видео не найден после завершения danser"
                        self._log(f"Ошибка: {next_item.error}")
                else:
                    next_item.status = "Ошибка"
                    next_item.error = f"Danser завершился с кодом {code}"
                    self._log(f"Ошибка danser: код {code}")

            except Exception as exc:
                next_item.status = "Ошибка"
                next_item.error = str(exc)
                self._log(f"Исключение при рендере: {exc}")

            if self.on_queue_update:
                self.on_queue_update(self.get_queue_snapshot())

        self.current_item = None
        self.is_running = False
        self._log("Обработка очереди рендера завершена.")
        if self.on_queue_update:
            self.on_queue_update(self.get_queue_snapshot())

    def _compute_queue_stats(self, current: RenderItem, current_speed: float) -> dict:
        total_items = len(self.queue)
        completed_items = sum(1 for item in self.queue if item.status in ("Готово", "Ошибка", "Отменено"))
        current_idx = completed_items + 1

        overall_pct = (completed_items + (current.progress / 100.0)) / max(1, total_items) * 100.0

        # Estimate remaining duration of pending items in queue
        upcoming_duration_s = sum(
            item.duration_s for item in self.queue if item.status == "В очереди"
        )
        upcoming_eta_s = upcoming_duration_s / max(1.0, current_speed)
        total_eta_s = current.eta_s + upcoming_eta_s

        elapsed_total = time.perf_counter() - self.total_queue_start_time

        return {
            "current_map_title": f"{current.artist} — {current.title} [{current.version}]",
            "current_map_progress": current.progress,
            "current_map_eta_s": current.eta_s,
            "current_map_speed": current.speed,
            "current_map_fps": current.fps,
            "queue_current_idx": min(total_items, current_idx),
            "queue_total_items": total_items,
            "queue_overall_progress": overall_pct,
            "queue_total_eta_s": total_eta_s,
            "queue_elapsed_s": elapsed_total,
        }

    def _create_dataset_metadata(self, item: RenderItem, video_path: Path):
        """Create matching stamps.npy and policy_manifest.json alongside the video."""
        try:
            session_dir = self.output_dir / video_path.stem
            session_dir.mkdir(parents=True, exist_ok=True)
            target_video = session_dir / "video.mp4"
            if not target_video.exists():
                shutil.copy2(str(video_path), str(target_video))

            # Approximate total frames from duration and fps
            total_frames = int(round(item.duration_s * item.fps))
            if total_frames <= 0:
                total_frames = 1000

            frame_interval_ms = 1000.0 / float(item.fps)
            stamps = np.arange(total_frames, dtype=np.float32) * frame_interval_ms
            np.save(str(session_dir / "stamps.npy"), stamps)

            manifest = {
                "version": 8,
                "created_at": datetime.now().isoformat(),
                "synthetic": True,
                "renderer": "danser-go",
                "fps": item.fps,
                "beatmap": {
                    "path": item.path,
                    "title": item.title,
                    "artist": item.artist,
                    "version": item.version,
                    "ar": item.ar,
                    "cs": item.cs,
                    "objects": item.objects,
                    "duration_s": item.duration_s,
                },
                "clock": {
                    "offset_ms": 0.0,
                    "scale": 1.0,
                    "click_safe": True,
                    "max_time_uncertainty_ms": 0.0,
                },
                "playfield_roi": [295, 35, 1625, 1075],
            }
            (session_dir / "policy_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _log(self, msg: str):
        if self.on_log:
            self.on_log(f"[{datetime.now():%H:%M:%S}] {msg}")
