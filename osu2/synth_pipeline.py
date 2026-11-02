import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"C:\Selgis OSU AI Player")
DANSER_DIR = Path(r"C:\Users\Game PC\Desktop\Danser OSU")
DANSER_EXE = DANSER_DIR / "danser-cli.exe"
SYNTH_REC_DIR = ROOT / "data" / "synthetic_recordings"
SYNTH_NPZ_DIR = ROOT / "data" / "dataset_v8_synth"
COMBINED_DIR = ROOT / "data" / "dataset_v8_combined"
RENDER_LIST_FILE = ROOT / "data" / "selected_render_list.json"

for p in [str(ROOT), r"C:\Gamma\Osu_module"]:
    if p not in sys.path:
        sys.path.insert(0, p)

from osu2.build_v8 import build_dataset_v8
from osu2.fast_train import train as fast_train


def prepare_danser_config(fps: int = 60, songs_dir: str = r"C:\Semga\lazer_songs") -> str:
    preset_name = "ml_synth_x264"
    src_preset = DANSER_DIR / "settings" / f"{preset_name}.json"
    if not src_preset.exists():
        src_preset = DANSER_DIR / "settings" / "ml_synth_shift.json"

    cfg_data = json.loads(src_preset.read_text(encoding="utf-8"))
    cfg_data.setdefault("General", {})["OsuSongsDir"] = songs_dir
    cfg_data.setdefault("Playfield", {})["OsuShift"] = True
    cfg_data["Playfield"]["DrawCursors"] = False

    rec = cfg_data.setdefault("Recording", {})
    rec["FPS"] = int(fps)
    rec["FrameWidth"] = 1920
    rec["FrameHeight"] = 1080
    rec["Encoder"] = "libx264"

    target_preset = f"ml_active_{fps}fps"
    target_path = DANSER_DIR / "settings" / f"{target_preset}.json"
    target_path.write_text(json.dumps(cfg_data, indent=2), encoding="utf-8")
    return target_preset


def render_all_maps(render_list_path: Path = RENDER_LIST_FILE):
    SYNTH_REC_DIR.mkdir(parents=True, exist_ok=True)
    with open(render_list_path, "r", encoding="utf-8") as f:
        maps = json.load(f)

    print(f"=== 1. STARTING HEADLESS DANSER RENDERING ({len(maps)} MAPS) ===", flush=True)
    preset_name = prepare_danser_config(fps=60, songs_dir=r"C:\Semga\lazer_songs")

    for i, m in enumerate(maps, 1):
        safe_title = "".join(c if c.isalnum() else "_" for c in f"{m['title']}_{m['difficulty']}")[:45].strip("_")
        session_name = f"synth_{safe_title}_{int(m['stars']*10)}star"
        target_session = SYNTH_REC_DIR / session_name
        target_video = target_session / "video.mp4"
        target_stamps = target_session / "stamps.npy"
        target_manifest = target_session / "policy_manifest.json"

        if target_video.exists() and target_stamps.exists() and target_manifest.exists():
            print(f"[{i}/{len(maps)}] Already rendered: {session_name}", flush=True)
            continue

        print(f"\n[{i}/{len(maps)}] Rendering: {m['artist']} - {m['title']} [{m['difficulty']}] ({m['stars']}*)", flush=True)
        t0 = time.perf_counter()
        out_stem = f"tmp_render_{i}_{int(time.time())}"

        cmd = [
            str(DANSER_EXE),
            f"-settings={preset_name}",
            f"-title={m['title']}",
            f"-difficulty={m['difficulty']}",
            f"-out={out_stem}",
            "-record",
            "-noupdatecheck",
            "-nodbcheck",
            "-preciseprogress"
        ]

        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        proc = subprocess.Popen(
            cmd,
            cwd=str(DANSER_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=flags
        )

        for line in proc.stdout:
            line = line.strip()
            if "Progress:" in line and ("50%" in line or "100%" in line):
                print(f"  {line}", flush=True)

        code = proc.wait()
        dt = time.perf_counter() - t0

        danser_out = DANSER_DIR / "videos" / f"{out_stem}.mp4"
        if code != 0 or not danser_out.exists():
            print(f"  FAILED to render {session_name} (code={code})", flush=True)
            continue

        target_session.mkdir(parents=True, exist_ok=True)
        shutil.move(str(danser_out), str(target_video))

        # Inspect video frame count
        cap = cv2.VideoCapture(str(target_video))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_real = cap.get(cv2.CAP_PROP_FPS) or 60.0
        cap.release()

        frame_interval_ms = 1000.0 / fps_real
        stamps = np.arange(n_frames, dtype=np.float32) * frame_interval_ms
        np.save(str(target_stamps), stamps)

        manifest = {
            "version": 8,
            "synthetic": True,
            "renderer": "danser-go",
            "fps": fps_real,
            "beatmap": {
                "path": m["osu_path"],
                "artist": m["artist"],
                "title": m["title"],
                "version": m["difficulty"]
            },
            "clock": {
                "formula": "beatmap_ms = scale * recording_ms + offset_ms",
                "offset_ms": 0.0,
                "scale": 1.0,
                "max_time_uncertainty_ms": 0.0,
                "click_safe": True
            },
            "frames": n_frames,
            "playfield_roi": [295, 35, 1625, 1075]
        }
        target_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Rendered {n_frames} frames ({n_frames/fps_real:.1f}s) in {dt:.1f}s ({dt/(n_frames/fps_real):.2f}x real-time) -> {target_session.name}", flush=True)


def build_synthetic_npz():
    print(f"\n=== 2. BUILDING SYNTHETIC NPZ DATASET IN {SYNTH_NPZ_DIR} ===", flush=True)
    build_dataset_v8(
        source_root=str(SYNTH_REC_DIR),
        output_root=str(SYNTH_NPZ_DIR),
        width=96,
        height=72
    )


def combine_datasets():
    print(f"\n=== 3. COMBINING DATASETS INTO {COMBINED_DIR} ===", flush=True)
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)

    v8_index_file = ROOT / "data" / "dataset_v8" / "index.json"
    synth_index_file = SYNTH_NPZ_DIR / "index.json"

    combined_sessions = []
    total_frames = 0

    if v8_index_file.exists():
        v8_data = json.loads(v8_index_file.read_text(encoding="utf-8"))
        for s in v8_data.get("sessions", []):
            npz_path = Path(s["path"])
            if npz_path.exists():
                combined_sessions.append({"name": s["name"], "path": str(npz_path.resolve()), "frames": s["frames"]})
                total_frames += s["frames"]
        print(f"Loaded {len(v8_data.get('sessions', []))} baseline sessions ({total_frames:,} frames)", flush=True)

    synth_frames = 0
    if synth_index_file.exists():
        synth_data = json.loads(synth_index_file.read_text(encoding="utf-8"))
        for s in synth_data.get("sessions", []):
            npz_path = Path(s["path"])
            if npz_path.exists():
                combined_sessions.append({"name": s["name"], "path": str(npz_path.resolve()), "frames": s["frames"]})
                synth_frames += s["frames"]
                total_frames += s["frames"]
        print(f"Added {len(synth_data.get('sessions', []))} synthetic sessions ({synth_frames:,} frames)", flush=True)

    combined_index = {
        "dataset": "dataset_v8_combined",
        "version": 8,
        "total_frames": total_frames,
        "sessions": combined_sessions
    }
    out_idx = COMBINED_DIR / "index.json"
    out_idx.write_text(json.dumps(combined_index, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Combined dataset ready: {len(combined_sessions)} sessions, {total_frames:,} frames -> {out_idx}", flush=True)
    return str(COMBINED_DIR)


def train_model_0_4(dataset_path: str):
    print(f"\n=== 4. TRAINING Semga-OSU-0.4-SYNTHETIC-EXP ON {dataset_path} ===", flush=True)
    out_path = str(ROOT / "data" / "models" / "Semga-OSU-0.4-SYNTHETIC-EXP.pt")

    fast_train(
        root=dataset_path,
        output=out_path,
        batch=128,
        stride=3,
        lr=1e-3,
        use_unet=True,
        action_fusion="tc_roi",
        tth_scale_ms=500.0,
        pos_weight=2.0,
        weighting="decoupled",
        version="Semga-OSU-0.4-SYNTHETIC-EXP",
        method="two_stage_freeze",
        stage1_epochs=2,
        stage2_epochs=15
    )
    print("=== TRAINING COMPLETE! ===", flush=True)


if __name__ == "__main__":
    t_start = time.perf_counter()
    render_all_maps()
    build_synthetic_npz()
    combined_path = combine_datasets()
    train_model_0_4(combined_path)
    t_total = time.perf_counter() - t_start
    print(f"\nFULL PIPELINE COMPLETED IN {t_total:.1f}s ({t_total/60.0:.2f} min)", flush=True)
