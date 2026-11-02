import json
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _nearest(source, query):
    right = np.searchsorted(source, query).clip(0, len(source) - 1)
    left = np.maximum(0, right - 1)
    return np.where(np.abs(source[left] - query) <= np.abs(source[right] - query), left, right)


def build_dataset(source_root, output_root, width=96, height=72):
    source_root, output_root = Path(source_root), Path(output_root)
    legacy_root = source_root.parents[1]
    if str(legacy_root) not in sys.path:
        sys.path.insert(0, str(legacy_root))
    from perception.beatmap import OSU_H, OSU_W, parse_osu_file, slider_ball_pos
    from training.action_timeline import TimeTransform, build_action_targets
    output_root.mkdir(parents=True, exist_ok=True)
    sessions = []
    for session in sorted(source_root.iterdir()):
        out = output_root / f"{session.name}.npz"
        images = None
        if out.exists():
            z = np.load(out)
            if "version" in z and int(z["version"]) == 6 and "targets" in z and z["targets"].shape[1] == 6:
                n = len(z["images"])
                sessions.append({"name": session.name, "path": str(out), "frames": n})
                print(f"cached {session.name}: {n} frames", flush=True)
                continue
            elif "images" in z:
                images = z["images"]
        required = [session / "policy_targets.npz", session / "vision_stamps.npy",
                    session / "policy_manifest.json", session / "vision_frames"]
        if not all(p.exists() for p in required):
            continue
        manifest = json.loads(required[2].read_text(encoding="utf-8"))
        uncertainty = float(manifest.get("clock", {}).get("max_time_uncertainty_ms", 999.0))
        if not manifest.get("clock", {}).get("click_safe", False) or uncertainty > 16.5:
            print(f"skipping dirty session {session.name} (uncertainty={uncertainty:.1f}ms)", flush=True)
            continue
        beatmap = parse_osu_file(manifest["beatmap"]["path"])
        stamps = np.load(required[1])
        pngs = sorted(required[3].glob("*.png"))
        n = min(len(stamps), len(pngs)); stamps, pngs = stamps[:n], pngs[:n]
        transform = TimeTransform(float(manifest["clock"]["offset_ms"]), float(manifest["clock"]["scale"]))
        action_targets = build_action_targets(beatmap, stamps, transform)
        if images is None or len(images) != n:
            images = np.empty((n, 2, height, width), np.uint8)
            for i, path in enumerate(pngs):
                frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if frame is None: raise RuntimeError(f"cannot read {path}")
                images[i, 0] = cv2.resize(frame[:, :, 0], (width, height), interpolation=cv2.INTER_AREA)
                images[i, 1] = cv2.resize(frame[:, :, 1], (width, height), interpolation=cv2.INTER_AREA)
        clicks = np.zeros(n, dtype=np.float32)
        key_events_file = session / "key_events.json"
        if key_events_file.exists():
            events = json.loads(key_events_file.read_text(encoding="utf-8"))
            downs = np.array([float(e["record_ms"]) for e in events if e.get("kind") == "down"], dtype=np.float64)
            if len(downs) > 0:
                idx = _nearest(stamps, downs)
                for down_t, peak_idx in zip(downs, idx):
                    dt = np.abs(stamps - down_t)
                    mask = dt <= 14.0
                    if np.any(mask):
                        soft = np.exp(-0.5 * (dt[mask] / 6.0) ** 2).astype(np.float32)
                        soft = soft / max(1e-5, soft.max())
                        clicks[mask] = np.maximum(clicks[mask], soft)
                    if abs(stamps[peak_idx] - down_t) <= 12.0:
                        clicks[peak_idx] = 1.0
        else:
            for obj in beatmap.objects:
                if obj.is_spinner:
                    continue
                record_t = (obj.time - manifest["clock"]["offset_ms"]) / manifest["clock"]["scale"]
                if stamps[0] - 20.0 <= record_t <= stamps[-1] + 20.0:
                    peak_idx = int(_nearest(stamps, np.array([record_t]))[0])
                    dt = np.abs(stamps - record_t)
                    mask = dt <= 14.0
                    if np.any(mask):
                        soft = np.exp(-0.5 * (dt[mask] / 6.0) ** 2).astype(np.float32)
                        soft = soft / max(1e-5, soft.max())
                        clicks[mask] = np.maximum(clicks[mask], soft)
                    if abs(stamps[peak_idx] - record_t) <= 12.0:
                        clicks[peak_idx] = 1.0

        targets = np.empty((n, 6), dtype=np.float32)
        for i, at in enumerate(action_targets[:n]):
            targets[i, 0] = at.cursor_x * 0.865111 + 0.0652055
            targets[i, 1] = at.cursor_y * 0.814956 + 0.0926181
            targets[i, 2] = at.next_hit_ms
            targets[i, 3] = clicks[i]
            targets[i, 4] = 1.0 if at.hold else 0.0
            targets[i, 5] = float(at.object_index)

        visible_preempt_ms = float(beatmap.preempt_ms)
        valid = np.zeros(n, dtype=np.float32)
        for i in range(n):
            obj_idx = int(targets[i, 5])
            if not (0 <= obj_idx < len(beatmap.objects)):
                continue
            obj = beatmap.objects[obj_idx]
            if obj.is_spinner:
                continue
            map_t = transform.to_map(stamps[i])
            is_slider_active = (targets[i, 4] > 0.5) or (obj.is_slider and obj.time <= map_t <= obj.end_time)
            if is_slider_active or (-80.0 < targets[i, 2] <= visible_preempt_ms):
                valid[i] = 1.0
        out = output_root / f"{session.name}.npz"
        np.savez_compressed(out, images=images, targets=targets, valid=valid,
                            stamps=stamps.astype(np.float32), version=6)
        sessions.append({"name": session.name, "path": str(out), "frames": n})
        print(f"built {session.name}: {n} frames (valid={int(np.sum(valid))}, clicks={int(np.sum(clicks > 0.5))}, hold={int(np.sum(targets[:, 4] > 0.5))})", flush=True)
    index = {"version": 2, "shape": [2, height, width],
             "channels": ["gray", "absdiff_16ms"], "sessions": sessions}
    (output_root / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index


class SequenceDataset(Dataset):
    def __init__(self, paths, sequence=6, stride=2):
        self.sequence, self.rows, self.arrays = sequence, [], []
        for path in paths:
            z = np.load(path)
            item = (z["images"], z["targets"], z["valid"])
            slot = len(self.arrays); self.arrays.append(item)
            n_frames = len(item[0])
            ends = set(range(sequence - 1, n_frames, stride))
            if len(item[1]) > 0 and item[1].shape[1] >= 4:
                click_indices = np.where(item[1][:, 3] > 0.25)[0]
                for ci in click_indices:
                    if ci >= sequence - 1:
                        ends.add(int(ci))
                hold_indices = np.where(item[1][:, 4] > 0.5)[0]
                for hi in hold_indices[::2]:
                    if hi >= sequence - 1:
                        ends.add(int(hi))
            self.rows += [(slot, end) for end in sorted(ends)]
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        slot, end = self.rows[index]; images, targets, valid = self.arrays[slot]
        x = images[end-self.sequence+1:end+1].astype(np.float32) / 255.0
        return (torch.from_numpy(x), torch.from_numpy(targets[end].astype(np.float32)),
                torch.tensor(np.float32(valid[end])))


def build_dataset_v7_1(*args, **kwargs):
    from .build_v7_1 import build_dataset_v7_1 as _build
    return _build(*args, **kwargs)

