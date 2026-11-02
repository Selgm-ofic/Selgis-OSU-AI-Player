"""Verification script for dataset_v7_1."""

import json
from pathlib import Path
import sys
import numpy as np

legacy_root = Path(r"C:\Selgis OSU AI Player")
if str(legacy_root) not in sys.path:
    sys.path.insert(0, str(legacy_root))
from perception.beatmap import parse_osu_file
from training.action_timeline import TimeTransform, cursor_target_at


def verify():
    data_dir = Path(r"C:\Selgis OSU AI Player\data\dataset_v7_1")
    index_file = data_dir / "index.json"

    assert index_file.exists(), "index.json missing"
    index = json.loads(index_file.read_text(encoding="utf-8"))

    print("=== DATASET V7.1 VERIFICATION ===")
    print(f"Total sessions in index: {len(index['sessions'])}")
    print(f"Total frames recorded in index: {index['total_frames']}")
    print(f"Channels: {index['channels']}")
    print(f"Lookahead: {index['lookahead_ms']} ms")
    print(f"FPS Target: {index['fps_target']}")

    total_frames_counted = 0
    total_valid_counted = 0
    total_clicks_counted = 0
    total_holds_counted = 0
    max_cursor_spike_val = 0

    print("\nDetailed per-session audit:")
    print(f"{'Session':<55} | {'Frames':<7} | {'Valid':<7} | {'Clicks':<6} | {'Holds':<6} | {'CursorLeak':<10}")
    print("-" * 105)

    for s in index["sessions"]:
        p = Path(s["path"])
        assert p.exists(), f"Missing file: {p}"
        z = np.load(p)

        for key in ["images", "targets", "valid", "stamps", "version"]:
            assert key in z, f"Key {key} missing in {p.name}"

        images = z["images"]
        targets = z["targets"]
        valid = z["valid"]
        stamps = z["stamps"]
        version = int(z["version"])

        n = len(images)
        total_frames_counted += n

        assert images.shape == (n, 2, 72, 96), f"Bad images shape: {images.shape}"
        assert images.dtype == np.uint8, f"Bad images dtype: {images.dtype}"
        assert targets.shape == (n, 6), f"Bad targets shape: {targets.shape}"
        assert targets.dtype == np.float32, f"Bad targets dtype: {targets.dtype}"
        assert valid.shape == (n,), f"Bad valid shape: {valid.shape}"
        assert valid.dtype == np.float32, f"Bad valid dtype: {valid.dtype}"
        assert stamps.shape == (n,), f"Bad stamps shape: {stamps.shape}"
        assert version == 7, f"Bad version: {version}"

        assert np.all(np.isfinite(targets)), f"NaNs or Infs found in targets of {p.name}"
        assert np.all(np.isfinite(valid)), f"NaNs or Infs found in valid of {p.name}"
        assert np.all(np.isfinite(stamps)), f"NaNs or Infs found in stamps of {p.name}"

        assert np.all(targets[:, 0] >= 0.0) and np.all(targets[:, 0] <= 1.0), "targets[:,0] out of bounds"
        assert np.all(targets[:, 1] >= 0.0) and np.all(targets[:, 1] <= 1.0), "targets[:,1] out of bounds"
        assert np.all(targets[:, 2] >= 0.0), "targets[:,2] negative"
        assert np.all(targets[:, 3] >= 0.0) and np.all(targets[:, 3] <= 1.0), "targets[:,3] out of bounds"
        assert np.all(targets[:, 4] >= 0.0) and np.all(targets[:, 4] <= 1.0), "targets[:,4] out of bounds"
        assert np.all((valid == 0.0) | (valid == 1.0)), "valid mask contains non-binary values"

        n_valid = int(np.sum(valid > 0.5))
        n_clicks = int(np.sum(targets[:, 3] > 0.5))
        n_holds = int(np.sum(targets[:, 4] > 0.5))

        total_valid_counted += n_valid
        total_clicks_counted += n_clicks
        total_holds_counted += n_holds

        # Cursor leakage test: sample every 25th frame
        sess_source = Path(r"C:\Gamma\Osu_module\data\recordings") / s["name"]
        mf = json.loads((sess_source / "policy_manifest.json").read_text(encoding="utf-8"))
        bm = parse_osu_file(mf["beatmap"]["path"])
        tf = TimeTransform(float(mf["clock"]["offset_ms"]), float(mf["clock"]["scale"]))

        ch1 = images[:, 1]
        sample_indices = np.arange(10, n - 10, 25)
        leak_vals = []
        for idx in sample_indices:
            map_t = tf.to_map(float(stamps[idx]))
            cx, cy, _ = cursor_target_at(bm, map_t)
            px = int(np.clip((cx * 0.865111 + 0.0652055) * 96, 0, 95))
            py = int(np.clip((cy * 0.814956 + 0.0926181) * 72, 0, 71))
            center_val = int(ch1[idx, py, px])
            leak_vals.append(center_val)
            if center_val > max_cursor_spike_val:
                max_cursor_spike_val = center_val

        max_leak_session = max(leak_vals) if leak_vals else 0
        name_trunc = s["name"][:55]
        print(f"{name_trunc:<55} | {n:<7} | {n_valid:<7} | {n_clicks:<6} | {n_holds:<6} | {max_leak_session:<10}")

    print("-" * 105)
    print(f"TOTALS: frames={total_frames_counted}, valid={total_valid_counted} ({total_valid_counted/total_frames_counted*100:.1f}%), clicks={total_clicks_counted}, holds={total_holds_counted}")
    print(f"Max cursor spike brightness across all checked samples: {max_cursor_spike_val} (target: 0)")
    assert max_cursor_spike_val == 0, f"Cursor leakage not zero: {max_cursor_spike_val}"
    print("\nALL VERIFICATION CHECKS PASSED WITH ZERO ERRORS!")


if __name__ == "__main__":
    verify()
