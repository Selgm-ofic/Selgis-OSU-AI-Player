"""Create click-safe v5 labels directly from temporal geometry evidence."""
import json, sys
from pathlib import Path


def label_session(session):
    legacy = Path(r"C:\Gamma\Osu_module")
    if str(legacy) not in sys.path: sys.path.insert(0, str(legacy))
    from perception.beatmap import parse_osu_file
    from training.geometry_sync import estimate_geometry_sync, save_geometry_sync
    from training.time_sync import SyncResult
    from training.session_labels import write_policy_labels
    import numpy as np
    session = Path(session)
    meta = json.loads((session / "meta.json").read_text(encoding="utf-8"))
    beatmap = parse_osu_file(meta["beatmap_osu"])
    geometry = estimate_geometry_sync(str(session), beatmap, float(meta.get("rate", 1.0)))
    save_geometry_sync(str(session), beatmap, geometry)
    if not geometry.is_safe(max_uncertainty_ms=30.0) or geometry.uncertainty_ms > 30.0:
        raise RuntimeError(f"geometry sync not click-safe: uncertainty={geometry.uncertainty_ms:.1f}ms coverage={geometry.covered_objects}/{geometry.total_objects}")
    sync = SyncResult(geometry.transform, geometry.score, geometry.runner_up_score,
                      geometry.anchors, min(17.0, geometry.uncertainty_ms), geometry.method)
    stamps = np.load(session / "vision_stamps.npy")
    keep = stamps[(stamps >= geometry.record_start_ms) & (stamps <= geometry.record_end_ms)]
    manifest = write_policy_labels(str(session), beatmap, keep, sync)
    print(f"click-safe labels: {session.name} frames={manifest['frames']} uncertainty={geometry.uncertainty_ms:.1f}ms", flush=True)
    return manifest
