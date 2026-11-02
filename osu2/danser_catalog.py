import os
import glob
import re
import time
from pathlib import Path
from typing import Optional

STABLE_SONGS_DIR = Path(r"C:\Users\Game PC\AppData\Local\osu!\Songs")
LAZER_SONGS_DIR = Path(r"C:\Semga\lazer_songs")


def parse_osu_header(file_path: Path) -> Optional[dict]:
    """Fast parse of .osu beatmap header metadata and object count."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as fp:
            content = fp.read()
        parts = content.split("[HitObjects]")
        if len(parts) < 2:
            return None
        header = parts[0]
        hits = parts[1].strip().splitlines()
        if not hits:
            return None

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
        return {
            "title": meta.get("Title", file_path.stem),
            "artist": meta.get("Artist", "Unknown"),
            "version": meta.get("Version", "Normal"),
            "ar": float(meta.get("ApproachRate", 5.0)),
            "cs": float(meta.get("CircleSize", 4.0)),
            "objects": len(hits),
            "duration_s": dur_s,
            "path": str(file_path.resolve()),
            "folder": str(file_path.parent.resolve()),
        }
    except Exception:
        return None


def scan_single_songs_dir(songs_dir: Path, source_label: str) -> list[dict]:
    """Scan all .osu files in a songs directory (1 level of set folders)."""
    if not songs_dir.is_dir():
        return []

    maps = []
    osu_files = list(songs_dir.glob("*/*.osu"))
    for f in osu_files:
        info = parse_osu_header(f)
        if info is not None and info["objects"] > 0:
            info["source"] = source_label
            info["songs_dir"] = str(songs_dir.resolve())
            maps.append(info)
    return maps


def scan_all_catalogs(include_lazer: bool = True, include_stable: bool = True) -> list[dict]:
    """Scan and merge beatmaps from osu! stable and osu! lazer."""
    all_maps = []
    if include_stable and STABLE_SONGS_DIR.exists():
        all_maps.extend(scan_single_songs_dir(STABLE_SONGS_DIR, "osu! stable"))

    if include_lazer and LAZER_SONGS_DIR.exists():
        all_maps.extend(scan_single_songs_dir(LAZER_SONGS_DIR, "osu! lazer"))

    all_maps.sort(key=lambda m: (m["artist"].casefold(), m["title"].casefold(), m["ar"]))
    return all_maps


def scan_danser_maps(songs_dir: Optional[str] = None) -> list[dict]:
    """Backward compatible signature for scanning danser maps."""
    if songs_dir is None:
        return scan_all_catalogs(include_lazer=True, include_stable=True)
    p = Path(songs_dir)
    src_label = "osu! lazer" if "lazer" in str(p).lower() else "osu! stable"
    return scan_single_songs_dir(p, src_label)


if __name__ == "__main__":
    t0 = time.perf_counter()
    maps = scan_all_catalogs(include_lazer=True, include_stable=True)
    t1 = time.perf_counter()
    stable_count = sum(1 for m in maps if m["source"] == "osu! stable")
    lazer_count = sum(1 for m in maps if m["source"] == "osu! lazer")
    print(f"Total: {len(maps)} maps in {t1 - t0:.2f}s (Stable: {stable_count}, Lazer: {lazer_count})")
    for m in maps[:3] + maps[-3:]:
        print(f"[{m['source']}] {m['artist']} - {m['title']} [{m['version']}] | AR={m['ar']} CS={m['cs']} Dur={m['duration_s']:.1f}s")
