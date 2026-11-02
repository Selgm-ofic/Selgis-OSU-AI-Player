import os
import json
import time
import subprocess
from pathlib import Path
from typing import Optional


LAZER_REALM_PATH = Path(os.environ.get("APPDATA", "")) / "osu" / "client.realm"
LAZER_FILES_DIR = Path(os.environ.get("APPDATA", "")) / "osu" / "files"
LAZER_SONGS_DIR = Path(r"C:\Semga\lazer_songs")
MANIFEST_PATH = Path(r"C:\Selgis OSU AI Player\data\lazer_manifest.json")
ROOT_DIR = Path(__file__).resolve().parent.parent
EXTRACTOR_DIR = ROOT_DIR / "tools" / "lazer_extractor"
if not EXTRACTOR_DIR.exists():
    EXTRACTOR_DIR = ROOT_DIR / "scratch" / "lazer_test"


def export_realm_manifest(output_path: Optional[Path] = None) -> Optional[dict]:
    """Execute the .NET dynamic Realm extractor to dump lazer_manifest.json."""
    if output_path is None:
        output_path = MANIFEST_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not LAZER_REALM_PATH.exists():
        return None

    try:
        cmd = ["dotnet", "run", "--", str(output_path.resolve())]
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        res = subprocess.run(
            cmd,
            cwd=str(EXTRACTOR_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=flags,
            timeout=30
        )
        if res.returncode == 0 and output_path.exists():
            with open(output_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None


def sync_lazer_hardlinks(force_export: bool = False) -> tuple[int, int, str]:
    """
    Ensure all beatmaps from osu! lazer client.realm are hardlinked into LAZER_SONGS_DIR.
    Returns (linked_count, total_maps, status_message).
    """
    LAZER_SONGS_DIR.mkdir(parents=True, exist_ok=True)

    # Check if manifest needs update
    need_export = force_export or not MANIFEST_PATH.exists()
    if not need_export and LAZER_REALM_PATH.exists():
        realm_mtime = LAZER_REALM_PATH.stat().st_mtime
        manifest_mtime = MANIFEST_PATH.stat().st_mtime
        if realm_mtime > manifest_mtime:
            need_export = True

    if need_export:
        data = export_realm_manifest(MANIFEST_PATH)
    else:
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = export_realm_manifest(MANIFEST_PATH)

    if not data or "sets" not in data:
        return 0, 0, "Не удалось прочитать манифест osu! lazer"

    linked = 0
    skipped = 0
    total_maps = data.get("total_standard_beatmaps", 0)

    for s in data["sets"]:
        folder_name = s.get("folder_name", "")
        if not folder_name:
            continue
        target_dir = LAZER_SONGS_DIR / folder_name
        target_dir.mkdir(parents=True, exist_ok=True)

        for f in s.get("files", []):
            fname = f.get("filename", "")
            h = f.get("hash", "")
            if not fname or not h or len(h) < 3:
                continue
            src = LAZER_FILES_DIR / h[0] / h[:2] / h
            dst = target_dir / fname
            if dst.exists():
                skipped += 1
                continue
            if not src.exists():
                continue
            try:
                os.link(str(src), str(dst))
                linked += 1
            except Exception:
                pass

    msg = f"Lazer синхронизирован: {linked} новых ссылок, {skipped} уже создано, {total_maps} карт доступно"
    return linked, total_maps, msg


if __name__ == "__main__":
    t0 = time.perf_counter()
    new_links, total, message = sync_lazer_hardlinks(force_export=False)
    t1 = time.perf_counter()
    print(f"{message} ({t1 - t0:.2f}s)")
