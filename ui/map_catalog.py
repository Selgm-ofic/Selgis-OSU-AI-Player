"""Fast metadata index for hash-addressed osu!lazer beatmap files."""
from dataclasses import dataclass
import os
import re


@dataclass(frozen=True)
class MapEntry:
    path: str
    title: str
    artist: str
    version: str
    approach_rate: float
    circle_size: float
    objects: int
    aliases: str = ""

    @property
    def search_text(self):
        return f"{self.artist} {self.title} {self.version} {self.aliases}".casefold()


def default_lazer_root():
    return os.path.join(os.environ.get("APPDATA", ""), "osu", "files")


def _value(text, key, default=""):
    match = re.search(rf"(?m)^{re.escape(key)}\s*:(.*)$", text)
    return match.group(1).strip() if match else default


def read_entry(path):
    try:
        with open(path, "rb") as handle:
            prefix = handle.read(64)
            if not prefix.startswith(b"osu file format"):
                return None
            remainder = handle.read()
        text = (prefix + remainder).decode("utf-8", errors="ignore")
    except OSError:
        return None
    mode = _value(text, "Mode", "0")
    if mode != "0" or "[HitObjects]" not in text:
        return None
    title_ascii = _value(text, "Title", "?")
    artist_ascii = _value(text, "Artist", "?")
    title_unicode = _value(text, "TitleUnicode")
    artist_unicode = _value(text, "ArtistUnicode")
    title = title_unicode or title_ascii
    artist = artist_unicode or artist_ascii
    version = _value(text, "Version", "?")
    try:
        ar = float(_value(text, "ApproachRate", "5"))
        cs = float(_value(text, "CircleSize", "4"))
    except ValueError:
        ar, cs = 5.0, 4.0
    hit_section = text.split("[HitObjects]", 1)[1]
    objects = sum(bool(line.strip()) and not line.lstrip().startswith("//")
                  for line in hit_section.splitlines())
    aliases = f"{artist_ascii} {title_ascii} {artist_unicode} {title_unicode}"
    return MapEntry(path, title, artist, version, ar, cs, objects, aliases)


def scan_lazer_maps(root=None, progress=None):
    root = root or default_lazer_root()
    entries = []
    checked = 0
    if not os.path.isdir(root):
        return entries
    for folder, _, names in os.walk(root):
        for name in names:
            checked += 1
            entry = read_entry(os.path.join(folder, name))
            if entry is not None:
                entries.append(entry)
            if progress and checked % 1000 == 0:
                progress(checked, len(entries))
    entries.sort(key=lambda e: (e.artist.casefold(), e.title.casefold(),
                                e.version.casefold()))
    return entries


def filter_maps(entries, query):
    words = query.casefold().split()
    if not words:
        return list(entries)
    return [entry for entry in entries
            if all(word in entry.search_text for word in words)]
