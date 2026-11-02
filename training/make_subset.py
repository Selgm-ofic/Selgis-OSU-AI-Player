"""Стратифицированный сабсет датасета: ~2к кадров пропорционально картам.

Логика (подQuant-вопрос: маленького датасета хватает, но редкие классы
терять нельзя):
  - база: все НЕ-дубликаты (_spx*) семплируем стратифицированно по картам
    (по ~100 с карты -> ~2к);
  - спиннеры: все 91 оригинальный кадр со спиннером + дубли x4;
  - выход: data/yolo2k/{images,labels} + dataset.yaml (исходник не трогаем).

  python training/make_subset.py --total 2000 --spinner-dup 4
"""
import os
import sys
import glob
import shutil
import random
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from capture.manager import load_cfg


def map_tag(fname: str) -> str:
    # synth_<map>_<diff>_<NNNNN>[_spxNN].jpg -> synth_<map>_<diff>
    base = os.path.splitext(os.path.basename(fname))[0]
    if "_spx" in base:
        base = base.split("_spx")[0]
    parts = base.split("_")
    return "_".join(parts[:-1]) if len(parts) > 2 else base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--total", type=int, default=2000)
    ap.add_argument("--spinner-dup", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/yolo2k")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    src = cfg.get("yolo", {}).get("dir", "data/yolo")
    rng = random.Random(args.seed)

    all_imgs = sorted(glob.glob(os.path.join(src, "images", "*.jpg")))
    dupes = [p for p in all_imgs if "_spx" in os.path.basename(p)]
    base = [p for p in all_imgs if "_spx" not in os.path.basename(p)]

    # кадры со спиннером (класс 4) среди базы
    def has_spinner(img_path: str) -> bool:
        lbl = os.path.join(src, "labels",
                           os.path.splitext(os.path.basename(img_path))[0] + ".txt")
        try:
            with open(lbl) as f:
                return any(line.startswith("4 ") for line in f)
        except OSError:
            return False

    spin_base = [p for p in base if has_spinner(p)]
    rest = [p for p in base if p not in set(spin_base)]

    # стратификация остатка по картам
    by_map = defaultdict(list)
    for p in rest:
        by_map[map_tag(p)].append(p)
    per_map = max(1, (args.total - len(spin_base)) // max(len(by_map), 1))
    picked = list(spin_base)
    for tag, lst in sorted(by_map.items()):
        picked += rng.sample(lst, min(per_map, len(lst)))
    # добить до total если не хватило
    if len(picked) < args.total:
        pool = [p for p in rest if p not in set(picked)]
        picked += rng.sample(pool, min(args.total - len(picked), len(pool)))

    # дубли спиннеров: ищем _spx-копии оригиналов (по префиксу имени)
    spin_names = {os.path.splitext(os.path.basename(p))[0] for p in spin_base}
    spin_dupes = [p for p in dupes
                  if os.path.basename(p).split("_spx")[0] in spin_names]
    rng.shuffle(spin_dupes)
    picked += spin_dupes[:len(spin_base) * args.spinner_dup]

    img_out = os.path.join(args.out, "images")
    lbl_out = os.path.join(args.out, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)
    for p in picked:
        b = os.path.basename(p)
        lbl = os.path.join(src, "labels", os.path.splitext(b)[0] + ".txt")
        shutil.copy(p, os.path.join(img_out, b))
        if os.path.exists(lbl):
            shutil.copy(lbl, os.path.join(lbl_out, os.path.basename(lbl)))

    from perception.schema import CLASS_NAMES
    with open(os.path.join(args.out, "dataset.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(args.out)}\n")
        f.write("train: images\nval: images\n\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write(f"names: {CLASS_NAMES}\n")

    from collections import Counter
    c = Counter()
    for b in os.listdir(lbl_out):
        if not b.endswith(".txt"):
            continue
        with open(os.path.join(lbl_out, b)) as f:
            for line in f:
                line = line.strip()
                if line:
                    c[int(line.split()[0])] += 1
    print(f"subset: {len(picked)} frames -> {args.out}")
    print("boxes:", dict(sorted(c.items())), "total:", sum(c.values()))


if __name__ == "__main__":
    main()
