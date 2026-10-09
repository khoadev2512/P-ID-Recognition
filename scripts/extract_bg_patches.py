"""Extract a pool of EMPTY background patches from the DigitizePID sheets.

The synthetic P&ID renderer (scripts/render_pid.py) pastes clean ISO symbols onto a
realistic background instead of a blank white canvas. Real P&ID "empty" space isn't
white — it carries pipe lines, note tables, borders and scan noise, which is exactly
the context we want the detector to learn to ignore.

DigitizePID symbols cover only ~2% of each 7168x4561 sheet, so 98% is usable empty
space. We use the YOLO labels to crop patches that contain NO symbol, then keep only
patches whose ink ratio (non-white pixels = lines/text/border) falls in a target band:
too white (<min) is no better than a blank canvas; too busy (>max) would clutter the
scene and hide pasted symbols.

Output: data/raw/bg_patches/bg_<i>.png  (+ a count log). Run once:
    uv run python scripts/extract_bg_patches.py --n 300
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image


def _read_yolo_boxes(
    label_path: Path, img_w: int, img_h: int
) -> list[tuple[float, float, float, float]]:
    """Symbol boxes as (cx, cy, w, h) in PIXELS, from a YOLO label file."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cx, cy, w, h = (float(v) for v in parts[1:5])
        boxes.append((cx * img_w, cy * img_h, w * img_w, h * img_h))
    return boxes


def _patch_has_symbol(
    x: int, y: int, size: int, boxes: list[tuple[float, float, float, float]]
) -> bool:
    """True if any symbol box overlaps the patch [x, x+size] x [y, y+size]."""
    px0, py0, px1, py1 = x, y, x + size, y + size
    for cx, cy, bw, bh in boxes:
        bx0, by0, bx1, by1 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
        if bx0 < px1 and bx1 > px0 and by0 < py1 and by1 > py0:
            return True
    return False


def _ink_ratio(patch: np.ndarray) -> float:
    """Fraction of non-white (ink) pixels — lines/text/border density of the patch."""
    return float((patch < 200).mean())


def extract(
    src: Path,
    out_dir: Path,
    n: int,
    size: int = 1024,
    ink_min: float = 0.01,
    ink_max: float = 0.10,
    tries_per_image: int = 20,
    seed: int = 42,
) -> int:
    """Crop up to `n` empty, ink-in-band patches from src/images/** into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    image_files: list[tuple[Path, Path]] = []
    for split in ("train", "val"):
        img_dir = src / "images" / split
        lbl_dir = src / "labels" / split
        if img_dir.is_dir():
            for img in sorted(img_dir.glob("*.jpg")):
                image_files.append((img, lbl_dir / f"{img.stem}.txt"))
    rng.shuffle(image_files)

    saved = 0
    for img_path, lbl_path in image_files:
        if saved >= n:
            break
        im = Image.open(img_path).convert("L")
        W, H = im.size
        if W < size or H < size:
            continue
        boxes = _read_yolo_boxes(lbl_path, W, H)
        for _ in range(tries_per_image):
            if saved >= n:
                break
            x = rng.randint(0, W - size)
            y = rng.randint(0, H - size)
            if _patch_has_symbol(x, y, size, boxes):
                continue
            patch = np.array(im.crop((x, y, x + size, y + size)))
            ink = _ink_ratio(patch)
            if ink_min <= ink <= ink_max:
                Image.fromarray(patch).save(out_dir / f"bg_{saved:04d}.png")
                saved += 1

    return saved


def main() -> None:
    p = argparse.ArgumentParser(description="Extract empty background patches from DigitizePID")
    p.add_argument("--src", default="data/raw/digitize_pid_yolo")
    p.add_argument("--out", default="data/raw/bg_patches")
    p.add_argument("--n", type=int, default=300, help="how many patches to collect")
    p.add_argument("--size", type=int, default=1024, help="patch size (px)")
    p.add_argument("--ink-min", type=float, default=0.01, help="min ink ratio (some line/text)")
    p.add_argument("--ink-max", type=float, default=0.10, help="max ink ratio (avoid clutter)")
    args = p.parse_args()

    saved = extract(
        Path(args.src), Path(args.out), n=args.n, size=args.size,
        ink_min=args.ink_min, ink_max=args.ink_max,
    )
    print(f"Saved {saved} background patches -> {args.out} (ink {args.ink_min}-{args.ink_max})")
    if saved < args.n:
        print(f"  note: wanted {args.n} but only {saved} patches fell in the ink band.")


if __name__ == "__main__":
    main()
