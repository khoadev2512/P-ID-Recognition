"""Render ISO P&IDs by REPLACING DigitizePID symbols in place (SynthPID-style).

Instead of scattering ISO symbols on a cropped background (scripts/render_pid.py), this
keeps each real DigitizePID sheet — its pipes, text, title block, and realistic layout —
and only swaps every symbol for an ISO-standard one of the same coarse class. The result
is a P&ID that is visually real in every respect EXCEPT the symbols, which are now 100%
ISO. This gives true process topology (inherited from the real sheet) and no legend
overlap (symbols land exactly where real symbols were), the two weaknesses of the
scatter approach.

Per symbol box in a DigitizePID YOLO label:
  1. erase the old symbol (paint its box white),
  2. pick a random ISO template whose coarse matches (via _COARSE_MAP), fit it to the box,
  3. paste it, and emit a COCO annotation with the ISO fine_name.

Coarse mapping DigitizePID -> ISO (from docs/symbol_mapping/symbol_map.csv + vocab.csv):
  valve/blind_disc/heat_exchanger/safety_valve -> same coarse
  reducer/flange/flow_direction                 -> piping (they are piping fines in ISO)
  instrument (ISA bubbles, no ISO equivalent)    -> agitator (the chosen ISO FGC stand-in)

Run (after extract_iso_symbols; needs data/raw/digitize_pid_yolo + iso_symbols_clean):
    uv run python scripts/render_pid_replace.py --out data/canonical_iso
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# DigitizePID coarse -> ISO coarse. Keys are the `coarse` column of symbol_map.csv.
_COARSE_MAP = {
    "valve": "valve",
    "blind_disc": "blind_disc",
    "heat_exchanger": "heat_exchanger",
    "safety_valve": "safety_valve",
    "reducer": "piping",
    "flange": "piping",
    "flow_direction": "piping",
    "instrument": "agitator",  # ISO has no instrument bubble; agitator is the FGC stand-in
}


@dataclass
class IsoTemplate:
    fine_name: str
    coarse: str
    is_fgc: bool
    img: Image.Image  # RGBA, white -> transparent


def _white_to_transparent(img: Image.Image, thresh: int = 235) -> Image.Image:
    rgba = img.convert("RGBA")
    arr = np.array(rgba)
    white = (arr[:, :, 0] > thresh) & (arr[:, :, 1] > thresh) & (arr[:, :, 2] > thresh)
    arr[white, 3] = 0
    return Image.fromarray(arr)


def load_iso_library(symbols_dir: Path, vocab_csv: Path) -> dict[str, list[IsoTemplate]]:
    """ISO templates grouped by coarse, so we can pick a same-class replacement."""
    by_coarse: dict[str, list[IsoTemplate]] = defaultdict(list)
    for row in csv.DictReader(vocab_csv.open()):
        png = symbols_dir / f"{row['reg_number']}.png"
        if not png.exists():
            continue
        t = IsoTemplate(
            fine_name=row["fine_name"],
            coarse=row["coarse"],
            is_fgc=row["fgc_group"].strip().lower() == "yes",
            img=_white_to_transparent(Image.open(png)),
        )
        by_coarse[row["coarse"]].append(t)
    return by_coarse


def _digitizepid_class_to_coarse(symbol_map_csv: Path) -> dict[int, str]:
    """YOLO class_id (0-indexed) -> DigitizePID coarse, from symbol_map.csv."""
    out = {}
    for row in csv.DictReader(symbol_map_csv.open()):
        out[int(row["class_id"])] = row["coarse"]
    return out


def build_categories(
    iso_by_coarse: dict[str, list[IsoTemplate]],
) -> tuple[list[dict], dict[str, int]]:
    """COCO categories for every ISO fine class actually reachable via _COARSE_MAP, with
    an `fgc` flag (so build_classes marks the right FGC families)."""
    reachable = set(_COARSE_MAP.values())
    cats, name_to_id = [], {}
    i = 0
    for coarse, templates in iso_by_coarse.items():
        if coarse not in reachable:
            continue
        for t in templates:
            if t.fine_name in name_to_id:
                continue
            i += 1
            cats.append({"id": i, "name": t.fine_name, "supercategory": coarse, "fgc": t.is_fgc})
            name_to_id[t.fine_name] = i
    return cats, name_to_id


def _read_yolo(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    rows = []
    if not label_path.exists():
        return rows
    for line in label_path.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        rows.append((int(float(p[0])), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    return rows


def replace_symbols_on_image(
    img: Image.Image,
    boxes: list[tuple[int, int, int, int]],  # pixel x0,y0,x1,y1
    coarses: list[str],
    iso_by_coarse: dict[str, list[IsoTemplate]],
    name_to_id: dict[str, int],
    rng,
) -> tuple[Image.Image, list[dict]]:
    """Erase each old symbol, paste an ISO replacement; return (canvas, COCO anns no-id)."""
    canvas = img.convert("RGB")
    anns = []
    from PIL import ImageDraw

    draw = ImageDraw.Draw(canvas)
    for (x0, y0, x1, y1), dp_coarse in zip(boxes, coarses, strict=True):
        iso_coarse = _COARSE_MAP.get(dp_coarse)
        templates = iso_by_coarse.get(iso_coarse, []) if iso_coarse else []
        if not templates:
            continue
        t = rng.choice(templates)
        bw, bh = x1 - x0, y1 - y0
        if bw < 4 or bh < 4:
            continue
        # erase old symbol (white box), then fit ISO into the same box (keep aspect)
        draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255))
        sym = t.img
        scale = min(bw / sym.width, bh / sym.height)
        nw, nh = max(2, int(sym.width * scale)), max(2, int(sym.height * scale))
        sym = sym.resize((nw, nh), Image.LANCZOS)
        px = x0 + (bw - nw) // 2
        py = y0 + (bh - nh) // 2
        canvas.paste(sym, (px, py), sym)
        anns.append(
            {"category_id": name_to_id[t.fine_name],
             "bbox": [px, py, nw, nh], "area": nw * nh, "iscrowd": 0}
        )
    return canvas, anns


def render(
    dpid_root: Path,
    iso_by_coarse: dict[str, list[IsoTemplate]],
    cls_to_coarse: dict[int, str],
    out_dir: Path,
    seed: int = 42,
) -> None:
    rng = random.Random(seed)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    cats, name_to_id = build_categories(iso_by_coarse)

    coco_images, coco_anns, split_map = [], [], {}
    ann_id = 0
    img_id = 0
    for split in ("train", "val"):
        img_dir = dpid_root / "images" / split
        lbl_dir = dpid_root / "labels" / split
        if not img_dir.is_dir():
            continue
        for img_path in sorted(img_dir.glob("*.jpg"), key=lambda p: p.stem):
            rows = _read_yolo(lbl_dir / f"{img_path.stem}.txt")
            if not rows:
                continue
            im = Image.open(img_path)
            W, H = im.size
            boxes, coarses = [], []
            for cid, cx, cy, w, h in rows:
                coarse = cls_to_coarse.get(cid)
                if coarse is None:
                    continue
                x0 = int((cx - w / 2) * W)
                y0 = int((cy - h / 2) * H)
                x1 = int((cx + w / 2) * W)
                y1 = int((cy + h / 2) * H)
                boxes.append((x0, y0, x1, y1))
                coarses.append(coarse)
            canvas, anns = replace_symbols_on_image(
                im, boxes, coarses, iso_by_coarse, name_to_id, rng
            )
            img_id += 1
            out_name = f"iso_{img_path.stem}"
            canvas.save(images_dir / f"{out_name}.jpg", quality=92)
            split_map[out_name] = split
            coco_images.append(
                {"id": img_id, "file_name": f"synthetic/{out_name}.jpg", "width": W, "height": H}
            )
            for a in anns:
                ann_id += 1
                coco_anns.append({"id": ann_id, "image_id": img_id, **a})

    coco = {"images": coco_images, "annotations": coco_anns, "categories": cats}
    (out_dir / "annotations.coco.json").write_text(json.dumps(coco))
    (out_dir / "digitizepid_splits.json").write_text(json.dumps(split_map, indent=2))
    n_coarse = len({c["supercategory"] for c in cats})
    print(f"Rendered {img_id} images, {ann_id} annotations -> {out_dir}")
    print(f"  categories: {len(cats)} fine, {n_coarse} coarse "
          f"(real DigitizePID topology, symbols replaced with ISO)")


def main() -> None:
    p = argparse.ArgumentParser(description="Replace DigitizePID symbols with ISO (in place)")
    p.add_argument("--dpid-root", default="data/raw/digitize_pid_yolo")
    p.add_argument("--symbols-dir", default="data/raw/iso_symbols_clean")
    p.add_argument("--vocab", default="data/raw/iso_symbols/vocab.csv")
    p.add_argument("--symbol-map", default="docs/symbol_mapping/symbol_map.csv")
    p.add_argument("--out", default="data/canonical_iso")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    iso_by_coarse = load_iso_library(Path(args.symbols_dir), Path(args.vocab))
    cls_to_coarse = _digitizepid_class_to_coarse(Path(args.symbol_map))
    print(f"ISO coarse groups: {sorted(iso_by_coarse)}")
    print(f"DigitizePID classes mapped: {len(cls_to_coarse)}")
    render(Path(args.dpid_root), iso_by_coarse, cls_to_coarse, Path(args.out), seed=args.seed)


if __name__ == "__main__":
    main()
