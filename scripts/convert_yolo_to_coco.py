"""Convert the DigitizePID YOLO dataset (data/raw/digitize_pid_yolo) into the canonical
COCO layout the pipeline consumes (data/canonical/).

DigitizePID ships symbol boxes as 0-indexed YOLO labels (`class cx cy w h`, normalized)
with images already split into train/ and val/. This script:

  1. maps each of the 32 YOLO class ids -> a fine name (Symbol_<n>, 1-indexed per the
     Digitize-PID paper / Fig. 3) AND a coarse supercategory (COARSE_MAP below), derived
     by matching each symbol to ISO 10628-2 groups (equipment/piping) or ISA-5.1
     (instrument bubbles c25-c31). See docs / memory for the full provenance table.
  2. converts YOLO center-xywh (normalized) -> COCO bbox [x, y, w, h] (top-left, pixels),
     reading each image's true width/height (they are NOT all identical: 7168x4561 vs
     7168x4562).
  3. copies images into canonical/images/ and prefixes every COCO file_name with
     `synthetic/` so build_manifest's source inference (`_source_of`) treats them as the
     synthetic source (-> train/val split, never forced to test). The original train/val
     assignment is preserved separately in a split map (see --split-map-out) because
     build_manifest re-derives its own split; a follow-up step wires that back in.

`build_classes` then derives classes.yaml (fine<->coarse, fgc_groups) straight from the
COCO `categories` we emit here — so fgc_groups fall out automatically as the coarse
classes with >1 fine child (valve / blind_disc / heat_exchanger / instrument).

Standalone utility (scripts/), not a Hydra pipeline stage. Run:
    uv run python scripts/convert_yolo_to_coco.py
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

# --- 32-class coarse map (class_id 0-indexed -> coarse supercategory) -------------- #
# Fine name is Symbol_<class_id+1> (paper/Fig.3 numbering). Coarse per the provenance
# table: c0-c15 valve (ISO G21), c16-18 blind_disc (ISO G25 disc), c19 reducer, c20
# flange, c21-22 heat_exchanger (ISO G3), c23 flow_direction (ISO G25 annotation, REG#241),
# c24 safety_valve (ISO G23 ERV), c25-31 instrument (ISA-5.1 bubbles).
COARSE_MAP: dict[int, str] = {
    **{c: "valve" for c in range(0, 16)},
    16: "blind_disc",
    17: "blind_disc",
    18: "blind_disc",
    19: "reducer",
    20: "flange",
    21: "heat_exchanger",
    22: "heat_exchanger",
    23: "flow_direction",
    24: "safety_valve",
    **{c: "instrument" for c in range(25, 32)},
}
NUM_CLASSES = 32


def fine_name(class_id: int) -> str:
    """YOLO class id (0-indexed) -> fine class name Symbol_<n> (1-indexed per paper)."""
    return f"Symbol_{class_id + 1}"


def _read_yolo_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    rows: list[tuple[int, float, float, float, float]] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        cid = int(float(parts[0]))
        cx, cy, w, h = (float(v) for v in parts[1:5])
        rows.append((cid, cx, cy, w, h))
    return rows


def _yolo_to_coco_bbox(
    cx: float, cy: float, w: float, h: float, img_w: int, img_h: int
) -> list[float]:
    """Normalized YOLO center-xywh -> COCO [x, y, w, h] (top-left corner, pixels)."""
    bw = w * img_w
    bh = h * img_h
    x = cx * img_w - bw / 2.0
    y = cy * img_h - bh / 2.0
    return [x, y, bw, bh]


def build_categories() -> tuple[list[dict], dict[int, int]]:
    """COCO categories (one per YOLO class) + a map from YOLO class id -> COCO category id.

    Category ids are 1-indexed (COCO convention) and ordered by YOLO class id so fine
    ids stay stable/readable.
    """
    categories: list[dict] = []
    yolo_to_cat: dict[int, int] = {}
    for cid in range(NUM_CLASSES):
        cat_id = cid + 1
        categories.append(
            {
                "id": cat_id,
                "name": fine_name(cid),
                "supercategory": COARSE_MAP[cid],
            }
        )
        yolo_to_cat[cid] = cat_id
    return categories, yolo_to_cat


def convert(
    src: Path, out: Path, splits: tuple[str, ...] = ("train", "val")
) -> tuple[dict, dict[str, str]]:
    """Build the COCO dict + an image_id->split map from the YOLO dataset at `src`."""
    categories, yolo_to_cat = build_categories()
    images: list[dict] = []
    annotations: list[dict] = []
    split_map: dict[str, str] = {}

    coco_image_id = 0
    ann_id = 0
    class_hist: Counter[str] = Counter()

    out_images = out / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    for split in splits:
        img_dir = src / "images" / split
        lbl_dir = src / "labels" / split
        if not img_dir.is_dir():
            raise FileNotFoundError(f"Missing image dir: {img_dir}")

        for img_path in sorted(img_dir.glob("*.jpg"), key=lambda p: int(p.stem)):
            image_id = img_path.stem  # e.g. "0" (manifest-style id, no extension)
            with Image.open(img_path) as im:
                img_w, img_h = im.size

            coco_image_id += 1
            # file_name carries the synthetic/ prefix so build_manifest classifies the
            # source correctly; the actual file is copied to canonical/images/<id>.jpg.
            images.append(
                {
                    "id": coco_image_id,
                    "file_name": f"synthetic/{img_path.name}",
                    "width": img_w,
                    "height": img_h,
                }
            )
            split_map[image_id] = split
            shutil.copy2(img_path, out_images / img_path.name)

            for cid, cx, cy, w, h in _read_yolo_label(lbl_dir / f"{image_id}.txt"):
                if cid not in yolo_to_cat:
                    raise ValueError(f"{img_path}: class id {cid} outside 0..{NUM_CLASSES - 1}")
                bbox = _yolo_to_coco_bbox(cx, cy, w, h, img_w, img_h)
                ann_id += 1
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": coco_image_id,
                        "category_id": yolo_to_cat[cid],
                        "bbox": [round(v, 2) for v in bbox],
                        "area": round(bbox[2] * bbox[3], 2),
                        "iscrowd": 0,
                    }
                )
                class_hist[COARSE_MAP[cid]] += 1

    coco = {"images": images, "annotations": annotations, "categories": categories}
    print(f"Images: {len(images)} | annotations: {len(annotations)}")
    print("Coarse distribution:")
    for name, n in sorted(class_hist.items(), key=lambda kv: -kv[1]):
        print(f"  {name:16s} {n}")
    return coco, split_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert DigitizePID YOLO -> canonical COCO")
    parser.add_argument("--src", default="data/raw/digitize_pid_yolo", help="YOLO dataset root")
    parser.add_argument("--out", default="data/canonical", help="canonical/ output dir")
    parser.add_argument(
        "--split-map-out",
        default="data/canonical/digitizepid_splits.json",
        help="where to write the original image_id->split map (for split preservation)",
    )
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    coco, split_map = convert(src, out)

    coco_path = out / "annotations.coco.json"
    with coco_path.open("w") as f:
        json.dump(coco, f)
    print(f"Wrote {coco_path}")

    split_path = Path(args.split_map_out)
    with split_path.open("w") as f:
        json.dump(split_map, f, indent=2)
    print(f"Wrote {split_path} ({len(split_map)} images)")


if __name__ == "__main__":
    main()
