"""Visualize Stage-1 detector output: draw predicted (and optionally ground-truth)
boxes onto the full-resolution canonical P&ID images.

The pipeline only writes detections as YOLO-format .txt (derived/detections/<split>/),
with no visual output — this standalone utility closes that gap for qualitative review
and thesis figures.

Two output kinds per image (both, by default):
  1. full   — the whole diagram with every box drawn, downscaled to a viewable width.
  2. crops  — a few zoomed-in tiles around dense box regions, so small symbols on a
              7168px sheet are actually legible.

Prediction boxes come from derived/detections/<split>/<id>.txt (6 cols:
class_id cx cy w h score, YOLO-normalized — detect_infer's output contract). Optional
ground-truth boxes come from canonical/annotations.coco.json. Coarse class names and
colors come from canonical/classes.yaml.

Standalone (scripts/), not a Hydra stage. Example:
    uv run python scripts/visualize_detections.py --split val --max-images 10
    uv run python scripts/visualize_detections.py --split val --mode compare   # GT vs pred
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import yaml

# Distinct BGR colors cycled per class id (OpenCV is BGR). 12 is plenty for 8 coarse.
_PALETTE = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (128, 0, 255), (0, 128, 255),
    (0, 200, 128), (200, 128, 0), (128, 128, 255), (255, 128, 128),
]
_GT_COLOR = (0, 200, 0)      # ground-truth = green
_PRED_COLOR = (0, 0, 255)    # prediction = red (used in compare mode)


def _load_class_names(classes_yaml: Path) -> list[str]:
    """Coarse class names in id order (matches coco_to_yolo's coarse_id indexing)."""
    with classes_yaml.open() as f:
        raw = yaml.safe_load(f)
    return list(raw["coarse"])


def _color_for(class_id: int) -> tuple[int, int, int]:
    return _PALETTE[class_id % len(_PALETTE)]


def _read_pred_txt(
    path: Path, img_w: int, img_h: int
) -> list[tuple[int, tuple[int, int, int, int], float]]:
    """Parse derived/detections/<id>.txt -> [(class_id, (x1,y1,x2,y2) px, score)].

    6-col (det_only) or 7-col (det_plus_fgc, trailing fine_score ignored) — both share
    the first 6 columns' meaning: class_id cx cy w h score (YOLO-normalized).
    """
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        cid = int(float(parts[0]))
        cx, cy, w, h, score = (float(v) for v in parts[1:6])
        x1 = int((cx - w / 2) * img_w)
        y1 = int((cy - h / 2) * img_h)
        x2 = int((cx + w / 2) * img_w)
        y2 = int((cy + h / 2) * img_h)
        rows.append((cid, (x1, y1, x2, y2), score))
    return rows


def _read_gt_coco(
    coco: dict, image_id: str, cat_id_to_coarse_id: dict[int, int]
) -> list[tuple[int, tuple[int, int, int, int]]]:
    """Ground-truth coarse boxes for one image_id, as [(coarse_id, (x1,y1,x2,y2) px)]."""
    # map stem(file_name) -> coco numeric image id
    coco_img_id = None
    for img in coco.get("images", []):
        if Path(img["file_name"]).stem == image_id:
            coco_img_id = img["id"]
            break
    if coco_img_id is None:
        return []
    rows = []
    for ann in coco.get("annotations", []):
        if ann["image_id"] != coco_img_id:
            continue
        x, y, w, h = ann["bbox"]
        coarse_id = cat_id_to_coarse_id.get(ann["category_id"])
        if coarse_id is None:
            continue
        rows.append((coarse_id, (int(x), int(y), int(x + w), int(y + h))))
    return rows


def _draw_boxes(img, boxes, names, color=None, with_score=True) -> None:
    """Draw boxes in-place. boxes: [(class_id, (x1,y1,x2,y2), [score])]."""
    # Line/font scale with image size so they're visible on a 7168px sheet.
    h = img.shape[0]
    thick = max(2, h // 1000)
    font_scale = max(0.5, h / 2500)
    for item in boxes:
        cid, (x1, y1, x2, y2) = item[0], item[1]
        score = item[2] if len(item) > 2 and with_score else None
        c = color if color is not None else _color_for(cid)
        cv2.rectangle(img, (x1, y1), (x2, y2), c, thick)
        label = names[cid] if 0 <= cid < len(names) else str(cid)
        if score is not None:
            label = f"{label} {score:.2f}"
        cv2.putText(img, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, c, thick)


def _save_full(img, out_path: Path, max_width: int) -> None:
    """Downscale (if wider than max_width) and save the whole annotated diagram."""
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def _save_crops(img, boxes, out_dir: Path, image_id: str, n_crops: int, crop_size: int) -> int:
    """Save up to n_crops zoomed windows centered on box clusters (first boxes used)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = img.shape[:2]
    saved = 0
    for i, item in enumerate(boxes[:n_crops]):
        (x1, y1, x2, y2) = item[1]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        half = crop_size // 2
        cx1, cy1 = max(0, cx - half), max(0, cy - half)
        cx2, cy2 = min(w, cx + half), min(h, cy + half)
        crop = img[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue
        cv2.imwrite(str(out_dir / f"{image_id}_crop{i}.jpg"), crop)
        saved += 1
    return saved


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize Stage-1 detections on P&ID images")
    p.add_argument("--data-root", default="data", help="data/ root")
    p.add_argument("--split", default="val", help="manifest split to visualize")
    p.add_argument("--mode", default="pred", choices=["pred", "gt", "compare"],
                   help="pred=detections only, gt=ground-truth only, compare=both overlaid")
    p.add_argument("--max-images", type=int, default=10, help="how many images to render")
    p.add_argument("--max-width", type=int, default=2000, help="downscale full image to this width")
    p.add_argument("--n-crops", type=int, default=4, help="zoomed crops per image")
    p.add_argument("--crop-size", type=int, default=1024, help="crop window size (px)")
    p.add_argument("--no-crops", action="store_true", help="skip zoomed crops (full image only)")
    p.add_argument("--out", default=None, help="output dir (default: derived/viz/<split>)")
    args = p.parse_args()

    root = Path(args.data_root)
    canonical = root / "canonical"
    names = _load_class_names(canonical / "classes.yaml")
    det_dir = root / "derived" / "detections" / args.split
    out_dir = Path(args.out) if args.out else (root / "derived" / "viz" / args.split)

    # image ids for this split from the manifest
    import csv
    image_ids = []
    with (canonical / "manifest.csv").open() as f:
        for row in csv.DictReader(f):
            if row["split"] == args.split:
                image_ids.append(str(row["id"]))
    image_ids = image_ids[: args.max_images]

    # GT setup (only if needed)
    coco = cat_id_to_coarse_id = None
    if args.mode in ("gt", "compare"):
        with (canonical / "annotations.coco.json").open() as f:
            coco = json.load(f)
        coarse_index = {name: i for i, name in enumerate(names)}
        cat_id_to_coarse_id = {
            c["id"]: coarse_index[c["supercategory"]]
            for c in coco.get("categories", [])
            if c["supercategory"] in coarse_index
        }

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    rendered = 0
    for image_id in image_ids:
        img_path = next((canonical / "images" / f"{image_id}{e}" for e in exts
                         if (canonical / "images" / f"{image_id}{e}").exists()), None)
        if img_path is None:
            print(f"skip {image_id}: no image file")
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"skip {image_id}: unreadable")
            continue
        h, w = img.shape[:2]

        # choose which boxes to draw
        all_boxes_for_crops = []
        if args.mode in ("gt", "compare"):
            gt = _read_gt_coco(coco, image_id, cat_id_to_coarse_id)
            _draw_boxes(img, gt, names, color=_GT_COLOR if args.mode == "compare" else None,
                        with_score=False)
            all_boxes_for_crops = gt
        if args.mode in ("pred", "compare"):
            preds = _read_pred_txt(det_dir / f"{image_id}.txt", w, h)
            _draw_boxes(img, preds, names, color=_PRED_COLOR if args.mode == "compare" else None)
            all_boxes_for_crops = preds or all_boxes_for_crops

        _save_full(img.copy(), out_dir / f"{image_id}_full.jpg", args.max_width)
        n = 0
        if not args.no_crops:
            n = _save_crops(img, all_boxes_for_crops, out_dir / "crops", image_id,
                            args.n_crops, args.crop_size)
        rendered += 1
        print(f"{image_id}: {len(all_boxes_for_crops)} boxes -> full + {n} crops")

    legend = " (compare: GT=green, pred=red)" if args.mode == "compare" else ""
    print(f"\nRendered {rendered} images -> {out_dir}{legend}")


if __name__ == "__main__":
    main()
