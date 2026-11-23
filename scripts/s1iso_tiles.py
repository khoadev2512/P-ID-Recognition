"""PLAN_PHASE_C §2N rules 3-4: the one-class (`symbol`) tile set Stage 1 is trained on.

Sources: ISO-train (`--iso-root`, the export of configs/render_iso_s1train.yaml: degraded pages,
COCO, `variants/` + `degraded/` json) and DigitizePID (`--dpid-root`: `canonical/` train split,
val split for YOLO's val). Every COCO box is class 0, ignore regions (leader arrows, flags) too
(rule 3). Each page is cut like `data.tiling` (1024 px, 20% overlap; a box belongs to a tile when
>= 60% of its area lies inside, clipped to it), then tiles are kept by rule 4:

  - every tile with a symbol;
  - tiles without one: <= NEG of the page's kept tiles, frame / text tiles first (hard
    negatives), then tiles with other ink (pipes); blank tiles <= BLANK; drawn with
    Random("s1iso/tiles/<page>").

A tile's kind is read from the page: `sym` (a box), `frametext` (ISO pages only: it reaches
outside the drawing (`drawing_box`) or holds a label (`texts` of the variant, moved by the page's
affine)), `ink` (dark pixels of the clean drawing at the same place: the degradation's small
affine moves a pipe by at most a few px inside a tile; DigitizePID: of the page itself), `blank`.
Pages are never altered. ISO pages of `yolo_val` sheets (s1iso_prepare) and DigitizePID `val`
go to YOLO's val (early stopping only, rule 1).

    python scripts/s1iso_tiles.py --iso-root data/iso_s1train --dpid-root data \
        --out data/s1iso_tiles
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.iso_degrade import corner_box  # noqa: E402

TILE, OVERLAP, MIN_VIS = 1024, 0.2, 0.6
NEG, BLANK = 0.30, 0.05
SYM_SHARE = 1.0 - NEG - BLANK
INK_LEVEL, INK_MIN = 128, 20  # a tile has ink when >= INK_MIN pixels are darker than INK_LEVEL
JPEG_Q = 90
Box = tuple[float, float, float, float]


def starts(dim: int, tile: int = TILE, overlap: float = OVERLAP) -> list[int]:
    """Tile starts along one axis, the last one flush with the far edge (as `pipeline.tiling`)."""
    step = int(tile * (1 - overlap))
    s = list(range(0, max(dim - tile, 1), step))
    last = max(dim - tile, 0)
    if not s or s[-1] != last:
        s.append(last)
    return s


def tile_boxes(boxes: list[Box], x: int, y: int, tile: int = TILE) -> list[Box]:
    """Boxes with >= MIN_VIS of their area in the tile, clipped, in tile coordinates."""
    out = []
    for b in boxes:
        x0, y0 = max(b[0], x), max(b[1], y)
        x1, y1 = min(b[2], x + tile), min(b[3], y + tile)
        if x1 <= x0 or y1 <= y0:
            continue
        area = max((b[2] - b[0]) * (b[3] - b[1]), 1e-6)
        if (x1 - x0) * (y1 - y0) / area >= MIN_VIS:
            out.append((x0 - x, y0 - y, x1 - x, y1 - y))
    return out


def _overlaps(b: Box, x: int, y: int, tile: int = TILE) -> bool:
    return b[0] < x + tile and b[2] > x and b[1] < y + tile and b[3] > y


def tile_kind(
    n_boxes: int, x: int, y: int, ink: np.ndarray, drawing: Box | None, texts: list[Box]
) -> str:
    if n_boxes:
        return "sym"
    if drawing is not None and not (
        x >= drawing[0] and y >= drawing[1] and x + TILE <= drawing[2] and y + TILE <= drawing[3]
    ):
        return "frametext"
    if any(_overlaps(t, x, y) for t in texts):
        return "frametext"
    win = ink[y : y + TILE, x : x + TILE]
    return "ink" if int((win < INK_LEVEL).sum()) >= INK_MIN else "blank"


def select(tiles: list[tuple[int, int, str]], key: str) -> list[tuple[int, int, str]]:
    """Rule 4 for one page: all `sym` tiles; negatives <= NEG of the kept tiles (frametext first,
    then ink), blank <= BLANK. A page with no symbol keeps nothing."""
    rng = random.Random(f"s1iso/tiles/{key}")
    by = {k: [t for t in tiles if t[2] == k] for k in ("sym", "frametext", "ink", "blank")}
    n_sym = len(by["sym"])
    if not n_sym:
        return []
    total = n_sym / SYM_SHARE
    n_neg, n_blank = round(NEG * total), round(BLANK * total)
    neg = rng.sample(by["frametext"], min(n_neg, len(by["frametext"])))
    neg += rng.sample(by["ink"], min(n_neg - len(neg), len(by["ink"])))
    blank = rng.sample(by["blank"], min(n_blank, len(by["blank"])))
    return by["sym"] + neg + blank


def _yolo(boxes: list[Box], tile: int = TILE) -> str:
    return "".join(
        f"0 {(b[0] + b[2]) / 2 / tile:.6f} {(b[1] + b[3]) / 2 / tile:.6f}"
        f" {(b[2] - b[0]) / tile:.6f} {(b[3] - b[1]) / tile:.6f}\n"
        for b in boxes
    )


def cut_page(job: dict) -> list[dict]:
    """Cut, select and write the tiles of one page; one row per kept tile."""
    img = cv2.imread(job["image"], cv2.IMREAD_GRAYSCALE)
    ink = cv2.imread(job["ink"], cv2.IMREAD_GRAYSCALE) if job.get("ink") else img
    if ink.shape != img.shape:
        raise ValueError(f"{job['image']}: clean drawing {ink.shape} != page {img.shape}")
    boxes = [tuple(b) for b in job["boxes"]]
    texts = [tuple(t) for t in job.get("texts", [])]
    drawing = tuple(job["drawing"]) if job.get("drawing") else None
    h, w = img.shape
    tiles, inside = [], {}
    for y in starts(h):
        for x in starts(w):
            tb = tile_boxes(boxes, x, y)
            inside[(x, y)] = tb
            tiles.append((x, y, tile_kind(len(tb), x, y, ink, drawing, texts)))
    rows = []
    out = Path(job["out"])
    for x, y, kind in select(tiles, job["key"]):
        name = f"{job['key']}_{x}_{y}"
        crop = img[y : y + TILE, x : x + TILE]
        if crop.shape != (TILE, TILE):  # a page smaller than a tile: pad with paper
            pad = np.full((TILE, TILE), 255, np.uint8)
            pad[: crop.shape[0], : crop.shape[1]] = crop
            crop = pad
        cv2.imwrite(str(out / "images" / job["split"] / f"{name}.jpg"), crop,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])  # fmt: skip
        (out / "labels" / job["split"] / f"{name}.txt").write_text(_yolo(inside[(x, y)]))
        rows.append({"source": job["source"], "split": job["split"], "page": job["key"],
                     "tile": name, "kind": kind, "n_boxes": len(inside[(x, y)])})  # fmt: skip
    summary = Counter(t[2] for t in tiles)
    rows.append({"source": job["source"], "split": job["split"], "page": job["key"], "tile": "",
                 "kind": "_page", "n_boxes": json.dumps(dict(summary))})  # fmt: skip
    return rows


def _xyxy(b) -> list[float]:
    return [b[0], b[1], b[0] + b[2], b[1] + b[3]]


def iso_jobs(root: Path, out: Path, per_sheet: int | None = None) -> list[dict]:
    coco = json.loads((root / "canonical/annotations.coco.json").read_text())
    split = (
        {
            int(r["sheet"]): r.get("split", "train")
            for r in csv.DictReader(open(root / "src" / "mapping.csv"))
        }
        if (root / "src" / "mapping.csv").exists()
        else {}
    )
    boxes: dict[int, list] = {}
    for a in coco["annotations"]:  # rule 3: ignore regions are `symbol` too
        boxes.setdefault(a["image_id"], []).append(_xyxy(a["bbox"]))
    # §2N rule 2: the `per_sheet` lowest kept variant indices of each sheet (= what a build with
    # `variants.count = per_sheet` keeps: V6 tries indices in order and keeps the first that pass)
    first: dict[int, list[int]] = {}
    for im in coco["images"]:
        if im["degraded"]:
            first.setdefault(im["sheet"], []).append(im["variant"])
    chosen = {s: set(sorted(v)[:per_sheet] if per_sheet else v) for s, v in first.items()}
    jobs = []
    for im in coco["images"]:
        if not im["degraded"] or im["variant"] not in chosen[im["sheet"]]:
            continue
        stem = f"v{im['variant']:03d}"
        var = json.loads((root / "variants" / f"sheet{im['sheet']}" / f"{stem}.json").read_text())
        deg = json.loads((root / "degraded" / f"sheet{im['sheet']}" / f"{stem}.json").read_text())
        texts = [list(corner_box(deg["affine"], tuple(t["box"]))) for t in var.get("texts", [])]
        jobs.append({
            "source": "iso", "key": Path(im["file_name"]).stem,
            "image": str(root / "canonical" / "images" / Path(im["file_name"]).name),
            "ink": str(root / "variants" / f"sheet{im['sheet']}" / f"{stem}.png"),
            "boxes": boxes.get(im["id"], []), "texts": texts, "drawing": im.get("drawing_box"),
            "split": "val" if split.get(im["sheet"]) == "yolo_val" else "train", "out": str(out),
        })  # fmt: skip
    return jobs


def dpid_jobs(root: Path, out: Path) -> list[dict]:
    coco = json.loads((root / "canonical/annotations.coco.json").read_text())
    split = {r["id"]: r["split"] for r in csv.DictReader(open(root / "canonical/manifest.csv"))}
    boxes: dict[int, list] = {}
    for a in coco["annotations"]:
        boxes.setdefault(a["image_id"], []).append(_xyxy(a["bbox"]))
    jobs = []
    for im in coco["images"]:
        stem = Path(im["file_name"]).stem
        sp = split.get(stem)
        if sp not in ("train", "val"):
            continue
        jobs.append({
            "source": "digitizepid", "key": f"dpid{stem}",
            "image": str(root / "canonical" / "images" / Path(im["file_name"]).name),
            "boxes": boxes.get(im["id"], []), "split": sp, "out": str(out),
        })  # fmt: skip
    return jobs


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--iso-root", type=Path, required=True)
    ap.add_argument("--dpid-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="pages per source (a dry run)")
    ap.add_argument(
        "--per-sheet", type=int, default=1, help="ISO variants per sheet, lowest indices (rule 2)"
    )
    args = ap.parse_args(argv)
    if (args.out / "data.yaml").exists():
        raise SystemExit(f"{args.out} already holds a tile set")
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    jobs = (
        iso_jobs(args.iso_root, args.out, args.per_sheet)[: args.limit]
        + dpid_jobs(args.dpid_root, args.out)[: args.limit]
    )
    rows: list[dict] = []
    with ProcessPoolExecutor(args.workers) as ex:
        for r in ex.map(cut_page, jobs, chunksize=4):
            rows += r
    with open(args.out / "tiles.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    (args.out / "data.yaml").write_text(
        "# PLAN_PHASE_C §2N: one class. `path` is set where the set is unpacked.\n"
        "path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames: [symbol]\n"
    )
    kept = Counter((r["source"], r["split"], r["kind"]) for r in rows if r["kind"] != "_page")
    pages = Counter((r["source"], r["split"]) for r in rows if r["kind"] == "_page")
    summary = {
        "pages": {f"{s}/{sp}": n for (s, sp), n in sorted(pages.items())},
        "tiles": {f"{s}/{sp}/{k}": n for (s, sp, k), n in sorted(kept.items())},
        "boxes": sum(int(r["n_boxes"]) for r in rows if r["kind"] != "_page"),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
