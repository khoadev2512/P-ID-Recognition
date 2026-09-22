"""Pixel/normalized box conversions and clipping shared by data/detect/eval stages.

Ported from PID_Symbol_Detection's `utils/bbox_utils.py` (BBoxUtils), adapted to this
project's conventions: module-level functions (not a static-method class) operating on
plain tuples / this project's `BBox` dataclass, and YOLO rows keyed by an explicit
`class_id: int` rather than an ad-hoc column-count sniff.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from utils.types import BBox


def yolo_to_xyxy(box: BBox, image_w: int, image_h: int) -> tuple[float, float, float, float]:
    """Normalized YOLO center-xywh -> pixel-space corner-xyxy."""
    x1, y1, x2, y2 = box.to_xyxy()
    return (x1 * image_w, y1 * image_h, x2 * image_w, y2 * image_h)


def xyxy_to_yolo(xyxy: tuple[float, float, float, float], image_w: int, image_h: int) -> BBox:
    """Pixel-space corner-xyxy -> normalized YOLO center-xywh."""
    x1, y1, x2, y2 = xyxy
    box = BBox.from_xyxy(x1, y1, x2, y2)
    return BBox(x=box.x / image_w, y=box.y / image_h, w=box.w / image_w, h=box.h / image_h)


def clip_xyxy(
    xyxy: tuple[float, float, float, float], image_w: int, image_h: int
) -> tuple[int, int, int, int]:
    """Clip a pixel-space xyxy box to valid image bounds, returning ints."""
    x1, y1, x2, y2 = xyxy
    x1 = max(0, min(int(x1), image_w - 1))
    y1 = max(0, min(int(y1), image_h - 1))
    x2 = max(0, min(int(x2), image_w - 1))
    y2 = max(0, min(int(y2), image_h - 1))
    return (x1, y1, x2, y2)


def crop_xyxy(image: np.ndarray, xyxy: tuple[int, int, int, int]) -> np.ndarray:
    """Crop `image` (H, W, ...) to a pixel-space xyxy box."""
    x1, y1, x2, y2 = xyxy
    return image[y1:y2, x1:x2]


def read_yolo_txt(path: str | Path) -> list[tuple[int, BBox]]:
    """Read a YOLO label file into (class_id, BBox) rows. Ignores a trailing confidence column."""
    rows: list[tuple[int, BBox]] = []
    with Path(path).open("r") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            class_id = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:5])
            rows.append((class_id, BBox(x=cx, y=cy, w=w, h=h)))
    return rows


def write_yolo_txt(path: str | Path, rows: list[tuple[int, BBox]]) -> None:
    """Write (class_id, BBox) rows as a YOLO label file, one line per row."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for class_id, box in rows:
            f.write(f"{class_id} {box.x:.6f} {box.y:.6f} {box.w:.6f} {box.h:.6f}\n")
