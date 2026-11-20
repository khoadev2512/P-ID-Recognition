"""V7 geometry check: do the boxes moved by the affine still fit the moved ink?

For a few variants this redraws the page, turns it by the largest rotation and shear V7 allows
(both signs) and compares, for every annotation, two ways of moving its box with the box found on
the warped image itself (the symbol's own ink, run through cv2.warpAffine, thresholded at 10 %
coverage):

  corners  the four corners moved by the affine, then their axis-aligned box (loose)
  ink      the tight box of the symbol's own ink pixels, each moved by the affine (what V7 stores)

Per way: IoU with the box on the warped image, mean / max edge error (px), and the share of the
warped ink inside the box (enclosure). Also the share of all symbol ink of the warped page that
lies outside every moved box, against the same share on the unrotated page. Writes a CSV row per
annotation and prints a summary by symbol kind. Read-only on data/raw/.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.iso_degrade import _cv_matrix, affine_for, corner_box, ink_box_after  # noqa: E402
from utils.iso_mapping import load_config  # noqa: E402
from utils.iso_templates import load_templates  # noqa: E402
from utils.iso_variants import plan_variant, render_variant  # noqa: E402
from utils.page_svg import rasterize  # noqa: E402
from utils.pid2graph import load_graphml, pipe_runs  # noqa: E402


def own_ink_box(
    m, symbols: np.ndarray, box
) -> tuple[tuple[float, float, float, float], np.ndarray] | None:
    """(box, mask) of the ink of `box` on the image warped by `m`: the ink inside `box` is cut out
    alone (neighbours blanked), warped as an image and thresholded. The mask is the pixels'
    (x, y) page coordinates, N x 2."""
    h, w = symbols.shape
    sx0, sy0 = max(int(np.floor(box[0])), 0), max(int(np.floor(box[1])), 0)
    sx1, sy1 = min(int(np.ceil(box[2])), w), min(int(np.ceil(box[3])), h)
    src = ((symbols[sy0:sy1, sx0:sx1] < 245) * 255).astype(np.uint8)
    if not src.any():
        return None
    cb = corner_box(m, (sx0, sy0, sx1, sy1))
    dx0, dy0 = int(np.floor(cb[0])) - 3, int(np.floor(cb[1])) - 3
    dw, dh = int(np.ceil(cb[2])) - dx0 + 3, int(np.ceil(cb[3])) - dy0 + 3
    mc = _cv_matrix(m)
    a, t = mc[:, :2], mc[:, 2]
    shift = a @ [sx0, sy0] + t - [dx0, dy0]
    warped = cv2.warpAffine(src, np.hstack([a, shift[:, None]]), (dw, dh), flags=cv2.INTER_LINEAR)
    ys, xs = np.nonzero(warped >= 26)  # 10 % coverage
    if xs.size == 0:
        return None
    pts = np.stack([xs + dx0 + 0.5, ys + dy0 + 0.5], axis=1)
    return (
        float(pts[:, 0].min() - 0.5), float(pts[:, 1].min() - 0.5),
        float(pts[:, 0].max() + 0.5), float(pts[:, 1].max() + 0.5),
    ), pts  # fmt: skip


def iou(a, b) -> float:
    iw = max(min(a[2], b[2]) - max(a[0], b[0]), 0)
    ih = max(min(a[3], b[3]) - max(a[1], b[1]), 0)
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def inside(pts: np.ndarray, box) -> float:
    return float(
        (
            (pts[:, 0] >= box[0])
            & (pts[:, 0] <= box[2])
            & (pts[:, 1] >= box[1])
            & (pts[:, 1] <= box[3])
        ).mean()
    )


def kind_of(a: dict, h_over_w: float) -> str:
    name = a["fine_name"] or a["kind"]
    if name.startswith("agitator"):
        return "agitator"
    if name == "column":
        return "column"
    if name in ("vessel_dished_ends", "hx_straight_tube"):
        return "vessel tall" if h_over_w >= 2.0 else "vessel"
    if a["kind"] in ("bubble", "arrow", "leader", "flag"):
        return a["kind"]
    if name == "flanged_connection":
        return "flanged_connection"
    return "iso symbol"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=Path("data/raw/pid2graph/open100"))
    ap.add_argument(
        "--out", type=Path, default=Path("docs/experiments/iso_render/v7/geometry_errors.csv")
    )
    ap.add_argument("--sheets", type=int, nargs="*", default=[0, 3, 5, 6, 9, 10, 11])
    ap.add_argument("--indices", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--rot", type=float, default=1.0)
    ap.add_argument("--shear", type=float, default=0.5)
    args = ap.parse_args()
    cfg = load_config()
    templates = load_templates(cfg["template_dir"], cfg["taxonomy_csv"], cfg["scope"])
    rows, leak = [], []
    for i in args.sheets:
        sheet = load_graphml(args.src / f"{i}.graphml", Image.open(args.src / f"{i}.png").size)
        runs = pipe_runs(sheet)
        for idx in args.indices:
            draft = plan_variant(sheet, runs, templates, cfg, i, idx)
            rendered = render_variant(draft, sheet, runs, cfg)
            symbols = rasterize(rendered.page.layer("symbols"))
            h, w = symbols.shape
            for sign in (1, -1):
                m = affine_for(w, h, sign * args.rot, sign * args.shear)
                moved = []
                for a in rendered.page.annotations:
                    box = tuple(a["bbox"])
                    gt = own_ink_box(m, symbols, box)
                    if gt is None:
                        continue
                    gbox, pts = gt
                    corners = corner_box(m, box)
                    ink = ink_box_after(m, symbols, box)
                    moved.append(ink)
                    hw = (box[3] - box[1]) / max(box[2] - box[0], 1e-6)
                    row = {"sheet": i, "index": idx, "sign": sign, "id": a["id"],
                           "kind": kind_of(a, hw), "h_over_w": round(hw, 2),
                           "box_h": round(box[3] - box[1], 1)}  # fmt: skip
                    for name, b in (("corners", corners), ("ink", ink)):
                        edges = np.abs(np.array(b) - np.array(gbox))
                        row[f"{name}_iou"] = round(iou(b, gbox), 4)
                        row[f"{name}_edge_mean"] = round(float(edges.mean()), 2)
                        row[f"{name}_edge_max"] = round(float(edges.max()), 2)
                        row[f"{name}_enclosed"] = round(inside(pts, b), 4)
                    rows.append(row)
                warped = cv2.warpAffine(
                    ((symbols < 245) * 255).astype(np.uint8), _cv_matrix(m), (w, h)
                )
                mask = warped >= 26
                covered = np.zeros_like(mask)
                for b in moved:
                    covered[
                        max(int(b[1]), 0) : int(b[3]) + 1, max(int(b[0]), 0) : int(b[2]) + 1
                    ] = True
                leak.append((i, idx, sign, float((mask & ~covered).sum() / mask.sum())))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["kind"]].append(r)
    print(f"rotation {args.rot} deg, shear {args.shear} deg, both signs; {len(rows)} annotations")
    print("kind                  n  | IoU (median) corners  ink | max edge error px corners  ink"
          " | enclosed (min) corners  ink")  # fmt: skip

    def stat(g, fn, key):
        return fn(r[key] for r in g)

    for kind, g in sorted(groups.items(), key=lambda t: -len(t[1])):
        med = [float(np.median([r[k] for r in g])) for k in ("corners_iou", "ink_iou")]
        mx = [stat(g, max, k) for k in ("corners_edge_max", "ink_edge_max")]
        mn = [stat(g, min, k) for k in ("corners_enclosed", "ink_enclosed")]
        print(
            f"{kind:<20}{len(g):>5}  | {med[0]:>10.3f} {med[1]:>5.3f} | {mx[0]:>17.1f} "
            f"{mx[1]:>5.1f} | {mn[0]:>19.3f} {mn[1]:>5.3f}"
        )
    print("\nsymbol ink outside every moved box (share of all symbol ink):")
    base = np.mean([v for *_, v in leak])
    print(f"  mean {base:.5f}, max {max(v for *_, v in leak):.5f} over {len(leak)} warped pages")


if __name__ == "__main__":
    main()
