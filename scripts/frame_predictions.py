"""How many predictions fall in the sheet frame (PLAN_PHASE_C §2M item 7, §2N rule 7)?

A framed OPEN100-ISO v2 / P4-val v2 image carries `drawing_box` = [x0, y0, x1, y1] (px) in its COCO
entry: the drawing's rectangle (on a degraded image, the box round its moved corners, so slightly
loose). Nothing outside it is annotated, so a prediction whose box centre lies outside is a false
positive caused by the frame (title block, notes, revision table, zone band).

Reads a folder of detection files named `<image stem>.txt`, either
  Stage 1      `<class> <cx> <cy> <w> <h> <score>`                       (normalised), or
  one-shot     `<label> <best> <cx> <cy> <w> <h> <score> <cos> <cos2> <coarse>`
and counts, over the degraded images (the main view) with score >= --score-thr (eval.score_thr):
predictions, those in the frame and, for one-shot files, those in the frame that Stage 2 still
labelled as a library symbol (not `__other__` / `__unrefined__`): the frame's share of e2e FP.

    python scripts/frame_predictions.py --coco <root>/canonical/annotations.coco.json \\
        --dets runs/v2_csv/detections_agnostic runs/v2_csv/detections_oneshot_dinov2_ft --out x.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

NOT_A_SYMBOL = {"__other__", "__unrefined__"}  # utils.classmap / pipeline.oneshot_infer


def _rows(path: Path) -> list[tuple[str, float, float, float]]:
    """(label or '', cx, cy, score) per line; label only for one-shot files."""
    out = []
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) == 6:
            out.append(("", float(p[1]), float(p[2]), float(p[5])))
        elif len(p) == 10:
            out.append((p[0], float(p[2]), float(p[3]), float(p[6])))
        elif p:
            raise ValueError(f"{path}: expected 6 or 10 columns, got {len(p)}: {line!r}")
    return out


def count(coco: dict, det_dir: Path, score_thr: float = 0.25, degraded_only: bool = True) -> dict:
    images = [im for im in coco["images"] if im.get("degraded") or not degraded_only]
    missing = [im["file_name"] for im in images if "drawing_box" not in im]
    if missing:
        raise ValueError(f"{len(missing)} images without drawing_box (not a framed export?)")
    n = in_frame = sym_in_frame = 0
    labels: Counter[str] = Counter()
    per_image = {}
    oneshot = None
    for im in images:
        stem = Path(im["file_name"]).stem
        f = det_dir / f"{stem}.txt"
        if not f.exists():
            raise FileNotFoundError(f"{f} (detections of {im['file_name']})")
        x0, y0, x1, y1 = im["drawing_box"]
        k = 0
        for label, cx, cy, score in _rows(f):
            if score < score_thr:
                continue
            oneshot = bool(label) if oneshot is None else oneshot
            n += 1
            x, y = cx * im["width"], cy * im["height"]
            if x0 <= x <= x1 and y0 <= y <= y1:
                continue
            in_frame += 1
            k += 1
            if label and label not in NOT_A_SYMBOL:
                sym_in_frame += 1
                labels[label] += 1
        per_image[stem] = k
    out = {
        "dir": str(det_dir), "n_images": len(images), "score_thr": score_thr,
        "n_predictions": n, "in_frame": in_frame, "share_in_frame": in_frame / n if n else None,
        "per_image_in_frame": per_image,
    }  # fmt: skip
    if oneshot:
        out["symbol_in_frame"] = sym_in_frame
        out["labels_in_frame"] = dict(labels.most_common())
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--coco", type=Path, required=True)
    ap.add_argument("--dets", type=Path, nargs="+", required=True)
    ap.add_argument("--score-thr", type=float, default=0.25)
    ap.add_argument("--all-views", action="store_true", help="also the clean images")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    coco = json.loads(args.coco.read_text())
    res = [count(coco, d, args.score_thr, not args.all_views) for d in args.dets]
    for r in res:
        extra = (
            f", gán nhãn ký hiệu {r['symbol_in_frame']} (top: "
            + ", ".join(f"{k} {v}" for k, v in list(r["labels_in_frame"].items())[:5])
            + ")"
            if "symbol_in_frame" in r
            else ""
        )
        share = f"{r['share_in_frame']:.1%}" if r["share_in_frame"] is not None else "-"
        print(
            f"{Path(r['dir']).name}: {r['n_predictions']} dự đoán / {r['n_images']} ảnh,"
            f" trong khung {r['in_frame']} ({share}){extra}"
        )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
