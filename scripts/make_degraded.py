"""V7: degrade every kept V6 variant and move its geometry with the pixels.

Reads <variants>/sheet<i>/v<NNN>.{png,json} (make_variants.py), redraws the page from the seeds
(the clean PNG must match the redrawn page: checked), degrades the clean PNG with parameters drawn
from the variant's own `degrade` seed and writes, next to the clean variants:

  <out>/sheet<i>/v<NNN>.jpg    the degraded image (the JPEG of the last degradation step)
  <out>/sheet<i>/v<NNN>.json   params, the clean -> degraded affine (page coordinates), COCO
                               annotations with the moved ink boxes, node boxes / centres, ports,
                               signal lines

Read-only on the clean variants and on data/raw/.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.iso_degrade import (  # noqa: E402
    degrade_image,
    moved_geometry,
    params_dict,
    sample_params,
)
from utils.iso_mapping import load_config  # noqa: E402
from utils.iso_templates import load_templates  # noqa: E402
from utils.iso_variants import derive_seed, plan_variant, render_variant  # noqa: E402
from utils.page_svg import rasterize  # noqa: E402
from utils.pid2graph import load_graphml, pipe_runs  # noqa: E402


def degrade_one(
    meta: dict, clean_png: Path, sheet, runs, templates, cfg: dict
) -> tuple[dict, bytes, dict]:
    """(degraded json, jpeg bytes, seconds per phase) of one clean variant."""
    t0 = time.perf_counter()
    clean = cv2.imread(str(clean_png), cv2.IMREAD_GRAYSCALE)
    draft = plan_variant(sheet, runs, templates, cfg, meta["sheet"], meta["index"])
    rendered = render_variant(draft, sheet, runs, cfg)
    redrawn = rasterize(rendered.page.svg)
    if redrawn.shape != clean.shape or not np.array_equal(redrawn, clean):
        raise RuntimeError(f"{clean_png}: not the picture its seeds produce")
    symbols = rasterize(rendered.page.layer("symbols"))
    t1 = time.perf_counter()
    base = int(cfg["variants"]["base_seed"])
    seed = derive_seed(base, meta["sheet"], meta["index"], "degrade")
    params = sample_params(cfg, meta["stroke_px"], random.Random(seed))
    out, m, jpeg = degrade_image(clean, params)
    t2 = time.perf_counter()
    geo = moved_geometry(m, rendered, symbols, params)
    h, w = clean.shape
    outside = sum(
        b[0] < 0 or b[1] < 0 or b[2] > w or b[3] > h
        for b in (a["bbox"] for a in geo["annotations"])
    )
    doc = {
        "sheet": meta["sheet"], "index": meta["index"], "stroke_px": meta["stroke_px"],
        "seed": seed, "width": w, "height": h, "params": params_dict(params),
        "affine": [list(map(float, m[0])), list(map(float, m[1]))],
        "boxes_outside_page": outside, **geo,
    }  # fmt: skip
    t3 = time.perf_counter()
    return doc, jpeg, {"redraw": t1 - t0, "degrade": t2 - t1, "geometry": t3 - t2}


def run_sheet(job: tuple) -> list[dict]:
    source, variants, out, cfg = job
    templates = load_templates(cfg["template_dir"], cfg["taxonomy_csv"], cfg["scope"])
    src = Path(cfg["_src"])
    sheet = load_graphml(src / f"{source}.graphml", Image.open(src / f"{source}.png").size)
    runs = pipe_runs(sheet)
    (out / f"sheet{source}").mkdir(parents=True, exist_ok=True)
    rows = []
    for jf in sorted((variants / f"sheet{source}").glob("v*.json")):
        t0 = time.perf_counter()
        meta = json.loads(jf.read_text())
        doc, jpeg, secs = degrade_one(meta, jf.with_suffix(".png"), sheet, runs, templates, cfg)
        stem = out / f"sheet{source}" / jf.stem
        stem.with_suffix(".jpg").write_bytes(jpeg)
        stem.with_suffix(".json").write_text(json.dumps(doc, indent=1))
        rows.append(
            {
                "sheet": source, "index": meta["index"], "stroke_px": meta["stroke_px"],
                **{k: v for k, v in doc["params"].items() if k != "noise_seed"},
                "boxes_outside_page": doc["boxes_outside_page"],
                "jpeg_kb": round(len(jpeg) / 1000, 1),
                "json_kb": round(stem.with_suffix(".json").stat().st_size / 1000, 1),
                **{f"sec_{k}": round(v, 3) for k, v in secs.items()},
                "sec_total": round(time.perf_counter() - t0, 3),
            }
        )  # fmt: skip
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=Path("data/raw/pid2graph/open100"))
    ap.add_argument("--variants", type=Path, default=None, help="default: <data_root>/variants")
    ap.add_argument("--out", type=Path, default=None, help="default: <data_root>/degraded")
    ap.add_argument("--stats", type=Path, default=None, help="default: export.degrade_stats")
    ap.add_argument("--overwrite", action="store_true", help="allow replacing degraded images")
    ap.add_argument("--sheets", type=int, nargs="*", default=list(range(12)))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    cfg = load_config()
    cfg["_src"] = str(args.src)
    root = Path(cfg["export"]["data_root"])
    args.variants = args.variants or root / "variants"
    args.out = args.out or root / "degraded"
    args.stats = args.stats or Path(cfg["export"]["degrade_stats"])
    done = [s for s in args.sheets if any((args.out / f"sheet{s}").glob("v*.json"))]
    if done and not args.overwrite:  # v1 stays as it is (PLAN_PHASE_C §2M rule 2)
        raise SystemExit(f"{args.out}: sheets {done} already degraded (--overwrite to redo)")
    t0 = time.perf_counter()
    rows: list[dict] = []
    with ProcessPoolExecutor(args.workers) as pool:
        for r in pool.map(run_sheet, [(s, args.variants, args.out, cfg) for s in args.sheets]):
            print(f"sheet {r[0]['sheet'] if r else '-'}: {len(r)} degraded", flush=True)
            rows += r
    args.stats.mkdir(parents=True, exist_ok=True)
    with open(args.stats / "degraded.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    wall = time.perf_counter() - t0
    jpg = sum(r["jpeg_kb"] for r in rows) / len(rows)
    print(f"{len(rows)} images, {wall:.0f}s wall ({args.workers} workers), {jpg:.0f} KB jpg each")


if __name__ == "__main__":
    main()
