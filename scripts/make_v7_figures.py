"""V7 evidence pictures (local only: *.png under docs/experiments/ is gitignored).

before_after.png     a clean crop, the same crop at the mildest and at the strongest parameters
thin_details.png     flange pair, valve stem and agitator, clean / strongest erosion / dilation
bbox_overlay.png     a rotated degraded page with the moved boxes (ink = red, corners = blue)
ncc_summary.png      margin of every tested template (own NCC less the best other) per scenario
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.iso_degrade import degrade_image, extreme_params, ink_box_after  # noqa: E402
from utils.iso_mapping import load_config  # noqa: E402
from utils.iso_templates import load_templates  # noqa: E402
from utils.iso_variants import plan_variant, render_variant  # noqa: E402
from utils.page_svg import rasterize  # noqa: E402
from utils.pid2graph import load_graphml, pipe_runs  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def tile(img: np.ndarray, box, zoom: int, label: str, half: int = 40) -> Image.Image:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    x0, y0 = int(cx - half), int(cy - half)
    crop = img[max(y0, 0) : y0 + 2 * half, max(x0, 0) : x0 + 2 * half]
    im = (
        Image.fromarray(crop)
        .convert("RGB")
        .resize((2 * half * zoom,) * 2, Image.Resampling.NEAREST)
    )
    ImageDraw.Draw(im).text((6, 4), label, fill=(200, 0, 0))
    return im


def grid(tiles, cols, pad=8):
    rows = [tiles[i : i + cols] for i in range(0, len(tiles), cols)]
    w = max(sum(t.width + pad for t in r) for r in rows)
    h = sum(max(t.height for t in r) + pad for r in rows)
    out = Image.new("RGB", (w, h), (120, 120, 120))
    y = 0
    for r in rows:
        x = 0
        for t in r:
            out.paste(t, (x, y))
            x += t.width + pad
        y += max(t.height for t in r) + pad
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=Path("data/raw/pid2graph/open100"))
    ap.add_argument("--degraded", type=Path, default=Path("data/iso_open100/degraded"))
    ap.add_argument("--out", type=Path, default=Path("docs/experiments/iso_render/v7"))
    ap.add_argument(
        "--survival", type=Path, default=Path("docs/experiments/iso_render/v7/survival.csv")
    )
    args = ap.parse_args()
    cfg = load_config()
    templates = load_templates(cfg["template_dir"], cfg["taxonomy_csv"], cfg["scope"])

    def page(sheet_id: int, idx: int):
        sheet = load_graphml(
            args.src / f"{sheet_id}.graphml", Image.open(args.src / f"{sheet_id}.png").size
        )
        runs = pipe_runs(sheet)
        r = render_variant(
            plan_variant(sheet, runs, templates, cfg, sheet_id, idx), sheet, runs, cfg
        )
        return r, plan_variant(sheet, runs, templates, cfg, sheet_id, idx)

    # 1. before / after at the mildest and the strongest parameters (sheet 3, variant 0)
    r, draft = page(3, 0)
    clean = rasterize(r.page.svg)
    w = draft.stroke_px
    mild = replace(extreme_params(cfg, w, only="__none__"), noise_seed=1)
    strong = extreme_params(cfg, w, morph=-1)
    strong_d = extreme_params(cfg, w, morph=+1)
    panels = []
    anns = [
        a
        for a in r.page.annotations
        if a["fine_name"] in ("valve_gate", "valve_globe", "pump_gear")
    ]
    for a in anns[:2]:
        row = [tile(clean, a["bbox"], 5, "clean")]
        for label, p in (
            ("mildest", mild),
            ("strongest, eroding", strong),
            ("strongest, dilating", strong_d),
        ):
            out, m, _ = degrade_image(clean, p)
            box = ink_box_after(m, rasterize(r.page.layer("symbols")), tuple(a["bbox"]))
            row.append(tile(out, box, 5, f"{label} w={w}"))
        panels += row
    grid(panels, 4).save(args.out / "before_after.png")

    # 2. thin details at the strongest setting: flange pair, a valve stem, an agitator
    rows = []
    for sheet_id, idx, names in (
        (9, 12, ("agitator_general", "agitator_impeller", "agitator_turbine")),
        (3, 0, ("flanged_connection", "valve_control", "valve_gate")),
    ):
        r, draft = page(sheet_id, idx)
        clean = rasterize(r.page.svg)
        symbols = rasterize(r.page.layer("symbols"))
        w = draft.stroke_px
        chosen = {}
        for a in r.page.annotations:
            if a["fine_name"] in names and a["fine_name"] not in chosen:
                chosen[a["fine_name"]] = a
        outs = {
            k: degrade_image(clean, extreme_params(cfg, w, morph=sg))
            for k, sg in (("erode", -1), ("dilate", 1))
        }
        for name, a in chosen.items():
            half = 60 if sheet_id == 9 else 30
            rows.append(
                tile(clean, a["bbox"], 4 if sheet_id == 9 else 6, f"{name} clean w={w}", half)
            )
            for k, (out, m, _) in outs.items():
                box = ink_box_after(m, symbols, tuple(a["bbox"]))
                rows.append(tile(out, box, 4 if sheet_id == 9 else 6, f"max {k}", half))
    grid(rows, 3).save(args.out / "thin_details.png")

    # 3. a rotated degraded page with the boxes the JSON stores
    best = max(
        (
            json.loads(f.read_text()) | {"file": f}
            for f in (args.degraded / "sheet9").glob("v*.json")
        ),
        key=lambda d: abs(d["params"]["rotate_deg"]),
    )
    img = np.asarray(Image.open(best["file"].with_suffix(".jpg")).convert("RGB")).copy()
    ag = next(a for a in best["annotations"] if (a["fine_name"] or "").startswith("agitator"))
    x0, y0, x1, y1 = ag["bbox"]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    reg = (int(cx - 700), int(cy - 600), int(cx + 700), int(cy + 700))
    crop = Image.fromarray(img[reg[1] : reg[3], reg[0] : reg[2]])
    d = ImageDraw.Draw(crop)
    for a in best["annotations"]:
        for key, colour in (("bbox_corners", (40, 90, 255)), ("bbox", (230, 0, 0))):
            b = a[key]
            if b[2] > reg[0] and b[0] < reg[2] and b[3] > reg[1] and b[1] < reg[3]:
                d.rectangle(
                    [b[0] - reg[0], b[1] - reg[1], b[2] - reg[0], b[3] - reg[1]],
                    outline=colour,
                    width=2,
                )
    d.text(
        (10, 10),
        f"sheet 9 {best['file'].stem} rotation {best['params']['rotate_deg']:.2f} deg, "
        f"shear {best['params']['shear_deg']:.2f} deg; red = ink box (stored), blue = corner box",
        fill=(0, 0, 0),
    )
    crop.resize((crop.width * 3 // 4, crop.height * 3 // 4), Image.Resampling.LANCZOS).save(
        args.out / "bbox_overlay.png"
    )

    # 4. NCC summary
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows_s = [x for x in csv.DictReader(open(args.survival)) if float(x["k"]) >= 3.5]
    scen = [
        "clean", "all_erode", "all_dilate", "only_rotate_deg", "only_shear_deg", "only_erode",
        "only_dilate", "only_blur", "only_pixelate", "only_lighting", "only_noise_sigma",
        "only_jpeg_quality",
    ]  # fmt: skip
    names = sorted({x["template"] + " " + x["fine_name"] for x in rows_s})
    mat = np.full((len(names), len(scen)), np.nan)
    for x in rows_s:
        i, j = names.index(x["template"] + " " + x["fine_name"]), scen.index(x["scenario"])
        v = float(x["margin"])
        mat[i, j] = v if np.isnan(mat[i, j]) else min(mat[i, j], v)
    fig, ax = plt.subplots(figsize=(9, 11))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-0.1, vmax=0.4, aspect="auto")
    ax.set_xticks(range(len(scen)), scen, rotation=70, ha="right")
    ax.set_yticks(range(len(names)), names, fontsize=7)
    fig.colorbar(im, label="own NCC - best other class (worst of k >= 3.5, w = 2, 3, 4)")
    fig.tight_layout()
    fig.savefig(args.out / "ncc_summary.png", dpi=110)


if __name__ == "__main__":
    main()
