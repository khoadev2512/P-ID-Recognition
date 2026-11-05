"""V6 evidence pictures (local only, *.png under docs/experiments/ is gitignored).

Needs the variants written by make_variants.py (default data/iso_open100/variants) and its
attempts.csv. Writes montage, duplicate pair, stroke / size crops and a Hamming histogram.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.diversity import ahash, dhash, hamming  # noqa: E402
from utils.iso_mapping import load_config  # noqa: E402
from utils.iso_templates import load_templates  # noqa: E402
from utils.iso_variants import plan_variant, render_variant, variant_settings  # noqa: E402
from utils.page_svg import rasterize  # noqa: E402
from utils.pid2graph import load_graphml, pipe_runs  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def small(img: Image.Image, width: int) -> Image.Image:
    img = img.convert("L")
    return img.resize((width, round(img.height * width / img.width)), Image.Resampling.BOX)


def label(img: Image.Image, text: str) -> Image.Image:
    out = img.convert("RGB")
    ImageDraw.Draw(out).text((8, 6), text, fill=(200, 0, 0))
    return out


def grid(tiles: list[Image.Image], cols: int, pad: int = 10) -> Image.Image:
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
    ap.add_argument("--variants", type=Path, default=Path("data/iso_open100/variants"))
    ap.add_argument("--src", type=Path, default=Path("data/raw/pid2graph/open100"))
    ap.add_argument("--out", type=Path, default=Path("docs/experiments/iso_render/v6"))
    ap.add_argument(
        "--attempts", type=Path, default=Path("docs/experiments/iso_render/v6/attempts.csv")
    )
    ap.add_argument("--sheet", type=int, default=3)
    args = ap.parse_args()
    cfg = load_config()
    st = variant_settings(cfg)
    templates = load_templates(cfg["template_dir"], cfg["taxonomy_csv"], cfg["scope"])
    rows = list(csv.DictReader(open(args.attempts)))
    sheet_rows = [r for r in rows if int(r["sheet"]) == args.sheet]
    kept = [r for r in sheet_rows if r["status"] == "kept"]

    # 1. six kept variants side by side
    tiles = []
    for r in kept[:6]:
        img = Image.open(args.variants / f"sheet{args.sheet}" / f"v{int(r['index']):03d}.png")
        tiles.append(
            label(
                small(img, 900),
                f"v{int(r['index']):03d} w={r['stroke_px']} size={r['size_factor']}",
            )
        )
    grid(tiles, 3).save(args.out / "montage_sheet3.png")

    # 2. a duplicate dropped, next to the kept picture it matched
    dups = [r for r in rows if r["reason"] == "duplicate"]
    dups.sort(key=lambda r: int(r["nearest"]))
    for rank, r in enumerate((dups[0], dups[len(dups) // 2])):
        i, idx = int(r["sheet"]), int(r["index"])
        sheet = load_graphml(args.src / f"{i}.graphml", Image.open(args.src / f"{i}.png").size)
        runs = pipe_runs(sheet)
        draft = plan_variant(sheet, runs, templates, cfg, i, idx)
        img = rasterize(render_variant(draft, sheet, runs, cfg).page.svg)
        a, d = ahash(img, st.hash_size), dhash(img, st.hash_size)
        best = None
        for k in (
            x
            for x in rows
            if int(x["sheet"]) == i and x["status"] == "kept" and int(x["index"]) < idx
        ):
            kim = Image.open(args.variants / f"sheet{i}" / f"v{int(k['index']):03d}.png")
            dist = max(hamming(a, ahash(kim, st.hash_size)), hamming(d, dhash(kim, st.hash_size)))
            if best is None or dist < best[0]:
                best = (dist, k, kim)
        dist, k, kim = best
        pair = [
            label(small(kim, 1000), f"kept v{int(k['index']):03d}"),
            label(
                small(Image.fromarray(img), 1000),
                f"dropped v{idx:03d}: duplicate, distance {dist} < {st.min_distance}",
            ),
        ]
        grid(pair, 2).save(args.out / f"duplicate_pair_{rank}_sheet{i}.png")

    # 3. one symbol under w = 2 / 4 and size 0.85 / 1.15 (same assignment and layout)
    sheet = load_graphml(
        args.src / f"{args.sheet}.graphml", Image.open(args.src / f"{args.sheet}.png").size
    )
    runs = pipe_runs(sheet)
    draft = plan_variant(sheet, runs, templates, cfg, args.sheet, 0)
    shots = {}
    for tag, (w, f) in {
        "w=2": (2.0, 1.0),
        "w=3": (3.0, 1.0),
        "w=4": (4.0, 1.0),
        "size 0.85": (3.0, 0.85),
        "size 1.15": (3.0, 1.15),
    }.items():
        draft.stroke_px, draft.size_factor = w, f
        r = render_variant(draft, sheet, runs, cfg)
        shots[tag] = (rasterize(r.page.svg), {a["id"]: a for a in r.page.annotations})
    ref = shots["w=3"][1]
    ids = [
        n
        for n, a in ref.items()
        if a["fine_name"]
        in ("valve_globe", "check_valve_swing", "pump_gear", "valve_gate", "reducer")
    ][:3]
    crops = []
    for nid in ids:
        x0, y0, x1, y1 = ref[nid]["bbox"]
        cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, 60
        row = []
        for tag, (img, _) in shots.items():
            c = Image.fromarray(img).crop(
                (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
            )
            row.append(
                label(
                    c.resize((360, 360), Image.Resampling.NEAREST), f"{ref[nid]['fine_name']} {tag}"
                )
            )
        crops += row
    grid(crops, len(shots)).save(args.out / "style_crops.png")

    # 4. Hamming distance of every hashed candidate to its nearest kept variant
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    near = [(int(r["nearest"]), r["reason"] == "duplicate") for r in rows if r["nearest"] != ""]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    top = max(n for n, _ in near)
    bins = np.arange(0, top + 2) - 0.5
    ax.hist(
        [n for n, dup in near if not dup], bins=bins, label="kept / not duplicate", color="#2a6fb0"
    )
    ax.hist([n for n, dup in near if dup], bins=bins, label="dropped: duplicate", color="#d9822b")
    ax.axvline(st.min_distance - 0.5, color="k", ls="--", lw=1)
    ax.set_xlabel(f"max(Hamming ahash, dhash) to the nearest kept variant ({st.hash_size**2} bits)")
    ax.set_ylabel("candidates")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.out / "hamming_hist.png", dpi=130)


if __name__ == "__main__":
    main()
