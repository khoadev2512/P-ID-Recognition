"""V6 evidence pictures (local only, *.png under docs/experiments/ is gitignored).

Needs the variants written by make_variants.py (default data/iso_open100/variants) and its
attempts.csv. Writes montage, the closest kept pairs (by content, by picture hash), stroke /
size crops and the nearest-distance plot (content share vs Hamming distance).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.diversity import content_distance  # noqa: E402
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

    # 2. pairs of kept variants: the closest by content (fewest nodes with another symbol) and
    # the closest by picture hash (what the old hash filter would have dropped first)
    def kept_pair(i: int, idx: int) -> tuple[Image.Image, Image.Image, int, float]:
        sheet = load_graphml(args.src / f"{i}.graphml", Image.open(args.src / f"{i}.png").size)
        runs = pipe_runs(sheet)
        mine = plan_variant(sheet, runs, templates, cfg, i, idx).choices
        best = None
        for k in (
            int(x["index"])
            for x in rows
            if int(x["sheet"]) == i and x["status"] == "kept" and int(x["index"]) < idx
        ):
            dist = content_distance(mine, plan_variant(sheet, runs, templates, cfg, i, k).choices)
            if best is None or dist < best[1]:
                best = (k, dist)
        k, dist = best
        load = lambda j: Image.open(args.variants / f"sheet{i}" / f"v{j:03d}.png")  # noqa: E731
        return load(k), load(idx), k, dist

    later = [r for r in rows if r["status"] == "kept" and r["nearest_content"] != ""]
    for tag, key in (("content", "nearest_content"), ("hash", "nearest_hash")):
        r = min(later, key=lambda x: float(x[key]))
        i, idx = int(r["sheet"]), int(r["index"])
        kim, img, k, dist = kept_pair(i, idx)
        pair = [
            label(small(kim, 1000), f"kept v{k:03d} (closest earlier kept variant by content)"),
            label(
                small(img, 1000),
                f"kept v{idx:03d}: {dist:.0%} of nodes differ from v{k:03d}; "
                f"hash distance to the closest kept picture {r['nearest_hash']}",
            ),
        ]
        grid(pair, 2).save(args.out / f"closest_pair_{tag}_sheet{i}.png")

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

    # 4. every drawn candidate: share of nodes differing from its closest kept variant against
    # the Hamming distance of its picture to the closest kept picture
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = [r for r in rows if r["nearest_content"] != "" and r["nearest_hash"] != ""]
    dup = [r for r in rows if r["reason"] == "duplicate"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    shares = [100 * float(r["nearest_content"]) for r in rows if r["nearest_content"] != ""]
    ax1.hist(shares, bins=np.arange(0, max(shares) + 2.5, 2.5), color="#2a6fb0")
    ax1.axvline(100 * st.min_node_diff, color="k", ls="--", lw=1)
    ax1.set_xlabel("% of nodes with another symbol than the closest kept variant")
    ax1.set_ylabel("candidates")
    ax1.set_title(f"content: {len(dup)} dropped below {st.min_node_diff:.0%}", fontsize=10)
    ax2.scatter(
        [int(r["nearest_hash"]) for r in pts],
        [100 * float(r["nearest_content"]) for r in pts],
        s=8, color="#2a6fb0",
    )  # fmt: skip
    ax2.axhline(100 * st.min_node_diff, color="k", ls="--", lw=1)
    ax2.axvline(8 - 0.5, color="#d9822b", ls=":", lw=1, label="old hash filter: < 8 bits dropped")
    bits = st.hash_size**2
    ax2.set_xlabel(f"max(Hamming ahash, dhash) to the closest kept picture ({bits} bits)")
    ax2.set_ylabel("% nodes differing")
    ax2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out / "dedup_distances.png", dpi=130)


if __name__ == "__main__":
    main()
