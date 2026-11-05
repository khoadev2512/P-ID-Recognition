"""V8 evidence: statistics and pictures of an exported ISO data root (read only).

Writes into --out: per_class.csv (instances, images, clean / degraded, share), per_sheet.csv
(images, symbols, split), summary.txt (symbols per image, share of `other_*`), class_hist.png,
overlay_coco.png (COCO boxes on a clean and a degraded exported image) and overlay_graphml.png
(GraphML nodes and edges on an exported image).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.iso_export import read_source_graphml  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
OTHERS = ("other_iso", "other_isa", "other_generic")


def stem_of(img: dict) -> str:
    return img["file_name"].split("/")[-1].rsplit(".", 1)[0]


def crop_overlay(path: Path, region, draw_fn, scale: float, label: str) -> Image.Image:
    im = Image.open(path).convert("RGB").crop(region)
    d = ImageDraw.Draw(im)
    draw_fn(d)
    d.text((10, 8), label, fill=(0, 0, 0))
    return im.resize((round(im.width * scale), round(im.height * scale)), Image.Resampling.LANCZOS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("data/iso_open100"))
    ap.add_argument("--out", type=Path, default=Path("docs/experiments/iso_render/v8"))
    ap.add_argument("--sheet", type=int, default=9)
    ap.add_argument("--variant", type=int, default=12)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    coco = json.loads((args.root / "canonical" / "annotations.coco.json").read_text())
    names = {c["id"]: c["name"] for c in coco["categories"]}
    imgs = {i["id"]: i for i in coco["images"]}
    split = json.loads((args.root / "splits.json").read_text())

    inst, in_imgs = Counter(), defaultdict(set)
    by_kind = {False: Counter(), True: Counter()}
    per_image = Counter()
    for a in coco["annotations"]:
        n = names[a["category_id"]]
        inst[n] += 1
        in_imgs[n].add(a["image_id"])
        by_kind[imgs[a["image_id"]]["degraded"]][n] += 1
        per_image[a["image_id"]] += 1
    total = sum(inst.values())
    with open(args.out / "per_class.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["class", "instances", "share_pct", "images_with", "clean", "degraded"])
        for n in [c["name"] for c in coco["categories"]]:
            wr.writerow([n, inst[n], round(100 * inst[n] / total, 2), len(in_imgs[n]),
                         by_kind[False][n], by_kind[True][n]])  # fmt: skip
    sheets = sorted({i["sheet"] for i in coco["images"]})
    with open(args.out / "per_sheet.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["sheet", "split", "images_clean", "images_degraded", "symbols_per_image_mean"])
        for s in sheets:
            ids = [i["id"] for i in coco["images"] if i["sheet"] == s]
            wr.writerow([s, split["sheets"][str(s)],
                         sum(not imgs[i]["degraded"] for i in ids),
                         sum(imgs[i]["degraded"] for i in ids),
                         round(float(np.mean([per_image[i] for i in ids])), 1)])  # fmt: skip
    pi = np.array([per_image[i["id"]] for i in coco["images"]])
    other = sum(inst[o] for o in OTHERS)
    lines = [
        f"images {len(coco['images'])} (clean {sum(not i['degraded'] for i in coco['images'])}, "
        f"degraded {sum(i['degraded'] for i in coco['images'])}); annotations {total}",
        f"symbols per image: mean {pi.mean():.1f}, min {pi.min()}, max {pi.max()}",
        f"other_* share: {100 * other / total:.1f}% "
        f"({', '.join(f'{o} {inst[o]}' for o in OTHERS)})",
        f"vocabulary classes with >= 1 instance: "
        f"{sum(1 for n in names.values() if n not in OTHERS and inst[n])} of "
        f"{sum(1 for n in names.values() if n not in OTHERS)}",
        f"agitators (added): {sum(1 for a in coco['annotations'] if a.get('added'))}",
        f"split by sheet: {split['sheets']} (mode {split['mode']})",
    ]
    (args.out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = sorted(names.values(), key=lambda n: -inst[n])
    order = [n for n in order if inst[n]]
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(order) + 1))
    y = np.arange(len(order))
    colours = ["#d9822b" if n in OTHERS else "#2a6fb0" for n in order]
    ax.barh(y, [inst[n] for n in order], color=colours)
    ax.set_yticks(y, order, fontsize=7)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("instances over all exported images (log scale); orange = other_*")
    fig.tight_layout()
    fig.savefig(args.out / "class_hist.png", dpi=120)

    # overlays on one exported variant, clean and degraded
    ids = {stem_of(i): i for i in coco["images"]}
    clean_stem = f"iso{args.sheet:02d}_v{args.variant:03d}"
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)
    cat_colour = {"flow_direction": (0, 150, 0)}
    tiles = []
    ref = [a for a in anns[ids[clean_stem]["id"]] if a.get("added")][0]
    x, y, w, h = ref["bbox"]
    region = (int(x - 500), int(y - 150), int(x + w + 1100), int(y + h + 150))
    for stem in (clean_stem, clean_stem + "_deg"):
        if stem not in ids:
            continue
        info = ids[stem]

        def draw(d, info=info, region=region):
            for a in anns[info["id"]]:
                bx, by, bw, bh = a["bbox"]
                col = (
                    (230, 0, 0)
                    if a.get("added")
                    else cat_colour.get(names[a["category_id"]], (40, 90, 255))
                )
                x0, y0 = bx - region[0], by - region[1]
                d.rectangle([x0, y0, x0 + bw, y0 + bh], outline=col, width=3)

        ext = Path(info["file_name"]).suffix
        path = args.root / "canonical" / "images" / (stem + ext)
        label = f"{stem}  (red = agitator, green = flow arrow, blue = every other box)"
        tiles.append(crop_overlay(path, region, draw, 0.5, label))
    pic = Image.new(
        "RGB", (sum(t.width for t in tiles) + 10, max(t.height for t in tiles)), (120,) * 3
    )
    xo = 0
    for t in tiles:
        pic.paste(t, (xo, 0))
        xo += t.width + 10
    pic.save(args.out / "overlay_coco.png")

    # GraphML: boxes and edges on a degraded image
    stem = clean_stem + "_deg" if clean_stem + "_deg" in ids else clean_stem
    nodes = {}
    import xml.etree.ElementTree as ET

    root = ET.parse(args.root / "graphml" / f"{stem}.graphml").getroot()
    ns = "{http://graphml.graphdrawing.org/xmlns}"
    keys = {k.attrib["id"]: k.attrib["attr.name"] for k in root.iter(f"{ns}key")}
    for n in root.iter(f"{ns}node"):
        nodes[n.attrib["id"]] = {
            keys[d.attrib["key"]]: d.text or "" for d in n.findall(f"{ns}data")
        }
    _, edges = read_source_graphml(args.root / "graphml" / f"{stem}.graphml")
    ext = Path(ids[stem]["file_name"]).suffix

    def draw_graph(d):
        def centre(v):
            return ((float(v["xmin"]) + float(v["xmax"])) / 2 - region[0],
                    (float(v["ymin"]) + float(v["ymax"])) / 2 - region[1])  # fmt: skip

        for s, t, _ in edges:
            d.line([centre(nodes[s]), centre(nodes[t])], fill=(0, 140, 0), width=2)
        for v in nodes.values():
            col = (230, 0, 0) if v["drawn"] == "1" else (160, 160, 160)
            box = [
                float(v[k]) - o
                for k, o in (
                    ("xmin", region[0]),
                    ("ymin", region[1]),
                    ("xmax", region[0]),
                    ("ymax", region[1]),
                )
            ]
            d.rectangle(box, outline=col, width=2)

    label = f"{stem}: GraphML nodes (red drawn, grey connector / crossing) and edges (green)"
    path = args.root / "canonical" / "images" / (stem + ext)
    crop_overlay(path, region, draw_graph, 0.6, label).save(args.out / "overlay_graphml.png")


if __name__ == "__main__":
    main()
