"""PLAN_PHASE_C §2F: why S6 / S8 / S14 (and the other weak P1-val classes) stay at 0 — a sheet
for the user (the P&ID expert) to judge. Descriptive, no decision, nothing selected.

One row per class: its mapped ISO template | 5 DigitizePID crops from TRAIN sheets (the probe's
"center" crops, 30 per class; P1-val is not used) | the 3 templates DINOv2-S / A4 / CL2N names most
often for the 30 crops, with their share. Text: symbol_map status / notes, top-1 on the 30 crops,
median rank of the mapped template among the 42, mean cos to it and to the top guess.

    python scripts/variant_contact_sheet.py --out runs/phase_c/variants
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import oneshot_probe as probe  # noqa: E402

from pipeline.iso_prototypes import load_templates, read_vocab  # noqa: E402

CLASSES = [
    "Symbol_6",
    "Symbol_8",
    "Symbol_14",
    "Symbol_22",
    "Symbol_20",
    "Symbol_23",
    "Symbol_17",
    "Symbol_1",  # control: ISO, top-1 0.99
]
N_CROPS, N_GUESS = 5, 3
THUMB, LABEL_W, LINE = 140, 330, 16


def _font(size=13):
    for path in ("/System/Library/Fonts/Supplemental/Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _thumb(img: Image.Image, frame=None) -> Image.Image:
    im = img.convert("RGB")
    im.thumbnail((THUMB - 8, THUMB - 24))
    cell = Image.new("RGB", (THUMB, THUMB), (235, 235, 235))
    cell.paste(im, ((THUMB - im.width) // 2, 4 + (THUMB - 24 - im.height) // 2))
    if frame:
        ImageDraw.Draw(cell).rectangle([0, 0, THUMB - 1, THUMB - 1], outline=frame, width=3)
    return cell


def _wrap(text: str, width: int = 46) -> list[str]:
    out, line = [], ""
    for w in text.split():
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    return out + ([line] if line else [])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--probe", type=Path, default=ROOT / "runs/phase_c/probe")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(ROOT / "configs/config.yaml")
    cfg.data_root = str(ROOT / "data")
    world = probe.load_world(cfg)
    crops = probe.pick_crops(world)
    cache = torch.load(args.probe / "emb_dinov2_a4.pt", weights_only=False)
    emb, protos, names = cache["emb"], cache["protos"], cache["names"]
    centers = [c for c in crops if c["set"] == "p1" and c["role"] == "center"]
    center = torch.cat([emb[c["ann"]["id"]] for c in centers]).mean(0)
    p = F.normalize(protos - center, dim=-1)
    images, _ = load_templates(
        ROOT / "data/raw/iso_symbols_clean", read_vocab(ROOT / "data/raw/iso_symbols/vocab.csv")
    )
    smap = {
        f"Symbol_{r['symbol_n']}": r
        for r in csv.DictReader((ROOT / "docs/symbol_mapping/symbol_map.csv").open())
    }
    groups = world["groups"]

    stats, rows = {}, []
    for cls in CLASSES:
        cs = [c for c in centers if c["class"] == cls]
        q = torch.stack(
            [
                F.normalize(F.normalize(emb[c["ann"]["id"]] - center, dim=-1).mean(0), dim=0)
                for c in cs
            ]
        )
        cos = q @ p.T  # (30, 42)
        want = names.index(groups[cls].template)
        order = cos.argsort(dim=1, descending=True)
        ranks = (order == want).nonzero()[:, 1] + 1
        top1 = [names[i] for i in order[:, 0].tolist()]
        guess = Counter(top1).most_common(N_GUESS)
        g0 = names.index(guess[0][0])
        stats[cls] = {
            "template": names[want],
            "status": smap[cls]["status"],
            "n": len(cs),
            "top1": float(np.mean([t == names[want] for t in top1])),
            "median_rank": float(ranks.float().median()),
            "cos_template": float(cos[:, want].mean()),
            "cos_top_guess": float(cos[:, g0].mean()),
            "guesses": [[n, k / len(cs)] for n, k in guess],
            "ann_ids": [c["ann"]["id"] for c in cs[:N_CROPS]],
        }
        rows.append((cls, cs[:N_CROPS], want, guess))

    pixels = probe.cut_crops([c for _, cs, _, _ in rows for c in cs])
    cols = 1 + N_CROPS + N_GUESS
    canvas = Image.new("RGB", (LABEL_W + cols * THUMB, 26 + len(rows) * THUMB), "white")
    draw = ImageDraw.Draw(canvas)
    font, small = _font(13), _font(11)
    head = ["ảnh mẫu ánh xạ"] + [f"crop train {i + 1}" for i in range(N_CROPS)]
    head += [f"đoán #{i + 1}" for i in range(N_GUESS)]
    for c, h in enumerate(head):
        draw.text((LABEL_W + c * THUMB + 6, 6), h, fill="black", font=font)
    for r, (cls, cs, want, guess) in enumerate(rows):
        y = 26 + r * THUMB
        s = stats[cls]
        text = [
            f"{cls}  [{s['status']}]  → {s['template']}",
            f"top-1 {s['top1']:.2f} trên {s['n']} crop train; hạng TV {s['median_rank']:.0f}/42",
            f"cos ảnh mẫu {s['cos_template']:.3f} | đoán #1 {s['cos_top_guess']:.3f}",
        ]
        text += _wrap(smap[cls]["notes"])[:4]
        for i, t in enumerate(text):
            draw.text((6, y + 6 + i * LINE), t, fill="black", font=font if i < 3 else small)
        canvas.paste(_thumb(images[names[want]], (40, 140, 40)), (LABEL_W, y))
        for j, c in enumerate(cs):
            canvas.paste(
                _thumb(Image.fromarray(pixels[c["ann"]["id"]])), (LABEL_W + (1 + j) * THUMB, y)
            )
        for j, (n, k) in enumerate(guess):
            x = LABEL_W + (1 + N_CROPS + j) * THUMB
            frame = (40, 140, 40) if n == names[want] else (200, 60, 60)
            canvas.paste(_thumb(images[n], frame), (x, y))
            draw.text(
                (x + 4, y + THUMB - 18), f"{n[:17]} {k / s['n']:.0%}", fill="black", font=small
            )
        draw.line([(0, y), (canvas.width, y)], fill=(200, 200, 200))
    canvas.save(args.out / "variant_sheet.png")
    (args.out / "stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    for cls, s in stats.items():
        g = ", ".join(f"{n} {k:.0%}" for n, k in s["guesses"])
        print(
            f"{cls:10s} {s['status']:8s} {s['template']:32s} top1 {s['top1']:.2f}"
            f"  hạng TV {s['median_rank']:4.0f}"
            f"  cos {s['cos_template']:.3f} vs {s['cos_top_guess']:.3f}"
            f"  | đoán: {g}"
        )


if __name__ == "__main__":
    main()
