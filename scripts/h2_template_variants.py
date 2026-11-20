"""PLAN_PHASE_C §2D.1 "H2": ISO templates made drawing-like before embedding (B9 + stroke + scan).

Two steps, in this order (the contact sheet is approved by the user before any score):

    python scripts/h2_template_variants.py --out runs/phase_c/h2 --sheet
    python scripts/h2_template_variants.py --out runs/phase_c/h2 --score --approved

--sheet  one row per vocab template: clean | 3 dirty views | the first dirty view after A4
         (what the encoder sees), plus `ports.csv` (the SVG drawing used, the ports, where each
         pipe met the ink) -> `contact_sheet.png`.
--score  prototypes = mean over [clean] + K_VIEWS dirty views, each through A4, embedded by the
         frozen DINOv2-S (C3 did not pass, §2C.3b); queries, CL2N center and the baseline come
         from the Bước 1 probe cache (`runs/phase_c/probe/emb_dinov2_a4.pt`, same crops). The
         §1.2 rule on P1-val, baseline = dinov2 / A4 / CL2N with the clean view only.

Reads data/raw/iso_symbols*, data/raw/template_iso_svg, the probe cache; writes only --out.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import oneshot_probe as probe  # noqa: E402

from pipeline.iso_prototypes import load_templates, read_vocab  # noqa: E402
from utils.input_norm import ink_letterbox, ink_mask  # noqa: E402
from utils.iso_templates import load_templates as load_svg_templates  # noqa: E402
from utils.oneshot import aggregate_views, embed_images  # noqa: E402
from utils.template_variants import (  # noqa: E402
    K_VIEWS,
    _pad,
    axes_of,
    contact,
    ink_box,
    pick_template,
    template_views,
)

logger = logging.getLogger("h2")
SEED = 0
SHEET_DIRTY = 3
THUMB = 150
ENCODER = "dinov2"
INPUT = "a4"


def vocab_templates():
    """(fine_name -> PNG, fine_name -> coarse, fine_name -> (svg template | None, reg))."""
    rows = read_vocab(ROOT / "data/raw/iso_symbols/vocab.csv")
    images, coarse_of = load_templates(ROOT / "data/raw/iso_symbols_clean", rows)
    svg = load_svg_templates(
        ROOT / "data/raw/template_iso_svg", ROOT / "docs/symbol_mapping/iso_taxonomy.csv", "all"
    )
    chosen = {}
    for row in rows:
        name, reg = row["fine_name"].strip(), row["reg_number"].strip()
        x0, y0, x1, y1 = ink_box(ink_mask(np.asarray(images[name].convert("L"))))
        cands = [t for t in svg.values() if t.reg_number == reg]
        chosen[name] = (pick_template(cands, (x1 - x0) / (y1 - y0)), reg)
    return images, coarse_of, chosen


def _font():
    """A font with Vietnamese diacritics when the system has one (labels only)."""
    for path in ("/System/Library/Fonts/Supplemental/Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, 13)
        except OSError:
            continue
    return ImageFont.load_default()


def _thumb(img: Image.Image) -> Image.Image:
    im = img.convert("RGB")
    im.thumbnail((THUMB - 6, THUMB - 6))
    cell = Image.new("RGB", (THUMB, THUMB), (235, 235, 235))
    cell.paste(im, ((THUMB - im.width) // 2, (THUMB - im.height) // 2))
    return cell


def sheet(out: Path) -> None:
    images, _, chosen = vocab_templates()
    label_w = 230
    cols = 2 + SHEET_DIRTY
    canvas = Image.new("RGB", (label_w + cols * THUMB, len(images) * THUMB), "white")
    draw = ImageDraw.Draw(canvas)
    font = _font()
    with (out / "ports.csv").open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["fine_name", "reg", "svg_file", "side", "at", "contact_x", "contact_y"])
        for r, (name, img) in enumerate(images.items()):
            tpl, reg = chosen[name]
            axes = axes_of(tpl)
            views = template_views(img, axes, SHEET_DIRTY, SEED, name)
            row = views + [ink_letterbox(views[1])]
            for c, v in enumerate(row):
                canvas.paste(_thumb(v), (label_w + c * THUMB, r * THUMB))
            sides = "".join(a.side for a in axes) or "không cổng"
            draw.text((6, r * THUMB + 8), f"{name}", fill="black", font=font)
            draw.text(
                (6, r * THUMB + 26),
                f"REG {reg}  svg {tpl.file if tpl else '-'}",
                fill="black",
                font=font,
            )
            draw.text((6, r * THUMB + 44), f"cổng {sides}", fill="black", font=font)
            gray = np.asarray(img.convert("L"))
            pad, box, _, _ = _pad(gray, ink_box(ink_mask(gray)))
            for a in axes:
                x, y = contact(ink_mask(pad), box, a)
                wr.writerow([name, reg, tpl.file, a.side, f"{a.at:.3f}", x, y])
            if not axes:
                logger.info("%s (REG %s): no ports -> stroke + scan only", name, reg)
    head = ["sạch"] + [f"bẩn {i + 1}" for i in range(SHEET_DIRTY)] + ["bẩn 1 sau A4"]
    top = Image.new("RGB", (canvas.width, 24), "white")
    d2 = ImageDraw.Draw(top)
    for c, h in enumerate(head):
        d2.text((label_w + c * THUMB + 6, 6), h, fill="black", font=font)
    full = Image.new("RGB", (canvas.width, canvas.height + 24), "white")
    full.paste(top, (0, 0))
    full.paste(canvas, (0, 24))
    full.save(out / "contact_sheet.png")
    logger.info("wrote %s (%d templates)", out / "contact_sheet.png", len(images))


def score(out: Path, probe_dir: Path) -> None:
    cfg = OmegaConf.load(ROOT / "configs/config.yaml")
    cfg.data_root = str(ROOT / "data")
    world = probe.load_world(cfg)
    crops = probe.pick_crops(world)
    cache = torch.load(probe_dir / f"emb_{ENCODER}_{INPUT}.pt", weights_only=False)
    images, _, chosen = vocab_templates()
    names = list(images)
    assert cache["names"] == names, "probe cache names differ from the vocab"
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cpu")
    encoder, tf = probe.load_encoder(ENCODER, cfg, device)
    vectors, clean = [], []
    with torch.inference_mode():
        for name in names:
            tpl, _ = chosen[name]
            views = [
                ink_letterbox(v)
                for v in template_views(images[name], axes_of(tpl), K_VIEWS, SEED, name)
            ]
            emb = embed_images(encoder, views, tf, device)
            clean.append(emb[0])
            vectors.append(aggregate_views(emb))
    diff = float((torch.stack(clean) - cache["protos"]).abs().max())
    logger.info("clean view vs probe prototypes: max |diff| %.2e", diff)
    assert diff < 1e-4, diff
    protos = torch.stack(vectors)
    torch.save({"protos": protos, "names": names, "k": K_VIEWS, "seed": SEED}, out / "protos_h2.pt")
    base = probe.score_config(world, crops, cache["emb"], cache["protos"], names, cache["emb"])
    cand = probe.score_config(world, crops, cache["emb"], protos, names, cache["emb"])
    results = {"base": base, "h2": cand}
    (out / "results.json").write_text(json.dumps(results, indent=1))
    lines = [f"H2 (PLAN_PHASE_C §2D.1), encoder {ENCODER} / {INPUT}, K = {K_VIEWS}, seed {SEED}"]
    lines.append(
        f"{'dòng':22s} {'p1val top1':>10s} {'top3':>6s} {'AUROC':>6s} {'lề':>6s} {'hub':>6s}"
        f"   | {'p1 top1':>8s} {'top3':>6s} {'AUROC':>6s}"
    )
    for tag, res in results.items():
        for c, r in res.items():
            a, b = r["p1val"], r["p1"]
            lines.append(
                f"{tag + '/' + c:22s} {a['macro_top1']:10.3f} {a['macro_top3']:6.3f}"
                f" {a['auroc_cos']:6.3f} {a['auroc_margin']:6.3f} {a['hub_share']:6.3f}"
                f"   | {b['macro_top1']:8.3f}"
                f" {b['macro_top3']:6.3f} {b['auroc_cos']:6.3f}"
            )
    lines.append("")
    lines.append("Từng lớp P1-val (CL2N), top-1 mốc -> H2:")
    for cls, p in base["cl2n"]["p1val"]["per_class"].items():
        q = cand["cl2n"]["p1val"]["per_class"][cls]
        lines.append(
            f"  {cls:10s} {p['expected']:28s} {p['top1']:.3f} -> {q['top1']:.3f}"
            f"  (đoán nhiều nhất {q['most_predicted']})"
        )
    d = probe.decide(base["cl2n"]["p1val"], cand["cl2n"]["p1val"])
    lines.append("")
    lines.append(
        f"Luật §1.2 (mốc dinov2/a4/cl2n 1 view sạch): lớp tăng {d['n_up']}/10"
        f"  Δtop1 {d['d_macro_top1']:+.3f}"
        f"  ΔAUROC {d['d_auroc']:+.3f}  -> {'ĐẠT' if d['pass'] else 'KHÔNG ĐẠT'}"
    )
    text = "\n".join(lines)
    (out / "report.txt").write_text(text + "\n")
    print(text)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--probe", type=Path, default=ROOT / "runs/phase_c/probe")
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--approved", action="store_true", help="the user approved contact_sheet.png")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.sheet:
        sheet(args.out)
    if args.score:
        if not args.approved:
            ap.error("--score needs --approved: the contact sheet must be approved first (§2D.1)")
        score(args.out, args.probe)


if __name__ == "__main__":
    main()
