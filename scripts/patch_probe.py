"""PLAN_PHASE_C §2H (hướng A): DINOv2 patch matching against the ISO templates, on P1-val.

FoundPose (Örnek et al. ECCV 2024) / DN4 (Li et al. CVPR 2019): instead of one pooled vector per
image, compare the PATCH descriptors of a crop with those of each template. Frozen DINOv2-S/14,
A4 on crops and templates (as 2B), 41 templates (§2G: `pipeline` dropped), 8 query views
(oneshot.query_views), 1 template view.

    score(view, template) = 1/2 (mean_q max_t cos + mean_t max_q cos)   (ink patches only)
    score(crop, template) = mean over the 8 views

Four variants: layer {9, 12} x patch center {off, on} (center = mean ink-patch descriptor of the
probe's 570 "center" crops, train images of the 19 R1a classes). The best by P1-val macro top-1
goes against the 2B global baseline (DINOv2-S / A4 / CL2N, same 41 templates, from the probe
cache) under the §1.2 rule (oneshot_probe.decide).

    python scripts/patch_probe.py --out runs/phase_c/patch [--device mps]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import oneshot_probe as probe  # noqa: E402

from pipeline.iso_prototypes import load_templates, read_vocab  # noqa: E402
from utils.oneshot import ViewSpec, make_views  # noqa: E402

logger = logging.getLogger("patch_probe")

DROPPED = "pipeline"  # §2G
LAYERS = (9, 12)  # 1-based block index
CENTERED = (False, True)
VARIANTS = [(layer, c) for layer in LAYERS for c in CENTERED]  # tie -> this order
GRID = 16
INK = 128


def variant_name(layer: int, centered: bool) -> str:
    return f"L{layer}/{'center' if centered else 'raw'}"


# --------------------------------------------------------------------------- #
# pure functions (tests/test_patch_probe.py)
# --------------------------------------------------------------------------- #


def ink_grid(img: Image.Image, grid: int = GRID) -> torch.Tensor:
    """(grid*grid,) bool: patch holds >= 1 ink pixel, from the image at its own resolution
    (max-pooled onto the grid the encoder's square resize produces). Blank -> all True."""
    ink = torch.from_numpy(np.asarray(img.convert("L")) < INK).float()[None, None]
    m = F.adaptive_max_pool2d(ink, (grid, grid))[0, 0].flatten() > 0
    return m if m.any() else torch.ones_like(m)


def chamfer(
    q: torch.Tensor, qmask: torch.Tensor, t: torch.Tensor, tmask: torch.Tensor
) -> torch.Tensor:
    """q (V, P, D), qmask (V, P); t (T, P', D), tmask (T, P') — unit descriptors.
    -> (V, T): 1/2 (mean over query ink patches of the best template patch + mean over template
    ink patches of the best query patch)."""
    sim = torch.einsum("vqd,tpd->vtqp", q, t)
    neg = torch.finfo(sim.dtype).min
    qm, tm = qmask[:, None, :, None], tmask[None, :, None, :]
    q2t = sim.masked_fill(~tm, neg).amax(dim=3)  # (V, T, Q)
    t2q = sim.masked_fill(~qm, neg).amax(dim=2)  # (V, T, P)
    qw, tw = qmask[:, None, :].float(), tmask[None, :, :].float()
    a = (q2t * qw).sum(-1) / qw.sum(-1)
    b = (t2q * tw).sum(-1) / tw.sum(-1)
    return 0.5 * (a + b)


def prep_desc(x: torch.Tensor, center: torch.Tensor | None) -> torch.Tensor:
    return F.normalize(x - center if center is not None else x, dim=-1)


def rows_from_scores(crops, scores: torch.Tensor, names, expected_of) -> list[dict]:
    """scores (n, T) -> probe.summarize rows (top-3 names, best, second)."""
    top = scores.topk(3, dim=1)
    return [
        {
            "class": c["class"],
            "known": c["role"] == "known",
            "expected": expected_of(c["class"]),
            "top": [names[i] for i in ii],
            "best": vv[0],
            "second": vv[1],
            "stem": c["stem"],
        }
        for c, ii, vv in zip(crops, top.indices.tolist(), top.values.tolist(), strict=True)
    ]


def pick_best(summaries: dict[str, dict]) -> str:
    """Highest macro top-1; ties keep the first in VARIANTS order."""
    order = [variant_name(*v) for v in VARIANTS]
    return max(order, key=lambda k: (summaries[k]["macro_top1"], -order.index(k)))


# --------------------------------------------------------------------------- #
# embedding
# --------------------------------------------------------------------------- #


@torch.inference_mode()
def tokens(model, tf, images: list[Image.Image], device) -> dict[int, torch.Tensor]:
    """layer -> (n, 256, D) patch tokens (final LayerNorm applied), on the CPU."""
    x = torch.stack([tf(im.convert("RGB")) for im in images]).to(device)
    out = model.forward_intermediates(
        x, indices=[i - 1 for i in LAYERS], norm=True, output_fmt="NLC", intermediates_only=True
    )
    return {layer: o.float().cpu() for layer, o in zip(LAYERS, out, strict=True)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--probe", type=Path, default=ROOT / "runs/phase_c/probe")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    cfg = OmegaConf.load(ROOT / "configs/config.yaml")
    cfg.data_root = str(ROOT / "data")
    world = probe.load_world(cfg)
    crops = probe.pick_crops(world)
    groups = world["groups"]
    expected_of = lambda n: groups[n].template or ""  # noqa: E731
    p1val = [c for c in crops if c["set"] == "p1val"]
    centers = [c for c in crops if c["set"] == "p1" and c["role"] == "center"]
    spec_q = ViewSpec.from_cfg(cfg.oneshot.query_views)
    spec_t = ViewSpec.from_cfg(cfg.oneshot.template_views)
    bg = tuple(int(c) for c in cfg.oneshot.background)
    a4 = probe.norm_fn("a4")

    # baseline: 2B global from the probe cache, same 41 templates
    cache = torch.load(args.probe / "emb_dinov2_a4.pt", weights_only=False)
    keep = [i for i, n in enumerate(cache["names"]) if n != DROPPED]
    names = [cache["names"][i] for i in keep]
    emb = cache["emb"]
    gcenter = torch.cat([emb[c["ann"]["id"]] for c in centers]).mean(0)
    base = probe.summarize(
        probe.rows_for(p1val, emb, cache["protos"][keep], names, gcenter, expected_of, "known")
    )
    logger.info("baseline 2B (41): top-1 %.3f AUROC %.3f", base["macro_top1"], base["auroc_cos"])

    score_file = args.out / "scores.pt"
    if score_file.exists():
        scores = torch.load(score_file, weights_only=False)
    else:
        encoder, tf = probe.load_encoder("dinov2", cfg, device)
        model = encoder.model
        rows = read_vocab(ROOT / "data/raw/iso_symbols/vocab.csv")
        templates, _ = load_templates(ROOT / "data/raw/iso_symbols_clean", rows)
        t_imgs = [make_views(a4(templates[n]), spec_t, bg)[0] for n in names]
        t_tok = tokens(model, tf, t_imgs, device)
        t_mask = torch.stack([ink_grid(im) for im in t_imgs])
        pixels = probe.cut_crops(p1val + centers)

        def views_of(c):
            return make_views(a4(Image.fromarray(pixels[c["ann"]["id"]])), spec_q, bg)

        # pass 1: patch centers (mean ink-patch descriptor of the center crops, every view)
        sums = {layer: 0.0 for layer in LAYERS}
        count = 0
        for k in range(0, len(centers), 16):
            imgs = [v for c in centers[k : k + 16] for v in views_of(c)]
            tok, mask = tokens(model, tf, imgs, device), torch.stack([ink_grid(v) for v in imgs])
            for layer in LAYERS:
                sums[layer] = sums[layer] + tok[layer][mask].sum(0)
            count += int(mask.sum())
        pcenter = {layer: sums[layer] / count for layer in LAYERS}
        logger.info("patch centers from %d ink patches of %d crops", count, len(centers))

        # pass 2: P1-val scores per variant
        t_desc = {
            v: prep_desc(t_tok[v[0]], pcenter[v[0]] if v[1] else None).to(device) for v in VARIANTS
        }
        t_mask_d = t_mask.to(device)
        out = {variant_name(*v): torch.empty(len(p1val), len(names)) for v in VARIANTS}
        n_v = spec_q.n_views
        for k in range(0, len(p1val), 8):
            part = p1val[k : k + 8]
            imgs = [v for c in part for v in views_of(c)]
            tok, mask = tokens(model, tf, imgs, device), torch.stack([ink_grid(v) for v in imgs])
            mask = mask.to(device)
            for v in VARIANTS:
                q = prep_desc(tok[v[0]], pcenter[v[0]] if v[1] else None).to(device)
                for j in range(len(part)):
                    sl = slice(j * n_v, (j + 1) * n_v)
                    s = chamfer(q[sl], mask[sl], t_desc[v], t_mask_d).mean(0)
                    out[variant_name(*v)][k + j] = s.cpu()
            if (k // 8) % 25 == 0:
                logger.info("scored %d / %d P1-val crops", k + len(part), len(p1val))
        scores = {"names": names, "ann_ids": [c["ann"]["id"] for c in p1val], "scores": out}
        torch.save(scores, score_file)
    assert scores["names"] == names and scores["ann_ids"] == [c["ann"]["id"] for c in p1val]

    summ = {
        k: probe.summarize(rows_from_scores(p1val, s, names, expected_of))
        for k, s in scores["scores"].items()
    }
    best = pick_best(summ)
    verdict = probe.decide(base, summ[best])
    results = {"baseline_2b_41": base, "variants": summ, "best": best, "decision": verdict}
    (args.out / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    report(results, args.out)


def report(r: dict, out: Path) -> None:
    f = probe._fmt
    lines = [
        "Hướng A (PLAN_PHASE_C §2H): so khớp patch DINOv2-S / A4, 41 ảnh mẫu, P1-val",
        "",
        f"{'dòng':26s} {'top-1':>6s} {'top-3':>6s} {'AUROC':>6s} {'lề':>6s} {'hub':>6s}",
    ]
    rows = [("mốc 2B toàn ảnh (CL2N)", r["baseline_2b_41"])]
    rows += [
        (f"patch {k}" + (" ← dự thi" if k == r["best"] else ""), s)
        for k, s in r["variants"].items()
    ]
    for label, s in rows:
        lines.append(
            f"{label:26s} {f(s['macro_top1'])} {f(s['macro_top3'])} {f(s['auroc_cos'])}"
            f" {f(s['auroc_margin'])} {f(s['hub_share'])}"
        )
    d = r["decision"]
    lines += [
        "",
        f"Luật §1.2 ({r['best']} vs mốc): {d['n_up']}/10 lớp tăng, Δtop-1 {d['d_macro_top1']:+.3f},"
        f" ΔAUROC {d['d_auroc']:+.3f} -> {'ĐẠT' if d['pass'] else 'KHÔNG ĐẠT'}",
        "",
        "Từng lớp top-1 (mốc -> dự thi) [đoán nhiều nhất của dự thi]:",
    ]
    b, c = r["baseline_2b_41"]["per_class"], r["variants"][r["best"]]["per_class"]
    for k in b:
        lines.append(
            f"  {k:10s} {b[k]['top1']:.3f} -> {c[k]['top1']:.3f}  [{c[k]['most_predicted']}]"
        )
    text = "\n".join(lines)
    (out / "report.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
