"""PLAN_PHASE_C §2E.1 "F0": linear template -> crop alignment on frozen DINOv2-S / A4 features.

Reads only the Bước 1 probe cache (`runs/phase_c/probe/emb_dinov2_a4.pt`, same crops, nothing is
re-embedded) and writes `--out` (report.txt, results.json). CL2N space as the baseline: every
query view minus the crop center c, unit; only the templates move.

    python scripts/f0_linear_align.py --out runs/phase_c/f0
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import oneshot_probe as probe  # noqa: E402

from utils.linear_align import (  # noqa: E402
    ResidualRidge,
    align_templates,
    choose_lambda,
    domain_center,
    load_pairs,
    pair_shift,
    save_pairs,
)

logger = logging.getLogger("f0")
ENCODER, INPUT = "dinov2", "a4"
SPLITS = (0, 1, 2, 3, 4)
PAIRS_FILE = "pairs_dinov2_a4.pt"


def _summ(rows):
    return probe.summarize(rows)


def loco(scored_rows, known, unknown, fit_protos):
    """Leave-one-class-out over `known` classes. fit_protos(train_classes) -> (42, D) protos.
    Returns a summary shaped like probe.summarize (per_class from each class' own fold, AUROC =
    mean over folds of "class k vs every unknown crop") and the per-fold lambdas."""
    per, aucs, margins, lams, top3 = {}, [], [], {}, []
    for k in known:
        protos, lam = fit_protos([c for c in known if c != k])
        s = _summ(scored_rows(protos, k, unknown))
        per[k] = s["per_class"][k]
        top3.append(s["per_class"][k]["top3"])
        aucs.append(s["auroc_cos"])
        margins.append(s["auroc_margin"])
        lams[k] = lam
    return {
        "macro_top1": float(np.mean([p["top1"] for p in per.values()])),
        "macro_top3": float(np.mean(top3)),
        "auroc_cos": float(np.mean(aucs)),
        "auroc_margin": float(np.mean(margins)),
        "per_class": per,
        "lambdas": lams,
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--probe", type=Path, default=ROOT / "runs/phase_c/probe")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(ROOT / "configs/config.yaml")
    cfg.data_root = str(ROOT / "data")
    world = probe.load_world(cfg)
    crops = probe.pick_crops(world)
    cache = torch.load(args.probe / f"emb_{ENCODER}_{INPUT}.pt", weights_only=False)
    emb, t_raw, names = cache["emb"], cache["protos"], cache["names"]
    groups = world["groups"]
    template = lambda n: groups[n].template or ""  # noqa: E731

    by_set = defaultdict(list)
    for c in crops:
        by_set[(c["set"], c["role"])].append(c)
    center = torch.cat([emb[c["ann"]["id"]] for c in by_set[("p1", "center")]]).mean(0)
    emb_c = {a: F.normalize(v - center, dim=-1) for a, v in emb.items()}  # CL2N views
    query = {a: F.normalize(v.mean(0), dim=-1) for a, v in emb_c.items()}  # as probe.rank

    known = list(probe.P1VAL_KNOWN)
    t_idx = {k: names.index(template(k)) for k in known}
    assert len(set(t_idx.values())) == len(known), "two P1-val classes share a template"
    x_all = F.normalize(t_raw - center, dim=-1)  # baseline protos in CL2N space
    centers_by, centers = defaultdict(list), defaultdict(list)
    for c in by_set[("p1", "center")]:
        centers_by[c["class"]].append(query[c["ann"]["id"]])
        centers[c["class"]].append(c)
    y = {k: F.normalize(torch.stack(centers_by[k]).mean(0), dim=-1) for k in known}
    assert all(len(centers_by[k]) == probe.P1_SEEN_PER_CLASS for k in known)

    p1val = by_set[("p1val", "known")] + by_set[("p1val", "unknown")]
    p1val_known = defaultdict(list)
    for c in by_set[("p1val", "known")]:
        p1val_known[c["class"]].append(c)
    p1val_unknown = by_set[("p1val", "unknown")]
    h_in = set(world["holdout"].h_in)
    p1 = [
        dict(c, role="known" if c["class"] in h_in else "unknown") for c in by_set[("p1", "test")]
    ]

    def rows(cs, protos):
        return probe.rows_for(cs, emb_c, protos, names, None, template, "known")

    def scored_rows(protos, k, unknown):
        return rows(p1val_known[k] + unknown, protos)

    def ridge(train, lam=None):
        x = torch.stack([x_all[t_idx[k]] for k in train])
        yy = torch.stack([y[k] for k in train])
        lam = choose_lambda(x, yy) if lam is None else lam
        return ResidualRidge.fit(x, yy, lam)(x_all), lam

    results = {}
    # baseline: must reproduce the probe's dinov2 / a4 / cl2n P1-val numbers
    base = _summ(rows(p1val, x_all))
    ref = json.loads((args.probe / "results.json").read_text())[f"{ENCODER}/{INPUT}"]["cl2n"]
    assert abs(base["macro_top1"] - ref["p1val"]["macro_top1"]) < 1e-9, base["macro_top1"]
    assert abs(base["auroc_cos"] - ref["p1val"]["auroc_cos"]) < 1e-9, base["auroc_cos"]
    results["base"] = base
    results["base_loco"] = loco(scored_rows, known, p1val_unknown, lambda tr: (x_all, None))

    # F0-a: each domain on its own center
    fa = domain_center(t_raw)
    results["f0a"] = _summ(rows(p1val, fa))
    results["f0a_decide"] = probe.decide(base, results["f0a"])

    # F0-b: residual ridge, leave-one-class-out, lambda chosen inside the 9 train classes
    results["f0b"] = loco(scored_rows, known, p1val_unknown, ridge)
    results["f0b_decide"] = probe.decide(results["base_loco"], results["f0b"])
    logger.info("F0-b done")

    # reported only: (i) shift only
    results["f0b_shift"] = loco(scored_rows, known, p1val_unknown, lambda tr: ridge(tr, math.inf))
    # (ii) 5 train classes: 5 fixed 5/5 splits, both directions
    per5 = defaultdict(list)
    for s in SPLITS:
        half = sorted(random.Random(f"f0/split/{s}").sample(sorted(known), 5))
        other = [k for k in known if k not in half]
        for train, test in ((half, other), (other, half)):
            protos, _ = ridge(train)
            for k in test:
                per5[k].append(_summ(scored_rows(protos, k, p1val_unknown))["per_class"][k]["top1"])
    results["f0b_5cls"] = {
        "per_class_top1": {k: float(np.mean(v)) for k, v in per5.items()},
        "macro_top1": float(np.mean([np.mean(v) for v in per5.values()])),
    }
    # (iii) P1 sample (holdout classes): F0-a; F0-b fitted on all 10
    protos10, lam10 = ridge(known)
    results["p1"] = {
        "base": _summ(rows(p1, x_all)),
        "f0a": _summ(rows(p1, fa)),
        "f0b_all10": _summ(rows(p1, protos10)),
        "f0b_all10_lambda": lam10,
    }
    assert abs(results["p1"]["base"]["macro_top1"] - ref["p1"]["macro_top1"]) < 1e-9

    # pairs file for the pipeline (§2E.3, oneshot.template_pairs): RAW embeddings of the 10
    # templates and of every view of their 30 train crops; check it reproduces the shift above
    pairs_path = save_pairs(
        args.out / PAIRS_FILE,
        [template(k) for k in known],
        torch.stack([t_raw[t_idx[k]] for k in known]),
        [torch.stack([emb[c["ann"]["id"]] for c in centers[k]]) for k in known],
        {
            "encoder": ENCODER,
            "input_norm": "ink_letterbox",
            "classes": known,
            "ann_ids": {k: [c["ann"]["id"] for c in centers[k]] for k in known},
            "source": "scripts/f0_linear_align.py (PLAN_PHASE_C §2E.3)",
        },
    )
    pairs = load_pairs(pairs_path)
    b = pair_shift(pairs["templates"], pairs["crop_views"], center)
    shift10, _ = ridge(known, math.inf)
    assert (align_templates(t_raw, center, "pair_shift", b) - shift10).abs().max() < 1e-5
    assert (align_templates(t_raw, center, "domain_center") - fa).abs().max() < 1e-6
    results["pairs"] = {"file": str(pairs_path), "shift_norm": float(b.norm())}
    results["p1"]["shift_all10"] = _summ(rows(p1, shift10))
    logger.info("pairs -> %s (|b| = %.4f)", pairs_path, float(b.norm()))

    def clean(o):
        if isinstance(o, float) and math.isinf(o):
            return "inf"
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        return o

    (args.out / "results.json").write_text(json.dumps(clean(results), indent=1))
    report(results, args.out)


def report(r: dict, out: Path) -> None:
    def line(tag, s):
        return (
            f"{tag:26s} {s['macro_top1']:7.3f} {s['macro_top3']:6.3f}"
            f" {s['auroc_cos']:6.3f} {s['auroc_margin']:6.3f}"
        )

    head = f"{'dòng':26s} {'top1':>7s} {'top3':>6s} {'AUROC':>6s} {'lề':>6s}"
    lines = [f"F0 (PLAN_PHASE_C §2E.1), {ENCODER} / {INPUT} / CL2N, P1-val", "", head]
    lines += [line("mốc", r["base"]), line("F0-a tâm riêng miền", r["f0a"]), ""]
    lines += [
        "LOCO (AUROC = TB 10 lần 'lớp k vs 7 OOV'):",
        head,
        line("mốc (cách LOCO)", r["base_loco"]),
        line("F0-b ridge", r["f0b"]),
        line("F0-b chỉ dịch b (báo cáo)", r["f0b_shift"]),
        "",
        "Từng lớp P1-val top-1: mốc | F0-a | F0-b | F0-b chỉ dịch | F0-b 5 lớp train   (λ F0-b)",
    ]
    for k, p in r["base"]["per_class"].items():
        lines.append(
            f"  {k:10s} {p['expected']:30s} {p['top1']:.3f}"
            f" | {r['f0a']['per_class'][k]['top1']:.3f}"
            f" | {r['f0b']['per_class'][k]['top1']:.3f}"
            f" | {r['f0b_shift']['per_class'][k]['top1']:.3f}"
            f" | {r['f0b_5cls']['per_class_top1'][k]:.3f}   ({r['f0b']['lambdas'][k]})"
        )
    lines.append(f"  F0-b 5 lớp train: macro top-1 {r['f0b_5cls']['macro_top1']:.3f}")
    lines.append("")
    for tag in ("f0a", "f0b"):
        d = r[f"{tag}_decide"]
        lines.append(
            f"Luật §1.2 {tag}: lớp tăng {d['n_up']}/10  Δtop1 {d['d_macro_top1']:+.3f}"
            f"  ΔAUROC {d['d_auroc']:+.3f}  -> {'ĐẠT' if d['pass'] else 'KHÔNG ĐẠT'}"
        )
    lines += ["", "Mẫu P1 (lớp holdout thật; KHÔNG quyết định):", head]
    for tag, s in r["p1"].items():
        if isinstance(s, dict):
            lines.append(line(tag, s))
    lines.append(f"  λ F0-b học trên 10 lớp: {r['p1']['f0b_all10_lambda']}")
    text = "\n".join(lines)
    (out / "report.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
