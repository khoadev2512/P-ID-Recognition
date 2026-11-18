"""PLAN_PHASE_C §2D.3 "H1": the k-shot curve with DigitizePID crops as the exemplars.

Descriptive, no decision. Directory = 28 classes (32 minus h_out), k crops per class from train
images, prototype = mean of the k crop embeddings (each crop: its query views averaged), unit;
k in K_SHOTS, nested (one sample of max(K_SHOTS) per class and draw, its first k), DRAWS draws.
Queries = the probe's P1 sample (h_in known, h_out unknown), minus every crop on a sheet that
holds one of the draw's exemplars, so every k of a draw scores the same queries. Queries and the
CL2N center come from the Bước 1 probe cache (same crops, same embeddings).

    python scripts/h1_kshot.py --out runs/phase_c/h1 [--score-only]
"""

from __future__ import annotations

import argparse
import json
import logging
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

from utils.oneshot import ViewSpec  # noqa: E402

logger = logging.getLogger("h1")
K_SHOTS = (1, 3, 5)
DRAWS = (0, 1, 2, 3, 4)
ENCODERS = ("lo", "imagenet", "dinov2")
INPUTS = ("raw", "a4")


def pick_exemplars(world: dict) -> list[dict]:
    """max(K_SHOTS) train crops per class (h_out left out) for every draw, in sample order."""
    coco, split = world["coco"], world["split"]
    cat = {c["id"]: c["name"] for c in coco["categories"]}
    stem = {i["id"]: Path(i["file_name"]).stem for i in coco["images"]}
    h_out = set(world["holdout"].h_out)
    by = defaultdict(list)
    for a in coco["annotations"]:
        name = cat[a["category_id"]]
        if not a.get("iscrowd") and name not in h_out and split.get(stem[a["image_id"]]) == "train":
            by[name].append(a)
    out = []
    for d in DRAWS:
        for name in sorted(by):
            anns = sorted(by[name], key=lambda a: a["id"])
            for rank, a in enumerate(
                random.Random(f"kshot/{d}/{name}").sample(anns, min(max(K_SHOTS), len(anns)))
            ):
                out.append(
                    {
                        "set": f"k{d}",
                        "role": "exemplar",
                        "class": name,
                        "ann": a,
                        "rank": rank,
                        "stem": stem[a["image_id"]],
                    }
                )
    return out


def kshot_protos(exemplars, emb, draw, k):
    """(names, (N, D) unit prototypes) of one draw at k shots."""
    by = defaultdict(list)
    for c in exemplars:
        if c["set"] == f"k{draw}" and c["rank"] < k:
            by[c["class"]].append(F.normalize(emb[c["ann"]["id"]].mean(0), dim=-1))
    names = sorted(by)
    return names, F.normalize(torch.stack([torch.stack(by[n]).mean(0) for n in names]), dim=-1)


def score(world, crops, exemplars, q_emb, ex_emb):
    seen = torch.cat([q_emb[c["ann"]["id"]] for c in crops if c["role"] == "center"])
    center = seen.mean(0)
    h_in = set(world["holdout"].h_in)
    p1 = [
        dict(c, role="known" if c["class"] in h_in else "unknown")
        for c in crops
        if c["set"] == "p1" and c["role"] == "test"
    ]
    out = {}
    for cl2n in (False, True):
        cen = center if cl2n else None
        per_k = {}
        for k in K_SHOTS:
            draws = []
            for d in DRAWS:
                used = {c["stem"] for c in exemplars if c["set"] == f"k{d}"}
                q = [c for c in p1 if c["stem"] not in used]
                names, protos = kshot_protos(exemplars, ex_emb, d, k)
                s = probe.summarize(
                    probe.rows_for(q, q_emb, protos, names, cen, lambda n: n, "known")
                )
                s["n_queries"] = len(q)
                draws.append(s)
            per_k[k] = {
                "draws": draws,
                "top1": float(np.mean([s["macro_top1"] for s in draws])),
                "top1_std": float(np.std([s["macro_top1"] for s in draws], ddof=1)),
                "top3": float(np.mean([s["macro_top3"] for s in draws])),
                "auroc": float(np.mean([s["auroc_cos"] for s in draws])),
                "auroc_std": float(np.std([s["auroc_cos"] for s in draws], ddof=1)),
            }
        out["cl2n" if cl2n else "raw"] = per_k
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--probe", type=Path, default=ROOT / "runs/phase_c/probe")
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(ROOT / "configs/config.yaml")
    cfg.data_root = str(ROOT / "data")
    world = probe.load_world(cfg)
    crops = probe.pick_crops(world)
    exemplars = pick_exemplars(world)
    logger.info("%d exemplar crops", len(exemplars))
    spec_q = ViewSpec.from_cfg(cfg.oneshot.query_views)
    bg = tuple(int(c) for c in cfg.oneshot.background)
    device = torch.device("cpu")
    pixels = None
    results = {}
    for enc in ENCODERS:
        for kind in INPUTS:
            cache = args.out / f"ex_{enc}_{kind}.pt"
            if not (args.score_only and cache.exists()):
                if pixels is None:
                    pixels = probe.cut_crops(exemplars)
                logger.info("embedding exemplars %s / %s", enc, kind)
                encoder, tf = probe.load_encoder(enc, cfg, device)
                emb = probe.embed_all(
                    encoder, tf, exemplars, pixels, probe.norm_fn(kind), spec_q, bg, device
                )
                torch.save(emb, cache)
            ex_emb = torch.load(cache, weights_only=False)
            q_emb = torch.load(args.probe / f"emb_{enc}_{kind}.pt", weights_only=False)["emb"]
            results[f"{enc}/{kind}"] = score(world, crops, exemplars, q_emb, ex_emb)
            logger.info("scored %s / %s", enc, kind)
    (args.out / "results.json").write_text(json.dumps(results, indent=1))
    ref = json.loads((args.probe / "results.json").read_text())
    lines = [
        "H1 k-shot (PLAN_PHASE_C §2D.3), mẫu P1: h_in biết (top-1 / top-3 macro, 28 ứng viên), "
        "h_out lạ (AUROC); trung bình ± độ lệch chuẩn qua 5 lần rút",
        f"{'dòng':22s} {'ISO top1':>8s} "
        + " ".join(f"{'k=' + str(k) + ' top1':>15s} {'top3':>6s} {'AUROC':>6s}" for k in K_SHOTS),
    ]
    for key, res in results.items():
        for c, per_k in res.items():
            iso = ref.get(key, {}).get(c, {}).get("p1", {}).get("macro_top1")
            row = f"{key + '/' + c:22s} {iso if iso is None else f'{iso:8.3f}'} "
            row += " ".join(
                f"{per_k[k]['top1']:7.3f} ± {per_k[k]['top1_std']:.3f} {per_k[k]['top3']:6.3f}"
                f" {per_k[k]['auroc']:6.3f}"
                for k in K_SHOTS
            )
            lines.append(row)
    text = "\n".join(lines)
    (args.out / "report.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
