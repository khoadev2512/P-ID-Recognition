"""PLAN_PHASE_C §2I: mid-level ImageNet CNN feature matching (Gan et al., Sensors 2023), on P1-val.

Gan et al., "Zero-Shot Traffic Sign Recognition Based on Midlevel Feature Matching": frozen ImageNet
CNNs, the WHOLE mid-level feature map of a target is compared with that of each template by a sum
of squared differences (no pooling, no training). Here: A4 on crops and templates (as 2B), plain
224 x 224 resize, ImageNet mean/std, 41 templates (§2G), 1 template view, 8 query views
(oneshot.query_views); score(crop, template) = mean over the 8 views.

    E: -||f_q - f_t||^2 on the raw flattened map (the paper's Eq. 3)
    C: cosine of the flattened maps

Competing variants: {resnet50, densenet121, resnet34} x mid (stride 16, 14 x 14) x {E, C}; the
best by P1-val macro top-1 goes against the 2B global baseline (DINOv2-S / A4 / CL2N, same 41
templates, from the probe cache) under the §1.2 rule (oneshot_probe.decide). The last stage
(stride 32, 7 x 7) is scored too, as a reference only ("mid > last" in the paper).

    python scripts/midlevel_probe.py --out runs/phase_c/midlevel [--device mps]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import oneshot_probe as probe  # noqa: E402
from patch_probe import rows_from_scores  # noqa: E402

from pipeline.iso_prototypes import load_templates, read_vocab  # noqa: E402
from utils.oneshot import ViewSpec, make_views  # noqa: E402

logger = logging.getLogger("midlevel_probe")

DROPPED = "pipeline"  # §2G
ARCHS = {  # tie -> this order
    "resnet50": "resnet50.tv_in1k",
    "densenet121": "densenet121.tv_in1k",
    "resnet34": "resnet34.a1_in1k",
}
STAGES = {"mid": 3, "last": 4}  # timm features_only index: stride 16 / stride 32
SCORINGS = ("E", "C")
SIZE = 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def variant_name(arch: str, stage: str, scoring: str) -> str:
    return f"{arch}/{stage}/{scoring}"


COMPETING = [variant_name(a, "mid", s) for a in ARCHS for s in SCORINGS]
REFERENCE = [variant_name(a, "last", s) for a in ARCHS for s in SCORINGS]


# --------------------------------------------------------------------------- #
# pure functions (tests/test_midlevel_probe.py)
# --------------------------------------------------------------------------- #


def flat_scores(q: torch.Tensor, t: torch.Tensor, scoring: str) -> torch.Tensor:
    """q (V, D), t (T, D) flattened maps -> (V, T). E = -squared L2 (float64), C = cosine."""
    if scoring == "E":
        q, t = q.double(), t.double()
        d = (q * q).sum(1, keepdim=True) + (t * t).sum(1)[None] - 2.0 * q @ t.T
        return (-d.clamp_min(0)).float()
    if scoring == "C":
        return F.normalize(q, dim=1) @ F.normalize(t, dim=1).T
    raise ValueError(scoring)


def pick_best(summaries: dict[str, dict]) -> str:
    """Highest macro top-1 among COMPETING; ties keep the first in COMPETING order."""
    return max(COMPETING, key=lambda k: (summaries[k]["macro_top1"], -COMPETING.index(k)))


def to_tensor(img: Image.Image) -> torch.Tensor:
    """Plain 224 x 224 resize (no crop), ImageNet mean / std -> (3, 224, 224)."""
    a = torch.frombuffer(
        bytearray(img.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR).tobytes()),
        dtype=torch.uint8,
    )
    x = a.view(SIZE, SIZE, 3).permute(2, 0, 1).float() / 255.0
    return (x - torch.tensor(MEAN).view(3, 1, 1)) / torch.tensor(STD).view(3, 1, 1)


# --------------------------------------------------------------------------- #
# embedding
# --------------------------------------------------------------------------- #


@torch.inference_mode()
def feats(model, images: list[Image.Image], device) -> dict[str, torch.Tensor]:
    """stage -> (n, C*H*W) flattened maps, float32 on the CPU."""
    x = torch.stack([to_tensor(im) for im in images]).to(device)
    out = model(x)
    return {s: out[i].flatten(1).float().cpu() for i, s in enumerate(STAGES)}


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
        import timm

        rows = read_vocab(ROOT / "data/raw/iso_symbols/vocab.csv")
        templates, _ = load_templates(ROOT / "data/raw/iso_symbols_clean", rows)
        t_imgs = [make_views(a4(templates[n]), spec_t, bg)[0] for n in names]
        pixels = probe.cut_crops(p1val)

        def views_of(c):
            return make_views(a4(Image.fromarray(pixels[c["ann"]["id"]])), spec_q, bg)

        out = {k: torch.empty(len(p1val), len(names)) for k in COMPETING + REFERENCE}
        shapes = {}
        n_v = spec_q.n_views
        for arch, tag in ARCHS.items():
            model = timm.create_model(
                tag, pretrained=True, features_only=True, out_indices=tuple(STAGES.values())
            )
            model = model.eval().to(device)
            shapes[arch] = {
                s: [int(c), SIZE // int(r), SIZE // int(r)]
                for s, c, r in zip(
                    STAGES,
                    model.feature_info.channels(),
                    model.feature_info.reduction(),
                    strict=True,
                )
            }
            t_f = feats(model, t_imgs, device)
            for k in range(0, len(p1val), 8):
                part = p1val[k : k + 8]
                q_f = feats(model, [v for c in part for v in views_of(c)], device)
                for stage in STAGES:
                    for scoring in SCORINGS:
                        s = flat_scores(q_f[stage], t_f[stage], scoring).view(len(part), n_v, -1)
                        out[variant_name(arch, stage, scoring)][k : k + len(part)] = s.mean(1)
                if (k // 8) % 25 == 0:
                    logger.info("%s: scored %d / %d P1-val crops", arch, k + len(part), len(p1val))
            logger.info("%s done, maps %s", arch, shapes[arch])
            del model
        scores = {
            "names": names,
            "ann_ids": [c["ann"]["id"] for c in p1val],
            "shapes": shapes,
            "scores": out,
        }
        torch.save(scores, score_file)
    assert scores["names"] == names and scores["ann_ids"] == [c["ann"]["id"] for c in p1val]

    summ = {
        k: probe.summarize(rows_from_scores(p1val, s, names, expected_of))
        for k, s in scores["scores"].items()
    }
    best = pick_best(summ)
    verdict = probe.decide(base, summ[best])
    results = {
        "baseline_2b_41": base,
        "shapes": scores["shapes"],
        "competing": {k: summ[k] for k in COMPETING},
        "reference_last": {k: summ[k] for k in REFERENCE},
        "best": best,
        "decision": verdict,
    }
    (args.out / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    report(results, args.out)


def report(r: dict, out: Path) -> None:
    f = probe._fmt
    head = f"{'dòng':30s} {'top-1':>6s} {'top-3':>6s} {'AUROC':>6s} {'lề':>6s} {'hub':>6s}"
    lines = [
        "§2I: tầng giữa CNN ImageNet (Gan et al. 2023), A4, 41 ảnh mẫu, P1-val",
        "",
        head,
    ]

    def row(label, s):
        lines.append(
            f"{label:30s} {f(s['macro_top1'])} {f(s['macro_top3'])} {f(s['auroc_cos'])}"
            f" {f(s['auroc_margin'])} {f(s['hub_share'])}"
        )

    row("mốc 2B toàn ảnh (CL2N)", r["baseline_2b_41"])
    for k, s in r["competing"].items():
        row(k + (" ← dự thi" if k == r["best"] else ""), s)
    lines += ["", "Tham khảo (tầng cuối, không dự thi):", head]
    for k, s in r["reference_last"].items():
        row(k, s)
    d = r["decision"]
    lines += [
        "",
        f"Luật §1.2 ({r['best']} vs mốc): {d['n_up']}/10 lớp tăng, Δtop-1 {d['d_macro_top1']:+.3f},"
        f" ΔAUROC {d['d_auroc']:+.3f} -> {'ĐẠT' if d['pass'] else 'KHÔNG ĐẠT'}",
        "",
        "Từng lớp top-1 (mốc -> dự thi) [đoán nhiều nhất của dự thi]:",
    ]
    b, c = r["baseline_2b_41"]["per_class"], r["competing"][r["best"]]["per_class"]
    for k in b:
        lines.append(
            f"  {k:10s} {b[k]['top1']:.3f} -> {c[k]['top1']:.3f}  [{c[k]['most_predicted']}]"
        )
    text = "\n".join(lines)
    (out / "report.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
