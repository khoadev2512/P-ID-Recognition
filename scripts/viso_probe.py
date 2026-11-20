"""PLAN_PHASE_C §2K step 1: V-ISO (queries generated from the 41 ISO templates), the
"is V-ISO representative of P4" gate, a diagnosis of where near-duplicate confusions come from,
and R2 (DINOv2-S at 448 px instead of 224).

V-ISO: 30 queries per template from utils.template_variants.dirty_view (seed 2026), through the
crop path: A4 -> 8 query views -> DINOv2-S -> CL2N (probe center crops) -> cosine to the 41
one-view prototypes. Diagnosis modes: clean (the template itself), pipes only, stroke + scan only,
full (= V-ISO), full with 1 query view. Each at 224 and 448 px. P1-val (DigitizePID crops) is
the style guard for R2.

    python scripts/viso_probe.py --out runs/phase_c/viso [--device mps]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import oneshot_probe as probe  # noqa: E402
from h2_template_variants import vocab_templates  # noqa: E402

from utils.oneshot import ViewSpec, embed_images, make_views  # noqa: E402
from utils.template_variants import axes_of, dirty_view, template_seed  # noqa: E402

logger = logging.getLogger("viso")

DROPPED = "pipeline"  # §2G
SEED, N_PER = 2026, 30
RESOLUTIONS = (224, 448)
MODES = {  # name -> (pipes, morph, degrade); None = the clean template itself
    "clean": None,
    "pipes": (True, False, False),
    "scan": (False, True, True),
    "full": (True, True, True),
}
CLOSE = [  # §2K: classes with another template at CL2N cos >= 0.85 (224 px), fixed before scoring
    "blind_disc",
    "breather_valve",
    "check_valve_general",
    "check_valve_globe",
    "check_valve_swing",
    "compressor_centrifugal",
    "compressor_vacuum",
    "interchangeable_disc",
    "interchangeable_disc_open",
    "open_disc",
    "pump_centrifugal",
    "pump_diaphragm",
    "pump_general",
    "valve_butterfly",
    "valve_globe",
    "valve_needle",
    "valve_three_way",
]
GATE_RHO, GATE_GB = 0.5, 0.20
R2_CLOSE, R2_DROP = 0.05, 0.01
WATCH = [  # pairs reported in the table (a -> b and b -> a)
    ("valve_globe", "valve_butterfly"),
    ("valve_butterfly", "check_valve_swing"),
    ("check_valve_globe", "check_valve_swing"),
    ("interchangeable_disc", "interchangeable_disc_open"),
    ("valve_needle", "valve_three_way"),
    ("pump_general", "pump_diaphragm"),
]


# --------------------------------------------------------------------------- #
# pure functions (tests/test_viso_probe.py)
# --------------------------------------------------------------------------- #


def per_class_top1(true: list[str], pred: list[str]) -> dict[str, float]:
    out: dict[str, list[bool]] = {}
    for t, p in zip(true, pred, strict=True):
        out.setdefault(t, []).append(t == p)
    return {k: float(np.mean(v)) for k, v in out.items()}


def macro(per: dict[str, float], classes) -> float:
    vals = [per[c] for c in classes if c in per]
    return float(np.mean(vals)) if vals else float("nan")


def pair_rate(true: list[str], pred: list[str], a: str, b: str) -> float:
    """Share of class-`a` queries named `b`."""
    sel = [p for t, p in zip(true, pred, strict=True) if t == a]
    return float(np.mean([p == b for p in sel])) if sel else float("nan")


def spearman(x, y) -> float:
    """Spearman rho with average ranks for ties."""
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def decide_r2(base: dict, cand: dict) -> dict:
    """§2K R2 rule: close +0.05 and V-ISO all and P1-val each not down by more than 0.01."""
    d_close = cand["close"] - base["close"]
    d_all = cand["all"] - base["all"]
    d_p1 = cand["p1val"] - base["p1val"]
    ok = d_close >= R2_CLOSE and d_all >= -R2_DROP and d_p1 >= -R2_DROP
    return {"d_close": d_close, "d_all": d_all, "d_p1val": d_p1, "pass": bool(ok)}


# --------------------------------------------------------------------------- #
# data / embedding
# --------------------------------------------------------------------------- #


def queries(images, chosen, names, mode: str) -> tuple[list[Image.Image], list[str]]:
    imgs, labels = [], []
    for name in names:
        if MODES[mode] is None:
            imgs.append(images[name].convert("RGB"))
            labels.append(name)
            continue
        pipes, morph, degrade = MODES[mode]
        rng = random.Random(template_seed(SEED, f"{mode}/{name}"))
        axes = axes_of(chosen[name][0])
        for _ in range(N_PER):
            imgs.append(
                dirty_view(images[name], axes, rng, pipes=pipes, morph=morph, degrade=degrade)
            )
            labels.append(name)
    return imgs, labels


def encoder_at(res: int, device):
    from torchvision import transforms

    from utils.dataset import _IMAGENET_MEAN, _IMAGENET_STD

    tf = transforms.Compose(
        [
            transforms.Resize((res, res)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )
    return probe.TimmEncoder(probe.DINOV2, res, device), tf


def embed_views(encoder, tf, imgs, spec, bg, a4, device, chunk=64) -> torch.Tensor:
    """images -> (n, V, D) unit view embeddings (A4 first, as every crop)."""
    out = []
    for k in range(0, len(imgs), chunk):
        views = [v for im in imgs[k : k + chunk] for v in make_views(a4(im), spec, bg)]
        e = embed_images(encoder, views, tf, device, batch_size=128)
        out.append(e.view(-1, spec.n_views, e.shape[-1]))
    return torch.cat(out)


def names_of(idx: torch.Tensor, names) -> list[str]:
    return [names[i] for i in idx[:, 0].tolist()]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--probe", type=Path, default=ROOT / "runs/phase_c/probe")
    ap.add_argument("--p4", type=Path, default=ROOT / "runs/oneshot_p4_crops_dinov2.csv")
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
    cache = torch.load(args.probe / "emb_dinov2_a4.pt", weights_only=False)
    keep = [i for i, n in enumerate(cache["names"]) if n != DROPPED]
    names = [cache["names"][i] for i in keep]
    assert all(c in names for c in CLOSE)
    not_close = [n for n in names if n not in CLOSE]
    images, _, chosen = vocab_templates()
    pixels = None

    emb = {}
    for res in RESOLUTIONS:
        f = args.out / f"emb_{res}.pt"
        if f.exists():
            emb[res] = torch.load(f, weights_only=False)
            continue
        encoder, tf = encoder_at(res, device)
        t_imgs = [make_views(a4(images[n]), spec_t, bg)[0] for n in names]
        protos = embed_images(encoder, t_imgs, tf, device)
        e = {"protos": protos}
        if res == 224:
            diff = float((protos - cache["protos"][keep]).abs().max())
            logger.info("224 prototypes vs probe cache: max |diff| %.2e", diff)
            if diff >= 1e-3:
                raise SystemExit(f"224 path does not reproduce the probe ({diff:.2e})")
            e["proto_diff"] = diff
            e["center"] = torch.cat([cache["emb"][c["ann"]["id"]] for c in centers]).mean(0)
            e["p1val"] = torch.stack([cache["emb"][c["ann"]["id"]] for c in p1val])
        else:
            if pixels is None:
                pixels = probe.cut_crops(p1val + centers)

            def as_img(cs, px=pixels):
                return [Image.fromarray(px[c["ann"]["id"]]) for c in cs]

            e["center"] = embed_views(encoder, tf, as_img(centers), spec_q, bg, a4, device)
            e["center"] = e["center"].flatten(0, 1).mean(0)
            logger.info("%d px: center done", res)
            e["p1val"] = embed_views(encoder, tf, as_img(p1val), spec_q, bg, a4, device)
            logger.info("%d px: P1-val done", res)
        for mode in MODES:
            imgs, labels = queries(images, chosen, names, mode)
            e[mode] = embed_views(encoder, tf, imgs, spec_q, bg, a4, device)
            e[f"{mode}_labels"] = labels
            logger.info("%d px: V-ISO %s done (%d queries)", res, mode, len(imgs))
        torch.save(e, f)
        emb[res] = e
        del encoder

    results = {"names": names, "close": CLOSE, "res": {}}
    for res, e in emb.items():
        r = {}
        for mode in [*MODES, "full_1view"]:
            src = "full" if mode == "full_1view" else mode
            views = e[src][:, :1] if mode == "full_1view" else e[src]
            idx, _ = probe.rank(views, e["protos"], e["center"])
            true, pred = e[f"{src}_labels"], names_of(idx, names)
            per = per_class_top1(true, pred)
            r[mode] = {
                "close": macro(per, CLOSE),
                "not_close": macro(per, not_close),
                "all": macro(per, names),
                "per_class": per,
                "pairs": {
                    f"{a} -> {b}": pair_rate(true, pred, a, b)
                    for x, y in WATCH
                    for a, b in ((x, y), (y, x))
                },
            }
        p1 = {c["ann"]["id"]: v for c, v in zip(p1val, e["p1val"], strict=True)}
        r["p1val"] = probe.summarize(
            probe.rows_for(p1val, p1, e["protos"], names, e["center"], expected_of, "known")
        )
        results["res"][str(res)] = r
    base = results["res"]["224"]
    logger.info("P1-val 224 here: top-1 %.3f (probe 0.477)", base["p1val"]["macro_top1"])

    # gate: is V-ISO representative of P4 (224, full)
    p4 = pd.read_csv(args.p4)
    p4 = p4[(p4.split == "degraded") & p4.expected.isin(names)]
    p4_per = (p4.top1 == p4.expected).groupby(p4.expected).mean().to_dict()
    common = sorted(set(p4_per) & set(base["full"]["per_class"]))
    rho = spearman([base["full"]["per_class"][c] for c in common], [p4_per[c] for c in common])
    gb = base["full"]["pairs"]["valve_globe -> valve_butterfly"]
    gate = {
        "rho": rho,
        "n_classes": len(common),
        "globe_to_butterfly": gb,
        "pass": bool(rho >= GATE_RHO and gb >= GATE_GB),
    }
    results["gate"] = gate
    results["p4_per_class"] = {c: p4_per[c] for c in common}
    if gate["pass"]:

        def key(r):
            return {
                "close": r["full"]["close"],
                "all": r["full"]["all"],
                "p1val": r["p1val"]["macro_top1"],
            }

        results["r2"] = decide_r2(key(base), key(results["res"]["448"]))
    else:
        results["r2"] = None
    (args.out / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    report(results, args.out)


def report(r: dict, out: Path) -> None:
    g = r["gate"]
    lines = [
        "§2K bước 1: V-ISO (30 truy vấn / ảnh mẫu, seed 2026), 41 ảnh mẫu, DINOv2-S / A4 / CL2N",
        "",
        f"Cổng đại diện (224, dạng đủ vs P4 suy giảm): Spearman ρ {g['rho']:.3f}"
        f" trên {g['n_classes']} lớp"
        f" (cần ≥ {GATE_RHO}); globe → butterfly {g['globe_to_butterfly']:.3f} (cần ≥ {GATE_GB})"
        f" -> {'QUA' if g['pass'] else 'TRƯỢT'}",
        "",
        f"{'dạng':12s} {'px':>4s} {'17 sát':>7s} {'24 khác':>8s} {'cả 41':>6s}",
    ]
    for mode in [*MODES, "full_1view"]:
        for res in ("224", "448"):
            s = r["res"][res][mode]
            lines.append(
                f"{mode:12s} {res:>4s} {s['close']:7.3f} {s['not_close']:8.3f} {s['all']:6.3f}"
            )
    lines += ["", f"{'P1-val (chặn hại phong cách)':30s} {'top-1':>6s} {'AUROC':>6s}"]
    for res in ("224", "448"):
        s = r["res"][res]["p1val"]
        lines.append(f"{res + ' px':30s} {s['macro_top1']:6.3f} {s['auroc_cos']:6.3f}")
    lines += ["", "Tỉ lệ nhầm từng cặp (dạng đủ; 224 -> 448):"]
    for k in r["res"]["224"]["full"]["pairs"]:
        a, b = r["res"]["224"]["full"]["pairs"][k], r["res"]["448"]["full"]["pairs"][k]
        lines.append(f"  {k:52s} {a:.2f} -> {b:.2f}")
    if r["r2"] is not None:
        d = r["r2"]
        lines += [
            "",
            f"Luật R2 (448 vs 224): Δ 17 sát {d['d_close']:+.3f} (cần ≥ +{R2_CLOSE}),"
            f" Δ cả 41 {d['d_all']:+.3f}, Δ P1-val {d['d_p1val']:+.3f} (mỗi cái ≥ −{R2_DROP})"
            f" -> {'ĐẠT' if d['pass'] else 'KHÔNG ĐẠT'}",
        ]
    else:
        lines += ["", "Cổng trượt -> không quyết định R2 trên V-ISO (§2K)."]
    lines += ["", "Từng lớp sát, dạng đủ (224 -> 448) | P4 suy giảm:"]
    for c in r["close"]:
        a = r["res"]["224"]["full"]["per_class"][c]
        b = r["res"]["448"]["full"]["per_class"][c]
        p = r["p4_per_class"].get(c)
        lines.append(f"  {c:28s} {a:.2f} -> {b:.2f} | {'—' if p is None else f'{p:.2f}'}")
    text = "\n".join(lines)
    (out / "report.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
