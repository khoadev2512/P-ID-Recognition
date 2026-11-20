"""PLAN_PHASE_C §2K.4: (i) a second valve_butterfly template (BK-2020 X8075 form B) and (ii) how the
8 query views are aggregated (mean = 2B, vote, max), on P4-val (naming) with P1-val as the style
guard. 2B encoder, nothing trained; P4-val view embeddings come from runs/phase_c/p4val/emb.pt.

    python scripts/k4_probe.py --out runs/phase_c/k4 [--device mps]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import oneshot_probe as probe  # noqa: E402
from extract_iso_symbols_svg import render  # noqa: E402
from viso_probe import encoder_at  # noqa: E402

from utils.oneshot import ViewSpec, embed_images, make_views  # noqa: E402

logger = logging.getLogger("k4")

DROPPED = "pipeline"  # §2G
FORM_B = "valve_butterfly@B"
AGGS = ("mean", "vote", "max")  # mean = 2B; tie between candidates -> vote (listed first)
MIN_CROPS = 20
B_BOOT = 1000
I_GAIN, I_OTHER_DROP, P1_DROP = 0.10, 0.05, 0.01
II_GAIN = 0.03


# --------------------------------------------------------------------------- #
# pure functions (tests/test_k4_probe.py)
# --------------------------------------------------------------------------- #


def class_of(name: str) -> str:
    """`valve_butterfly@B` -> `valve_butterfly`."""
    return name.split("@", 1)[0]


def class_scores(views: torch.Tensor, protos: torch.Tensor, center, agg: str, classes, owner):
    """views (n, V, D) unit, protos (P, D) unit, `owner[p]` = index into `classes` of prototype p.
    -> (n, C) class scores; ranking by them gives the aggregation's answer.
    mean: cos of the renormalized mean of CL2N views (probe.rank); max: max over views of the view
    cosine; vote: per-view nearest prototype votes for its class, ranked by (votes, summed cosine of
    those votes, max cosine) encoded in one float. A class scores the max over its prototypes."""
    v = F.normalize(views - center, dim=-1) if center is not None else views
    p = F.normalize(protos - center, dim=-1) if center is not None else protos
    owner = torch.as_tensor(owner)
    n_cls = len(classes)
    cos = torch.einsum("nvd,pd->nvp", v, p)  # (n, V, P)
    if agg == "mean":
        q = F.normalize(v.mean(1), dim=-1)
        per_proto = q @ p.T
    elif agg == "max":
        per_proto = cos.max(1).values
    elif agg == "vote":
        best_cos, best = cos.max(2)  # (n, V)
        cls = owner[best]  # (n, V)
        votes = F.one_hot(cls, n_cls).float().sum(1)  # (n, C)
        summed = torch.zeros_like(votes).scatter_add_(1, cls, best_cos)
        mx = _class_max(cos.max(1).values, owner, n_cls)
        return votes * 1e4 + summed * 10 + mx  # votes <= V, |summed| <= V, |mx| <= 1
    else:
        raise ValueError(agg)
    return _class_max(per_proto, owner, n_cls)


def _class_max(per_proto: torch.Tensor, owner: torch.Tensor, n_cls: int) -> torch.Tensor:
    out = torch.full((per_proto.shape[0], n_cls), -torch.inf)
    return out.scatter_reduce(1, owner.expand_as(per_proto), per_proto, reduce="amax")


def confidence(views: torch.Tensor, protos: torch.Tensor, center, agg: str) -> torch.Tensor:
    """(n,) the cosine each aggregation reports for its answer (the Other gate's input): mean ->
    best cosine of the mean vector; max -> best view cosine; vote -> mean cosine of the winning
    prototype's votes (classify_views)."""
    v = F.normalize(views - center, dim=-1) if center is not None else views
    p = F.normalize(protos - center, dim=-1) if center is not None else protos
    if agg == "mean":
        return (F.normalize(v.mean(1), dim=-1) @ p.T).max(1).values
    cos = torch.einsum("nvd,pd->nvp", v, p)
    if agg == "max":
        return cos.amax(dim=(1, 2))
    best_cos, best = cos.max(2)
    out = []
    for bc, b in zip(best_cos, best, strict=True):
        votes: dict[int, list[float]] = {}
        for c, i in zip(bc.tolist(), b.tolist(), strict=True):
            votes.setdefault(i, []).append(c)
        win = max(votes.values(), key=lambda xs: (len(xs), sum(xs)))
        out.append(sum(win) / len(win))
    return torch.tensor(out)


def macro_top1(df: pd.DataFrame, classes) -> float:
    per = (df.top1 == df["class"]).groupby(df["class"]).mean()
    return float(per[[c for c in classes if c in per.index]].mean())


def boot_delta(base: pd.DataFrame, cand: pd.DataFrame, classes, b=B_BOOT, seed=0):
    """95% CI of macro_top1(cand) - macro_top1(base), resampling sheets (same draw for both)."""
    sheets = np.array(sorted(base.sheet.unique()))
    rng = np.random.default_rng(seed)
    gb, gc = dict(tuple(base.groupby("sheet"))), dict(tuple(cand.groupby("sheet")))
    out = []
    for _ in range(b):
        draw = rng.choice(sheets, len(sheets), replace=True)
        bb = pd.concat([gb[s] for s in draw])
        cc = pd.concat([gc[s] for s in draw])
        out.append(macro_top1(cc, classes) - macro_top1(bb, classes))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def _auroc(score: torch.Tensor, known: np.ndarray) -> float:
    r = pd.Series(score.numpy()).rank().to_numpy()
    n_pos, n_neg = int(known.sum()), int((~known).sum())
    return float((r[known].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# --------------------------------------------------------------------------- #


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--probe", type=Path, default=ROOT / "runs/phase_c/probe")
    ap.add_argument("--p4val", type=Path, default=ROOT / "runs/phase_c/p4val")
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
    p1val = [c for c in crops if c["set"] == "p1val"]
    centers = [c for c in crops if c["set"] == "p1" and c["role"] == "center"]
    spec_t = ViewSpec.from_cfg(cfg.oneshot.template_views)
    bg = tuple(int(c) for c in cfg.oneshot.background)
    a4 = probe.norm_fn("a4")
    cache = torch.load(args.probe / "emb_dinov2_a4.pt", weights_only=False)
    keep = [i for i, n in enumerate(cache["names"]) if n != DROPPED]
    names = [cache["names"][i] for i in keep]
    protos41 = cache["protos"][keep]
    center = torch.cat([cache["emb"][c["ann"]["id"]] for c in centers]).mean(0)

    # (i) form B template, rendered exactly as iso_symbols_clean
    store = args.out / "form_b.pt"
    if store.exists():
        fb = torch.load(store, weights_only=False)
    else:
        imgs = {}
        for form in ("X8075_1", "X8075_2"):
            svg = (ROOT / "data/raw/template_iso_svg" / f"{form}.svg").read_bytes()
            imgs[form] = render(svg, dpi=600, pad=6)
            imgs[form].save(args.out / f"{form}.png")
        encoder, tf = encoder_at(224, device)
        e = embed_images(
            encoder, [make_views(a4(imgs[f]), spec_t, bg)[0] for f in imgs], tf, device
        )
        fb = {"form_a_rerender": e[0], "form_b": e[1]}
        torch.save(fb, store)
    a_idx = names.index("valve_butterfly")
    a_cos = float(
        F.normalize(fb["form_a_rerender"] - center, dim=0)
        @ F.normalize(protos41[a_idx] - center, dim=0)
    )
    b_cos = float(
        F.normalize(fb["form_b"] - center, dim=0) @ F.normalize(protos41[a_idx] - center, dim=0)
    )
    logger.info(
        "CL2N cos: form A re-render vs template %.3f; form B vs template %.3f", a_cos, b_cos
    )

    classes = names  # 41 classes; the extra prototype maps onto valve_butterfly
    dirs = {
        "41": (protos41, list(range(len(names)))),
        "41+B": (torch.cat([protos41, fb["form_b"][None]]), list(range(len(names))) + [a_idx]),
    }

    st = torch.load(args.p4val / "emb.pt", weights_only=False)
    assert st["names"] == names
    rows = pd.DataFrame(st["rows"])
    known = rows["class"].isin(names).to_numpy()
    kv = st["emb"][torch.as_tensor(known)]
    kdf = rows[known].reset_index(drop=True)
    n = kdf["class"].value_counts()
    main_cls = sorted(c for c in n.index if n[c] >= MIN_CROPS)
    p1v = torch.stack([cache["emb"][c["ann"]["id"]] for c in p1val])
    p1_true = [groups[c["class"]].template or "" for c in p1val]
    p1_known = np.array([c["role"] == "known" for c in p1val])

    out, frames = {}, {}
    for dname, (protos, owner) in dirs.items():
        for agg in AGGS:
            key = f"{dname}/{agg}"
            s = class_scores(kv, protos, center, agg, classes, owner)
            top = s.topk(3, dim=1).indices
            df = kdf.copy()
            for j in range(3):
                df[f"top{j + 1}"] = [classes[i] for i in top[:, j].tolist()]
            frames[key] = df
            per = (df.top1 == df["class"]).groupby(df["class"]).mean()
            sp = class_scores(p1v, protos, center, agg, classes, owner)
            pred = [classes[i] for i in sp.argmax(1).tolist()]
            p1_per = (
                pd.Series([t == p for t, p, k in zip(p1_true, pred, p1_known, strict=True) if k])
                .groupby([c["class"] for c in p1val if c["role"] == "known"])
                .mean()
            )
            out[key] = {
                "p4val_macro": macro_top1(df, main_cls),
                "p4val_per_class": {c: float(per[c]) for c in main_cls},
                "p1val_macro": float(p1_per.mean()),
                "p1val_auroc": _auroc(confidence(p1v, protos, center, agg), p1_known),
            }
            logger.info(
                "%s: P4-val %.3f  P1-val %.3f",
                key,
                out[key]["p4val_macro"],
                out[key]["p1val_macro"],
            )

    base = out["41/mean"]
    # (i)
    ci = out["41+B/mean"]
    bfly = ci["p4val_per_class"]["valve_butterfly"] - base["p4val_per_class"]["valve_butterfly"]
    worst = min(
        (ci["p4val_per_class"][c] - base["p4val_per_class"][c], c)
        for c in main_cls
        if c != "valve_butterfly"
    )
    dec_i = {
        "d_butterfly": bfly,
        "worst_other": {"class": worst[1], "delta": worst[0]},
        "d_macro": ci["p4val_macro"] - base["p4val_macro"],
        "d_p1val": ci["p1val_macro"] - base["p1val_macro"],
    }
    dec_i["pass"] = bool(
        bfly >= I_GAIN
        and worst[0] >= -I_OTHER_DROP
        and dec_i["d_macro"] >= 0
        and dec_i["d_p1val"] >= -P1_DROP
    )
    # (ii)
    cand = max(("vote", "max"), key=lambda a: (out[f"41/{a}"]["p4val_macro"], a == "vote"))
    cii = out[f"41/{cand}"]
    lo, hi = boot_delta(frames["41/mean"], frames[f"41/{cand}"], main_cls)
    dec_ii = {
        "candidate": cand,
        "d_macro": cii["p4val_macro"] - base["p4val_macro"],
        "ci95": [lo, hi],
        "d_p1val": cii["p1val_macro"] - base["p1val_macro"],
    }
    dec_ii["pass"] = bool(dec_ii["d_macro"] >= II_GAIN and lo > 0 and dec_ii["d_p1val"] >= -P1_DROP)
    res = {
        "classes_main": main_cls,
        "consistency": {"cos_formA_rerender_vs_template": a_cos, "cos_formB_vs_template": b_cos},
        "rows": out,
        "decision_i": dec_i,
        "decision_ii": dec_ii,
    }
    (args.out / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    report(res, args.out)


def report(r: dict, out: Path) -> None:
    c = r["consistency"]
    lines = [
        "§2K.4: (i) ảnh mẫu dạng B của butterfly, (ii) cách gộp 8 view — P4-val (18 lớp ≥ 20 crop),"
        " P1-val chặn hại",
        f"kiểm: cos CL2N form A render lại vs ảnh mẫu {c['cos_formA_rerender_vs_template']:.3f};"
        f" form B vs ảnh mẫu {c['cos_formB_vs_template']:.3f}",
        "",
        f"{'thư mục / gộp':16s} {'P4-val':>7s} {'P1-val':>7s} {'P1 AUROC':>9s}"
        f" {'butterfly':>10s} {'globe':>6s}",
    ]
    for k, s in r["rows"].items():
        pc = s["p4val_per_class"]
        lines.append(
            f"{k:16s} {s['p4val_macro']:7.3f} {s['p1val_macro']:7.3f} {s['p1val_auroc']:9.3f}"
            f" {pc['valve_butterfly']:10.2f} {pc['valve_globe']:6.2f}"
        )
    d = r["decision_i"]
    lines += [
        "",
        f"(i) 41+B vs 41 (mean): Δ butterfly {d['d_butterfly']:+.3f} (≥ +{I_GAIN});"
        f" lớp khác tệ nhất {d['worst_other']['class']} {d['worst_other']['delta']:+.3f}"
        f" (≥ −{I_OTHER_DROP}); Δ macro {d['d_macro']:+.3f} (≥ 0); Δ P1-val {d['d_p1val']:+.3f}"
        f" (≥ −{P1_DROP}) -> {'ĐẠT' if d['pass'] else 'KHÔNG ĐẠT'}",
    ]
    d = r["decision_ii"]
    lines += [
        f"(ii) dự thi {d['candidate']} vs mean (41): Δ macro {d['d_macro']:+.3f} (≥ +{II_GAIN}),"
        f" CI95 theo tờ [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] (không chứa 0);"
        f" Δ P1-val {d['d_p1val']:+.3f} (≥ −{P1_DROP}) -> {'ĐẠT' if d['pass'] else 'KHÔNG ĐẠT'}",
        "",
        "Từng lớp P4-val (41/mean | 41+B/mean | 41/vote | 41/max):",
    ]
    rows = r["rows"]
    for cl in r["classes_main"]:
        vals = [
            rows[k]["p4val_per_class"][cl] for k in ("41/mean", "41+B/mean", "41/vote", "41/max")
        ]
        lines.append(f"  {cl:28s} " + "  ".join(f"{v:.2f}" for v in vals))
    text = "\n".join(lines)
    (out / "report.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
