"""PLAN_PHASE_C §2C.1: C1 (R1a gate + DINOv2 names) and C2 (Mahalanobis instead of cosine).

C1  reads the eval CSVs (no GPU, nothing selected): runs/oneshot_p1_crops.csv (R1a) with
    runs/oneshot_p1_crops_dinov2.csv, and runs/oneshot_p4_crops_{lo,dinov2}.csv. A crop is
    "Other" when R1a's best_cos < TAU_R1A (rule A of notebook 9a, chosen on val_sel); otherwise
    its label is DINOv2's top-1. Reported next to R1a alone and DINOv2 alone, with bootstrap CIs
    (P1: by image and by class; P4: by sheet).
C2  reads the Bước 1 probe cache (runs/phase_c/probe/emb_<encoder>_<input>.pt, written by
    scripts/oneshot_probe.py): in the CL2N space, a shared Ledoit-Wolf covariance of the 570 seen
    train crops; (a) whitened cosine, (b) Mahalanobis nearest prototype with the Relative
    Mahalanobis unknown score (Ren et al. 2021). Decided on P1-val by the §1.2 rule against
    cos + CL2N of the same encoder.

    python scripts/phase_c_c1c2.py --out runs/phase_c/c1c2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import oneshot_probe as probe  # noqa: E402

from utils.oneshot_metrics import auroc, oscr  # noqa: E402

TAU_R1A = 0.4699  # rule A, notebook 9a (R1a, val_sel)
TAU_DINO = 0.1979  # rule A, notebook 13b (DINOv2 + A4 + CL2N, val_sel)
C2_ENCODERS = ("dinov2/a4", "imagenet/raw")  # the first decides (§2C.1)
B = 1000

# --------------------------------------------------------------------------- #
# C1
# --------------------------------------------------------------------------- #


def open_set(correct, score, known, gate_score, tau) -> dict:
    """Metrics of one system on aligned crops: `correct` (named right, known crops only matter),
    `score` (higher = more known) for AUROC / OSCR, `gate_score` < tau = "Other"."""
    correct, score, known = (np.asarray(x) for x in (correct, score, known))
    gate = np.asarray(gate_score) >= tau
    return {
        "top1_ungated": float(correct[known].mean()),
        "accepted_correct": float((correct & gate)[known].mean()),
        "known_rejected": float((~gate)[known].mean()),
        "unknown_rejected": float((~gate)[~known].mean()),
        "auroc": auroc(score[known], score[~known]),
        "oscr": oscr(correct[known], score[known], score[~known])[2],
    }


def systems(lo: pd.DataFrame, dino: pd.DataFrame, known: np.ndarray) -> dict:
    """R1a alone, DINOv2 alone (gated by its own tau) and C1, on aligned frames."""
    ok_lo = (lo.top1 == lo.expected).to_numpy()
    ok_d = (dino.top1 == dino.expected).to_numpy()
    cos_lo, cos_d = lo.best_cos.to_numpy(), dino.best_cos.to_numpy()
    return {
        "R1a": open_set(ok_lo, cos_lo, known, cos_lo, TAU_R1A),
        "DINOv2": open_set(ok_d, cos_d, known, cos_d, TAU_DINO),
        "C1": open_set(ok_d, cos_lo, known, cos_lo, TAU_R1A),
    }


def p1_frames():
    lo = pd.read_csv(ROOT / "runs/oneshot_p1_crops.csv").set_index("path").sort_index()
    dino = pd.read_csv(ROOT / "runs/oneshot_p1_crops_dinov2.csv").set_index("path").sort_index()
    assert lo.index.equals(dino.index) and (lo["class"] == dino["class"]).all()
    return lo, dino


def c1_p1() -> dict:
    lo, dino = p1_frames()
    known = (lo.group == "H_chon").to_numpy()
    full = systems(lo, dino, known)
    cls = lo["class"].to_numpy()
    img = (lo.split + "/" + lo.image_id.astype(str)).to_numpy()
    rng = np.random.default_rng(0)
    classes = sorted(set(cls))
    by_img = {
        c: {i: np.flatnonzero((cls == c) & (img == i)) for i in sorted(set(img[cls == c]))}
        for c in classes
    }
    rows_of = {c: np.flatnonzero(cls == c) for c in classes}
    kc = sorted(set(cls[known]))
    uc = sorted(set(cls[~known]))

    def at(rows):
        return systems(lo.iloc[rows], dino.iloc[rows], known[rows])

    boot = {"by_image": [], "by_class": []}
    for _ in range(B):
        parts = []
        for ims in by_img.values():
            keys = list(ims)
            parts += [ims[keys[j]] for j in rng.choice(len(keys), len(keys))]
        boot["by_image"].append(at(np.concatenate(parts)))
        rows = [rows_of[kc[j]] for j in rng.choice(len(kc), len(kc))]
        rows += [rows_of[uc[j]] for j in rng.choice(len(uc), len(uc))]
        boot["by_class"].append(at(np.concatenate(rows)))
    return {"full": full, "ci": diffs(full, boot, ("auroc", "oscr", "accepted_correct"))}


def diffs(full, boot, keys) -> dict:
    out = {}
    for a, b in (("C1", "DINOv2"), ("C1", "R1a")):
        for k in keys:
            d = full[a][k] - full[b][k]
            ci = {
                lvl: [
                    float(x) for x in np.nanpercentile([s[a][k] - s[b][k] for s in bs], [2.5, 97.5])
                ]
                for lvl, bs in boot.items()
            }
            out[f"{a}-{b}/{k}"] = {"diff": float(d), **ci}
    return out


def c1_p4() -> dict:
    lo = pd.read_csv(ROOT / "runs/oneshot_p4_crops_lo.csv")
    dino = pd.read_csv(ROOT / "runs/oneshot_p4_crops_dinov2.csv")
    assert lo[["split", "ann_id"]].equals(dino[["split", "ann_id"]])
    out = {}
    for view in ("degraded", "clean"):
        sel = (lo.split == view) & lo.group.ne("other_isa") & lo.group.ne("other_generic")
        lv, dv = lo[sel].reset_index(drop=True), dino[sel].reset_index(drop=True)
        known = (~lv.group.str.startswith("other")).to_numpy()
        full = systems(lv, dv, known)
        # accepted + correct per group (macro over classes), the gate's cost where it lands
        groups = {}
        for g in ("S", "V", "H_chon", "H_tu_nhien"):
            m = (lv.group == g).to_numpy()
            ok_d = (dv.top1 == dv.expected).to_numpy()
            acc = {
                "C1": ok_d & (lv.best_cos.to_numpy() >= TAU_R1A),
                "DINOv2": ok_d & (dv.best_cos.to_numpy() >= TAU_DINO),
                "DINOv2_ungated": ok_d,
                "R1a": (lv.top1 == lv.expected).to_numpy() & (lv.best_cos.to_numpy() >= TAU_R1A),
            }
            cls = lv["class"].to_numpy()
            groups[g] = {
                k: float(np.mean([v[m & (cls == c)].mean() for c in sorted(set(cls[m]))]))
                for k, v in acc.items()
            }
        sheets = sorted(lv.sheet.unique())
        rng = np.random.default_rng(0)
        boot = []
        for _ in range(B):
            pick = [sheets[j] for j in rng.choice(len(sheets), len(sheets))]
            rows = np.concatenate([np.flatnonzero(lv.sheet.to_numpy() == s) for s in pick])
            boot.append(systems(lv.iloc[rows], dv.iloc[rows], known[rows]))
        out[view] = {
            "full": full,
            "groups": groups,
            "ci": diffs(full, {"by_sheet": boot}, ("auroc", "oscr", "accepted_correct")),
        }
    return out


# --------------------------------------------------------------------------- #
# C2
# --------------------------------------------------------------------------- #


def cl2n(views: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """(n, V, D) unit view embeddings -> (n, D): centered views, averaged, renormalized."""
    return F.normalize(F.normalize(views - center, dim=-1).mean(dim=1), dim=-1)


def whitening(samples: np.ndarray):
    """Ledoit-Wolf mean m and W = Sigma^(-1/2) of (n, D) samples."""
    from sklearn.covariance import LedoitWolf

    lw = LedoitWolf().fit(samples)
    vals, vecs = np.linalg.eigh(lw.covariance_)
    return lw.location_, (vecs / np.sqrt(vals)) @ vecs.T


def c2_scores(q: np.ndarray, p: np.ndarray, m: np.ndarray, w: np.ndarray, variant: str):
    """(n, N) scores, higher = nearer. 'white': cosine of whitened vectors; 'maha': Relative
    Mahalanobis, -(d_c - d_0) with d_c the squared Mahalanobis distance to prototype c and d_0 to
    the shared mean (the ranking over c is the plain Mahalanobis nearest prototype)."""
    qw, pw = (q - m) @ w, (p - m) @ w
    if variant == "white":
        qw /= np.linalg.norm(qw, axis=1, keepdims=True)
        pw /= np.linalg.norm(pw, axis=1, keepdims=True)
        return qw @ pw.T
    d = (qw**2).sum(1)[:, None] - 2 * qw @ pw.T + (pw**2).sum(1)[None, :]
    d0 = (qw**2).sum(1)[:, None]
    return -(d - d0)


def c2(world, crops) -> dict:
    groups = world["groups"]
    by = {}
    for c in crops:
        by.setdefault((c["set"], c["role"]), []).append(c)
    h_in = set(world["holdout"].h_in)
    sets = {
        "p1val": by[("p1val", "known")] + by[("p1val", "unknown")],
        "p1": [
            dict(c, role="known" if c["class"] in h_in else "unknown") for c in by[("p1", "test")]
        ],
    }
    out = {}
    for enc in C2_ENCODERS:
        name, kind = enc.split("/")
        data = torch.load(ROOT / f"runs/phase_c/probe/emb_{name}_{kind}.pt", weights_only=False)
        emb, protos, names = data["emb"], data["protos"], data["names"]
        seen = torch.stack([emb[c["ann"]["id"]] for c in by[("p1", "center")]])
        center = seen.reshape(-1, seen.shape[-1]).mean(dim=0)
        s_cl2n = cl2n(seen, center).numpy()
        p_cl2n = F.normalize(protos - center, dim=-1).numpy()
        m, w = whitening(s_cl2n)
        res = {}
        for set_name, cs in sets.items():
            q = cl2n(torch.stack([emb[c["ann"]["id"]] for c in cs]), center).numpy()
            res[set_name] = {
                "cos_cl2n": rows_summary(cs, q @ p_cl2n.T, names, groups),
                "white": rows_summary(cs, c2_scores(q, p_cl2n, m, w, "white"), names, groups),
                "maha": rows_summary(cs, c2_scores(q, p_cl2n, m, w, "maha"), names, groups),
            }
        best = max(("white", "maha"), key=lambda v: res["p1val"][v]["macro_top1"])
        res["chosen"] = best
        res["decision"] = probe.decide(res["p1val"]["cos_cl2n"], res["p1val"][best])
        out[enc] = res
    return out


def rows_summary(cs, scores: np.ndarray, names, groups) -> dict:
    order = np.argsort(-scores, axis=1)[:, :3]
    rows = []
    for c, o, s in zip(cs, order, scores, strict=True):
        rows.append(
            {
                "class": c["class"],
                "known": c["role"] == "known",
                "expected": groups[c["class"]].template or "",
                "top": [names[i] for i in o],
                "best": float(s[o[0]]),
                "second": float(s[o[1]]),
            }
        )
    return probe.summarize(rows)


# --------------------------------------------------------------------------- #


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(ROOT / "configs/config.yaml")
    cfg.data_root = str(ROOT / "data")
    world = probe.load_world(cfg)
    crops = probe.pick_crops(world)
    result = {"c1": {"p1": c1_p1(), "p4": c1_p4()}, "c2": c2(world, crops)}
    (args.out / "results.json").write_text(json.dumps(result, indent=1, default=float))
    report(result)


def report(r) -> None:
    f = lambda v: "   -  " if v is None else f"{v:6.3f}"  # noqa: E731
    keys = (
        "top1_ungated",
        "accepted_correct",
        "known_rejected",
        "unknown_rejected",
        "auroc",
        "oscr",
    )
    print("C1 — P1 (h_in / h_out)        " + "".join(f"{k[:14]:>15s}" for k in keys))
    for s, v in r["c1"]["p1"]["full"].items():
        print(f"  {s:28s}" + "".join(f"{f(v[k]):>15s}" for k in keys))
    for k, v in r["c1"]["p1"]["ci"].items():
        print(
            f"    {k:30s} {v['diff']:+.3f}  ảnh [{v['by_image'][0]:+.3f}, {v['by_image'][1]:+.3f}]"
            f"  lớp [{v['by_class'][0]:+.3f}, {v['by_class'][1]:+.3f}]"
        )
    for view, o in r["c1"]["p4"].items():
        print(f"C1 — P4 {view} (vocab / other_iso)")
        for s, v in o["full"].items():
            print(f"  {s:28s}" + "".join(f"{f(v[k]):>15s}" for k in keys))
        for g, v in o["groups"].items():
            print(f"    nhận+đúng {g:11s} " + "  ".join(f"{k} {x:.3f}" for k, x in v.items()))
        for k, v in o["ci"].items():
            print(
                f"    {k:30s} {v['diff']:+.3f}  tờ "
                f"[{v['by_sheet'][0]:+.3f}, {v['by_sheet'][1]:+.3f}]"
            )
    print("\nC2 — mốc = cos + CL2N cùng encoder; quyết định trên p1val (§2C.1)")
    for enc, res in r["c2"].items():
        for v in ("cos_cl2n", "white", "maha"):
            a, b = res["p1val"][v], res["p1"][v]
            print(
                f"  {enc:12s} {v:9s} p1val top1 {a['macro_top1']:.3f} "
                f"top3 {a['macro_top3']:.3f} AUROC {f(a['auroc_cos'])} lề {f(a['auroc_margin'])}"
                f"  | p1 top1 {b['macro_top1']:.3f} top3 {b['macro_top3']:.3f} "
                f"AUROC {f(b['auroc_cos'])}"
            )
        d = res["decision"]
        print(
            f"    -> chọn {res['chosen']}: lớp tăng {d['n_up']}/10, "
            f"Δtop1 {d['d_macro_top1']:+.3f}, ΔAUROC {d['d_auroc']:+.3f}"
            f" -> {'ĐẠT' if d['pass'] else 'KHÔNG ĐẠT'}"
        )


if __name__ == "__main__":
    main()
