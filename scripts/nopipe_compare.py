"""PLAN_PHASE_C §2G (Điểm 4): 2B with 41 templates (`pipeline` excluded) against 2B with 42.

Reads per-crop eval CSVs only (CPU). From notebook §15 (`runs/nopipe_csv/` on Drive, copied to
--dir): `oneshot_p2_crops_dinov2.csv` (42), `oneshot_{p1,p2,p4}_crops_dinov2_nopipe.csv` (41);
the 42-template P1 / P4 CSVs of §13 are the ones already in --dir (as for F0).

1. Consistency gate: removing one template leaves the other 41 cosines untouched (CL2N center from
   crops, each template normalized alone), so each 41 row must equal its 42 row with `pipeline`
   dropped from the top-3 (top-1 identical, best / second cos within 1e-4). No crop expects
   `pipeline`, so no crop may turn from right to wrong. A failure stops before any number.
2. P2 (clean evidence; 2B never ran P2): groups S / V macro top-1 / top-3, AUROC cos / margin with
   S + V known and OOV_seen unknown; 95 % CI bootstrap by image within class.
3. P1 / P4 (POST-HOC, §2G): the F0 comparisons (scripts/f0_colab_compare.py).

    python scripts/nopipe_compare.py --out runs/phase_c/nopipe [--dir runs]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import f0_colab_compare as f0  # noqa: E402

from utils.oneshot_metrics import auroc  # noqa: E402

DROPPED = "pipeline"
TOL = 1e-4
B = 1000
P2_GROUPS = ("S", "V")


def drop_template(df: pd.DataFrame, name: str = DROPPED) -> pd.DataFrame:
    """What the 41-template run must give, from the 42-template top-3: `name` removed and the
    rest moved up. Columns top1 / best_cos / second_cos are exact; top2 / top3 are not
    (the 4th guess is unknown) and are left as they come (top3 set to None when shifted)."""
    out = df.copy()
    names = df[["top1", "top2", "top3"]].to_numpy(dtype=object)
    cos = df[["cos1", "cos2", "cos3"]].to_numpy(dtype=float)
    keep = names != name
    hit = ~keep.all(axis=1)
    new_names = np.empty_like(names)
    new_cos = np.full_like(cos, np.nan)
    for i in np.flatnonzero(hit):
        k = np.flatnonzero(keep[i])
        new_names[i, : len(k)], new_cos[i, : len(k)] = names[i, k], cos[i, k]
        new_names[i, len(k) :] = None
    for j, (n, c) in enumerate((("top1", "cos1"), ("top2", "cos2"), ("top3", "cos3"))):
        out.loc[hit, n] = new_names[hit, j]
        out.loc[hit, c] = new_cos[hit, j]
    out.loc[hit, "best_cos"] = new_cos[hit, 0]
    out.loc[hit, "second_cos"] = new_cos[hit, 1]
    return out


def consistency(base: pd.DataFrame, cand: pd.DataFrame, protocol: str) -> dict:
    """Gate 1 of the docstring; raises on any violation."""
    if (base.expected == DROPPED).any():
        raise AssertionError(f"{protocol}: a crop expects {DROPPED!r}")
    if (cand[["top1", "top2", "top3"]] == DROPPED).any().any():
        raise AssertionError(f"{protocol}: the 41-template run still names {DROPPED!r}")
    want = drop_template(base)
    # near-ties (two templates within TOL) may flip between Colab sessions: not a violation
    tie = (np.abs(want.cos1.to_numpy() - want.cos2.to_numpy()) <= TOL) | (
        np.abs(base.cos1.to_numpy() - base.cos2.to_numpy()) <= TOL
    )
    top1 = (want.top1 != cand.top1).to_numpy() & ~tie
    d_best = np.abs(want.best_cos.to_numpy() - cand.best_cos.to_numpy())
    d_second = np.abs(want.second_cos.to_numpy() - cand.second_cos.to_numpy())
    worse = ((base.top1 == base.expected) & (cand.top1 != cand.expected)).to_numpy() & ~tie
    out = {
        "n_crops": len(base),
        "n_changed": int((base.top1 != cand.top1).sum()),
        "n_had_pipeline_top2": int(((base.top1 == DROPPED) | (base.top2 == DROPPED)).sum()),
        "n_ties": int(tie.sum()),
        "top1_mismatch": int(top1.sum()),
        "max_best_cos_diff": float(d_best.max()),
        "max_second_cos_diff": float(d_second.max()),
        "right_to_wrong": int(worse.sum()),
    }
    if out["top1_mismatch"] or out["right_to_wrong"] or max(d_best.max(), d_second.max()) > TOL:
        raise AssertionError(f"{protocol}: 41 != 42 minus {DROPPED!r}: {out}")
    return out


# --------------------------------------------------------------------------- #
# P2
# --------------------------------------------------------------------------- #


def p2_stats(df: pd.DataFrame, rows: np.ndarray | None = None) -> dict:
    d = df if rows is None else df.iloc[rows]
    ok1 = (d.top1 == d.expected).to_numpy()
    ok3 = ok1 | (d.top2 == d.expected).to_numpy() | (d.top3 == d.expected).to_numpy()
    grp, cls = d.group.to_numpy(), d["class"].to_numpy()
    out = {}
    for g in P2_GROUPS:
        classes = sorted(set(cls[grp == g]))
        out[f"{g}_top1"] = float(np.mean([ok1[cls == c].mean() for c in classes]))
        out[f"{g}_top3"] = float(np.mean([ok3[cls == c].mean() for c in classes]))
    known, unknown = np.isin(grp, P2_GROUPS), grp == "OOV_seen"
    cos, margin = d.best_cos.to_numpy(), (d.best_cos - d.second_cos).to_numpy()
    out["auroc_cos"] = auroc(cos[known], cos[unknown])
    out["auroc_margin"] = auroc(margin[known], margin[unknown])
    return out


def p2_per_class(df: pd.DataFrame) -> dict:
    d = df[df.group.isin(P2_GROUPS)]
    return {c: float((x.top1 == x.expected).mean()) for c, x in d.groupby("class", sort=True)}


def p2_compare(base: pd.DataFrame, cand: pd.DataFrame, seed: int = 0) -> dict:
    full_b, full_c = p2_stats(base), p2_stats(cand)
    cls = base["class"].to_numpy()
    img = base.image_id.astype(str).to_numpy()
    groups = []
    for k in sorted(set(cls)):
        m = cls == k
        groups.append([np.flatnonzero(m & (img == i)) for i in sorted(set(img[m]))])
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(B):
        rows = np.concatenate([g[j] for g in groups for j in rng.choice(len(g), len(g))])
        sb, sc = p2_stats(base, rows), p2_stats(cand, rows)
        boot.append({k: sc[k] - sb[k] for k in sb})
    res = {}
    for k in full_b:
        lo, hi = np.percentile([x[k] for x in boot], [2.5, 97.5])
        res[k] = {
            "base": full_b[k],
            "cand": full_c[k],
            "diff": full_c[k] - full_b[k],
            "ci": [float(lo), float(hi)],
            "verdict": "tốt hơn" if lo > 0 else "kém hơn" if hi < 0 else "không khác",
        }
    pb, pc = p2_per_class(base), p2_per_class(cand)
    res["changed_classes"] = {k: [pb[k], pc[k]] for k in pb if pc[k] != pb[k]}
    return res


# --------------------------------------------------------------------------- #


def load(path: Path, keys: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} (notebook §15 -> runs/nopipe_csv/ on Drive)")
    return pd.read_csv(path).sort_values(keys).reset_index(drop=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dir", type=Path, default=ROOT / "runs")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    keys = {"p1": ["path"], "p2": ["path"], "p4": ["split", "ann_id"]}
    base = {p: load(args.dir / f"oneshot_{p}_crops_dinov2.csv", keys[p]) for p in keys}
    cand = {p: load(args.dir / f"oneshot_{p}_crops_dinov2_nopipe.csv", keys[p]) for p in keys}
    r: dict = {"consistency": {}}
    for p in keys:
        assert base[p][keys[p]].equals(cand[p][keys[p]]), f"{p}: different crops"
        r["consistency"][p] = consistency(base[p], cand[p], p)
    r["p2"] = p2_compare(base["p2"], cand["p2"])
    r["p1_posthoc"] = f0.p1_compare(base["p1"], cand["p1"])
    r["p4_posthoc"] = f0.p4_compare(base["p4"], cand["p4"])
    (args.out / "results.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
    report(r, args.out)


def _line(k: str, x: dict) -> str:
    return (
        f"  {k:13s} {x['base']:.3f} -> {x['cand']:.3f}  Δ {x['diff']:+.3f}"
        f"  CI [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]  {x['verdict']}"
    )


def report(r: dict, out: Path) -> None:
    lines = ["Điểm 4 (PLAN_PHASE_C §2G): 2B 41 ảnh mẫu (bỏ pipeline) so với 2B 42", ""]
    lines.append("Kiểm nhất quán (41 = 42 bỏ pipeline khỏi top-3): ĐẠT")
    for p, c in r["consistency"].items():
        lines.append(
            f"  {p}: {c['n_crops']} crop, {c['n_had_pipeline_top2']} có pipeline trong top-2,"
            f" {c['n_changed']} đổi top-1, đúng->sai {c['right_to_wrong']},"
            f" lệch cos tối đa {max(c['max_best_cos_diff'], c['max_second_cos_diff']):.1e}"
        )
    lines += ["", "P2 (val, test; bằng chứng SẠCH — 2B chưa từng chạy P2), CI bootstrap theo ảnh:"]
    lines += [_line(k, x) for k, x in r["p2"].items() if k != "changed_classes"]
    lines.append(
        "  lớp đổi top-1: "
        + (
            ", ".join(f"{k} {a:.3f}->{b:.3f}" for k, (a, b) in r["p2"]["changed_classes"].items())
            or "không"
        )
    )
    p1 = r["p1_posthoc"]
    b, c = p1["base"], p1["cand"]
    lines += [
        "",
        "P1 đủ — HẬU KIỂM:",
        f"  h_in top-1 {b['macro_top1']:.3f} -> {c['macro_top1']:.3f}"
        f"  AUROC cos {b['auroc_cos']:.3f} -> {c['auroc_cos']:.3f}"
        f"  AUROC lề {b['auroc_margin']:.3f} -> {c['auroc_margin']:.3f}",
    ]
    for view in ("degraded", "clean"):
        lines.append(f"P4 {view} — HẬU KIỂM{' (số chính)' if view == 'degraded' else ''}:")
        lines += [_line(k, x) for k, x in r["p4_posthoc"][view].items()]
    worse = [k for k in ("auroc_cos", "auroc_margin") if r["p2"][k]["verdict"] == "kém hơn"]
    lines += [
        "",
        "Luật §2G: P2 AUROC kém hơn -> "
        + (f"CÓ ({', '.join(worse)}): báo user, ghi là đánh đổi" if worse else "không"),
    ]
    text = "\n".join(lines)
    (out / "report.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
