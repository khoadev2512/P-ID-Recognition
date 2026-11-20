"""PLAN_PHASE_C §2E.3: the Colab check of F0 (P1 full + P4), scored by the rule locked there.

Reads per-crop eval CSVs only (no GPU): the 2B baseline `oneshot_{p1,p4}_crops_dinov2.csv` and,
per variant v in VARIANTS, `oneshot_{p1,p4}_crops_dinov2_<v>.csv` (notebook §14, copied from
`eval_{p1,p4}_dinov2_<v>/*/`). A missing variant is skipped.

P1  known = group H_chon (h_in), unknown = the rest (OOV_heldout); macro top-1 / top-3 of the known
    classes; AUROC of best_cos and of the margin. Bootstrap by image within class (CI, reported).
P4  degraded = main, clean = reference; group top-1 = macro over the group's classes; AUROC with
    `other_iso` as the unknowns (other_isa / other_generic out, as notebook 13d / C1).
    Bootstrap by sheet (12 sheets).

    python scripts/f0_colab_compare.py --out runs/phase_c/f0_colab [--dir runs]
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

from utils.oneshot_metrics import auroc  # noqa: E402

VARIANTS = ("f0a", "shift")  # in the §2E.3 order of preference
GROUPS = ("S", "V", "H_chon", "H_tu_nhien")
B = 1000
MIN_AUROC_GAIN, MAX_TOP1_DROP, MIN_CLASSES_UP = 0.05, 0.01, 4


# --------------------------------------------------------------------------- #
# P1
# --------------------------------------------------------------------------- #


def p1_stats(df: pd.DataFrame) -> dict:
    known = (df.group == "H_chon").to_numpy()
    ok1 = (df.top1 == df.expected).to_numpy()
    ok3 = ok1 | (df.top2 == df.expected).to_numpy() | (df.top3 == df.expected).to_numpy()
    cls = df["class"].to_numpy()
    per = {c: float(ok1[known & (cls == c)].mean()) for c in sorted(set(cls[known]))}
    top3 = [float(ok3[known & (cls == c)].mean()) for c in per]
    cos, margin = df.best_cos.to_numpy(), (df.best_cos - df.second_cos).to_numpy()
    return {
        "macro_top1": float(np.mean(list(per.values()))),
        "macro_top3": float(np.mean(top3)),
        "auroc_cos": auroc(cos[known], cos[~known]),
        "auroc_margin": auroc(margin[known], margin[~known]),
        "per_class": per,
        "n_crops": len(df),
    }


def p1_compare(base: pd.DataFrame, cand: pd.DataFrame, seed: int = 0) -> dict:
    b, c = p1_stats(base), p1_stats(cand)
    up = [k for k in b["per_class"] if c["per_class"][k] > b["per_class"][k]]
    d_top1 = c["macro_top1"] - b["macro_top1"]
    d_cos = c["auroc_cos"] - b["auroc_cos"]
    # bootstrap by image within class, ΔAUROC (reported only)
    known = (base.group == "H_chon").to_numpy()
    cls = base["class"].to_numpy()
    img = (base.split.astype(str) + "/" + base.image_id.astype(str)).to_numpy()
    groups = []
    for k in sorted(set(cls)):
        m = cls == k
        groups.append([np.flatnonzero(m & (img == i)) for i in sorted(set(img[m]))])
    rng = np.random.default_rng(seed)
    cos_b, cos_c = base.best_cos.to_numpy(), cand.best_cos.to_numpy()
    mar_b = (base.best_cos - base.second_cos).to_numpy()
    mar_c = (cand.best_cos - cand.second_cos).to_numpy()
    d_boot = {"auroc_cos": [], "auroc_margin": []}
    for _ in range(B):
        rows = np.concatenate([g[j] for g in groups for j in rng.choice(len(g), len(g))])
        k = known[rows]
        for key, sb, sc in (("auroc_cos", cos_b, cos_c), ("auroc_margin", mar_b, mar_c)):
            xb, xc = sb[rows], sc[rows]
            d_boot[key].append(auroc(xc[k], xc[~k]) - auroc(xb[k], xb[~k]))
    ci = {k: [float(x) for x in np.percentile(v, [2.5, 97.5])] for k, v in d_boot.items()}
    return {
        "base": b,
        "cand": c,
        "classes_up": up,
        "n_up": len(up),
        "d_macro_top1": d_top1,
        "d_auroc_cos": d_cos,
        "d_auroc_margin": c["auroc_margin"] - b["auroc_margin"],
        "ci_by_image": ci,
        "separation_improved": d_cos >= MIN_AUROC_GAIN and d_top1 >= -MAX_TOP1_DROP,
        "naming_improved": len(up) >= MIN_CLASSES_UP and d_cos >= 0,
    }


# --------------------------------------------------------------------------- #
# P4
# --------------------------------------------------------------------------- #


def _p4_view(df: pd.DataFrame, view: str) -> pd.DataFrame:
    sel = (df.split == view) & ~df.group.isin(["other_isa", "other_generic"])
    return df[sel].reset_index(drop=True)


class _P4:
    """Per-sheet sufficient statistics for fast sheet bootstraps of one system."""

    def __init__(self, df: pd.DataFrame):
        self.sheet = df.sheet.to_numpy()
        self.sheets = sorted(set(self.sheet))
        self.known = (~df.group.str.startswith("other")).to_numpy()
        self.ok = (df.top1 == df.expected).to_numpy()
        self.cos = df.best_cos.to_numpy()
        self.margin = (df.best_cos - df.second_cos).to_numpy()
        self.groups = {}
        cls = df["class"].to_numpy()
        for g in GROUPS:
            m = (df.group == g).to_numpy()
            classes = sorted(set(cls[m]))
            # (sheet, class) -> (correct, total)
            cnt = np.zeros((len(self.sheets), len(classes), 2))
            for i, s in enumerate(self.sheets):
                ms = m & (self.sheet == s)
                for j, c in enumerate(classes):
                    mc = ms & (cls == c)
                    cnt[i, j] = (self.ok[mc].sum(), mc.sum())
            self.groups[g] = cnt
        self.rows = {s: np.flatnonzero(self.sheet == s) for s in self.sheets}

    def stats(self, pick: list[int] | None = None) -> dict:
        idx = list(range(len(self.sheets))) if pick is None else pick
        out = {}
        for g, cnt in self.groups.items():
            tot = cnt[idx].sum(axis=0)
            have = tot[:, 1] > 0
            out[g] = float((tot[have, 0] / tot[have, 1]).mean())
        rows = np.concatenate([self.rows[self.sheets[i]] for i in idx])
        k = self.known[rows]
        out["auroc_cos"] = auroc(self.cos[rows][k], self.cos[rows][~k])
        out["auroc_margin"] = auroc(self.margin[rows][k], self.margin[rows][~k])
        return out


def p4_compare(base: pd.DataFrame, cand: pd.DataFrame, seed: int = 0) -> dict:
    out = {}
    for view in ("degraded", "clean"):
        b, c = _P4(_p4_view(base, view)), _P4(_p4_view(cand, view))
        assert b.sheets == c.sheets
        full_b, full_c = b.stats(), c.stats()
        rng = np.random.default_rng(seed)
        boot = []
        for _ in range(B):
            pick = list(rng.choice(len(b.sheets), len(b.sheets)))
            sb, sc = b.stats(pick), c.stats(pick)
            boot.append({k: sc[k] - sb[k] for k in sb})
        res = {}
        for k in full_b:
            lo, hi = np.percentile([x[k] for x in boot], [2.5, 97.5])
            verdict = "tốt hơn" if lo > 0 else "kém hơn" if hi < 0 else "không khác"
            res[k] = {
                "base": full_b[k],
                "cand": full_c[k],
                "diff": full_c[k] - full_b[k],
                "ci": [float(lo), float(hi)],
                "verdict": verdict,
            }
        out[view] = res
    return out


# --------------------------------------------------------------------------- #


def load(dirpath: Path, protocol: str, tag: str | None) -> pd.DataFrame | None:
    name = f"oneshot_{protocol}_crops_dinov2{'' if tag is None else '_' + tag}.csv"
    path = dirpath / name
    if not path.exists():
        return None
    df = pd.read_csv(path)
    keys = ["path"] if protocol == "p1" else ["split", "ann_id"]
    return df.sort_values(keys).reset_index(drop=True)


def decide(p1: dict, p4: dict) -> dict:
    deg = p4["degraded"]
    worse_groups = [g for g in GROUPS if deg[g]["verdict"] == "kém hơn"]
    adopt = (
        p1["separation_improved"] and deg["auroc_cos"]["verdict"] != "kém hơn" and not worse_groups
    )
    return {"p4_worse_groups": worse_groups, "adopt": adopt}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dir", type=Path, default=ROOT / "runs")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    base1, base4 = load(args.dir, "p1", None), load(args.dir, "p4", None)
    assert base1 is not None and base4 is not None, "2B baseline CSVs missing"
    results = {}
    for v in VARIANTS:
        c1, c4 = load(args.dir, "p1", v), load(args.dir, "p4", v)
        if c1 is None:
            print(f"[{v}] thiếu CSV P1 -> bỏ qua")
            continue
        assert c1.path.equals(base1.path) and (c1["class"] == base1["class"]).all(), v
        p1 = p1_compare(base1, c1)
        if c4 is None:  # P1 only: reported, no decision (§2E.3 needs P4)
            print(f"[{v}] chưa có CSV P4 -> chỉ báo P1, CHƯA quyết")
            results[v] = {"p1": p1, "p4": None, "adopt": None, "p4_worse_groups": None}
            continue
        assert c4[["split", "ann_id"]].equals(base4[["split", "ann_id"]]), v
        p4 = p4_compare(base4, c4)
        results[v] = {"p1": p1, "p4": p4, **decide(p1, p4)}
    adopted = [v for v in VARIANTS if results.get(v, {}).get("adopt")]
    pending = any(results.get(v, {}).get("p4", 0) is None for v in VARIANTS if v in results)
    results["chosen"] = "pending" if pending else (adopted[0] if adopted else None)
    (args.out / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    report(results, args.out)


def report(r: dict, out: Path) -> None:
    lines = ["F0 trên Colab (PLAN_PHASE_C §2E.3), so với 2B DINOv2 + A4 + CL2N", ""]
    for v in VARIANTS:
        if v not in r:
            continue
        p1, p4 = r[v]["p1"], r[v]["p4"]
        b, c = p1["base"], p1["cand"]
        lines += [
            f"===== {v}",
            f"P1 đủ ({c['n_crops']} crop)   top-1 {b['macro_top1']:.3f} -> {c['macro_top1']:.3f}"
            f"  top-3 {b['macro_top3']:.3f} -> {c['macro_top3']:.3f}",
            f"  AUROC cos {b['auroc_cos']:.3f} -> {c['auroc_cos']:.3f}"
            f" (Δ {p1['d_auroc_cos']:+.3f}, CI theo ảnh {p1['ci_by_image']['auroc_cos'][0]:+.3f}"
            f" .. {p1['ci_by_image']['auroc_cos'][1]:+.3f})",
            f"  AUROC lề {b['auroc_margin']:.3f} -> {c['auroc_margin']:.3f}",
            "  từng lớp h_in: "
            + ", ".join(
                f"{k} {b['per_class'][k]:.3f}->{c['per_class'][k]:.3f}" for k in b["per_class"]
            ),
            f"  tách lạ cải thiện: {'CÓ' if p1['separation_improved'] else 'KHÔNG'}"
            f"   gọi tên cải thiện ({p1['n_up']}/6 lớp tăng): "
            f"{'CÓ' if p1['naming_improved'] else 'KHÔNG'}",
        ]
        if p4 is None:
            lines += ["P4: chưa có -> CHƯA quyết (luật §2E.3 cần cả P4)", ""]
            continue
        for view in ("degraded", "clean"):
            lines.append(f"P4 {view}{' (SỐ CHÍNH)' if view == 'degraded' else ' (tham chiếu)'}:")
            for k, x in p4[view].items():
                lines.append(
                    f"  {k:13s} {x['base']:.3f} -> {x['cand']:.3f}  Δ {x['diff']:+.3f}"
                    f"  CI [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]  {x['verdict']}"
                )
        lines.append(
            f"-> thay 2B: {'ĐẠT' if r[v]['adopt'] else 'KHÔNG ĐẠT'}"
            f"  (nhóm P4 kém hơn: {r[v]['p4_worse_groups'] or 'không'})"
        )
        lines.append("")
    chosen = r["chosen"]
    if chosen == "pending":
        chosen = "chưa quyết — thiếu P4"
    lines.append(f"Chọn: {chosen or 'không dòng nào — giữ 2B'}")
    text = "\n".join(lines)
    (out / "report.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
