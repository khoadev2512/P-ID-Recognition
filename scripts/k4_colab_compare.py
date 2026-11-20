"""PLAN_PHASE_C §2K.6: the Colab check of (i) form B of valve_butterfly and (ii) aggregate=max,
scored by the rule locked there. Per-crop CSVs only (no GPU).

Baseline `base41` = 2B with 41 templates, mean (notebook §15:
`oneshot_{p}_crops_dinov2_nopipe.csv`, in --base-dir). Candidates (notebook §16, in --dir):
`oneshot_{p}_crops_dinov2_{formB,max,formB_max}.csv` for p in p1, p2, p4. Bootstraps reuse
f0_colab_compare (P1 by image, P4 by 12 sheets) and nopipe_compare (P2 by image in class).

Consistency (blocking): every candidate scores the baseline's crops; a form-B line may change a
crop's top-1 only where valve_butterfly is in its own or its reference's top-3 (formB vs base41,
formB_max vs max).

    python scripts/k4_colab_compare.py --out runs/phase_c/k4_colab [--dir runs] [--base-dir runs]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import f0_colab_compare as f0  # noqa: E402
import nopipe_compare as npc  # noqa: E402

KEYS = {"p1": ["path"], "p2": ["path"], "p4": ["split", "ann_id"]}
LINES = ("formB", "max", "formB_max")
BASE_TAG = "nopipe"
FORM_CLASS = "valve_butterfly"
P1_DROP = 0.01
P4_GROUPS = f0.GROUPS
P2_KEYS = ("S_top1", "V_top1")


# --------------------------------------------------------------------------- #
# rule (pure; tests/test_k4_colab_compare.py)
# --------------------------------------------------------------------------- #


def form_consistency(ref: pd.DataFrame, cand: pd.DataFrame, cls: str = FORM_CLASS) -> dict:
    """Rows whose top-1 differs must have `cls` in the top-3 of ref or cand."""
    top = ["top1", "top2", "top3"]
    changed = (ref.top1 != cand.top1).to_numpy()
    near = (ref[top] == cls).any(axis=1).to_numpy() | (cand[top] == cls).any(axis=1).to_numpy()
    bad = int((changed & ~near).sum())
    return {"n_changed": int(changed.sum()), "n_changed_without_form": bad, "ok": bad == 0}


def _worse(x: dict) -> bool:
    return x["verdict"] == "kém hơn"


def decide(p1: dict, p2: dict, p4: dict, need_better: bool) -> dict:
    """§2K.6. need_better=True is rule (ii) (`max`, and the combination); False is rule (i)."""
    deg = p4["degraded"]
    worse_p4 = [g for g in P4_GROUPS if _worse(deg[g])]
    better_p4 = [g for g in P4_GROUPS if deg[g]["verdict"] == "tốt hơn"]
    worse_p2 = [k for k in P2_KEYS if _worse(p2[k])]
    checks = {
        "p4_no_group_worse": not worse_p4,
        "p4_auroc_cos_not_worse": not _worse(deg["auroc_cos"]),
        "p2_no_group_worse": not worse_p2,
        "p2_auroc_cos_not_worse": not _worse(p2["auroc_cos"]),
        "p1_top1": p1["d_macro_top1"] >= -P1_DROP,
    }
    if need_better:
        checks["p4_some_group_better"] = bool(better_p4)
        checks["p1_auroc_cos"] = p1["d_auroc_cos"] >= -P1_DROP
    return {
        "checks": checks,
        "p4_worse": worse_p4,
        "p4_better": better_p4,
        "p2_worse": worse_p2,
        "adopt": all(checks.values()),
    }


def final_config(adopt: dict) -> str:
    if adopt["formB"] and adopt["max"] and adopt["formB_max"]:
        return "formB_max"
    if adopt["max"]:
        return "max"
    if adopt["formB"]:
        return "formB"
    return "base41"


# --------------------------------------------------------------------------- #


def load(dirpath: Path, protocol: str, tag: str) -> pd.DataFrame:
    path = dirpath / f"oneshot_{protocol}_crops_dinov2_{tag}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} (notebook §15 / §16 -> runs/nopipe_csv, runs/k4_csv)")
    return pd.read_csv(path).sort_values(KEYS[protocol]).reset_index(drop=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dir", type=Path, default=ROOT / "runs")
    ap.add_argument("--base-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    base_dir = args.base_dir or args.dir
    base = {p: load(base_dir, p, BASE_TAG) for p in KEYS}
    cand = {t: {p: load(args.dir, p, t) for p in KEYS} for t in LINES}

    r: dict = {"consistency": {}, "lines": {}}
    for t in LINES:
        for p in KEYS:
            if not base[p][KEYS[p]].equals(cand[t][p][KEYS[p]]):
                raise SystemExit(f"{t}/{p}: not the baseline's crops")
    for t, ref in (("formB", base), ("formB_max", cand["max"])):
        for p in KEYS:
            c = form_consistency(ref[p], cand[t][p])
            r["consistency"][f"{t}/{p}"] = c
            if not c["ok"]:
                raise SystemExit(
                    f"{t}/{p}: {c['n_changed_without_form']} top-1 changes without {FORM_CLASS}"
                )

    for t in LINES:
        p1 = f0.p1_compare(base["p1"], cand[t]["p1"])
        p2 = npc.p2_compare(base["p2"], cand[t]["p2"])
        p4 = f0.p4_compare(base["p4"], cand[t]["p4"])
        r["lines"][t] = {
            "p1": p1,
            "p2": p2,
            "p4": p4,
            "decision": decide(p1, p2, p4, need_better=(t != "formB")),
        }
    adopt = {t: r["lines"][t]["decision"]["adopt"] for t in LINES}
    r["adopt"] = adopt
    r["final"] = final_config(adopt)
    r["p4_per_class_degraded"] = {t: _per_class(cand[t]["p4"]) for t in LINES} | {
        "base41": _per_class(base["p4"])
    }
    (args.out / "results.json").write_text(
        json.dumps(r, indent=1, ensure_ascii=False, default=float)
    )
    report(r, args.out)


def _per_class(df: pd.DataFrame) -> dict:
    d = df[(df.split == "degraded") & ~df.group.astype(str).str.startswith("other")]
    return (d.top1 == d.expected).groupby(d.expected).mean().round(4).to_dict()


def report(r: dict, out: Path) -> None:
    lines = [
        "§2K.6: (i) form B butterfly, (ii) aggregate=max — so với 2B 41 ảnh mẫu (mean)",
        "",
        "Kiểm nhất quán (dạng B chỉ đổi crop có butterfly trong top-3): ĐẠT",
    ]
    lines += [f"  {k}: {c['n_changed']} crop đổi top-1" for k, c in r["consistency"].items()]
    for t, x in r["lines"].items():
        d = x["decision"]
        lines += ["", f"===== {t}  -> {'NHẬN' if d['adopt'] else 'KHÔNG NHẬN'}"]
        lines.append("P4 suy giảm (CI theo 12 tờ):")
        lines += [
            npc._line(k, x["p4"]["degraded"][k]) for k in (*P4_GROUPS, "auroc_cos", "auroc_margin")
        ]
        lines.append("P2 (val, test; CI theo ảnh):")
        lines += [npc._line(k, x["p2"][k]) for k in (*P2_KEYS, "auroc_cos", "auroc_margin")]
        p1 = x["p1"]
        lines.append(
            f"P1 đủ: top-1 {p1['base']['macro_top1']:.3f} -> {p1['cand']['macro_top1']:.3f}"
            f" (Δ {p1['d_macro_top1']:+.3f}), AUROC cos {p1['base']['auroc_cos']:.3f} ->"
            f" {p1['cand']['auroc_cos']:.3f} (Δ {p1['d_auroc_cos']:+.3f})"
        )
        lines.append(
            "Điều kiện: " + ", ".join(f"{k} {'✓' if v else '✗'}" for k, v in d["checks"].items())
        )
    lines += [
        "",
        f"Cấu hình đề xuất theo luật: {r['final']}",
        "",
        "P4 suy giảm từng lớp (base41 | formB | max | formB_max):",
    ]
    pc = r["p4_per_class_degraded"]
    for c in sorted(pc["base41"]):
        lines.append(
            f"  {c:28s} "
            + "  ".join(f"{pc[t].get(c, float('nan')):.2f}" for t in ("base41", *LINES))
        )
    text = "\n".join(lines)
    (out / "report.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
