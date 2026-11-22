"""PLAN_PHASE_C §2L: the Colab confirmation of FT-triển khai (`ft` = DINOv2 fine-tuned on all 42
template images, WiSE alpha = 0.5) against the adopted `formB`, by rule (ii) of §2K.6 plus guard
(b) of §2L. Per-crop CSVs only (no GPU).

Baseline `formB` (notebook §16: `oneshot_{p}_crops_dinov2_formB.csv`, in --base-dir); candidate
(notebook §17: `oneshot_{p}_crops_dinov2_ft.csv`, in --dir), p in p1, p2, p4. Bootstraps reuse
f0_colab_compare (P1 by image, P4 by 12 sheets) and nopipe_compare (P2 by image in class).

Rule (ii) §2K.6 (k4_colab_compare.decide, need_better=True) and (b): on degraded P4, top-1 of
valve_three_way and breather_valve may not drop by more than 0.10 (blocks only, never chooses).
Consistency (blocking): the candidate scores the baseline's crops.
End-to-end (notebook 17d, `e2e_dinov2_{formB,ft}.json`, optional): reported, decides nothing.

    python scripts/ft_colab_compare.py --out runs/phase_c/ft_colab [--dir runs] [--base-dir runs]
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
import k4_colab_compare as k4  # noqa: E402
import nopipe_compare as npc  # noqa: E402

BASE_TAG, CAND_TAG = "formB", "ft"
GUARD_B = ("valve_three_way", "breather_valve")  # §2L (b): impossible in P4-val
GUARD_DROP = 0.10


# --------------------------------------------------------------------------- #
# rule (pure; tests/test_ft_colab_compare.py)
# --------------------------------------------------------------------------- #


def per_class_degraded(df: pd.DataFrame) -> dict[str, float]:
    return k4._per_class(df)


def guard_b(base: dict[str, float], cand: dict[str, float]) -> dict:
    """§2L (b): each GUARD_B class keeps top-1 within GUARD_DROP of the baseline."""
    missing = [c for c in GUARD_B if c not in base or c not in cand]
    if missing:
        raise ValueError(f"guard (b) classes missing from the P4 CSV: {missing}")
    d = {c: cand[c] - base[c] for c in GUARD_B}
    return {"delta": d, "ok": all(v >= -GUARD_DROP for v in d.values())}


def decide(p1: dict, p2: dict, p4: dict, guard: dict) -> dict:
    d = k4.decide(p1, p2, p4, need_better=True)
    d["checks"]["guard_b"] = guard["ok"]
    d["adopt"] = all(d["checks"].values())
    return d


# --------------------------------------------------------------------------- #


def e2e_summary(path: Path) -> dict | None:
    """Degraded end-to-end numbers of one metrics.json['oneshot_e2e'] copy (notebook 17d)."""
    if not path.exists():
        return None
    o = json.loads(path.read_text())
    b = o["degraded"]
    return {
        "tau": o["tau"],
        "precision": b["precision"],
        "recall": b["recall"],
        "box_recall": b["box_recall"],
        "by_group": {k: r["recall"] for k, r in b["by_group"].items()},
    }


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
    base = {p: k4.load(base_dir, p, BASE_TAG) for p in k4.KEYS}
    cand = {p: k4.load(args.dir, p, CAND_TAG) for p in k4.KEYS}
    for p, keys in k4.KEYS.items():
        if not base[p][keys].equals(cand[p][keys]):
            raise SystemExit(f"{CAND_TAG}/{p}: not the baseline's crops")

    p1 = f0.p1_compare(base["p1"], cand["p1"])
    p2 = npc.p2_compare(base["p2"], cand["p2"])
    p4 = f0.p4_compare(base["p4"], cand["p4"])
    pc = {BASE_TAG: per_class_degraded(base["p4"]), CAND_TAG: per_class_degraded(cand["p4"])}
    guard = guard_b(pc[BASE_TAG], pc[CAND_TAG])
    r = {
        "p1": p1,
        "p2": p2,
        "p4": p4,
        "guard_b": guard,
        "decision": decide(p1, p2, p4, guard),
        "p4_per_class_degraded": pc,
        "e2e": {t: e2e_summary(args.dir / f"e2e_dinov2_{t}.json") for t in (BASE_TAG, CAND_TAG)},
    }
    r["final"] = CAND_TAG if r["decision"]["adopt"] else BASE_TAG
    (args.out / "results.json").write_text(
        json.dumps(r, indent=1, ensure_ascii=False, default=float)
    )
    report(r, args.out)


def report(r: dict, out: Path) -> None:
    d = r["decision"]
    lines = [
        "§2L xác nhận Colab: FT-triển khai (`ft`) so với `formB` — luật (ii) §2K.6 + chặn (b)",
        "",
        f"===== ft  -> {'NHẬN' if d['adopt'] else 'KHÔNG NHẬN'}",
        "P4 suy giảm (CI theo 12 tờ):",
    ]
    lines += [
        npc._line(k, r["p4"]["degraded"][k]) for k in (*k4.P4_GROUPS, "auroc_cos", "auroc_margin")
    ]
    lines.append("P2 (val, test; CI theo ảnh):")
    lines += [npc._line(k, r["p2"][k]) for k in (*k4.P2_KEYS, "auroc_cos", "auroc_margin")]
    p1 = r["p1"]
    lines.append(
        f"P1 đủ: top-1 {p1['base']['macro_top1']:.3f} -> {p1['cand']['macro_top1']:.3f}"
        f" (Δ {p1['d_macro_top1']:+.3f}), AUROC cos {p1['base']['auroc_cos']:.3f} ->"
        f" {p1['cand']['auroc_cos']:.3f} (Δ {p1['d_auroc_cos']:+.3f})"
    )
    pc = r["p4_per_class_degraded"]
    lines.append(
        "(b) P4 suy giảm, từng lớp: "
        + "; ".join(
            f"{c} {pc[BASE_TAG][c]:.2f} -> {pc[CAND_TAG][c]:.2f}"
            f" (Δ {r['guard_b']['delta'][c]:+.2f})"
            for c in GUARD_B
        )
        + f"  (mỗi lớp ≥ −{GUARD_DROP})"
    )
    lines.append(
        "Điều kiện: " + ", ".join(f"{k} {'✓' if v else '✗'}" for k, v in d["checks"].items())
    )
    lines += ["", f"Cấu hình đề xuất theo luật: {r['final']}", ""]
    if any(r["e2e"].values()):
        lines.append("End-to-end (agnostic_merge, suy giảm; MÔ TẢ, không quyết định):")
        for t, e in r["e2e"].items():
            if e is None:
                lines.append(f"  {t:6s} (chưa có)")
                continue
            lines.append(
                f"  {t:6s} τ={e['tau']:.4f}  P {e['precision']:.3f}  R {e['recall']:.3f}"
                f"  R box {e['box_recall']:.3f}  |  "
                + "  ".join(f"{k} {v:.3f}" for k, v in e["by_group"].items() if v is not None)
            )
        lines.append("")
    lines.append(f"P4 suy giảm từng lớp ({BASE_TAG} | {CAND_TAG}):")
    for c in sorted(pc[BASE_TAG]):
        lines.append(f"  {c:28s} {pc[BASE_TAG][c]:.2f}  {pc[CAND_TAG].get(c, float('nan')):.2f}")
    text = "\n".join(lines)
    (out / "report.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
