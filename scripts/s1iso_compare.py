"""PLAN_PHASE_C §2N rule 6: does the one-class Stage 1 trained on ISO pages (`new`) replace the
current DigitizePID detector (`old`)? Decided on P4-val v2 only (degraded view, class-agnostic
`det_only`, `agnostic_merge`), from the two `metrics.json['detection']` copies the Colab notebook
(`train_stage1_iso_colab.ipynb`) saves as `det_p4v2_{old,new}.json`.

NHẬN when all hold:
  (a) recall rises, and the lower end of the 95% CI of Δrecall (paired bootstrap over the
      source sheets, B resamples) is > 0;
  (b) precision drops by at most 0.02;
  (c) recall of `other_iso` (symbols outside the library: never drawn for the new detector)
      drops by at most 0.05.

    python scripts/s1iso_compare.py --dir runs/s1iso_csv --out runs/phase_c/s1iso
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

VIEW = "degraded"
P_DROP, UNSEEN_DROP = 0.02, 0.05
UNSEEN = "other_iso"
B, SEED = 10_000, 0


def sheets_of(det: dict) -> dict[str, dict]:
    return det[VIEW]["per_sheet"]


def _sums(per: dict[str, dict], names: list[str], group: str | None = None) -> np.ndarray:
    """(tp, fp, n) per sheet of `names` (or of one GT group: tp, 0, n)."""
    rows = []
    for s in names:
        b = per[s]
        if group is None:
            rows.append((b["tp"], b["fp"], b["n_gt"]))
        else:
            g = b.get("recall_by_group", {}).get(group, {"tp": 0, "n": 0})
            rows.append((g["tp"], 0, g["n"]))
    return np.array(rows, dtype=float)


def _rate(a, b):
    return float(a / b) if b else None


def summary(det: dict) -> dict:
    v = det[VIEW]
    groups = {k: r["recall"] for k, r in v.get("recall_by_group", {}).items()}
    return {
        "precision": v["precision"] or 0.0,  # None: no prediction at all
        "recall": v["recall"],
        "tp": v["tp"],
        "fp": v["fp"],
        "n_gt": v["n_gt"],
        "n_predictions": v["n_predictions"],
        "recall_by_group": groups,
    }


def paired_delta_recall(old: dict, new: dict, b: int = B, seed: int = SEED) -> dict:
    """Δrecall (new - old) and its 95% CI, resampling the sheets with replacement (paired)."""
    per_o, per_n = sheets_of(old), sheets_of(new)
    if set(per_o) != set(per_n):
        raise ValueError(f"not the same sheets: {sorted(set(per_o) ^ set(per_n))}")
    names = sorted(per_o)
    so, sn = _sums(per_o, names), _sums(per_n, names)
    if not np.array_equal(so[:, 2], sn[:, 2]):
        raise ValueError("not the same GT boxes per sheet")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(names), size=(b, len(names)))
    n = so[idx, 2].sum(1)
    d = (sn[idx, 0].sum(1) - so[idx, 0].sum(1)) / np.maximum(n, 1)
    point = (sn[:, 0].sum() - so[:, 0].sum()) / so[:, 2].sum()
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta": float(point), "ci95": [float(lo), float(hi)], "n_sheets": len(names), "B": b}


def decide(old: dict, new: dict, b: int = B) -> dict:
    so, sn = summary(old), summary(new)
    dr = paired_delta_recall(old, new, b)
    dp = sn["precision"] - so["precision"]
    u_o, u_n = so["recall_by_group"].get(UNSEEN), sn["recall_by_group"].get(UNSEEN)
    if u_o is None or u_n is None:
        raise ValueError(f"no `{UNSEEN}` group in the metrics (groups checkpoint missing?)")
    checks = {
        "a_recall_up_ci": dr["delta"] > 0 and dr["ci95"][0] > 0,
        "b_precision": dp >= -P_DROP,
        "c_unseen_recall": u_n - u_o >= -UNSEEN_DROP,
    }
    return {
        "old": so, "new": sn, "delta_recall": dr, "delta_precision": dp,
        "delta_unseen_recall": u_n - u_o, "checks": checks, "adopt": all(checks.values()),
    }  # fmt: skip


def _ok(v: bool) -> str:
    return "✓" if v else "✗"


def _line(tag: str, s: dict) -> str:
    return (
        f"  {tag}: P {s['precision']:.3f}  R {s['recall']:.3f}  ({s['n_predictions']} dự đoán,"
        f" TP {s['tp']}, FP {s['fp']}, {s['n_gt']} GT)"
    )


def report(r: dict) -> str:
    o, n, dr, ck = r["old"], r["new"], r["delta_recall"], r["checks"]
    groups = "; ".join(
        f"{k} {o['recall_by_group'][k]:.3f} -> {n['recall_by_group'].get(k, float('nan')):.3f}"
        for k in o["recall_by_group"]
        if o["recall_by_group"][k] is not None
    )
    lines = [
        "§2N luật 6 — Stage 1 mới (1 lớp, trang ISO) so với cũ (8 lớp DigitizePID),"
        " P4-val v2 suy giảm",
        _line("cũ ", o),
        _line("mới", n),
        f"  (a) Δrecall {dr['delta']:+.4f}, CI95 [{dr['ci95'][0]:+.4f}, {dr['ci95'][1]:+.4f}]"
        f" theo {dr['n_sheets']} tờ -> {_ok(ck['a_recall_up_ci'])}",
        f"  (b) Δprecision {r['delta_precision']:+.4f} (≥ −{P_DROP}) -> {_ok(ck['b_precision'])}",
        f"  (c) Δrecall {UNSEEN} {r['delta_unseen_recall']:+.4f} (≥ −{UNSEEN_DROP})"
        f" -> {_ok(ck['c_unseen_recall'])}",
        f"  recall theo nhóm (cũ -> mới): {groups}",
        "",
        f"Kết luận: {'NHẬN detector mới' if r['adopt'] else 'GIỮ detector cũ'}",
    ]
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dir", type=Path, default=Path("runs/s1iso_csv"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    load = lambda t: json.loads((args.dir / f"det_p4v2_{t}.json").read_text())  # noqa: E731
    r = decide(load("old"), load("new"))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
    text = report(r)
    (args.out / "report.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
