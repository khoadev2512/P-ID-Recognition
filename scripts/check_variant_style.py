"""V6 check: do the G1 criteria survive other stroke widths and symbol sizes?

check_g1.py's thresholds were tuned at w = 3 px. For every sheet and variant index this keeps the
V3 assignment and the V4 layout (their seeds) and redraws the page over a grid of stroke widths
and size factors (V5 only), then runs the automatic G1 criteria on each. Writes one CSV row per
(sheet, index, w, size) with pass / fail per criterion and the page log, and prints a summary.
Read-only on data/raw/.
"""

from __future__ import annotations

import argparse
import csv
import sys
from itertools import product
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_g1  # noqa: E402

from utils.iso_mapping import load_config  # noqa: E402
from utils.iso_templates import load_templates  # noqa: E402
from utils.iso_variants import plan_variant, render_variant  # noqa: E402
from utils.pid2graph import load_graphml, pipe_runs  # noqa: E402

CRITERIA = (1, 2, 3, 5, 6, 7)
LOGGED = ("shrunk", "unmatched_ports", "loose_ends", "bubbles_unsettled", "agitators_skipped")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=Path("data/raw/pid2graph/open100"))
    ap.add_argument("--out", type=Path, default=Path("docs/experiments/iso_render/v6/g1_style.csv"))
    ap.add_argument("--sheets", type=int, nargs="*", default=list(range(12)))
    ap.add_argument("--indices", type=int, default=3, help="variant indices 0..N-1 per sheet")
    ap.add_argument("--stroke", type=float, nargs="*", default=[2.0, 3.0, 4.0])
    ap.add_argument("--size", type=float, nargs="*", default=[0.85, 1.0, 1.15])
    args = ap.parse_args()

    cfg = load_config()
    templates = load_templates(cfg["template_dir"], cfg["taxonomy_csv"], cfg["scope"])
    rows = []
    for i in args.sheets:
        sheet = load_graphml(args.src / f"{i}.graphml", Image.open(args.src / f"{i}.png").size)
        runs = pipe_runs(sheet)
        for idx in range(args.indices):
            draft = plan_variant(sheet, runs, templates, cfg, i, idx)
            for w, f in product(args.stroke, args.size):
                draft.stroke_px, draft.size_factor = w, f
                r = render_variant(draft, sheet, runs, cfg)
                _, ok = check_g1.evaluate(r.page, r.runs, r.source, r.cfg)
                rows.append(
                    {"sheet": i, "index": idx, "stroke_px": w, "size": f}
                    | {f"c{c}": int(ok[c]) for c in CRITERIA}
                    | {k: r.page.log[k] for k in LOGGED}
                )
                bad = [c for c in CRITERIA if not ok[c]]
                verdict = "PASS" if not bad else f"FAIL {bad}"
                print(f"sheet {i} v{idx} w={w} size={f}: {verdict}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\n{'w':>5} {'size':>5} {'n':>4} " + " ".join(f"c{c:>2}" for c in CRITERIA) + "  all")
    for w, f in product(args.stroke, args.size):
        sel = [r for r in rows if r["stroke_px"] == w and r["size"] == f]
        fails = [sum(1 - r[f"c{c}"] for r in sel) for c in CRITERIA]
        allp = sum(all(r[f"c{c}"] for c in CRITERIA) for r in sel)
        cols = " ".join(f"{x:>3}" for x in fails)
        print(f"{w:>5} {f:>5} {len(sel):>4} {cols}  {allp}/{len(sel)}")
    print("(columns: number of failing renderings per criterion)")


if __name__ == "__main__":
    main()
