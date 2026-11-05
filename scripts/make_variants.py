"""V6: many variants of each OPEN100 sheet, near-duplicates and failed pictures removed.

Runs utils.iso_variants.generate_sheet on the chosen sheets (one process per sheet) with the
`variants` block of configs/render_iso.yaml, the automatic G1 criteria of check_g1.py as the
last filter. Kept variants go to <out>/sheet<i>/v<NNN>.{png,svg,json}; the attempt table
(kept and dropped, with the reason) goes to <out>/manifest.csv. With --stats, summary CSVs
(per sheet, per class, per attempt) are written there. Read-only on data/raw/.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_g1  # noqa: E402

from utils.iso_mapping import load_config  # noqa: E402
from utils.iso_templates import load_templates  # noqa: E402
from utils.iso_variants import Rendered, generate_sheet  # noqa: E402
from utils.pid2graph import load_graphml, pipe_runs  # noqa: E402

REASONS = ("signature", "error", "quality", "duplicate", "g1")


def _g1(rendered: Rendered) -> dict[int, bool]:
    return check_g1.evaluate(rendered.page, rendered.runs, rendered.source, rendered.cfg)[1]


def run_sheet(job: tuple) -> tuple[int, list[dict], float]:
    """Worker: all attempts of one sheet. Returns (sheet, records, wall seconds)."""
    source, src, out, cfg, use_g1 = job
    t0 = time.perf_counter()
    templates = load_templates(cfg["template_dir"], cfg["taxonomy_csv"], cfg["scope"])
    sheet = load_graphml(src / f"{source}.graphml", Image.open(src / f"{source}.png").size)
    records = generate_sheet(
        source, sheet, pipe_runs(sheet), templates, cfg, out, _g1 if use_g1 else None
    )
    return source, records, time.perf_counter() - t0


def write_stats(records: list[dict], walls: dict[int, float], out: Path, stats: Path) -> None:
    stats.mkdir(parents=True, exist_ok=True)
    flat = []
    for r in records:
        flat.append(
            {
                "sheet": r["sheet"],
                "index": r["index"],
                "status": r["status"],
                "reason": r["reason"] or "",
                "detail": r["detail"],
                "stroke_px": r["stroke_px"],
                "size_factor": r["size_factor"],
                "nearest": "" if r["nearest"] is None else r["nearest"],
                "signature": r["signature"],
                "g1_flags": ",".join(map(str, r.get("g1_flags", []))),
                "sec_total": r["seconds"].get("total", ""),
                **{f"sec_{k}": v for k, v in r["seconds"].items() if k != "total"},
            }
        )
    keys = list(dict.fromkeys(k for row in flat for k in row))
    for name, rows in (("attempts.csv", flat),):
        with open(stats / name, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=keys)
            wr.writeheader()
            wr.writerows(rows)
    with open(out / "manifest.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(flat)

    with open(stats / "per_sheet.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["sheet", "attempts", "kept", *REASONS, "g1_flagged", "wall_s", "png_kb_mean"])
        for s in sorted(walls):
            rs = [r for r in records if r["sheet"] == s]
            kept = [r for r in rs if r["status"] == "kept"]
            drops = Counter(r["reason"] for r in rs if r["status"] == "dropped")
            pngs = [f for r in kept for f in r.get("files", []) if f.endswith(".png")]
            sizes = [(out / f).stat().st_size / 1000 for f in pngs]
            wr.writerow(
                [s, len(rs), len(kept), *(drops[x] for x in REASONS),
                 sum(bool(r.get("g1_flags")) for r in kept), round(walls[s], 1),
                 round(sum(sizes) / len(sizes), 1) if sizes else ""]
            )  # fmt: skip

    classes: dict[str, Counter] = {}
    for r in records:
        if r["status"] == "kept":
            for label, n in r["classes"].items():
                classes.setdefault(label, Counter())[r["sheet"]] += n
    sheets = sorted(walls)
    with open(stats / "per_class.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["class", "total", "variants_with", *[f"sheet{s}" for s in sheets]])
        kept_recs = [r for r in records if r["status"] == "kept"]
        for label in sorted(classes):
            with_ = sum(label in r["classes"] for r in kept_recs)
            per_sheet = [classes[label][s] for s in sheets]
            wr.writerow([label, sum(classes[label].values()), with_, *per_sheet])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=Path("data/raw/pid2graph/open100"))
    ap.add_argument("--out", type=Path, default=Path("data/iso_open100/variants"))
    ap.add_argument("--stats", type=Path, default=None, help="write summary CSVs here")
    ap.add_argument("--sheets", type=int, nargs="*", default=list(range(12)))
    ap.add_argument("--count", type=int, help="variants kept per sheet (config: variants.count)")
    ap.add_argument("--max-attempts", type=int, help="candidates tried per sheet at most")
    ap.add_argument("--base-seed", type=int)
    ap.add_argument("--min-distance", type=int)
    ap.add_argument("--no-g1", action="store_true", help="skip the G1 criteria filter")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cfg = load_config()
    for key, val in (
        ("count", args.count),
        ("max_attempts", args.max_attempts),
        ("base_seed", args.base_seed),
        ("min_distance", args.min_distance),
    ):
        if val is not None:
            cfg["variants"][key] = val
    cfg["variants"]["max_attempts"] = max(cfg["variants"]["max_attempts"], cfg["variants"]["count"])
    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    jobs = [(s, args.src, args.out, cfg, not args.no_g1) for s in args.sheets]
    records: list[dict] = []
    walls: dict[int, float] = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for source, recs, wall in pool.map(run_sheet, jobs):
            kept = sum(r["status"] == "kept" for r in recs)
            drops = Counter(r["reason"] for r in recs if r["status"] == "dropped")
            print(
                f"sheet {source:>2}: {len(recs)} attempts, {kept} kept, dropped {dict(drops)}, "
                f"{wall:.0f}s",
                flush=True,
            )
            records += recs
            walls[source] = wall
    total = time.perf_counter() - t0
    kept = sum(r["status"] == "kept" for r in records)
    print(f"{kept} kept of {len(records)} attempts in {total:.0f}s wall ({args.workers} workers)")
    write_stats(records, walls, args.out, args.stats or args.out)


if __name__ == "__main__":
    main()
