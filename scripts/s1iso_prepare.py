"""PLAN_PHASE_C §2N rule 1: source sheets for ISO-train (Stage 1 learns on ISO pages drawn by the
OPEN100-ISO renderer). Read-only on PID2Graph.zip; writes only under --out.

Every `Dataset PID` and `PID2Graph Synthetic` sheet that is NOT one of the 40 P4-val sheets
(`--p4val` mapping.csv, `source_id` per subset), renumbered from 200 (render seeds hash the sheet
id: clear of OPEN100's 0-11 and P4-val's 100-139), scaled to OPEN100's size exactly as P4-val
(`p4val_prepare.rescale_graphml`, f from the median valve of the whole subset). 5% of the sheets
of each subset (`Random("s1iso/2026")`) are marked `yolo_val` (YOLO's own val: early stopping,
never a decision). Writes `<out>/<i>.graphml`, a white `<out>/<i>.png` of the scaled size (the
renderer reads only the size) and `<out>/mapping.csv`.

    python scripts/s1iso_prepare.py --zip ~/Downloads/PID2Graph.zip --out data/iso_s1train/src
"""

from __future__ import annotations

import argparse
import csv
import io
import random
import statistics
import sys
import zipfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p4val_prepare import OPEN100_VALVE, SUBSETS, rescale_graphml, valve_sizes  # noqa: E402

FIRST_ID = 200
VAL_FRACTION = 0.05
COUNTED = ("valve", "general", "pump", "tank", "instrumentation", "arrow")


def p4val_sources(mapping: Path) -> dict[str, set[int]]:
    """{subset short name: source ids used by P4-val}."""
    out: dict[str, set[int]] = {}
    for r in csv.DictReader(open(mapping)):
        out.setdefault(r["subset"], set()).add(int(r["source_id"]))
    return out


def split_val(ids: list[int], subset: str) -> set[int]:
    """The YOLO-val sheets of one subset (5%, at least one)."""
    rng = random.Random(f"s1iso/2026/{subset}")
    return set(rng.sample(ids, max(1, round(VAL_FRACTION * len(ids)))))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--p4val", type=Path, default=Path("data/iso_p4val/src/mapping.csv"))
    args = ap.parse_args(argv)
    if any(args.out.glob("*.graphml")):
        raise SystemExit(f"{args.out} already holds sheets")
    args.out.mkdir(parents=True, exist_ok=True)
    held = p4val_sources(args.p4val)
    rows = []
    new_id = FIRST_ID
    with zipfile.ZipFile(args.zip) as zf:
        names = set(zf.namelist())
        for folder, (short, ext) in SUBSETS.items():
            all_ids = sorted(
                int(Path(n).stem)
                for n in names
                if n.startswith(folder + "/") and n.endswith(".graphml")
            )
            sizes = [s for i in all_ids for s in valve_sizes(zf.read(f"{folder}/{i}.graphml"))]
            f = OPEN100_VALVE / statistics.median(sizes)  # same f as P4-val: whole subset
            ids = [i for i in all_ids if i not in held.get(short, set())]
            assert len(all_ids) - len(ids) == len(held.get(short, set())), short
            val = split_val(ids, short)
            print(
                f"{short}: {len(all_ids)} sheets, {len(ids)} after P4-val,"
                f" {len(val)} yolo_val, f={f:.4f}"
            )
            for src_id in ids:
                xml, counts = rescale_graphml(zf.read(f"{folder}/{src_id}.graphml"), f)
                with Image.open(io.BytesIO(zf.read(f"{folder}/{src_id}.{ext}"))) as im:
                    w, h = im.size
                nw, nh = round(w * f), round(h * f)
                (args.out / f"{new_id}.graphml").write_bytes(xml)
                Image.new("1", (nw, nh), 1).save(args.out / f"{new_id}.png")
                rows.append(
                    {
                        "sheet": new_id, "subset": short, "source_id": src_id, "f": f"{f:.5f}",
                        "src_size": f"{w}x{h}", "size": f"{nw}x{nh}",
                        "split": "yolo_val" if src_id in val else "train",
                        **{f"n_{k}": counts.get(k, 0) for k in COUNTED},
                    }
                )  # fmt: skip
                new_id += 1
    with open(args.out / "mapping.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"{len(rows)} sheets -> {args.out}")


if __name__ == "__main__":
    main()
