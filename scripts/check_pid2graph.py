"""V2 check for PLAN_ISO_RENDER_PID2GRAPH: port_config x label table + port overlays.

Runs utils.pid2graph over the 12 OPEN100 sheets and writes, under --out:
  port_config_by_label.csv   label, port_config(+axis), count  (the thesis table)
  size_class_by_port_config.csv  `general` nodes: size_class (V3 thresholds), port_config, count
  elbows.csv                 every pipe run that needed an L elbow, with its warning
  overlay_<i>.jpg            pipes (green = as annotated, red = elbowed), symbol bboxes
                             (orange), connectors (blue), crossings (magenta), ports (cyan dot)

Read-only on data/raw/; writes nothing outside --out.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.iso_mapping import load_config, size_class  # noqa: E402
from utils.pid2graph import load_graphml, pipe_runs, port_config, symbol_ports  # noqa: E402

_SKIP = ("connector", "background", "crossing")
_BOX_COLORS = {"connector": (0, 0, 255), "crossing": (255, 0, 255)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=Path("data/raw/pid2graph/open100"))
    ap.add_argument("--out", type=Path, default=Path("docs/experiments/iso_render/v2_ports"))
    ap.add_argument("--scale", type=float, default=0.5, help="overlay downscale")
    ap.add_argument("--no-overlay", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    table: Counter[tuple[str, str]] = Counter()
    size_table: Counter[tuple[str, str]] = Counter()
    thresholds = load_config()["size_class"]
    elbow_rows: list[tuple[int, str, str, str, int]] = []
    for i in range(12):
        image = Image.open(args.src / f"{i}.png")
        sheet = load_sheet(args.src, i, image.size)
        runs = pipe_runs(sheet)
        for r in runs:
            if r.elbows:
                la, lb = sheet.nodes[r.a].label, sheet.nodes[r.b].label
                elbow_rows.append(
                    (i, f"{r.a}({la})", f"{r.b}({lb})", "; ".join(r.warnings), len(r.connectors))
                )

        ports_all = []
        for nid, node in sheet.nodes.items():
            if node.label in _SKIP:
                continue
            ports = symbol_ports(sheet, nid, runs)
            cfg = port_config(ports)
            key = cfg.kind + (f"_{cfg.axis}" if cfg.axis else "")
            table[(node.label, key)] += 1
            if node.label == "general":
                size = size_class(node, cfg, thresholds["small_max"], thresholds["large_min"])
                size_table[(size, key)] += 1
            ports_all += ports

        if args.no_overlay:
            continue
        s = args.scale
        im = image.convert("RGB").resize((int(image.width * s), int(image.height * s)))
        d = ImageDraw.Draw(im)
        for r in runs:
            d.line(
                [(x * s, y * s) for x, y in r.points],
                fill=(255, 0, 0) if r.elbows else (0, 160, 0),
                width=2,
            )
        for node in sheet.nodes.values():
            if node.label == "background":
                continue
            color = _BOX_COLORS.get(node.label, (255, 140, 0))
            d.rectangle([node.xmin * s, node.ymin * s, node.xmax * s, node.ymax * s], outline=color)
        for p in ports_all:
            d.ellipse([p.x * s - 3, p.y * s - 3, p.x * s + 3, p.y * s + 3], fill=(0, 200, 220))
        im.save(args.out / f"overlay_{i}.jpg", quality=85)

    with (args.out / "port_config_by_label.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "port_config", "count"])
        for (label, cfg), n in sorted(table.items()):
            w.writerow([label, cfg, n])
    with (args.out / "size_class_by_port_config.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["size_class", "port_config", "count"])
        for (size, key), n in sorted(size_table.items()):
            w.writerow([size, key, n])
    with (args.out / "elbows.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sheet", "a", "b", "warnings", "n_connectors"])
        w.writerows(elbow_rows)

    labels = sorted({label for label, _ in table})
    for label in labels:
        row = {cfg: n for (lb, cfg), n in table.items() if lb == label}
        print(f"{label:16s} {dict(sorted(row.items()))}")
    for size in ("small", "medium", "large"):
        row = {k: n for (sz, k), n in size_table.items() if sz == size}
        print(f"general/{size:7s} {sum(row.values()):4d} {dict(sorted(row.items()))}")
    print(f"elbowed runs: {len(elbow_rows)}  -> {args.out}")


def load_sheet(src: Path, i: int, size: tuple[int, int]):
    return load_graphml(src / f"{i}.graphml", size)


if __name__ == "__main__":
    main()
