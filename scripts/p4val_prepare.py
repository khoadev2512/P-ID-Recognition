"""PLAN_PHASE_C §2K.2: source sheets for P4-val (the OPEN100-ISO renderer on other PID2Graph
skeletons). Read-only on PID2Graph.zip; writes only under --out.

Draws 20 `Dataset PID` + 20 `PID2Graph Synthetic` sheets (Random("p4val/2026")), renumbers them
100-139 (render seeds hash the sheet id, so they differ from OPEN100's 0-11), scales every
coordinate by f = OPEN100's median valve size / the subset's median valve size (long side of
`valve` nodes over ALL sheets of the subset) so the renderer's pixel constants and
symbol sizes match OPEN100, keeps `connector` / `crossing` markers at their own size around the new
centre, and writes `<out>/<i>.graphml` + a white `<out>/<i>.png` of the scaled size (the renderer
reads only the size) + `<out>/mapping.csv`.

    python scripts/p4val_prepare.py --zip PID2Graph.zip --out data/iso_p4val/src
"""

from __future__ import annotations

import argparse
import csv
import io
import random
import statistics
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from PIL import Image

NS = "http://graphml.graphdrawing.org/xmlns"
OPEN100_VALVE = 26.3  # px, median long side of `valve` nodes on the 12 OPEN100 sheets (measured)
SUBSETS = {  # zip folder -> (short name, image extension)
    "PID2Graph/Complete/Dataset PID": ("dataset_pid", "png"),
    "PID2Graph/Complete/PID2Graph Synthetic": ("synthetic", "jpg"),
}
PER_SUBSET = 20
FIRST_ID = 100  # render seeds hash the sheet id: keep clear of OPEN100's 0-11
MARKERS = ("connector", "crossing")
COORDS = ("xmin", "ymin", "xmax", "ymax")

ET.register_namespace("", NS)


def scale_box(box: tuple[float, float, float, float], f: float, keep_size: bool):
    """Scale a box about the origin by f; a marker keeps its own width / height round the new
    centre."""
    x0, y0, x1, y1 = box
    if not keep_size:
        return x0 * f, y0 * f, x1 * f, y1 * f
    cx, cy, hw, hh = (x0 + x1) / 2 * f, (y0 + y1) / 2 * f, (x1 - x0) / 2, (y1 - y0) / 2
    return cx - hw, cy - hh, cx + hw, cy + hh


def _nodes(root):
    """(node, {attr name: data element}); key ids are looked up one by one, since a file may
    declare two keys with the same name (PID2Graph: `xmin` as long and as double)."""
    names = {k.get("id"): k.get("attr.name") for k in root.iter(f"{{{NS}}}key")}
    for n in root.iter(f"{{{NS}}}node"):
        yield n, {names.get(d.get("key")): d for d in n.findall(f"{{{NS}}}data")}


def valve_sizes(xml: bytes) -> list[float]:
    out = []
    for _, d in _nodes(ET.fromstring(xml)):
        if (
            d.get("label") is not None
            and d["label"].text == "valve"
            and all(c in d for c in COORDS)
        ):
            x0, y0, x1, y1 = (float(d[c].text) for c in COORDS)
            out.append(max(x1 - x0, y1 - y0))
    return out


def rescale_graphml(xml: bytes, f: float) -> tuple[bytes, dict[str, int]]:
    root = ET.fromstring(xml)
    counts: dict[str, int] = {}
    for _, d in _nodes(root):
        label = d["label"].text if d.get("label") is not None else ""
        counts[label] = counts.get(label, 0) + 1
        if not all(c in d for c in COORDS):
            continue
        box = scale_box(tuple(float(d[c].text) for c in COORDS), f, label in MARKERS)
        for c, v in zip(COORDS, box, strict=True):
            d[c].text = repr(round(v, 3))
    return ET.tostring(root, encoding="utf-8", xml_declaration=True), counts


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--zip", type=Path, default=Path("PID2Graph.zip"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random("p4val/2026")
    rows = []
    with zipfile.ZipFile(args.zip) as zf:
        names = set(zf.namelist())
        new_id = FIRST_ID
        for folder, (short, ext) in SUBSETS.items():
            ids = sorted(
                int(Path(n).stem)
                for n in names
                if n.startswith(folder + "/") and n.endswith(".graphml")
            )
            sizes = [s for i in ids for s in valve_sizes(zf.read(f"{folder}/{i}.graphml"))]
            f = OPEN100_VALVE / statistics.median(sizes)
            print(
                f"{short}: {len(ids)} sheets, {len(sizes)} valves,"
                f" median {statistics.median(sizes):.1f} px, f = {f:.4f}"
            )
            for src_id in rng.sample(ids, PER_SUBSET):
                xml, counts = rescale_graphml(zf.read(f"{folder}/{src_id}.graphml"), f)
                with Image.open(io.BytesIO(zf.read(f"{folder}/{src_id}.{ext}"))) as im:
                    w, h = im.size
                nw, nh = round(w * f), round(h * f)
                (args.out / f"{new_id}.graphml").write_bytes(xml)
                Image.new("1", (nw, nh), 1).save(args.out / f"{new_id}.png")
                rows.append(
                    {
                        "sheet": new_id,
                        "subset": short,
                        "source_id": src_id,
                        "f": f"{f:.5f}",
                        "src_size": f"{w}x{h}",
                        "size": f"{nw}x{nh}",
                        **{
                            f"n_{k}": counts.get(k, 0)
                            for k in (
                                "valve",
                                "general",
                                "pump",
                                "tank",
                                "instrumentation",
                                "arrow",
                            )
                        },
                    }
                )
                new_id += 1
    with (args.out / "mapping.csv").open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"{len(rows)} sheets -> {args.out}")


if __name__ == "__main__":
    main()
