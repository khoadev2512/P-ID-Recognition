"""Export the V6 / V7 renderings as a data root the pipeline reads (V8 of the ISO render plan).

The root is laid out like `data/` (`utils.paths.DataPaths`):

    <root>/canonical/images/<stem>.<png|jpg>   hardlinks (or symlinks / copies) of the renderings
    <root>/canonical/annotations.coco.json     images / annotations / categories (+ `fgc`)
    <root>/canonical/classes.yaml              made by pipeline.build_classes from the COCO file
    <root>/canonical/manifest.csv              made by pipeline.build_manifest
    <root>/canonical/digitizepid_splits.json   {stem: split}: the split map build_manifest reads
    <root>/splits.json                         the split per source sheet and per image
    <root>/graphml/<stem>.graphml              the source graph, relabelled, at drawn coordinates

An image's stem is `iso<sheet>_v<NNN>` (clean) or `iso<sheet>_v<NNN>_deg` (degraded); the COCO
`file_name` is `synthetic/<stem>.<ext>` because `build_manifest` insists on a tier-0 prefix.

COCO category of an annotation = its V5 `label`: a `fine_name` of the Stage 2 vocabulary
(`vocab.csv`) or `other_iso` / `other_isa` / `other_generic`. An annotation with no label (the
inlet / outlet flags) or whose label is in `drop_labels` (leader arrows) is no class: the
vocabulary has none for it, and a detector asked for symbols should be neither rewarded nor
punished for finding one. It goes to COCO as an IGNORE REGION (B6d): `iscrowd: 1`, category
`utils.classmap.IGNORE_CATEGORY` (the last category, not in the vocabulary), `ignore_kind` =
`leader_arrow` / `flag`; the stages that make labels or crops skip it. In the GraphML these stay
nodes (label = their kind, `category` empty). An agitator (`added: true`) is in COCO and has no
GraphML node; a disc's flanges are a second box on the disc's node.

A degraded image is exported from its V7 json and JPEG as they are; `degrade_mismatch` checks
that the json's `params` are what the current `degrade` config draws from its seed, so a config
changed after make_degraded.py stops the export instead of slipping into the test set.
"""

from __future__ import annotations

import csv
import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from utils.classmap import IGNORE_CATEGORY
from utils.iso_degrade import params_dict, sample_params
from utils.iso_variants import derive_seed

Box = tuple[float, float, float, float]
_NS = "http://graphml.graphdrawing.org/xmlns"

# GraphML attribute keys written for every node / edge / graph
_NODE_KEYS = [
    ("d0", "label", "string"), ("d1", "xmin", "double"), ("d2", "xmax", "double"),
    ("d3", "ymin", "double"), ("d4", "ymax", "double"), ("d5", "orig_label", "string"),
    ("d6", "category", "string"), ("d7", "drawn", "string"),
]  # fmt: skip
_EDGE_KEY = ("d8", "edge_label", "string")
_GRAPH_KEYS = [("d9", "image", "string"), ("d10", "affine", "string")]


def stem_of(sheet: int, index: int, degraded: bool) -> str:
    return f"iso{sheet:02d}_v{index:03d}" + ("_deg" if degraded else "")


def load_vocab(path: str | Path) -> list[dict[str, Any]]:
    """`vocab.csv` rows as {fine, coarse, fgc}, in file order."""
    with open(path, newline="") as fh:
        return [
            {"fine": r["fine_name"], "coarse": r["coarse"], "fgc": r["fgc_group"] == "yes"}
            for r in csv.DictReader(fh)
        ]


def build_categories(
    vocab: Sequence[Mapping[str, Any]], others: Sequence[str], other_coarse: str
) -> list[dict[str, Any]]:
    """COCO categories: the vocabulary in file order, then the `other_*` classes."""
    cats = [
        {"id": i, "name": v["fine"], "supercategory": v["coarse"], "fgc": bool(v["fgc"])}
        for i, v in enumerate(vocab, start=1)
    ]
    for name in others:
        cats.append(
            {"id": len(cats) + 1, "name": name, "supercategory": other_coarse, "fgc": False}
        )
    return cats


def ignore_category(cat_id: int) -> dict[str, Any]:
    """The COCO category of the ignore regions (`iscrowd: 1`); not a class (see module doc)."""
    return {"id": cat_id, "name": IGNORE_CATEGORY, "supercategory": IGNORE_CATEGORY, "fgc": False}


def coco_category(ann: Mapping[str, Any], known: set[str], drop: set[str]) -> str | None:
    """The COCO category of a V5 annotation, or None when it is no class and is exported as an
    ignore region (see module doc). A label that is neither in the vocabulary nor an `other_*`
    is an error, not a guess."""
    label = ann.get("label")
    if label is None or ann.get("kind") == "flag" or label in drop:
        return None
    if label not in known:
        raise ValueError(f"annotation {ann['id']}: label {label!r} is not a known category")
    return str(label)


def ignore_kind(ann: Mapping[str, Any]) -> str:
    """What an ignore region is: its label (`leader_arrow`), else its drawn kind (`flag`)."""
    return str(ann.get("label") or ann["kind"])


def degrade_mismatch(doc: Mapping[str, Any], render_cfg: Mapping[str, Any]) -> list[str]:
    """The keys of a V7 json's `params` that the current config does not draw from its stored
    `seed` (and `seed` itself when `variants.base_seed` would no longer give it); empty when the
    degraded image is what make_degraded.py would make now."""
    seed = int(doc["seed"])
    want = params_dict(sample_params(render_cfg, float(doc["stroke_px"]), random.Random(seed)))
    have = doc["params"]
    bad = sorted(k for k in want.keys() | have.keys() if want.get(k) != have.get(k))
    base = int(render_cfg["variants"]["base_seed"])
    if seed != derive_seed(base, doc["sheet"], doc["index"], "degrade"):
        bad.insert(0, "seed")
    return bad


def union(boxes: Iterable[Sequence[float]]) -> Box:
    bs = list(boxes)
    return (
        min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)
    )  # fmt: skip


def node_geometry(
    annotations: Sequence[Mapping[str, Any]], fallback: Mapping[str, Sequence[float]]
) -> dict[str, dict[str, Any]]:
    """Per node: its drawn box and the annotations that stand for it.

    A node with annotations takes the union of their boxes (a disc and its flanges; a bubble: its
    own moved circle, so its centre is the COCO box centre); an agitator, which belongs to the
    vessel's node but is not part of it, is left out. A node without (connector, crossing,
    background) takes its `fallback` box (the V4 box, moved by the affine when degraded)."""
    own: dict[str, list[Mapping[str, Any]]] = {}
    for a in annotations:
        if not a.get("added"):
            own.setdefault(a.get("node", a["id"]), []).append(a)
    out = {}
    for nid, box in fallback.items():
        anns = own.get(nid, [])
        out[nid] = {
            "box": union(a["bbox"] for a in anns) if anns else tuple(box),
            "annotations": [a["id"] for a in anns],
        }
    return out


# --- GraphML ------------------------------------------------------------------------------------


def read_source_graphml(
    path: str | Path,
) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]]]:
    """The OPEN100 graph as ([(node id, label)] in file order, [(source, target, edge_label)])."""
    root = ET.parse(path).getroot()
    names = {
        k.attrib["id"]: k.attrib.get("attr.name", k.attrib["id"])
        for k in root.iter(f"{{{_NS}}}key")
    }
    nodes = []
    for n in root.iter(f"{{{_NS}}}node"):
        vals = {names[d.attrib["key"]]: (d.text or "") for d in n.findall(f"{{{_NS}}}data")}
        nodes.append((n.attrib["id"], vals["label"].strip()))
    edges = []
    for e in root.iter(f"{{{_NS}}}edge"):
        vals = {names[d.attrib["key"]]: (d.text or "") for d in e.findall(f"{{{_NS}}}data")}
        edges.append((e.attrib["source"], e.attrib["target"], vals.get("edge_label", "solid")))
    return nodes, edges


def write_graphml(
    path: Path,
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[tuple[str, str, str]],
    graph: Mapping[str, str],
) -> None:
    """One GraphML (same keys as OPEN100's, plus orig_label / category / drawn, and graph-level
    `image` / `affine`). `nodes`: dicts with id, label, orig_label, category, drawn, box."""
    ET.register_namespace("", _NS)
    root = ET.Element(f"{{{_NS}}}graphml")
    for kid, name, typ in [*_NODE_KEYS, _EDGE_KEY, *_GRAPH_KEYS]:
        scope = (
            "edge"
            if name == "edge_label"
            else "graph"
            if (kid, name, typ) in _GRAPH_KEYS
            else "node"
        )
        ET.SubElement(
            root,
            f"{{{_NS}}}key",
            id=kid,
            attrib={"for": scope, "attr.name": name, "attr.type": typ},
        )
    g = ET.SubElement(root, f"{{{_NS}}}graph", edgedefault="undirected")
    for kid, name, _ in _GRAPH_KEYS:
        ET.SubElement(g, f"{{{_NS}}}data", key=kid).text = str(graph[name])
    for n in nodes:
        x0, y0, x1, y1 = n["box"]
        el = ET.SubElement(g, f"{{{_NS}}}node", id=n["id"])
        values = [
            n["label"], f"{x0:.2f}", f"{x1:.2f}", f"{y0:.2f}", f"{y1:.2f}", n["orig_label"],
            n["category"] or "", "1" if n["drawn"] else "0",
        ]  # fmt: skip
        for (kid, _, _), v in zip(_NODE_KEYS, values, strict=True):
            ET.SubElement(el, f"{{{_NS}}}data", key=kid).text = v
    for s, t, lab in edges:
        el = ET.SubElement(g, f"{{{_NS}}}edge", source=s, target=t)
        ET.SubElement(el, f"{{{_NS}}}data", key=_EDGE_KEY[0]).text = lab
    ET.indent(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def graph_nodes(
    source_nodes: Sequence[tuple[str, str]],
    geometry: Mapping[str, Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
    categories: Mapping[str, str | None],
) -> list[dict[str, Any]]:
    """GraphML node dicts: the source nodes (no agitator is added), relabelled.

    label = the exported category, or the drawn kind (`flag`, `leader_arrow`, ...) of a symbol
    that is not in COCO; a node that is not a symbol (connector, crossing, background) keeps its
    PID2Graph label. `categories`: annotation id -> COCO category or None."""
    main = {a["id"]: a for a in annotations if a["id"] == a.get("node", a["id"])}
    out = []
    for nid, orig in source_nodes:
        a = main.get(nid)
        if a is None:
            label, cat = orig, None
        else:
            cat = categories.get(a["id"])
            label = cat or a.get("label") or a["kind"]
        out.append(
            {
                "id": nid, "label": label, "orig_label": orig, "category": cat,
                "drawn": a is not None, "box": geometry[nid]["box"],
            }
        )  # fmt: skip
    return out


# --- splits and files ---------------------------------------------------------------------------


def assign_splits(
    sheets: Sequence[int], mode: str, test_fraction: float = 0.5, seed: int = 0
) -> dict[int, str]:
    """`test` / `train` per source sheet. `all_test`: every sheet is test. `by_source`: a seeded
    shuffle of the sheets, the first `round(test_fraction * n)` are test (at least one of each
    side when there are two sheets or more). The variants of one sheet always share its split."""
    if mode == "all_test":
        return {s: "test" for s in sheets}
    if mode != "by_source":
        raise ValueError(f"split.mode must be all_test or by_source, got {mode!r}")
    order = sorted(sheets)
    random.Random(seed).shuffle(order)
    n_test = round(test_fraction * len(order))
    if len(order) >= 2:
        n_test = min(max(n_test, 1), len(order) - 1)
    test = set(order[:n_test])
    return {s: ("test" if s in test else "train") for s in sorted(sheets)}


def link_file(src: Path, dst: Path, how: str) -> str:
    """Put `src` at `dst` by hardlink, symlink or copy; a hardlink across devices falls back to
    a copy. Returns what was done."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if how == "hardlink":
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            shutil.copy2(src, dst)
            return "copy"
    if how == "symlink":
        os.symlink(os.path.relpath(src.resolve(), dst.parent.resolve()), dst)
        return "symlink"
    if how == "copy":
        shutil.copy2(src, dst)
        return "copy"
    raise ValueError(f"link must be hardlink, symlink or copy, got {how!r}")


def coco_box(box: Sequence[float]) -> list[float]:
    """[x0, y0, x1, y1] -> COCO [x, y, w, h], to 2 decimals."""
    return [
        round(box[0], 2),
        round(box[1], 2),
        round(box[2] - box[0], 2),
        round(box[3] - box[1], 2),
    ]


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1))
