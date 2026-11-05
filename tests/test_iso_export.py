"""Tests for utils/iso_export.py and pipeline/render_iso.py (V8): the exported root loads through
the pipeline's own readers, keeps the source topology, and its boxes agree with its graph."""

from __future__ import annotations

import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf
from PIL import Image

from pipeline.render_iso import RenderIsoPipeline
from utils import iso_export as ex
from utils.classmap import load_classmap
from utils.iso_mapping import load_config
from utils.iso_templates import load_templates
from utils.iso_variants import generate_sheet
from utils.paths import DataPaths
from utils.pid2graph import load_graphml, pipe_runs

ROOT = Path(__file__).resolve().parent.parent
_OPEN100 = ROOT / "data" / "raw" / "pid2graph" / "open100"
_TEMPLATES = ROOT / "data" / "raw" / "template_iso_svg"
_VOCAB = ROOT / "data" / "raw" / "iso_symbols" / "vocab.csv"
needs_data = pytest.mark.skipif(
    not (_OPEN100.exists() and _TEMPLATES.exists() and _VOCAB.exists()), reason="data missing"
)

OTHERS = ["other_iso", "other_isa", "other_generic"]


def _ann(aid, label, kind="iso", node=None, box=(0, 0, 10, 10), **extra):
    return {
        "id": aid,
        "node": node or aid,
        "label": label,
        "kind": kind,
        "bbox": list(box),
        **extra,
    }


@needs_data
def test_categories_are_the_vocabulary_then_other():
    vocab = ex.load_vocab(_VOCAB)
    cats = ex.build_categories(vocab, OTHERS, "other")
    assert [c["name"] for c in cats][-3:] == OTHERS
    assert [c["id"] for c in cats] == list(range(1, len(vocab) + 4))
    assert {c["name"] for c in cats[: len(vocab)]} == {v["fine"] for v in vocab}
    by = {c["name"]: c for c in cats}
    assert by["valve_gate"]["fgc"] is True and by["pump_general"]["fgc"] is False
    assert by["other_isa"]["supercategory"] == "other" and by["other_isa"]["fgc"] is False


def test_coco_category_rules():
    known, drop = {"valve_gate", "flow_direction", *OTHERS}, {"leader_arrow"}
    assert ex.coco_category(_ann("a", "valve_gate"), known, drop) == "valve_gate"
    assert ex.coco_category(_ann("a", "other_iso"), known, drop) == "other_iso"
    assert ex.coco_category(_ann("a", "flow_direction", "arrow"), known, drop) == "flow_direction"
    assert ex.coco_category(_ann("f", None, "flag"), known, drop) is None  # inlet / outlet
    assert ex.coco_category(_ann("l", "leader_arrow", "leader"), known, drop) is None
    with pytest.raises(ValueError, match="not a known category"):
        ex.coco_category(_ann("x", "mystery"), known, drop)


def test_node_geometry_unions_flanges_and_leaves_the_agitator_out():
    anns = [
        _ann("disc1", "blind_disc", box=(10, 10, 20, 100)),
        _ann("disc1/flanges", "flanged_connection", node="disc1", box=(5, 90, 25, 110)),
        _ann("tank2", "other_iso", box=(0, 0, 50, 200)),
        _ann("tank2/agitator", "agitator_general", node="tank2", box=(20, -5, 30, 150), added=True),
    ]
    fallback = {"disc1": (0, 0, 1, 1), "tank2": (0, 0, 1, 1), "conn3": (7, 8, 9, 10)}
    g = ex.node_geometry(anns, fallback)
    assert g["disc1"]["box"] == (5, 10, 25, 110)
    assert g["tank2"]["box"] == (0, 0, 50, 200)  # not stretched by the agitator
    assert g["conn3"]["box"] == (7, 8, 9, 10) and g["conn3"]["annotations"] == []


def test_assign_splits_by_source():
    sheets = list(range(12))
    assert set(ex.assign_splits(sheets, "all_test").values()) == {"test"}
    a = ex.assign_splits(sheets, "by_source", 0.25, seed=1)
    assert a == ex.assign_splits(sheets, "by_source", 0.25, seed=1)  # deterministic
    assert list(a.values()).count("test") == 3 and set(a) == set(sheets)
    assert ex.assign_splits(sheets, "by_source", 0.25, seed=2) != a
    two = ex.assign_splits([3, 9], "by_source", 0.0)
    assert sorted(two.values()) == ["test", "train"]  # one of each side
    with pytest.raises(ValueError):
        ex.assign_splits(sheets, "random")


def test_link_file_modes(tmp_path):
    src = tmp_path / "a" / "x.png"
    src.parent.mkdir()
    src.write_bytes(b"data")
    for how in ("hardlink", "symlink", "copy"):
        dst = tmp_path / how / "y.png"
        assert ex.link_file(src, dst, how) == how
        assert dst.read_bytes() == b"data" and dst.exists()
    assert os.stat(tmp_path / "hardlink" / "y.png").st_nlink >= 2
    assert (tmp_path / "symlink" / "y.png").is_symlink()
    ex.link_file(src, tmp_path / "hardlink" / "y.png", "hardlink")  # again: replaced, no error
    with pytest.raises(ValueError):
        ex.link_file(src, tmp_path / "z.png", "rsync")


@needs_data
def test_graphml_writer_round_trips_through_the_pipeline_reader(tmp_path):
    nodes, edges = ex.read_source_graphml(_OPEN100 / "3.graphml")
    sheet = load_graphml(_OPEN100 / "3.graphml", Image.open(_OPEN100 / "3.png").size)
    assert [n for n, _ in nodes] == list(sheet.nodes) and len(edges) == len(sheet.edges)
    out = [
        {"id": n, "label": "x", "orig_label": o, "category": None, "drawn": False,
         "box": (1.0, 2.0, 3.0, 4.0)}
        for n, o in nodes
    ]  # fmt: skip
    ex.write_graphml(tmp_path / "g.graphml", out, edges, {"image": "a.png", "affine": "identity"})
    back = load_graphml(tmp_path / "g.graphml", (10, 10))  # the reader the pipeline code uses
    assert list(back.nodes) == list(sheet.nodes) and back.edges == sheet.edges
    n0 = back.nodes[nodes[0][0]]
    assert (n0.label, n0.xmin, n0.ymin, n0.xmax, n0.ymax) == ("x", 1.0, 2.0, 3.0, 4.0)
    assert {e[2] for e in edges} == {"solid"}


# --- the whole export on two real sheets --------------------------------------------------------


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """Two sheets (3: bubbles, discs; 9: reactors with agitators), two variants each, clean and
    degraded, exported with `by_source`."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_degraded

    root = tmp_path_factory.mktemp("iso_root")
    cfg = load_config()
    cfg["variants"].update(count=2, max_attempts=4)
    cfg["export"].update(data_root=str(root), workers=1)
    cfg["export"]["split"].update(mode="by_source", test_fraction=0.5, seed=0)
    templates = load_templates(cfg["template_dir"], cfg["taxonomy_csv"], cfg["scope"])
    for s in (3, 9):
        sheet = load_graphml(_OPEN100 / f"{s}.graphml", Image.open(_OPEN100 / f"{s}.png").size)
        runs = pipe_runs(sheet)
        generate_sheet(s, sheet, runs, templates, cfg, root / "variants")
        out = root / "degraded" / f"sheet{s}"
        out.mkdir(parents=True)
        for jf in sorted((root / "variants" / f"sheet{s}").glob("v*.json")):
            meta = json.loads(jf.read_text())
            doc, jpeg, _ = make_degraded.degrade_one(
                meta, jf.with_suffix(".png"), sheet, runs, templates, cfg
            )
            (out / jf.name).write_text(json.dumps(doc))
            (out / jf.name).with_suffix(".jpg").write_bytes(jpeg)
    top = OmegaConf.create({"data_root": str(root), "run_dir": str(root / "runs")})
    top.render_iso = OmegaConf.create(cfg)
    RenderIsoPipeline(top).run()
    return root, cfg


@needs_data
def test_export_loads_through_the_pipeline_readers(exported):
    root, _ = exported
    paths = DataPaths(root=root)
    coco = json.loads(paths.coco_json.read_text())
    classmap = load_classmap(paths.classes_yaml)
    assert len(coco["images"]) == 8  # 2 sheets x 2 variants x (clean, degraded)
    assert {c["name"] for c in coco["categories"]} == set(classmap.fine_classes)
    manifest = pd.read_csv(paths.manifest_csv)
    assert len(manifest) == 8 and set(manifest["source"]) == {"synthetic"}
    for img in coco["images"]:
        stem = img["file_name"].split("/")[-1].rsplit(".", 1)[0]
        path = paths.find_image(stem)  # the lookup every stage uses
        assert Image.open(path).size == (img["width"], img["height"])
        assert (root / "graphml" / f"{stem}.graphml").exists()
    assert sum(img["degraded"] for img in coco["images"]) == 4
    assert any(c["fgc"] for c in coco["categories"]) and "other" in classmap.coarse_classes


@needs_data
def test_exported_categories_are_vocabulary_or_other(exported):
    root, _ = exported
    coco = json.loads(DataPaths(root=root).coco_json.read_text())
    vocab = {v["fine"] for v in ex.load_vocab(_VOCAB)}
    names = {c["id"]: c["name"] for c in coco["categories"]}
    used = {names[a["category_id"]] for a in coco["annotations"]}
    assert used <= vocab | set(OTHERS) and {"other_isa", "other_iso"} <= used
    assert not used & {"leader_arrow", "flag", "vessel_dished_ends"}
    assert all(a["bbox"][2] > 0 and a["bbox"][3] > 0 for a in coco["annotations"])


@needs_data
def test_graphml_topology_equals_the_source(exported):
    root, _ = exported
    for s in (3, 9):
        src_nodes, src_edges = ex.read_source_graphml(_OPEN100 / f"{s}.graphml")
        for g in sorted((root / "graphml").glob(f"iso{s:02d}_*.graphml")):
            nodes, edges = ex.read_source_graphml(g)
            assert [n for n, _ in nodes] == [n for n, _ in src_nodes], g.name  # no extra nodes
            assert edges == src_edges, g.name
            assert not any("agitator" in n for n, _ in nodes)


def _boxes(graphml: Path):
    root = ET.parse(graphml).getroot()
    ns = "{http://graphml.graphdrawing.org/xmlns}"
    names = {k.attrib["id"]: k.attrib["attr.name"] for k in root.iter(f"{ns}key")}
    out = {}
    for n in root.iter(f"{ns}node"):
        v = {names[d.attrib["key"]]: d.text or "" for d in n.findall(f"{ns}data")}
        out[n.attrib["id"]] = v
    return out


@needs_data
def test_bubble_centre_in_the_graph_is_the_centre_of_its_coco_box(exported):
    root, _ = exported
    coco = json.loads(DataPaths(root=root).coco_json.read_text())
    ids = {i["id"]: i["file_name"].split("/")[-1].rsplit(".", 1)[0] for i in coco["images"]}
    checked = set()
    for a in coco["annotations"]:
        if a["node"] != a["ann_id"] or not a["node"].startswith("instrumentation"):
            continue
        stem = ids[a["image_id"]]
        v = _boxes(root / "graphml" / f"{stem}.graphml")[a["node"]]
        if v["category"] != "other_isa":
            continue  # a bubble replaced by a device
        cx = (float(v["xmin"]) + float(v["xmax"])) / 2
        cy = (float(v["ymin"]) + float(v["ymax"])) / 2
        x, y, w, h = a["bbox"]
        assert cx == pytest.approx(x + w / 2, abs=0.02) and cy == pytest.approx(y + h / 2, abs=0.02)
        checked.add(stem)
    assert any(s.endswith("_deg") for s in checked) and any(not s.endswith("_deg") for s in checked)


@needs_data
def test_agitators_are_in_coco_without_a_node_and_flanges_share_their_node(exported):
    root, _ = exported
    coco = json.loads(DataPaths(root=root).coco_json.read_text())
    added = [a for a in coco["annotations"] if a.get("added")]
    assert added and all(a["ann_id"] == a["node"] + "/agitator" for a in added)
    flanges = [a for a in coco["annotations"] if a["ann_id"].endswith("/flanges")]
    assert flanges
    ids = {i["id"]: i["file_name"].split("/")[-1].rsplit(".", 1)[0] for i in coco["images"]}
    a = flanges[0]
    own = [
        b for b in coco["annotations"] if b["image_id"] == a["image_id"] and b["node"] == a["node"]
    ]
    assert len(own) >= 2  # the disc and its flanges: two boxes, one node
    assert a["node"] in _boxes(root / "graphml" / f"{ids[a['image_id']]}.graphml")


@needs_data
def test_splits_are_disjoint_by_source_and_match_the_manifest(exported):
    root, _ = exported
    sp = json.loads((root / "splits.json").read_text())
    assert sorted(sp["sheets"].values()) == ["test", "train"]
    by_split: dict[str, set[int]] = {"train": set(), "test": set()}
    for stem, split in sp["images"].items():
        by_split[split].add(int(stem[3:5]))
    assert not by_split["train"] & by_split["test"] and by_split["train"] and by_split["test"]
    manifest = pd.read_csv(root / "canonical" / "manifest.csv").set_index("id")
    assert {k: manifest.loc[k, "split"] for k in sp["images"]} == sp["images"]
    assert json.loads((root / "canonical" / "digitizepid_splits.json").read_text()) == sp["images"]


@needs_data
def test_images_are_hardlinks_and_the_root_guard_holds(exported, tmp_path):
    root, cfg = exported
    img = root / "canonical" / "images" / "iso03_v000.png"
    assert os.path.samefile(img, root / "variants" / "sheet3" / "v000.png")
    assert os.stat(img).st_nlink >= 2
    top = OmegaConf.create({"data_root": str(ROOT / "data"), "run_dir": "x"})
    top.render_iso = OmegaConf.create(cfg)
    with pytest.raises(ValueError, match="refusing"):
        RenderIsoPipeline(top).validate()
