"""scripts/p4val_prepare.py (PLAN_PHASE_C §2K.2): box scaling and GraphML rescaling."""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from p4val_prepare import NS, rescale_graphml, scale_box, valve_sizes  # noqa: E402

GRAPH = f"""<?xml version='1.0' encoding='utf-8'?>
<graphml xmlns="{NS}">
  <key id="d0" for="node" attr.name="label" attr.type="string"/>
  <key id="d1" for="node" attr.name="xmin" attr.type="double"/>
  <key id="d2" for="node" attr.name="ymin" attr.type="double"/>
  <key id="d3" for="node" attr.name="xmax" attr.type="double"/>
  <key id="d4" for="node" attr.name="ymax" attr.type="double"/>
  <key id="d5" for="edge" attr.name="edge_label" attr.type="string"/>
  <key id="d6" for="node" attr.name="xmin" attr.type="long"/>
  <graph edgedefault="undirected">
    <node id="v"><data key="d0">valve</data><data key="d1">100</data><data key="d2">200</data>
      <data key="d3">180</data><data key="d4">240</data></node>
    <node id="c"><data key="d0">connector</data><data key="d1">96</data><data key="d2">216</data>
      <data key="d3">104</data><data key="d4">224</data></node>
    <edge source="v" target="c"><data key="d5">solid</data></edge>
  </graph>
</graphml>""".encode()


def test_scale_box_scales_symbols_and_keeps_marker_size():
    assert scale_box((10, 20, 30, 60), 0.5, False) == (5, 10, 15, 30)
    assert scale_box((96, 216, 104, 224), 0.5, True) == (46, 106, 54, 114)


def test_valve_sizes_reads_long_sides():
    assert valve_sizes(GRAPH) == [80.0]


def test_rescale_graphml_changes_only_coordinates():
    out, counts = rescale_graphml(GRAPH, 0.25)
    assert counts == {"valve": 1, "connector": 1}
    root = ET.fromstring(out)
    vals = {
        n.get("id"): [float(d.text) for d in n.findall(f"{{{NS}}}data")[1:]]
        for n in root.iter(f"{{{NS}}}node")
    }
    assert vals["v"] == pytest.approx([25, 50, 45, 60])
    assert vals["c"] == pytest.approx([21, 51, 29, 59])  # 8 x 8 marker round the scaled centre
    assert [e.find(f"{{{NS}}}data").text for e in root.iter(f"{{{NS}}}edge")] == ["solid"]
