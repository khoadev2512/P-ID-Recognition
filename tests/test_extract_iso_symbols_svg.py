"""Unit tests for scripts/extract_iso_symbols_svg.py — the vector ISO template renderer.

A tiny hand-written SVG in the BK-2020-04 layout (Inkscape-labelled REG# groups) stands
in for the real sheets, so the tests need pymupdf but not the GitLab repo. What they
pin is the part that is easy to get subtly wrong: which 5-unit segments are connection
stubs (removed) versus part of the symbol (kept), and dashed lines surviving a renderer
that ignores stroke-dasharray.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pymupdf")

_SCRIPTS = str(Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import extract_iso_symbols_svg as ext  # noqa: E402

_STYLE = "fill:none;stroke:#000000;stroke-width:0.5"

# A bow-tie valve with a centre stem (5 units, INSIDE the body: must stay) and two pipe
# stubs (5 units, OUTSIDE: must go); a symbol made only of 5-unit bars (a flange: must
# keep everything); a dashed line; and an unrelated symbol that must not leak in.
_SVG = f"""<svg xmlns="http://www.w3.org/2000/svg"
  xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  width="200mm" height="100mm" viewBox="0 0 200 100">
  <g inkscape:label="REG#:V1_symbol">
    <path style="{_STYLE}" d="m 20,10 v 5 l 10,-5 v 5 z"/>
    <path style="{_STYLE}" d="m 25,10 v 5"/>
    <path style="{_STYLE}" d="m 20,12.5 h -5"/>
    <path style="{_STYLE}" d="M 30,12.5 H 35"/>
  </g>
  <g inkscape:label="REG#:F1_symbol">
    <path style="{_STYLE}" d="m 60,10 v 5"/>
    <path style="{_STYLE}" d="m 62,10 v 5"/>
  </g>
  <g inkscape:label="REG#:D1_symbol">
    <path style="{_STYLE};stroke-dasharray:4, 4" d="m 100,10 h 20"/>
    <path style="{_STYLE}" d="m 100,8 h 20"/>
  </g>
  <g inkscape:label="REG#:OTHER_symbol">
    <path style="{_STYLE}" d="m 150,50 h 30 v 30"/>
  </g>
</svg>
"""


def _root() -> ET.Element:
    return ET.fromstring(_SVG)


def _group(root, reg):
    return next(e for e in root.iter() if e.get(ext._INK_LABEL) == ext.symbol_label(reg))


def _paths(group) -> list[str]:
    return [e.get("d") for e in group.iter() if e.tag.endswith("path")]


# ------------------------------------------------------------------ #
# path geometry
# ------------------------------------------------------------------ #


@pytest.mark.parametrize(
    "d, expected",
    [
        ("m 20,12.5 h -5", ((20, 12.5), (15, 12.5))),
        ("M 30,12.5 H 35", ((30, 12.5), (35, 12.5))),
        ("m 25,10 v 5", ((25, 10), (25, 15))),
        ("m 1,2 V 9", ((1, 2), (1, 9))),  # first moveto is absolute even when lowercase
        ("m 0,0 l 3,4", ((0, 0), (3, 4))),
        ("M 1,1 L 4,5", ((1, 1), (4, 5))),
    ],
)
def test_segment_endpoints(d, expected):
    assert ext.segment_endpoints(d) == expected


@pytest.mark.parametrize("d", [None, "", "m 0,0 h 5 v 5", "m 0,0 c 1,1 2,2 3,3"])
def test_non_segments_have_no_endpoints(d):
    assert ext.segment_endpoints(d) is None


def test_segment_length():
    assert ext.segment_length("m 0,0 l 3,4") == pytest.approx(5.0)
    assert ext.segment_length("m 0,0 h -5") == pytest.approx(5.0)


def test_only_five_unit_straight_paths_are_stub_candidates():
    make = lambda d: ET.Element("{http://www.w3.org/2000/svg}path", {"d": d})  # noqa: E731
    assert ext.is_stub(make("m 0,0 h 5"))
    assert ext.is_stub(make("m 0,0 v -4.99999"))
    assert not ext.is_stub(make("m 0,0 h 20"))
    assert not ext.is_stub(make("m 0,0 h 5 v 5"))
    assert not ext.is_stub(ET.Element("{http://www.w3.org/2000/svg}circle"))


# ------------------------------------------------------------------ #
# isolate: pruning, stubs, dashes
# ------------------------------------------------------------------ #


def test_isolate_keeps_only_the_requested_symbol():
    root = _root()
    assert ext.isolate(root, "V1", keep_stubs=True)
    labels = {e.get(ext._INK_LABEL) for e in root.iter() if e.get(ext._INK_LABEL)}
    assert labels == {"REG#:V1_symbol"}


def test_isolate_reports_a_missing_symbol():
    assert not ext.isolate(_root(), "NOPE")


def test_outside_stubs_go_and_the_inside_stem_stays():
    """The gate-valve case (X8074): its stem is a 5-unit segment too, but it sits inside
    the body, so it is part of the symbol."""
    root = _root()
    ext.isolate(root, "V1")
    paths = _paths(_group(root, "V1"))
    assert "m 25,10 v 5" in paths  # stem kept
    assert "m 20,12.5 h -5" not in paths and "M 30,12.5 H 35" not in paths  # stubs gone
    assert len(paths) == 2


def test_keep_stubs_renders_everything():
    root = _root()
    ext.isolate(root, "V1", keep_stubs=True)
    assert len(_paths(_group(root, "V1"))) == 4


def test_a_symbol_made_only_of_bars_keeps_them():
    """Flange (511): no body to be outside of — the bars ARE the symbol."""
    root = _root()
    ext.isolate(root, "F1")
    assert len(_paths(_group(root, "F1"))) == 2


def test_dashed_segments_become_explicit_dashes():
    root = _root()
    ext.isolate(root, "D1")
    dashed = _paths(_group(root, "D1"))[0]
    # 20 units of "4 on, 4 off" -> dashes at 0-4, 8-12, 16-20
    assert dashed.count("M ") == 3
    assert "M 100.00000,10.00000 L 104.00000,10.00000" in dashed
    style = next(e for e in _group(root, "D1").iter() if e.get("d") == dashed).get("style")
    assert "stroke-dasharray:none" in style


# ------------------------------------------------------------------ #
# render
# ------------------------------------------------------------------ #


def test_render_crops_to_ink_on_white():
    root = _root()
    ext.isolate(root, "V1")
    img = ext.render(ET.tostring(root), dpi=150, pad=4)
    arr = np.asarray(img)
    assert img.mode == "RGB"
    assert arr[0, 0].tolist() == [255, 255, 255]
    assert arr.min() < 100
    # Without stubs the symbol is the 10x5-unit bow-tie (+ padding), not 20 units wide.
    assert img.width < 2.6 * img.height


def test_rendering_with_stubs_is_wider():
    a, b = _root(), _root()
    ext.isolate(a, "V1")
    ext.isolate(b, "V1", keep_stubs=True)
    without = ext.render(ET.tostring(a), dpi=150, pad=4)
    with_stubs = ext.render(ET.tostring(b), dpi=150, pad=4)
    assert with_stubs.width > without.width * 1.5


def test_extract_writes_every_vocab_row_and_copies_overrides(tmp_path):
    svg_dir = tmp_path / "DWG"
    svg_dir.mkdir()
    (svg_dir / "BK-2020-04-PID-1-SHT1.svg").write_text(_SVG)
    vocab = tmp_path / "vocab.csv"
    vocab.write_text("reg_number,desc\nV1,valve\nD1,dashed\nF1,flange\n")
    old = tmp_path / "old_F1.png"
    from PIL import Image

    Image.new("RGB", (7, 7), "white").save(old)

    rows = ext.extract(svg_dir, vocab, tmp_path / "out", dpi=150, copy_from_old={"F1": old})
    assert [r["reg_number"] for r in rows] == ["V1", "D1", "F1"]
    assert rows[2]["source"] == "old_pdf_crop"
    assert Image.open(tmp_path / "out" / "F1.png").size == (7, 7)
    assert (tmp_path / "out" / "symbols.csv").read_text().splitlines()[0] == (
        "reg_number,desc,source,file"
    )


def test_extract_fails_loudly_on_a_missing_symbol(tmp_path):
    svg_dir = tmp_path / "DWG"
    svg_dir.mkdir()
    (svg_dir / "BK-2020-04-PID-1-SHT1.svg").write_text(_SVG)
    vocab = tmp_path / "vocab.csv"
    vocab.write_text("reg_number,desc\nV1,valve\nMISSING,x\n")
    with pytest.raises(RuntimeError, match="MISSING"):
        ext.extract(svg_dir, vocab, tmp_path / "out", dpi=150)
