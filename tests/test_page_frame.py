"""Unit tests for utils/page_frame.py (PLAN_PHASE_C §2M item 7): the sheet frame round a page.
The drawing must come out of the frame pixel for pixel, moved by the offset; every recorded
coordinate moves with it; nothing of the frame lies on the drawing; v1 (no `frame`) is untouched."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from utils.iso_mapping import load_config
from utils.iso_templates import load_templates
from utils.iso_variants import plan_variant, render_variant
from utils.page_frame import (
    FrameSettings,
    add_frame,
    draw_frame,
    frame_layout,
    frame_settings,
    make_content,
    shift_runs,
    shift_sheet,
)
from utils.page_svg import Page, rasterize
from utils.pid2graph import Node, PipeRun, Sheet, load_graphml, pipe_runs

ROOT = Path(__file__).resolve().parent.parent
_OPEN100 = ROOT / "data" / "raw" / "pid2graph" / "open100"
_TEMPLATES = ROOT / "data" / "raw" / "template_iso_svg"
needs_open100 = pytest.mark.skipif(
    not (_OPEN100.exists() and _TEMPLATES.exists()), reason="OPEN100 data missing"
)

ST = FrameSettings(
    cap_px=(13, 19), column=(0.16, 0.22), notes=(4, 18), revisions=(1, 6), rows=(6, 8),
    p_references=0.5, p_centring=0.5,
)  # fmt: skip


def _page() -> Page:
    """A 600 x 400 page with a box symbol, a pipe and a label."""
    return Page(
        600, 400,
        pipes='<polyline points="50,200 550,200" fill="none" stroke="black" stroke-width="3"/>',
        symbols='<rect x="280" y="180" width="40" height="40" fill="none" stroke="black" '
        'stroke-width="3"/>',
        annotations=[{"id": "n1", "kind": "iso", "bbox": [278.5, 178.5, 321.5, 221.5]}],
        ports=[{"id": "n1", "kind": "iso", "x": 280.0, "y": 200.0, "loose": False}],
        stems=[{"id": "n1", "bubble": "b", "points": [(300.0, 180.0), (300.0, 100.0)]}],
        text='<path d="M10 10L20 10L20 20Z" fill="black"/>',
        texts=[{"kind": "line", "box": [10, 10, 20, 20]}],
    )  # fmt: skip


def test_settings_off_unless_enabled():
    assert frame_settings({}) is None
    assert frame_settings({"frame": {"enabled": False}}) is None
    cfg = load_config()
    assert cfg["frame"]["enabled"] is False  # v1 is drawn without a frame
    st = frame_settings({"frame": {**cfg["frame"], "enabled": True}})
    assert st is not None and st.rows == (6, 8)


def test_layout_keeps_the_drawing_clear_of_the_frame():
    lay = frame_layout(6000, 4000, 0.2)
    cx0, cy0, cx1, cy1 = lay.content
    assert (cx1 - cx0, cy1 - cy0) == (6000, 4000)
    assert lay.offset == (cx0, cy0) and all(isinstance(v, int) for v in lay.offset)
    bx0, by0, bx1, by1 = lay.border
    assert bx0 < cx0 and by0 < cy0 and cy1 < by1  # a gap round the drawing inside the border
    assert cx1 < lay.column[0] < lay.column[2] == bx1  # the column beside it, not over it
    assert lay.trim[0] < bx0 and lay.trim[2] > bx1 and lay.trim[2] < lay.width


def test_content_is_reproducible_and_made_up():
    a, b = make_content(ST, random.Random(5)), make_content(ST, random.Random(5))
    assert a == b
    assert a.title[0] == "PIPING AND INSTRUMENTATION DIAGRAM"
    assert 1 <= len(a.revisions) <= 6 and 4 <= len(a.notes) <= 18
    assert [n.split(".")[0] for n in a.notes] == [str(i + 1) for i in range(len(a.notes))]
    revs = [r[0] for r in a.revisions]
    assert revs == sorted(revs, reverse=True)  # newest on top
    assert make_content(ST, random.Random(6)) != a


def test_frame_ink_never_on_the_drawing():
    for seed in range(4):
        fr = draw_frame(900, 600, 3.0, ST, random.Random(seed))
        lay = fr.layout
        doc = Page(lay.width, lay.height, "", "", frame=fr.svg).layer("frame")
        ink = rasterize(doc) < 250
        x0, y0, x1, y1 = (int(v) for v in lay.content)
        assert not ink[y0:y1, x0:x1].any()
        assert ink.sum() > 0 and fr.notes_written >= 1


def test_add_frame_moves_drawing_and_coordinates_together():
    page = _page()
    framed, fr = add_frame(page, 3.0, ST, random.Random(1))
    dx, dy = fr.layout.offset
    assert (framed.width, framed.height) == (fr.layout.width, fr.layout.height)
    assert framed.annotations[0]["bbox"] == [278.5 + dx, 178.5 + dy, 321.5 + dx, 221.5 + dy]
    assert (framed.ports[0]["x"], framed.ports[0]["y"]) == (280.0 + dx, 200.0 + dy)
    assert framed.stems[0]["points"][1] == (300.0 + dx, 100.0 + dy)
    assert framed.texts[0]["box"] == [10 + dx, 10 + dy, 20 + dx, 20 + dy]
    assert page.annotations[0]["bbox"][0] == 278.5  # the input page is left as it was
    # each layer is the same picture, moved by a whole number of pixels
    for layer in ("pipes", "symbols", "text"):
        before = rasterize(page.layer(layer)).astype(int)
        after = rasterize(framed.layer(layer))[dy : dy + 400, dx : dx + 600]
        assert np.abs(after - before).max() <= 32
    assert framed.frame and framed.frame in framed.svg and not page.frame


def test_shift_sheet_and_runs():
    sheet = Sheet("s", 100, 80, {"a": Node("a", "valve", 1, 2, 3, 4)}, [])
    moved = shift_sheet(sheet, 10, 20, 300, 200)
    n = moved.nodes["a"]
    assert (n.xmin, n.ymin, n.xmax, n.ymax, moved.width, moved.height) == (11, 22, 13, 24, 300, 200)
    run = PipeRun("a", "b", ((0.0, 0.0), (5.0, 0.0)), ())
    assert shift_runs([run], 1, 2)[0].points == ((1.0, 2.0), (6.0, 2.0))


@needs_open100
def test_render_variant_with_frame_is_the_unframed_drawing_moved():
    cfg = load_config()
    templates = load_templates(cfg["template_dir"], cfg["taxonomy_csv"], cfg["scope"])
    sheet = load_graphml(_OPEN100 / "3.graphml", Image.open(_OPEN100 / "3.png").size)
    runs = pipe_runs(sheet)
    draft = plan_variant(sheet, runs, templates, cfg, 3, 0)
    plain = render_variant(draft, sheet, runs, cfg)
    assert plain.frame is None and plain.page.frame == ""
    cfg_f = {**cfg, "frame": {**cfg["frame"], "enabled": True}}
    framed = render_variant(draft, sheet, runs, cfg_f)
    dx, dy = framed.frame.layout.offset
    h, w = plain.page.height, plain.page.width
    sym0 = rasterize(plain.page.layer("symbols"))
    sym1 = rasterize(framed.page.layer("symbols"))
    # the same picture moved by whole pixels; nested transforms may shift anti-aliasing a little
    diff = np.abs(sym1[dy : dy + h, dx : dx + w].astype(int) - sym0)
    assert diff.max() <= 32 and (diff > 0).sum() < 1e-3 * (sym0 < 128).sum()
    a0 = {a["id"]: a["bbox"] for a in plain.page.annotations}
    for a in framed.page.annotations:
        b = a0[a["id"]]
        assert a["bbox"] == pytest.approx([b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy], abs=0.011)
    nid = next(iter(plain.sheet.nodes))
    assert framed.sheet.nodes[nid].xmin == pytest.approx(plain.sheet.nodes[nid].xmin + dx)
    assert framed.runs[0].points[0][1] == pytest.approx(plain.runs[0].points[0][1] + dy)
    # its own seed: drawing again gives the same frame
    assert render_variant(draft, sheet, runs, cfg_f).page.svg == framed.page.svg


def test_v2_overlay_changes_only_the_fixes_and_the_frame(monkeypatch):
    from utils.iso_mapping import CONFIG_ENV

    monkeypatch.delenv(CONFIG_ENV, raising=False)
    monkeypatch.delenv("PID_DATA_ROOT", raising=False)
    v1 = load_config()
    v2 = load_config(ROOT / "configs" / "render_iso_v2.yaml")
    monkeypatch.setenv(CONFIG_ENV, str(ROOT / "configs" / "render_iso_v2.yaml"))
    assert load_config() == v2  # the env var picks the overlay
    assert v2["frame"] == {**v1["frame"], "enabled": True}
    assert v2["page"] == {
        **v1["page"], "place_by_axis": True, "drop_hanging": True, "clip_leaders": True,
    }  # fmt: skip
    assert v2["layout"] == {**v1["layout"], "straighten_pipes": True}
    assert v2["arrow_chains"] is True and "arrow_chains" not in v1
    assert v2["export"]["data_root"] == "data/iso_open100_v2" != v1["export"]["data_root"]
    same = {"rules", "variants", "degrade", "flanges", "agitators", "text", "size_class", "scope"}
    assert all(v1[k] == v2[k] for k in same)  # sources, mapping, budget, seeds, degradation


def test_drawing_box_clean_and_degraded():
    from pipeline.render_iso import drawing_box

    assert drawing_box({}, None) is None
    meta = {"frame": {"content": [100, 50, 700, 450]}}
    assert drawing_box(meta, None) == [100, 50, 700, 450]
    moved = drawing_box(meta, {"affine": [[1, 0, 5], [0, 1, -3]]})
    assert moved == [105, 47, 705, 447]


def test_dates_exist():
    from utils.page_frame import _date

    assert _date(random.Random(0), 360 * 3 + 30 + 29, 2) == "2018-02-28"
    assert _date(random.Random(0), 0, 1) == "01-JAN-2015"
