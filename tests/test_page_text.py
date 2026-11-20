"""Unit tests for utils/page_text.py (V5c): glyph outlines, the made-up P&ID strings, placement
in blank space only, and that text leaves the rest of a V5 page as it was."""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from utils.iso_mapping import load_config
from utils.iso_templates import load_templates
from utils.iso_variants import plan_variant, render_variant, scale_symbols, sized_nodes
from utils.page_svg import Page, rasterize, render_page
from utils.page_text import (
    TextSettings,
    equipment_prefix,
    equipment_tag,
    line_number,
    loop_tags,
    neighbour_role,
    place_texts,
    text_path,
    text_settings,
    text_size,
)
from utils.pid2graph import load_graphml, pipe_runs

ROOT = Path(__file__).resolve().parent.parent
_OPEN100 = ROOT / "data" / "raw" / "pid2graph" / "open100"
_TEMPLATES = ROOT / "data" / "raw" / "template_iso_svg"
needs_open100 = pytest.mark.skipif(
    not (_OPEN100.exists() and _TEMPLATES.exists()), reason="OPEN100 data missing"
)

ST = TextSettings(
    cap=(15.0, 15.0), gap=0.5, margin=0.35, bubbles=1.0, lines=1.0, leaders=1.0,
    equipment=1.0, flags=1.0, min_cap=8.0,
)  # fmt: skip


def _ink_box(svg_body: str, w: int, h: int):
    doc = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
        f'<rect width="{w}" height="{h}" fill="white"/>{svg_body}</svg>'
    )
    ys, xs = np.nonzero(rasterize(doc) < 128)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def test_a_text_path_fills_the_box_text_size_gives():
    w, h = text_size("FIC 1203", 20)
    assert h == pytest.approx(20, abs=1.0)  # the cap height; round digits overshoot it a little
    x0, y0, x1, y1 = _ink_box(f'<path d="{text_path("FIC 1203", 20, 30, 40)}"/>', 300, 120)
    assert (x0, y0) == pytest.approx((30, 40), abs=1.5)
    assert (x1 - x0, y1 - y0) == pytest.approx((w, h), abs=1.5)


def test_a_vertical_text_reads_bottom_to_top_in_a_turned_box():
    w, h = text_size('2"-P-1203-A1', 15, vertical=True)
    ww, hh = text_size('2"-P-1203-A1', 15)
    assert (w, h) == pytest.approx((hh, ww))
    x0, y0, x1, y1 = _ink_box(
        f'<path d="{text_path(chr(50) + "-P", 15, 20, 30, vertical=True)}"/>', 80, 120
    )
    assert x0 >= 19 and y0 >= 29  # inside the box that starts at (20, 30)


def test_made_up_strings_follow_the_p_and_id_conventions():
    rng = random.Random(1)
    for _ in range(50):
        assert re.fullmatch(r'[\d/-]+"-[A-Z]+-4\d{3}-[A-E]\d[A]?(-[HPC])?', line_number(rng, 4))
        assert re.fullmatch(r"P-4\d\d[AB]?", equipment_tag(rng, "P", 4))
    assert equipment_prefix("pump_gear") == "P" and equipment_prefix("column") == "C"
    assert equipment_prefix("vessel_dished_ends") == "V"
    assert equipment_prefix("vessel_dished_ends", column=True) == "C"
    assert (
        equipment_prefix("compressor_vacuum") == "K" and equipment_prefix("hx_double_pipe") == "E"
    )
    assert equipment_prefix("valve_gate") is None and equipment_prefix(None) is None


def test_what_a_bubble_is_joined_to_is_read_from_the_symbol():
    assert neighbour_role("bubble", None) == "bubble"
    assert neighbour_role("iso", "valve_control") == "control"
    assert neighbour_role("iso", "safety_valve_spring_angle") == "safety"
    assert neighbour_role("iso", "valve_gate") == "valve"
    assert neighbour_role("vessel", "vessel_dished_ends") == "vessel"
    assert neighbour_role(None, None) == "pipe" and neighbour_role("iso", "reducer") == "pipe"


def test_bubbles_of_one_loop_share_variable_and_number_and_fit_what_they_are_on():
    links = {
        "cv": [("v1", "control"), ("tic", "bubble")],  # a control valve's bubble ...
        "tic": [("cv", "bubble")],  # ... and the controller joined to it
        "tt": [("x1", "pipe"), ("tic2", "bubble")],  # a transmitter on a line ...
        "tic2": [("tt", "bubble")],  # ... and its readout
        "psv": [("v2", "safety")],
        "hv": [("v3", "valve")],
        "pos": [("v3", "valve")],  # a second bubble on the same valve: the same loop
        "lone": [],
    }
    for seed in range(20):
        t = loop_tags(links, random.Random(seed), 6)
        var = t["cv"][0][0]
        assert t["cv"][0] in (f"{var}V", f"{var}CV") and t["tic"] == (f"{var}IC", t["cv"][1])
        v2 = t["tt"][0][0]
        assert t["tt"][1] == t["tic2"][1] and t["tic2"][0][0] == v2 and t["tt"][0][1] in "TEIGW"
        assert t["psv"][0] == "PSV"
        assert t["hv"][1].rstrip("AB") == t["pos"][1].rstrip("AB")  # one loop through valve v3
        assert {t["hv"][0], t["pos"][0]} & {"HV", "XV"}  # one of them is the valve's own tag
        loops = {t[b][1].rstrip("AB") for b in ("cv", "tt", "psv", "hv", "lone")}
        assert len(loops) == 5 and all(n.startswith("6") for n in loops)  # one number per loop


def test_two_bubbles_with_the_same_letters_in_one_loop_get_a_and_b():
    links = {"a": [("v", "safety")], "b": [("v", "safety")]}  # two relief valve bubbles
    t = loop_tags(links, random.Random(0), 3)
    assert {t["a"][0], t["b"][0]} == {"PSV"} and {t["a"][1][-1], t["b"][1][-1]} == {"A", "B"}


def test_text_settings_are_read_from_the_config_and_checked():
    cfg = load_config()
    st = text_settings(cfg)
    assert st is not None and 0 < st.min_cap <= st.cap[0] <= st.cap[1]
    assert text_settings({k: v for k, v in cfg.items() if k != "text"}) is None
    bad = {**cfg, "text": {**cfg["text"], "min_cap_px": 99}}
    with pytest.raises(ValueError, match="min_cap"):
        text_settings(bad)


def _canvas():
    """A pipe (y = 100, x 20..580) and a bubble (centre (300, 250), r 40) as ink on a page."""
    img = np.full((400, 600), 255, np.uint8)
    cv2.line(img, (20, 100), (580, 100), 0, 3)
    cv2.circle(img, (300, 250), 40, 0, 3)
    return img < 250


def test_labels_go_only_where_the_page_is_blank():
    ink = _canvas()
    svg, labels, log = place_texts(
        ink, 3.0, random.Random(0), ST, {"b": ((300.0, 250.0), 40.0)}, {}, {},
        [[(20.0, 100.0), (580.0, 100.0)]], {},
    )  # fmt: skip
    kinds = {lb.kind for lb in labels}
    assert kinds == {"bubble", "line"} and log["text_bubble_placed"] == 1
    bubble = next(lb for lb in labels if lb.kind == "bubble")
    x0, y0, x1, y1 = bubble.box
    assert all(np.hypot(x - 300, y - 250) < 40 for x in (x0, x1) for y in (y0, y1))
    line = next(lb for lb in labels if lb.kind == "line")
    assert line.box[3] < 100 or line.box[1] > 100  # beside the pipe, not on it
    # drawn, the text touches none of the ink (one stroke of margin)
    doc = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="600" height="400" viewBox="0 0 600 400">'
        f'<rect width="600" height="400" fill="white"/>{svg}</svg>'
    )
    text = cv2.dilate((rasterize(doc) < 128).astype(np.uint8), np.ones((3, 3), np.uint8))
    assert not (text.astype(bool) & ink).any()


def test_a_label_with_no_blank_place_is_left_out():
    ink = np.ones((400, 600), bool)  # all ink
    svg, labels, log = place_texts(
        ink, 3.0, random.Random(0), ST, {"b": ((300.0, 250.0), 40.0)}, {}, {},
        [[(20.0, 100.0), (580.0, 100.0)]], {},
    )  # fmt: skip
    assert svg == "" and labels == []
    assert log["text_bubble_skipped"] == 1 and log["text_line_skipped"] == 1


@pytest.fixture(scope="module")
def sheet3():
    cfg = load_config()
    templates = load_templates(cfg["template_dir"], cfg["taxonomy_csv"], cfg["scope"])
    sheet = load_graphml(_OPEN100 / "3.graphml", Image.open(_OPEN100 / "3.png").size)
    return cfg, templates, sheet, pipe_runs(sheet)


@needs_open100
def test_text_leaves_everything_else_on_the_page_as_it_was(sheet3):
    cfg, templates, sheet, runs = sheet3
    draft = plan_variant(sheet, runs, templates, cfg, 3, 0)
    r = render_variant(draft, sheet, runs, cfg)  # V6: with its "text" seed
    sized = scale_symbols(sheet, draft.size_factor, sized_nodes(draft.assignments))
    plain = render_page(r.sheet, r.runs, draft.assignments, r.cfg, sized, touch_source=sheet)
    page = r.page
    assert page.text and plain.text == "" and plain.texts == []
    # pipes, symbols and their COCO boxes are the same with or without text
    assert (page.pipes, page.symbols, page.annotations) == (
        plain.pipes, plain.symbols, plain.annotations,
    )  # fmt: skip
    assert {t["kind"] for t in page.texts} >= {"bubble", "equipment", "flag"}
    assert render_variant(draft, sheet, runs, cfg).page.text == page.text  # same seed, same text
    sys.path.insert(0, str(ROOT / "scripts"))
    import check_g1

    _, ok = check_g1.evaluate(page, r.runs, sheet, r.cfg)
    assert all(ok.values())  # criterion 8 (text off the ink) with the others


def test_criterion_8_catches_text_on_ink():
    sys.path.insert(0, str(ROOT / "scripts"))
    import check_g1

    pipes = '<g stroke="black" stroke-width="3"><polyline points="10,50 190,50"/></g>'
    on = f'<g fill="black"><path d="{text_path("FT", 12, 60, 44)}"/></g>'  # across the pipe
    off = f'<g fill="black"><path d="{text_path("FT", 12, 60, 10)}"/></g>'  # well above it
    for text, touched in ((on, True), (off, False)):
        page = Page(200, 100, pipes, "", text=text, texts=[{"kind": "line"}])
        p = rasterize(page.layer("pipes")) < 128
        s = rasterize(page.layer("symbols")) < 128
        assert (check_g1.text_hits(page, p, s) > 0) == touched
