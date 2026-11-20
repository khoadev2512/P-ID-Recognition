"""Unit tests for utils/diversity.py and utils/iso_variants.py (V6): hashes, content dedup,
sub-seeds, the `variants` config, the quality filter and a small end-to-end run on one OPEN100
sheet."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from utils.diversity import ContentDeduper, Deduper, ahash, content_distance, dhash, hamming
from utils.iso_mapping import Assignment, load_config
from utils.iso_templates import load_templates
from utils.iso_variants import (
    STAGES,
    assignment_signature,
    derive_seed,
    generate_sheet,
    node_choices,
    plan_variant,
    quality_failures,
    render_variant,
    scale_symbols,
    sized_nodes,
    variant_settings,
)
from utils.page_svg import rasterize
from utils.pid2graph import Node, Sheet, load_graphml, pipe_runs

ROOT = Path(__file__).resolve().parent.parent
_OPEN100 = ROOT / "data" / "raw" / "pid2graph" / "open100"
_TEMPLATES = ROOT / "data" / "raw" / "template_iso_svg"
needs_open100 = pytest.mark.skipif(
    not (_OPEN100.exists() and _TEMPLATES.exists()), reason="OPEN100 data missing"
)


def _blank(n=64):
    return np.full((n, n), 255, np.uint8)


def test_hamming_counts_differing_bits():
    assert hamming(0b1011, 0b0010) == 2
    assert hamming(5, 5) == 0


def test_ahash_known_images():
    img = _blank()
    img[:, :32] = 0  # left half black
    # 4x4 grid: the two left columns have ink above the mean -> bits 1100 per row
    assert ahash(img, 4) == int("1100" * 4, 2)
    assert ahash(_blank(), 4) == 0  # flat image: nothing above the mean


def test_dhash_known_images():
    img = _blank()
    img[:, :32] = 0
    # 5 cells per row over 64 px: ink 255, 255, ~half, 0, 0 -> "darker than the right
    # neighbour" is 0, 1, 1, 0 per row
    assert dhash(img, 4) == int("0110" * 4, 2)
    assert dhash(img[:, ::-1].copy(), 4) == 0  # black on the right: nothing falls to the right
    assert dhash(_blank(), 4) == 0


def test_hashes_ignore_size_and_see_thin_lines():
    small = _blank(64)
    big = _blank(256)
    small[10, :] = 0
    big[40:42, :] = 0  # the same line, drawn thicker at 4x
    assert ahash(small, 8) == ahash(big, 8)
    assert ahash(small, 8) != ahash(_blank(64), 8) or dhash(small, 8) != dhash(_blank(64), 8)


def test_deduper_needs_both_hashes_close():
    d = Deduper(min_distance=4)
    assert d.offer(0b0000, 0b0000) == (True, None)
    ok, near = d.offer(0b0001, 0b0011)  # 1 and 2 bits away: duplicate
    assert (ok, near) == (False, 2)
    ok, near = d.offer(0b0001, 0b11111)  # close ahash, far dhash: kept
    assert ok and near == 5
    assert len(d.kept) == 2
    assert d.is_duplicate(0b0001, 0b11111)[0]  # now a duplicate of the one just kept


def test_content_distance_is_the_share_of_differing_nodes():
    a = {f"n{i}": "valve_gate" for i in range(20)}
    b = {**a, "n0": "valve_globe"}
    assert content_distance(a, a) == 0.0 and content_distance({}, {}) == 0.0
    assert content_distance(a, b) == content_distance(b, a) == 0.05
    assert content_distance(a, {**a, "extra": "x"}) == 1 / 21  # a node in one map only differs


def test_content_deduper_drops_near_copies_of_the_symbols_not_of_the_picture():
    a = {f"n{i}": "valve_gate" for i in range(20)}
    d = ContentDeduper(min_node_diff=0.1)
    assert d.is_duplicate(a) == (False, None)
    d.add(a)
    one = {**a, "n0": "valve_globe"}  # 1 of 20 nodes: a near-copy
    assert d.is_duplicate(one) == (True, 0.05)
    # same nodes, same layout, a quarter of them drawn with other symbols: a variant worth keeping
    quarter = {**a, **{f"n{i}": "check_valve_swing" for i in range(5)}}
    assert d.is_duplicate(quarter) == (False, 0.25)
    d.add(quarter)
    assert d.distance({**quarter, "n19": "valve_ball"}) == 0.05  # the closest kept one counts


def test_node_choices_see_which_symbol_not_how_it_is_drawn():
    a = {"n1": Assignment("arrow", "flow_direction", axis="H", forward=True),
         "n2": Assignment("arrow", "flow_direction", axis="V", forward=False)}  # fmt: skip
    # the same symbols in other places: a label histogram sees no change
    swapped = {"n1": a["n2"], "n2": a["n1"]}
    assert content_distance(node_choices(a), node_choices(swapped)) == 1.0
    # the agitator's shaft length is how a symbol is drawn, not which
    n1 = Assignment("arrow", "flow_direction", axis="H", forward=True, clearance=0.6)
    shaft = {**a, "n1": n1}
    assert node_choices(shaft) == node_choices(a)
    assert assignment_signature(shaft) != assignment_signature(a)


def test_derive_seed_is_deterministic_and_separates_stages():
    a = derive_seed(0, 3, 5, "assign")
    assert a == derive_seed(0, 3, 5, "assign")
    assert a == 1341535754545934703  # pinned: stored seeds must keep meaning the same draws
    seeds = {derive_seed(0, 3, 5, s) for s in STAGES}
    assert len(seeds) == len(STAGES)
    others = {derive_seed(1, 3, 5, "assign"), derive_seed(0, 4, 5, "assign"),
              derive_seed(0, 3, 6, "assign")}  # fmt: skip
    assert a not in others and len(others) == 3
    assert 0 <= a < 2**64


def test_variant_settings_reads_config_and_scales_stroke():
    cfg = {
        "layout": {"scale": 6},
        "variants": {
            "count": 5, "max_attempts": 9, "base_seed": 7, "stroke": [2, 4],
            "size": [0.85, 1.15], "hash_size": 8, "min_node_diff": 0.05,
            "quality": {"max_log": {"bubbles_unsettled": 0}},
            "g1": {"drop": [1, 2], "flag": [5]},
        },
    }  # fmt: skip
    s = variant_settings(cfg)
    assert s.stroke == (4.0, 8.0)  # [2, 4] x scale / 3
    assert (s.count, s.max_attempts, s.base_seed, s.hash_size) == (5, 9, 7, 8)
    assert s.min_node_diff == 0.05
    assert s.g1_drop == (1, 2) and s.g1_flag == (5,) and s.max_log == {"bubbles_unsettled": 0}
    bad = json.loads(json.dumps(cfg))
    bad["variants"]["max_attempts"] = 2
    with pytest.raises(ValueError, match="max_attempts"):
        variant_settings(bad)
    bad = json.loads(json.dumps(cfg))
    bad["variants"]["g1"]["flag"] = [1]
    with pytest.raises(ValueError, match="both"):
        variant_settings(bad)
    bad = json.loads(json.dumps(cfg))
    bad["variants"]["min_node_diff"] = 5  # a share, not a node count
    with pytest.raises(ValueError, match="min_node_diff"):
        variant_settings(bad)


def test_repo_config_has_a_valid_variants_block():
    s = variant_settings(load_config())
    assert s.stroke == (2.0, 4.0) and s.size == (0.85, 1.15)
    assert s.count <= s.max_attempts and 0 not in s.g1_drop


def test_quality_failures_lists_counters_over_their_limit():
    log = {"bubbles_unsettled": 2, "unmatched_ports": 0, "loose_ends": 9}
    limits = {"bubbles_unsettled": 0, "unmatched_ports": 0}
    assert quality_failures(log, limits) == ["bubbles_unsettled=2"]
    assert quality_failures(log, {"bubbles_unsettled": 2}) == []
    assert quality_failures({}, limits) == []


def test_signature_changes_with_any_choice():
    a = {"n1": Assignment("arrow", "flow_direction", axis="H", forward=True)}
    b = {"n1": Assignment("arrow", "flow_direction", axis="H", forward=False)}
    c = {"n1": Assignment("arrow", "flow_direction", axis="H", forward=True, clearance=0.4)}
    assert assignment_signature(a) == assignment_signature(dict(a))
    assert len({assignment_signature(x) for x in (a, b, c)}) == 3


def test_scale_symbols_scales_only_the_given_nodes_about_their_centres():
    sheet = Sheet(
        "s", 100, 100,
        {"a": Node("a", "valve", 10, 10, 30, 20), "b": Node("b", "valve", 50, 50, 60, 60)},
        [],
    )  # fmt: skip
    out = scale_symbols(sheet, 1.5, {"a"})
    assert out.nodes["a"].center == (20, 15) and out.nodes["a"].width == 30
    assert out.nodes["b"] == sheet.nodes["b"] and sheet.nodes["a"].width == 20
    assert scale_symbols(sheet, 1.0, {"a"}) is sheet


# --- on a real sheet ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sheet3():
    cfg = load_config()
    cfg["variants"].update(count=3, max_attempts=6)
    templates = load_templates(cfg["template_dir"], cfg["taxonomy_csv"], cfg["scope"])
    sheet = load_graphml(_OPEN100 / "3.graphml", Image.open(_OPEN100 / "3.png").size)
    return cfg, templates, sheet, pipe_runs(sheet)


@needs_open100
def test_plan_variant_is_reproducible_and_stages_are_independent(sheet3):
    cfg, templates, sheet, runs = sheet3
    a = plan_variant(sheet, runs, templates, cfg, 3, 4)
    b = plan_variant(sheet, runs, templates, cfg, 3, 4)
    assert (a.seeds, a.stroke_px, a.size_factor, a.signature) == (
        b.seeds, b.stroke_px, b.size_factor, b.signature,
    )  # fmt: skip
    lo, hi = variant_settings(cfg).stroke
    assert lo <= a.stroke_px <= hi and 0.85 <= a.size_factor <= 1.15
    # another base seed moves every stage; the style draw does not depend on the assignment
    cfg2 = {**cfg, "variants": {**cfg["variants"], "base_seed": 1}}
    c = plan_variant(sheet, runs, templates, cfg2, 3, 4)
    assert all(a.seeds[s] != c.seeds[s] for s in STAGES)
    assert sized_nodes(a.assignments) <= set(sheet.nodes)


@needs_open100
def test_render_variant_uses_its_stroke_and_leaves_cfg_alone(sheet3):
    cfg, templates, sheet, runs = sheet3
    draft = plan_variant(sheet, runs, templates, cfg, 3, 0)
    draft.stroke_px, draft.size_factor = 2.0, 1.0
    thin = render_variant(draft, sheet, runs, cfg)
    draft.stroke_px = 4.0
    thick = render_variant(draft, sheet, runs, cfg)
    assert cfg["page"]["stroke_px"] == 3
    assert thin.cfg["page"]["stroke_px"] == 2.0 and thick.cfg["page"]["stroke_px"] == 4.0
    assert thin.page.annotations[0]["id"] == thick.page.annotations[0]["id"]
    ink = [(rasterize(r.page.svg) < 128).sum() for r in (thin, thick)]
    assert ink[1] > 1.3 * ink[0]  # the same picture, drawn with a thicker pen
    # G1 is given the data's boxes, not the enlarged ones V5 fitted the symbols to
    draft.size_factor = 1.15
    big = render_variant(draft, sheet, runs, cfg)
    assert big.source is sheet


@needs_open100
def test_generate_sheet_writes_kept_variants_and_is_reproducible(sheet3, tmp_path):
    cfg, templates, sheet, runs = sheet3
    recs = generate_sheet(3, sheet, runs, templates, cfg, tmp_path)
    kept = [r for r in recs if r["status"] == "kept"]
    assert len(kept) == 3 and len(recs) <= 6
    assert all(r["reason"] is None for r in kept)
    # each kept variant differs from the earlier kept ones in at least min_node_diff of its nodes;
    # its picture's hash distance is recorded, not filtered on
    st = variant_settings(cfg)
    assert all(r["nearest_content"] >= st.min_node_diff for r in kept[1:])
    assert all(isinstance(r["nearest_hash"], int) for r in kept[1:])
    meta = json.loads((tmp_path / "sheet3" / "v000.json").read_text())
    assert meta["seeds"] == recs[0]["seeds"] and meta["annotations"]
    assert Image.open(tmp_path / "sheet3" / "v000.png").size == (meta["width"], meta["height"])
    again = generate_sheet(3, sheet, runs, templates, cfg, None)
    assert [r["signature"] for r in again] == [r["signature"] for r in recs]
    # the picture is a function of (seeds, parameters) alone
    draft = plan_variant(sheet, runs, templates, cfg, 3, 0)
    svg = render_variant(draft, sheet, runs, cfg).page.svg
    assert svg == (tmp_path / "sheet3" / "v000.svg").read_text()


@needs_open100
def test_generate_sheet_drops_duplicates_and_failed_checks(sheet3):
    cfg, templates, sheet, runs = sheet3
    # a threshold no two assignments reach: every variant after the first is a near-duplicate,
    # dropped before it is drawn
    huge = {**cfg, "variants": {**cfg["variants"], "min_node_diff": 1.0, "max_attempts": 4}}
    recs = generate_sheet(3, sheet, runs, templates, huge, None)
    assert [r["status"] for r in recs] == ["kept", "dropped", "dropped", "dropped"]
    assert {r["reason"] for r in recs[1:]} == {"duplicate"}
    assert all(0 < r["nearest_content"] < 1 for r in recs[1:])
    assert all("render" not in r["seconds"] for r in recs[1:])
    # a G1 check that fails criterion 7 drops (7 is in g1.drop); one that fails a flagged one keeps
    fail = lambda r: {1: True, 7: False}  # noqa: E731
    recs = generate_sheet(3, sheet, runs, templates, cfg, None, fail)
    assert {r["reason"] for r in recs} == {"g1"} and recs[0]["detail"] == "7"
    flag = {**cfg, "variants": {**cfg["variants"], "g1": {"drop": [1], "flag": [7]}}}
    recs = generate_sheet(3, sheet, runs, templates, flag, None, fail)
    assert recs[0]["status"] == "kept" and recs[0]["g1_flags"] == [7]


@needs_open100
def test_generate_sheet_survives_a_render_error(sheet3, monkeypatch):
    cfg, templates, sheet, runs = sheet3
    import utils.iso_variants as iv

    def boom(*a, **k):
        raise RuntimeError("no layout")

    monkeypatch.setattr(iv, "render_page", boom)
    recs = generate_sheet(3, sheet, runs, templates, cfg, None)
    assert {r["reason"] for r in recs} == {"error"} and "no layout" in recs[0]["detail"]


def test_stroke_thickness_reads_fractional_and_even_widths():
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import check_g1

    for w in (2.0, 2.65, 3.0, 4.0):
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="300" '
            'viewBox="0 0 300 300"><rect width="300" height="300" fill="white"/>'
            f'<path d="M20 100.3 H280 M40.4 20 V280" stroke="black" stroke-width="{w}" '
            'fill="none"/></svg>'
        )
        ink = rasterize(svg) < 128
        assert abs(check_g1.stroke_thickness(svg, ink) - w) <= 0.5
