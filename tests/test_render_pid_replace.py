"""Unit tests for scripts/render_pid_replace.py (SynthPID-style in-place symbol swap)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

pytest.importorskip("PIL")
pytest.importorskip("numpy")
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

_SCRIPTS = str(Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import render_pid_replace as rr  # noqa: E402


def _iso_template(fine, coarse, fgc):
    a = np.full((20, 20, 3), 255, dtype=np.uint8)
    a[6:14, 6:14] = 0
    return rr.IsoTemplate(fine, coarse, fgc, rr._white_to_transparent(Image.fromarray(a)))


def test_coarse_map_covers_all_digitizepid_coarses():
    # every coarse used in symbol_map.csv must have a mapping, or symbols silently vanish
    smap = Path("docs/symbol_mapping/symbol_map.csv")
    if not smap.exists():
        pytest.skip("symbol_map.csv not present")
    coarses = {r["coarse"] for r in csv.DictReader(smap.open())}
    assert coarses <= set(rr._COARSE_MAP), f"unmapped: {coarses - set(rr._COARSE_MAP)}"


def test_instrument_maps_to_agitator():
    assert rr._COARSE_MAP["instrument"] == "agitator"


def test_piping_fines_map_together():
    assert rr._COARSE_MAP["reducer"] == "piping"
    assert rr._COARSE_MAP["flange"] == "piping"
    assert rr._COARSE_MAP["flow_direction"] == "piping"


def test_build_categories_has_fgc_flag():
    iso = {
        "valve": [_iso_template("valve_gate", "valve", True)],
        "piping": [_iso_template("reducer", "piping", False)],
    }
    cats, name_to_id = rr.build_categories(iso)
    by_coarse = {c["supercategory"]: c for c in cats}
    assert by_coarse["valve"]["fgc"] is True
    assert by_coarse["piping"]["fgc"] is False
    assert name_to_id["valve_gate"] in {c["id"] for c in cats}


def test_build_categories_skips_unreachable_coarse():
    # a coarse not in _COARSE_MAP's values must not become a category
    iso = {"motor": [_iso_template("motor_electric", "motor", False)]}  # motor isn't a target
    cats, _ = rr.build_categories(iso)
    assert cats == []


def test_replace_symbols_swaps_and_annotates():
    img = Image.new("RGB", (200, 200), "white")
    boxes = [(50, 50, 90, 110)]  # one symbol box
    coarses = ["valve"]  # DigitizePID coarse -> ISO valve
    iso = {"valve": [_iso_template("valve_gate", "valve", True)]}
    _, name_to_id = rr.build_categories(iso)
    rng = __import__("random").Random(0)
    canvas, anns = rr.replace_symbols_on_image(img, boxes, coarses, iso, name_to_id, rng)
    assert len(anns) == 1
    assert anns[0]["category_id"] == name_to_id["valve_gate"]
    bx = anns[0]["bbox"]
    # the ISO symbol sits inside the original box
    assert 50 <= bx[0] and bx[0] + bx[2] <= 90
    assert 50 <= bx[1] and bx[1] + bx[3] <= 110
    # something was pasted (canvas no longer all white in that region)
    assert (np.array(canvas)[50:110, 50:90] != 255).any()


def test_replace_skips_unmapped_coarse():
    img = Image.new("RGB", (200, 200), "white")
    # 'instrument' maps to agitator, but if agitator templates are absent -> skip, no crash
    iso = {"valve": [_iso_template("valve_gate", "valve", True)]}  # no agitator
    _, name_to_id = rr.build_categories(iso)
    canvas, anns = rr.replace_symbols_on_image(
        img, [(10, 10, 40, 40)], ["instrument"], iso, name_to_id, __import__("random").Random(0)
    )
    assert anns == []  # instrument -> agitator, none available -> nothing placed
