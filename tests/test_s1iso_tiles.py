"""Unit tests for scripts/s1iso_tiles.py (PLAN_PHASE_C §2N rules 3-4): the tile grid matches
`pipeline.tiling`, boxes belong to a tile by the 60% rule, tile kinds, and the keep rule (all
symbol tiles; negatives <= 30% with frame / text first; blank <= 5%; seeded)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import s1iso_tiles as st  # noqa: E402


def test_starts_cover_the_page_like_the_tiling_pipeline():
    assert st.starts(1024) == [0]
    assert st.starts(500) == [0]  # smaller than a tile: one, padded when written
    s = st.starts(3000)
    assert (
        s[0] == 0
        and s[-1] == 3000 - 1024
        and all(b - a <= 819 for a, b in zip(s, s[1:], strict=False))
    )


def test_tile_boxes_keep_boxes_mostly_inside_and_clip_them():
    boxes = [(100, 100, 200, 200), (1000, 100, 1100, 200), (990, 100, 1010, 200)]
    tb = st.tile_boxes(boxes, 0, 0)
    assert (100, 100, 200, 200) in tb
    assert (1000, 100, 1024, 200) not in tb  # 24% inside: belongs to the next tile
    assert (990, 100, 1010, 200) in tb  # 100% inside (ends at 1010)
    assert st.tile_boxes(boxes, 819, 0) == [(181, 100, 281, 200), (171, 100, 191, 200)]


def test_tile_kind_order():
    ink = np.full((3000, 3000), 255, np.uint8)
    ink[500, 100:400] = 0  # a pipe in the first tile
    drawing = (50, 50, 2950, 2950)
    assert st.tile_kind(2, 0, 0, ink, drawing, []) == "sym"
    assert st.tile_kind(0, 0, 0, ink, drawing, []) == "frametext"  # reaches outside the drawing
    assert st.tile_kind(0, 819, 819, ink, drawing, [(900, 900, 950, 920)]) == "frametext"
    assert st.tile_kind(0, 819, 819, ink, drawing, []) == "blank"
    assert st.tile_kind(0, 0, 0, ink, None, []) == "ink"  # DigitizePID: no frame known


def test_select_keeps_all_symbols_and_caps_negatives():
    tiles = [(i, 0, "sym") for i in range(13)] + [(i, 1, "frametext") for i in range(40)]
    tiles += [(i, 2, "ink") for i in range(40)] + [(i, 3, "blank") for i in range(40)]
    kept = st.select(tiles, "page")
    kinds = [t[2] for t in kept]
    assert kinds.count("sym") == 13
    total = 13 / st.SYM_SHARE  # 20
    assert kinds.count("frametext") == round(st.NEG * total) and kinds.count("ink") == 0
    assert kinds.count("blank") == round(st.BLANK * total)
    assert st.select(tiles, "page") == kept  # seeded by the page
    assert st.select([(0, 0, "blank"), (1, 0, "ink")], "empty") == []  # no symbol: nothing


def test_select_takes_ink_tiles_when_frame_tiles_run_out():
    tiles = [(i, 0, "sym") for i in range(13)] + [(0, 1, "frametext")]
    tiles += [(i, 2, "ink") for i in range(40)]
    kinds = [t[2] for t in st.select(tiles, "p")]
    assert kinds.count("frametext") == 1 and kinds.count("ink") == round(st.NEG * 20) - 1


def test_yolo_label_line():
    assert st._yolo([(0, 0, 512, 256)]) == "0 0.250000 0.125000 0.500000 0.250000\n"
