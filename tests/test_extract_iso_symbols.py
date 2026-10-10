"""Unit tests for scripts/extract_iso_symbols.py — the ISO 10628-2 symbol extractor.

Only the pure text-parsing / naming helpers are unit-tested (REG#/DESC pairing, safe
filenames). The PDF-rendering path (extract / vector bbox) needs the real PDF + pymupdf
and is exercised by running the script, not here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("pymupdf")

_SCRIPTS = str(Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import extract_iso_symbols as ext  # noqa: E402


def test_safe_name_keeps_valid_chars():
    assert ext._safe_name("X8074") == "X8074"
    assert ext._safe_name("C0001") == "C0001"


def test_safe_name_sanitizes_separators():
    # REG#s with slashes/spaces become filesystem-safe (no path separators leak through).
    assert "/" not in ext._safe_name("X80/74")
    assert ext._safe_name("X80/74") == "X80_74"


def test_reg_regex_matches_forms():
    # the module's REG# regex should capture both 'REG#:X8074' and 'REG# : 2101' forms
    assert ext._REG_RE.search("REG#:X8074").group(1) == "X8074"
    assert ext._REG_RE.search("REG# : 2101").group(1) == "2101"


def test_desc_regex_captures_text():
    assert ext._DESC_RE.search("DESC:Valve, gate type").group(1) == "Valve, gate type"


class _FakeLine:
    def __init__(self, text, x, y):
        self._text = text
        self.bbox = (x, y, x + 100, y + 10)


class _FakePage:
    """Minimal stand-in exposing get_text('dict') with REG#/DESC lines in a column."""

    def __init__(self, lines):
        self._lines = lines

    def get_text(self, _mode):
        return {
            "blocks": [
                {"lines": [{"bbox": ln.bbox, "spans": [{"text": ln._text}]}] } for ln in self._lines
            ]
        }


def test_page_label_items_pairs_reg_and_desc():
    # a REG# at (260,60) with its DESC just below (260,78) -> one paired item
    page = _FakePage([
        _FakeLine("REG#:X8074", 260, 60),
        _FakeLine("DESC:Valve, gate type", 260, 78),
        _FakeLine("REG#:2101", 827, 60),
        _FakeLine("DESC:Valve (general)", 827, 78),
    ])
    items = ext._page_label_items(page)
    by_reg = {it["reg"]: it["desc"] for it in items}
    assert by_reg["X8074"] == "Valve, gate type"
    assert by_reg["2101"] == "Valve (general)"


def test_page_label_items_handles_missing_desc():
    # a REG# with no matching DESC still yields an item (empty desc), not a crash
    page = _FakePage([_FakeLine("REG#:X9999", 260, 60)])
    items = ext._page_label_items(page)
    assert len(items) == 1
    assert items[0]["reg"] == "X9999"
    assert items[0]["desc"] == ""
