"""Tests for scripts/check_b5.py — the rules the B5 held-out lists must satisfy."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SCRIPTS = str(Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import check_b5  # noqa: E402

_CLASSES = """
coarse: [valve, instrument, flange]
fine:
  Symbol_1: valve
  Symbol_2: valve
  Symbol_3: valve
  Symbol_4: instrument
  Symbol_5: instrument
  Symbol_6: flange
fgc_groups: [valve, instrument]
"""


def _setup(tmp_path, statuses: dict[int, tuple[str, str]], h_in, h_out, h_extra=()):
    (tmp_path / "classes.yaml").write_text(_CLASSES)
    lines = ["symbol_n,status,reg_number"] + [f"{n},{s},{r}" for n, (s, r) in statuses.items()]
    (tmp_path / "symbol_map.csv").write_text("\n".join(lines) + "\n")
    (tmp_path / "vocab.csv").write_text("reg_number,fine_name\nA1,valve_a\nB2,valve_b\n")
    sym = tmp_path / "sym"
    sym.mkdir(exist_ok=True)
    (sym / "A1.png").write_bytes(b"")
    (sym / "B2.png").write_bytes(b"")
    fmt = lambda xs: "[" + ", ".join(xs) + "]"  # noqa: E731
    (tmp_path / "config.yaml").write_text(
        f"oneshot:\n  holdout:\n    h_in: {fmt(h_in)}\n    h_out: {fmt(h_out)}\n"
        f"    h_extra: {fmt(h_extra)}\n"
    )
    return check_b5.check(
        tmp_path / "symbol_map.csv",
        tmp_path / "vocab.csv",
        sym,
        tmp_path / "classes.yaml",
        tmp_path / "config.yaml",
    )


_OK = {
    1: ("ISO", "A1"),
    2: ("ISO", "B2"),
    3: ("UNSURE", ""),
    4: ("OOV", ""),
    5: ("OOV", ""),
    6: ("OOV", ""),
}


def test_a_valid_selection_has_no_errors(tmp_path):
    errors, _ = _setup(tmp_path, _OK, ["Symbol_1"], ["Symbol_4"])
    assert errors == []


def test_leftover_todo_status_is_an_error(tmp_path):
    errors, _ = _setup(tmp_path, {**_OK, 3: ("TODO", "")}, ["Symbol_1"], ["Symbol_4"])
    assert any("Symbol_3" in e and "TODO" in e for e in errors)


@pytest.mark.parametrize(
    "row, needle",
    [(("ISO", ""), "reg_number is empty"), (("ISO", "Z9"), "not in vocab.csv")],
)
def test_iso_rows_need_a_template(tmp_path, row, needle):
    errors, _ = _setup(tmp_path, {**_OK, 2: row}, ["Symbol_1"], ["Symbol_4"])
    assert any(needle in e for e in errors)


def test_unsure_cannot_be_in_either_test_set(tmp_path):
    errors, _ = _setup(tmp_path, _OK, ["Symbol_3"], ["Symbol_4"])
    assert any("h_in Symbol_3" in e for e in errors)
    errors, _ = _setup(tmp_path, _OK, ["Symbol_1"], ["Symbol_3"])
    assert any("h_out Symbol_3" in e for e in errors)


def test_a_family_cannot_be_held_out_entirely(tmp_path):
    """flange has one class: holding it out removes the family from training."""
    errors, _ = _setup(tmp_path, _OK, ["Symbol_1"], ["Symbol_6"])
    assert any("'flange'" in e for e in errors)


def test_two_classes_on_one_iso_symbol_is_a_warning(tmp_path):
    _, warns = _setup(tmp_path, {**_OK, 2: ("ISO", "A1")}, ["Symbol_1"], ["Symbol_4"])
    assert any("A1" in w and "one ISO symbol" in w for w in warns)


def test_variant_needs_a_template_but_is_kept_out_of_h(tmp_path):
    """S2 case: a ball valve drawn differently from ISO X8071 — same reg_number as the
    ISO row, no duplicate warning, and never a P1 test class."""
    rows = {**_OK, 3: ("VARIANT", "A1")}
    errors, warns = _setup(tmp_path, rows, ["Symbol_2"], ["Symbol_4"])
    assert errors == [] and not any("one ISO symbol" in w for w in warns)
    errors, _ = _setup(tmp_path, rows, ["Symbol_3"], ["Symbol_4"])
    assert any("h_in Symbol_3" in e and "VARIANT" in e for e in errors)
    errors, _ = _setup(tmp_path, {**_OK, 3: ("VARIANT", "")}, ["Symbol_1"], ["Symbol_4"])
    assert any("VARIANT but reg_number is empty" in e for e in errors)


# VARIANT rule (PLAN B5b): S5 valve_ball in h_in needs S2 / S13 (same X8071) in h_extra.
_WITH_VARIANT = {**_OK, 3: ("VARIANT", "A1")}  # Symbol_3 draws Symbol_1's symbol differently


def test_h_in_with_its_variant_still_in_training_is_an_error(tmp_path):
    errors, _ = _setup(tmp_path, _WITH_VARIANT, ["Symbol_1"], ["Symbol_4"])
    assert any("h_in Symbol_1" in e and "Symbol_3" in e and "h_extra" in e for e in errors)


def test_h_in_with_its_variant_in_h_extra_is_clean(tmp_path):
    errors, _ = _setup(tmp_path, _WITH_VARIANT, ["Symbol_1"], ["Symbol_4"], ["Symbol_3"])
    assert errors == []


def test_a_variant_of_a_class_not_held_out_needs_nothing(tmp_path):
    """The rule follows h_in: Symbol_2 (B2) is held out, Symbol_3 varies A1 — unrelated."""
    errors, _ = _setup(tmp_path, _WITH_VARIANT, ["Symbol_2"], ["Symbol_4"])
    assert errors == []


@pytest.mark.parametrize(
    "rows, h_extra, needle",
    [
        (_WITH_VARIANT, ["Symbol_5"], "needs VARIANT"),  # an OOV class
        ({**_OK, 3: ("VARIANT", "B2")}, ["Symbol_3"], "not a VARIANT of an h_in class"),
    ],
)
def test_h_extra_must_be_a_variant_of_an_h_in_class(tmp_path, rows, h_extra, needle):
    errors, _ = _setup(tmp_path, rows, ["Symbol_1"], ["Symbol_4"], h_extra)
    assert any(needle in e for e in errors)


def test_h_extra_counts_towards_emptying_a_family(tmp_path):
    rows = {**_OK, 6: ("VARIANT", "A1")}  # flange's only class is a variant of Symbol_1
    errors, _ = _setup(tmp_path, rows, ["Symbol_1"], ["Symbol_4"], ["Symbol_6"])
    assert any("'flange'" in e for e in errors)


def test_the_shipped_lists_pass(tmp_path):
    """configs/config.yaml + docs/symbol_mapping/symbol_map.csv as committed: no error, so
    R1a on Colab never starts from a list that breaks a B5 rule."""
    root = Path(__file__).resolve().parents[1]
    vocab = root / "data/raw/iso_symbols/vocab.csv"
    if not vocab.exists():
        pytest.skip("ISO vocab not present")
    errors, _ = check_b5.check(
        root / "docs/symbol_mapping/symbol_map.csv",
        vocab,
        root / "data/raw/iso_symbols_clean",
        root / "data/canonical/classes.yaml",
        root / "configs/config.yaml",
    )
    assert errors == []
