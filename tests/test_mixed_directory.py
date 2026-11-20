"""scripts/mixed_directory.py (PLAN_PHASE_C §2F.1): the four directories."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mixed_directory import build_directory  # noqa: E402


def _setup():
    iso = torch.eye(4)
    iso_names = ["valve_gate", "valve_butterfly", "reducer", "tank"]
    ex = {"S1": torch.full((4,), 1.0), "S6": torch.full((4,), 2.0), "S11": torch.full((4,), 3.0)}
    meaning = {"S1": "valve_gate", "S6": "valve_butterfly", "S11": "S11"}
    return iso, iso_names, ex, meaning


def test_d0_is_the_iso_directory_untouched():
    iso, names, ex, meaning = _setup()
    p, n = build_directory("D0", iso, names, ex, meaning, ["S6"])
    assert torch.equal(p, iso) and n == names


def test_d1_adds_only_the_variant_exemplars_named_by_meaning():
    iso, names, ex, meaning = _setup()
    p, n = build_directory("D1", iso, names, ex, meaning, ["S6", "S8"])  # S8 has no exemplar
    assert n == names + ["valve_butterfly"] and len(p) == 5
    assert torch.equal(p[4], ex["S6"])


def test_d2_adds_every_exemplar_and_d3_keeps_only_them():
    iso, names, ex, meaning = _setup()
    p2, n2 = build_directory("D2", iso, names, ex, meaning, ["S6"])
    p3, n3 = build_directory("D3", iso, names, ex, meaning, ["S6"])
    assert n2 == names + ["valve_gate", "S11", "valve_butterfly"]  # sorted by class name
    assert n3 == ["valve_gate", "S11", "valve_butterfly"] and torch.equal(p2[4:], p3)
    with pytest.raises(ValueError):
        build_directory("D9", iso, names, ex, meaning, [])
