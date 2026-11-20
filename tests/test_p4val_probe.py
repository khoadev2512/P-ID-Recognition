"""scripts/p4val_probe.py (PLAN_PHASE_C §2K.2): representativeness gate and AUROC."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from p4val_probe import auroc, gate  # noqa: E402


def _classes(k):
    return [f"c{i}" for i in range(k)]


def test_gate_passes_with_enough_common_classes_matching_ranks_and_globe_confusion():
    cs = _classes(16)
    val = {c: i / 16 for i, c in enumerate(cs)}
    p4 = {c: i / 20 for i, c in enumerate(cs)}
    n = {c: 25 for c in cs}
    g = gate(val, n, p4, 0.3)
    assert g["pass"] and g["n_common"] == 16 and g["rho"] == pytest.approx(1.0)


def test_gate_fails_on_few_classes_small_counts_low_rho_or_globe():
    cs = _classes(16)
    val = {c: i / 16 for i, c in enumerate(cs)}
    p4 = {c: i / 20 for i, c in enumerate(cs)}
    n = {c: 25 for c in cs}
    assert not gate(val, n, p4, 0.1)["pass"]  # G2
    assert not gate(val, {**n, "c0": 5, "c1": 5}, p4, 0.3)["pass"]  # G0: 14 classes with >= 20
    rev = {c: -v for c, v in p4.items()}
    assert not gate(val, n, rev, 0.3)["pass"]  # G1


def test_auroc_counts_ties_as_half():
    assert auroc(np.array([0.9, 0.8]), np.array([0.1, 0.2])) == 1.0
    assert auroc(np.array([0.5]), np.array([0.5])) == 0.5
