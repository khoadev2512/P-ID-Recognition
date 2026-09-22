"""Unit tests for utils.classmap: load_classmap + ClassMap accessors.

Uses a tiny synthetic classes.yaml (schema mirrors configs/classes.example.yaml)
written to a tmp file — no dependency on the real data/ tree.
"""

from __future__ import annotations

import pytest
import yaml

from utils.classmap import load_classmap

TINY_CLASSES = {
    "coarse": ["valve", "instrument_bubble", "pump"],
    "fine": {
        "gate_valve": "valve",
        "globe_valve": "valve",
        "ball_valve": "valve",
        "pressure_transmitter": "instrument_bubble",
        "centrifugal_pump": "pump",
    },
    "fgc_groups": ["valve"],
    "rare_fine": ["ball_valve"],
}


def _write_classes(tmp_path, data: dict) -> str:
    path = tmp_path / "classes.yaml"
    with path.open("w") as f:
        # sort_keys=False: ClassMap.fine_classes order is defined by the `fine:`
        # block's insertion order in the YAML file, so the test fixture must not let
        # yaml.safe_dump silently alphabetize it.
        yaml.safe_dump(data, f, sort_keys=False)
    return path


def test_fine_classes_order_matches_yaml(tmp_path):
    path = _write_classes(tmp_path, TINY_CLASSES)
    cm = load_classmap(path)
    assert cm.fine_classes == [
        "gate_valve",
        "globe_valve",
        "ball_valve",
        "pressure_transmitter",
        "centrifugal_pump",
    ]


def test_coarse_and_fine_ids_are_stable_ints(tmp_path):
    path = _write_classes(tmp_path, TINY_CLASSES)
    cm = load_classmap(path)

    assert cm.coarse_id("valve") == 0
    assert cm.coarse_id("instrument_bubble") == 1
    assert cm.coarse_id("pump") == 2

    assert cm.fine_id("gate_valve") == 0
    assert cm.fine_id("globe_valve") == 1
    assert cm.fine_id("ball_valve") == 2
    assert cm.fine_id("pressure_transmitter") == 3
    assert cm.fine_id("centrifugal_pump") == 4

    # Stable across repeated calls / a fresh load of the same file.
    cm2 = load_classmap(path)
    assert cm2.coarse_id("pump") == cm.coarse_id("pump")
    assert cm2.fine_id("centrifugal_pump") == cm.fine_id("centrifugal_pump")


def test_needs_fgc(tmp_path):
    path = _write_classes(tmp_path, TINY_CLASSES)
    cm = load_classmap(path)

    assert cm.needs_fgc("valve") is True
    assert cm.needs_fgc("instrument_bubble") is False
    assert cm.needs_fgc("pump") is False


def test_fine_of_group(tmp_path):
    path = _write_classes(tmp_path, TINY_CLASSES)
    cm = load_classmap(path)

    assert cm.fine_of_group("valve") == ["gate_valve", "globe_valve", "ball_valve"]
    assert cm.fine_of_group("instrument_bubble") == ["pressure_transmitter"]
    assert cm.fine_of_group("pump") == ["centrifugal_pump"]
    assert cm.fine_of_group("nonexistent_group") == []


def test_load_classmap_raises_on_unknown_fgc_group(tmp_path):
    bad = dict(TINY_CLASSES)
    bad["fgc_groups"] = ["valve", "not_a_real_coarse_class"]
    path = _write_classes(tmp_path, bad)

    with pytest.raises(ValueError):
        load_classmap(path)


def test_load_classmap_raises_on_missing_required_key(tmp_path):
    bad = {"coarse": ["valve"], "fine": {"gate_valve": "valve"}}  # no fgc_groups
    path = _write_classes(tmp_path, bad)

    with pytest.raises(ValueError):
        load_classmap(path)
