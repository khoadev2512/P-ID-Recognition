"""Smoke tests — verify the package imports and the path layout resolves.

Real logic is TODO; these just guard the scaffold structure.
"""

from pathlib import Path

from pid.common.paths import DataPaths


def test_datapaths_layout():
    p = DataPaths(root=Path("data"))
    assert p.coco_json == Path("data/canonical/annotations.coco.json")
    assert p.classes_yaml == Path("data/canonical/classes.yaml")
    assert p.yolo_coarse == Path("data/derived/yolo_coarse")
    assert p.crops_fgc == Path("data/derived/crops_fgc")


def test_package_imports():
    import pid  # noqa: F401
    from pid.common import classmap, types  # noqa: F401
