"""Smoke tests — verify the top-level packages import and the path layout resolves."""

from pathlib import Path

from utils.paths import DataPaths


def test_datapaths_layout():
    p = DataPaths(root=Path("data"))
    assert p.coco_json == Path("data/canonical/annotations.coco.json")
    assert p.classes_yaml == Path("data/canonical/classes.yaml")
    assert p.yolo_coarse == Path("data/derived/yolo_coarse")
    assert p.crops_fgc == Path("data/derived/crops_fgc")


def test_package_imports():
    # Flat src/ layout, no top-level wrapper package (see README): just two sibling
    # packages, pipeline/ (stages) and utils/ (shared helpers).
    import pipeline  # noqa: F401
    import utils  # noqa: F401
    from utils import classmap, types  # noqa: F401
