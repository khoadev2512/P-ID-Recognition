"""Guards on notebooks/train_stage1_iso_colab.ipynb (PLAN_PHASE_C §2N): the notebook only runs on
Colab, so a slip (training on the test set, scoring with per-class merging, a wrong label set)
would only show after a GPU session. Checked as text, like tests/test_colab_notebook.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_NB = Path(__file__).resolve().parents[1] / "notebooks" / "train_stage1_iso_colab.ipynb"


@pytest.fixture(scope="module")
def cells() -> list[str]:
    nb = json.loads(_NB.read_text(encoding="utf-8"))
    return [
        "\n".join(ln for ln in "".join(c["source"]).splitlines() if not ln.lstrip().startswith("#"))
        for c in nb["cells"]
        if c["cell_type"] == "code"
    ]


def _one(cells: list[str], needle: str) -> str:
    hits = [c for c in cells if needle in c]
    assert len(hits) == 1, (needle, len(hits))
    return hits[0]


def test_never_touches_the_test_set(cells):
    code = "\n".join(cells)
    assert "iso_open100" not in code and "LOCAL_ISO_ROOT" not in code


def test_tiles_are_the_2n_set_and_one_class(cells):
    tiles = _one(cells, "S1_ZIP")
    assert "(n_train, n_val) == (50248, 6924)" in tiles
    assert "'nc': 1, 'names': ['symbol']" in tiles
    assert "S1_ROOT = '/content/s1iso'" in tiles and "derived/tiles" in tiles


def test_trains_from_the_old_detector_with_fixed_epochs(cells):
    train = _one(cells, "detector.epochs=$EPOCHS")
    assert "detect --train" in train and "detector.weights=$DET_CKPT_OLD" in train
    assert "data_root=$S1_ROOT" in train and "detector.imgsz=1024" in train
    assert "EPOCHS = 20" in train
    resume = _one(cells, "detector.resume=true")
    assert "data_root=$S1_ROOT" in resume and "detector.weights" not in resume
    final = _one(cells, "results.csv")
    assert "len(rows) == 20" in final and "== ['symbol']" in final


def test_decision_on_p4val_v2_old_and_new_alike(cells):
    p4 = _one(cells, "P4V2_ZIP")
    assert "iso_p4val_v2_canonical.zip" in p4
    assert "len(coco['images']) == 73 and n_sym == 4832" in p4 and "drawing_box" in p4
    assert "('old', DET_CKPT_OLD, DET_CLASSES), ('new', DET_CKPT_NEW, S1_CLASSES)" in p4
    assert p4.count("++detector.agnostic_merge=true") == 1  # one command, both detectors
    assert "++eval.class_agnostic=true" in p4 and "__global_lo__.pt" in p4
    assert "eval.split=test" in p4 and "det_p4v2_{tag}.json" in p4
    table = _one(cells, "s1iso_compare.py")
    assert "det_p4v2_{t}.json" in table
