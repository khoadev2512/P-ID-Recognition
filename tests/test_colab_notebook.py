"""Guards on notebooks/train_stage2_colab.ipynb — the traps B2 removed must stay removed.

The notebook only runs on Colab, so nothing else would notice a regression until a GPU
session is spent on it. Checked as text: which commands the code cells issue.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_NB = Path(__file__).resolve().parents[1] / "notebooks" / "train_stage2_colab.ipynb"


@pytest.fixture(scope="module")
def code_cells() -> list[str]:
    nb = json.loads(_NB.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _active(cells: list[str]) -> str:
    """All code, minus commented-out lines."""
    return "\n".join(
        line for cell in cells for line in cell.splitlines() if not line.lstrip().startswith("#")
    )


def test_never_reads_the_old_28_class_crop_zip(code_cells):
    code = _active(code_cells)
    assert "crops_fgc_global.zip" in code
    assert "crops_fgc.zip" not in code.replace("crops_fgc_global.zip", "")


def test_every_training_and_cropping_command_is_global(code_cells):
    for cell in code_cells:
        active = _active([cell])
        if "fgc --train" in active or "--step crops" in active:
            assert "fgc.global_encoder=true" in active, cell


def test_leave_out_runs_take_the_list_from_config_not_by_hand(code_cells):
    code = _active(code_cells)
    assert "holdout_classes(" in code
    r1a = [c for c in code_cells if "run_dir=$RUNS_DIR/r1a" in c]
    assert r1a and all('"fgc.exclude_classes=$EXCLUDE"' in c for c in r1a)
    r1b = [c for c in code_cells if "run_dir=$RUNS_DIR/r1b" in c]
    assert r1b and all('"fgc.exclude_classes=[]"' in c for c in r1b)


def test_the_crop_tree_is_checked_for_the_full_vocabulary(code_cells):
    assert "== 32" in _active(code_cells)


def test_prototypes_step_is_wired(code_cells):
    code = _active(code_cells)
    assert "run_pipeline.py prototypes" in code
    assert "oneshot.symbols_dir=$REPO_DATA/iso_symbols_clean" in code


def test_group_wise_evaluation_stays_disabled_until_b6b(code_cells):
    """evaluation.py iterates fgc_groups and would silently skip a global encoder."""
    assert "run_pipeline.py evaluate" not in _active(code_cells)
