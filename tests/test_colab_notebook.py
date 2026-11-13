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


def test_never_reads_an_old_crop_zip(code_cells):
    """crops_fgc.zip: 28 classes. crops_fgc_global.zip: crops without _boxes.csv (before
    utils.crop_box), which box jitter cannot use."""
    code = _active(code_cells)
    assert "crops_fgc_ctx.zip" in code
    assert "crops_fgc.zip" not in code and "crops_fgc_global.zip" not in code
    assert "_boxes.csv" in code  # checked after unzipping


def test_both_encoders_train_and_resume_with_the_same_augment(code_cells):
    """Scan degradation + box jitter (PLAN 2026-11-11), from one AUG string: a resume
    without it would silently train the rest of the run without them."""
    code = _active(code_cells)
    assert "++fgc.aug.degrade.p=" in code and "++fgc.aug.box_jitter.p=" in code
    train = [c for c in code_cells if "fgc --train" in _active([c])]
    assert len(train) == 4  # R1a, R1a resume, R1b, R1b resume
    assert all("$AUG" in _active([c]) and "$LOADER" in _active([c]) for c in train)


def test_leave_out_also_drops_h_extra_and_finalize_checks_it(code_cells):
    """h_extra (VARIANT of an h_in symbol) must leave R1a's training too, or the encoder
    still learns the held-out symbol through its variant."""
    code = _active(code_cells)
    assert "h_extra" in code and "HELD = H + H_EXTRA" in code
    assert "EXCLUDE = '[' + ','.join(HELD) + ']'" in code
    assert "'__global_lo__.pt', HELD)" in code


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


def test_evaluation_scores_the_global_head_closed_set(code_cells):
    """Protocol C (B6c): without fgc.global_encoder=true evaluation looks for per-group
    <group>.pt and skips the global encoder; fgc_only needs no detector output."""
    evals = [
        c
        for c in code_cells
        if "run_pipeline.py evaluate" in _active([c]) and "eval.mode=oneshot" not in _active([c])
    ]
    assert evals and all(
        "fgc.global_encoder=true" in _active([c]) and "eval.mode=fgc_only" in _active([c])
        for c in evals
    )
    # the exam half only: val_sel chose tau (and, with val_acc, the checkpoint)
    assert all("++eval.image_subset=test" in _active([c]) for c in evals)


def test_both_encoders_train_to_the_same_fixed_epochs(code_cells):
    """val_acc saturates at epoch 1 (R1b, 2026-11-12), so best-by-val_acc would pick an
    arbitrary epoch: R1a and R1b are compared at one epoch fixed in advance (PLAN
    2026-11-13), saved by ++fgc.save_epochs whatever val_acc does."""
    code = _active(code_cells)
    assert "fgc.epochs=60" not in code
    train = [c for c in code_cells if "fgc --train" in _active([c])]
    assert all("fgc.epochs=$EPOCHS" in _active([c]) and '"$SAVE"' in _active([c]) for c in train)
    assert "CMP_EPOCH = 3" in code and "++fgc.save_epochs=[3," in code


def test_finalize_is_defined_apart_from_its_calls(code_cells):
    """It used to be defined in R1a's 5b cell, which also called it: with R1b run first
    and no R1a yet, that cell raised and 6b then hit NameError."""
    (definition,) = [c for c in code_cells if "def finalize(" in c]
    assert "finalize('" not in _active([definition])
    calls = [c for c in code_cells if "finalize('" in _active([c])]
    assert len(calls) == 2 and all("def finalize(" not in c for c in calls)
    assert "newest(f'__global__.ep{epoch}.pt') + newest('__global__.pt')" in definition
    assert "st['epoch'] + 1 == epoch" in definition


def test_oneshot_eval_section_runs_tau_on_val_sel_then_p1_and_p2_on_the_right_data(code_cells):
    """B6b-1: tau on val_sel for both encoders, P1 on the leave-out encoder over crops it never
    trained on (no image subset), P2 on the full encoder over the test half; mean aggregate."""
    cells = [c for c in code_cells if "eval.mode=oneshot" in _active([c])]
    by = {p: [c for c in cells if f"protocol={p}" in _active([c])] for p in ("tau", "p1", "p2")}
    assert all(len(v) == 1 for v in by.values()) and len(cells) == 3
    tau, p1, p2 = (_active(by[p]) for p in ("tau", "p1", "p2"))
    assert "++eval.image_subset=val_sel" in tau and "test" not in tau.split("image_subset=")[1][:8]
    assert "__global_lo__.pt" in tau and "__global__.pt" in tau and "iso_prototypes_lo.pt" in tau
    assert "image_subset" not in p1
    assert "oneshot.encoder_ckpt=$ENC_DIR/__global_lo__.pt" in p1
    assert "oneshot.prototypes=$ENC_DIR/iso_prototypes_lo.pt" in p1
    assert "++eval.image_subset=test" in p2
    assert "oneshot.encoder_ckpt=$ENC_DIR/__global__.pt" in p2
    assert "oneshot.prototypes=$ENC_DIR/iso_prototypes.pt" in p2
    for code in (tau, p1, p2):
        assert "oneshot.aggregate=mean" in code and "data_root=$LOCAL_DATA_ROOT" in code
        assert "oneshot.vocab_csv=$REPO_DATA/iso_symbols/vocab.csv" in code
    order = [
        next(i for i, c in enumerate(code_cells) if c is by[p][0]) for p in ("tau", "p1", "p2")
    ]
    assert order == sorted(order)  # tau is chosen before the test half is touched
    evaluate_c = next(i for i, c in enumerate(code_cells) if "run_dir=$RUNS_DIR/eval_c" in c)
    assert evaluate_c < order[0]
