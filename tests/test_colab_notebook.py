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


def _oneshot_cells(cells: list[str]) -> list[str]:
    return [c for c in cells if "eval.mode=oneshot" in _active([c])]


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


def _detection_cells(cells: list[str]) -> list[str]:
    """§12: scored from detection files (Stage 1, end to end), not from crops."""
    return [c for c in cells if "eval.mode=det_" in _active([c])]


def test_evaluation_scores_the_global_head_closed_set(code_cells):
    """Protocol C (B6c): without fgc.global_encoder=true evaluation looks for per-group
    <group>.pt and skips the global encoder; fgc_only needs no detector output."""
    evals = [
        c
        for c in code_cells
        if "run_pipeline.py evaluate" in _active([c])
        and "eval.mode=oneshot" not in _active([c])
        and c not in _detection_cells(code_cells)
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
    cells = [
        c
        for c in _oneshot_cells(code_cells)
        if "__imagenet__" not in c and "protocol=p4" not in _active([c])
    ]
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


def test_imagenet_baseline_runs_tau_on_r1a_classes_then_p1_after_section_9(code_cells):
    """B4a: the untrained ImageNet encoder gets its own prototypes, tau on val_sel crops of the
    classes R1a's tau run used (groups from __global_lo__.pt), and P1 on the same crops as 9b,
    in run dirs §9's "latest metrics.json" globs never pick up."""
    code = _active(code_cells)
    assert "scripts/make_imagenet_encoder.py --out $ENC_DIR/__imagenet__.pt" in code
    (make,) = [c for c in code_cells if "make_imagenet_encoder.py" in _active([c])]
    assert "data_root=$LOCAL_DATA_ROOT" in _active([make])
    assert "run_pipeline.py prototypes" in _active([make])
    assert "oneshot.encoder_ckpt=$ENC_DIR/__imagenet__.pt" in _active([make])
    assert "oneshot.prototypes=$ENC_DIR/iso_prototypes_imagenet.pt" in _active([make])
    assert "oneshot.symbols_dir=$REPO_DATA/iso_symbols_clean" in _active([make])

    cells = [
        c
        for c in _oneshot_cells(code_cells)
        if "__imagenet__" in c
        and "protocol=p4" not in _active([c])
        and "center" not in _active([c])  # 10d (CL2N) has its own test
    ]
    by = {p: [c for c in cells if f"protocol={p}" in _active([c])] for p in ("tau", "p1")}
    assert all(len(v) == 1 for v in by.values()) and len(cells) == 2
    tau, p1 = (_active(by[p]) for p in ("tau", "p1"))
    assert "run_dir=$RUNS_DIR/eval_tau_imagenet " in tau
    assert "run_dir=$RUNS_DIR/eval_p1_imagenet " in p1
    assert "++eval.image_subset=val_sel" in tau
    assert "++eval.oneshot.groups_ckpt=$ENC_DIR/__global_lo__.pt" in tau
    assert "latest('eval_tau_lo')" in tau and "o['classes'] == o_lo['classes']" in tau
    assert "image_subset" not in p1 and "groups_ckpt" not in p1
    assert "latest('eval_p1')" in p1 and "im['n_crops'] == lo['n_crops']" in p1
    for c in (tau, p1):
        assert "oneshot.encoder_ckpt=$ENC_DIR/__imagenet__.pt" in c
        assert "oneshot.prototypes=$ENC_DIR/iso_prototypes_imagenet.pt" in c
        assert "oneshot.aggregate=mean" in c and "data_root=$LOCAL_DATA_ROOT" in c
        assert "oneshot.vocab_csv=$REPO_DATA/iso_symbols/vocab.csv" in c
    idx = {
        name: next(i for i, c in enumerate(code_cells) if c is cell)
        for name, cell in (("make", make), ("tau", by["tau"][0]), ("p1", by["p1"][0]))
    }
    last_9 = max(i for i, c in enumerate(code_cells) if "protocol=p2" in _active([c]))
    assert last_9 < idx["make"] < idx["tau"] < idx["p1"]


def test_cl2n_builds_a_center_per_encoder_then_reruns_tau_and_p1_with_it(code_cells):
    """10d (CL2N): the center comes from train crops of R1a's classes (protocol=center, groups
    from __global_lo__.pt), is copied next to the encoder, and every centered run (tau on
    val_sel, then P1) passes it; run dirs never collide with the raw ones §9 / §10 read."""
    (run,) = [c for c in _oneshot_cells(code_cells) if "protocol=center" in _active([c])]
    code = _active([run])
    assert "('imagenet', '__imagenet__.pt', 'iso_prototypes_imagenet.pt')" in code
    assert "('lo', '__global_lo__.pt', 'iso_prototypes_lo.pt')" in code
    assert "GROUPS = f'++eval.oneshot.groups_ckpt={ENC_DIR}/__global_lo__.pt'" in code
    assert "CENTER = f'{ENC_DIR}/center_{tag}.pt'" in code
    assert "shutil.copy2(c['center_file'], CENTER)" in code
    assert "('' if p == 'center' else '_cl2n')" in code
    lines = [ln for ln in code.split("\n") if "protocol=" in ln]
    proto = {p: next(ln for ln in lines if f"protocol={p}" in ln) for p in ("center", "tau", "p1")}
    assert "++oneshot.center" not in proto["center"] and "$RUN_C" in proto["center"]
    assert "++eval.image_subset=val_sel" in code and "$RUN_T" in proto["tau"]
    assert code.count("++oneshot.center=$CENTER") == 2 and "$RUN_P" in proto["p1"]
    assert code.index("protocol=center") < code.index("protocol=tau") < code.index("protocol=p1")
    assert "oneshot.aggregate=mean" in code and "data_root={LOCAL_DATA_ROOT}" in code
    assert "latest('eval_tau_lo')['classes']" in code

    (report,) = [c for c in code_cells if "eval_p1_imagenet_cl2n" in _active([c]) and c is not run]
    rep = _active([report])
    for r, t in (("eval_p1_imagenet", "imagenet"), ("eval_p1_imagenet_cl2n", "imagenet_cl2n"),
                 ("eval_p1", "lo"), ("eval_p1_lo_cl2n", "lo_cl2n")):  # fmt: skip
        assert f"'{r}', '{t}')" in rep
    assert "len({o['n_crops'] for _, o, _ in have}) == 1" in rep
    assert "at_chosen_taus(o, t, {'H_chon'}, {'OOV_heldout'})" in rep
    i = [next(k for k, c in enumerate(code_cells) if c is x) for x in (run, report)]
    raw_p1 = max(k for k, c in enumerate(code_cells) if "run_dir=$RUNS_DIR/eval_p1_imagenet " in c)
    first_11 = min(k for k, c in enumerate(code_cells) if "iso_open100_canonical.zip" in c)
    assert raw_p1 < i[0] < i[1] < first_11


def test_chosen_tau_report_adds_the_margin_gate_from_the_same_tau_run(code_cells):
    """Score form 2 (PLAN B8): delta comes from the tau run's metrics.json (val_sel), never from
    the P1 / P2 numbers, and the gate is cos < tau_A OR margin < delta."""
    helper = next(c for c in code_cells if "def at_chosen_taus" in c)
    assert "def chosen_delta" in helper
    assert ".get('margin_rule_a', {}).get('delta')" in helper
    assert "gated_other_prf" in helper and "margins(" in helper
    assert "chosen_taus(tag)['rule_a']" in helper


def test_p4_section_runs_the_three_encoders_on_open100_iso_after_section_10(code_cells):
    """§11 (PLAN B6b P4): OPEN100-ISO unpacked from Drive into its own data root, P4 for R1b, R1a
    and ImageNet (grouped like R1a, as in 10b) with no image subset and nothing selected on it:
    tau / delta come from 9a / 10b, and the report filters the CSV by view (never pooled)."""
    (unpack,) = [c for c in code_cells if "iso_open100_canonical.zip" in _active([c])]
    assert "LOCAL_ISO_ROOT = '/content/iso_open100'" in unpack and "unzip" in _active([unpack])
    assert "== 864" in unpack and "93800" in unpack

    (p4,) = [c for c in _oneshot_cells(code_cells) if "protocol=p4" in _active([c])]
    code = _active([p4])
    assert "data_root=$LOCAL_ISO_ROOT" in code and "run_dir=$RUNS_DIR/eval_p4_$tag" in code
    assert "image_subset" not in code and "other_min_cosine" not in code
    assert "oneshot.aggregate=mean" in code
    assert "oneshot.vocab_csv=$REPO_DATA/iso_symbols/vocab.csv" in code
    for enc, protos in (("__global__.pt", "iso_prototypes.pt"),
                        ("__global_lo__.pt", "iso_prototypes_lo.pt"),
                        ("__imagenet__.pt", "iso_prototypes_imagenet.pt")):  # fmt: skip
        assert f"'{enc}', '{protos}'" in code
    assert "('imagenet', '__imagenet__.pt', 'iso_prototypes_imagenet.pt', " in code
    assert "f'++eval.oneshot.groups_ckpt={ENC_DIR}/__global_lo__.pt'" in code
    # the P4 tags are the tau run tags of 9a / 10b, so chosen_taus(tag) reads the right encoder
    assert all(f"('{t}', " in code for t in ("full", "lo", "imagenet"))
    # tau is never chosen on OPEN100-ISO
    assert not any(
        "protocol=tau" in _active([c]) and "LOCAL_ISO_ROOT" in _active([c]) for c in code_cells
    )

    (report,) = [c for c in code_cells if "eval_p4_{tag}" in _active([c])]
    assert "for view in ('degraded', 'clean')" in report
    assert "keep=lambda r: r['split'] == view" in report
    assert "at_chosen_taus(o, tag, set(P4_GROUPS), {'other_iso'}" in report
    assert "sheet_mean_std" in report and "'other_isa'" in report
    helper = next(c for c in code_cells if "def at_chosen_taus" in c)
    assert "keep=None, open_set=None" in helper

    idx = [next(i for i, c in enumerate(code_cells) if c is x) for x in (unpack, p4, report)]
    last_10 = max(i for i, c in enumerate(code_cells) if "eval_p1_imagenet" in _active([c]))
    assert last_10 < idx[0] < idx[1] < idx[2]


def test_section_12_stage1_and_end_to_end_on_open100_iso_then_wrong_boxes(code_cells):
    """§12 (PLAN B6b-2): the DigitizePID coarse detector on OPEN100-ISO with and without
    agnostic_merge, scored class-agnostically with recall per group; end to end with the one-shot
    Stage 2 of R1b and R1a at the tau / delta of 9a (gated at eval time, never chosen on
    OPEN100-ISO); the wrong-box rate on DigitizePID val_sel only."""
    cells = _detection_cells(code_cells)
    det = [c for c in cells if "eval.mode=det_only" in _active([c])]
    e2e = [c for c in cells if "eval.mode=det_plus_fgc" in _active([c])]
    assert len(det) == 1 and len(e2e) == 2
    (det,) = det
    code = _active([det])
    assert "DET_CKPT = f'{RUNS_DIR}/" in code and "assert os.path.exists(DET_CKPT)" in code
    assert "detect --infer" in code and "data_root=$LOCAL_ISO_ROOT" in code
    assert "++detector.classes_yaml=$DET_CLASSES" in code
    assert "DET_CLASSES = f'{LOCAL_DATA_ROOT}/canonical/classes.yaml'" in code
    assert "('agnostic', '++detector.agnostic_merge=true')" in code and "('perlabel', '')" in code
    assert "++eval.class_agnostic=true" in code and "oneshot.encoder_ckpt=$ENC_DIR/$enc" in code
    assert "('full', '__global__.pt'), ('lo', '__global_lo__.pt')" in code
    assert "oneshot.vocab_csv=$REPO_DATA/iso_symbols/vocab.csv" in code
    # shared by 12b and 12c: the encoders of 9a and §9's view aggregation
    assert "('full', '__global__.pt', 'iso_prototypes.pt')" in code
    assert "('lo', '__global_lo__.pt', 'iso_prototypes_lo.pt')" in code
    assert (
        "OS_ARGS = f'oneshot.vocab_csv={REPO_DATA}/iso_symbols/vocab.csv oneshot.aggregate=mean'"
        in code
    )

    iso, dpid = sorted(e2e, key=lambda c: "LOCAL_ISO_ROOT" not in c)
    for c in (iso, dpid):
        code = _active([c])
        assert "fgc --infer" in code and code.count("++fgc.oneshot=true") == 2
        assert "chosen_taus(tag)['rule_a']" in code and "chosen_delta(tag)" in code
        assert "oneshot.other_min_cosine=$tau" in code
        assert "++eval.oneshot.margin_delta={delta}" in code
        assert "for tag, enc, protos in E2E_RUNS:" in code and "$OS_ARGS" in code
    code = _active([iso])
    assert "data_root=$LOCAL_ISO_ROOT" in code and "image_subset" not in code
    assert "++detector.classes_yaml=$DET_CLASSES" in code  # Stage-1 ids are DigitizePID coarse
    assert "restore(f'{KEEP_ISO}/detections_agnostic', ISO_DET, 864)" in code
    code = _active([dpid])
    assert "data_root=$LOCAL_DATA_ROOT" in code and "eval.split=val" in code
    assert "++eval.image_subset=val_sel" in code and "image_subset=test" not in code
    assert "os.symlink(f'{DATA_ROOT}/canonical/images'" in code
    # nothing in §12 writes under the Drive data root
    assert not any("data_root=$DATA_ROOT" in _active([c]) for c in cells)

    idx = {name: next(i for i, c in enumerate(code_cells) if c is cell)
           for name, cell in (("det", det), ("iso", iso), ("dpid", dpid))}  # fmt: skip
    last_11 = next(i for i, c in enumerate(code_cells) if "eval_p4_{tag}" in _active([c]))
    tau_9a = next(i for i, c in enumerate(code_cells) if "def at_chosen_taus" in c)
    assert tau_9a < last_11 < idx["det"] < idx["iso"] < idx["dpid"]


def test_unpack_counts_symbols_not_ignore_regions(code_cells):
    """After the B6d.2 re-export the COCO also holds iscrowd ignore regions: 93800 symbols."""
    (unpack,) = [c for c in code_cells if "iso_open100_canonical.zip" in _active([c])]
    assert "n_sym = sum(not a.get('iscrowd')" in unpack and "n_sym == 93800" in unpack


def test_notebook_round_trips_byte_for_byte():
    raw = _NB.read_text(encoding="utf-8")
    assert json.dumps(json.loads(raw), indent=1, ensure_ascii=False) + "\n" == raw
