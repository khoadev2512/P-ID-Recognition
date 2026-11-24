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


def _before_13(cells: list[str]) -> list[str]:
    """Code cells before §13 (PHASE C 2B): the section tests below select cells by the
    commands they issue, and §13 reruns the same protocols for the DINOv2 encoder."""
    first = next((i for i, c in enumerate(cells) if "__dinov2__" in _active([c])), len(cells))
    return cells[:first]


def _is_14(cell: str) -> bool:
    """§14 (PHASE C §2E.3, F0 on Colab): reruns §13's protocols with oneshot.template_align."""
    code = _active([cell])
    return "template_align" in code or "ROWS14" in code or "in F0:" in code


def _is_19(cell: str) -> bool:
    """§19 (PHASE C §2N rule 7): the new one-class Stage 1 on OPEN100-ISO v2, e2e formB / ft."""
    code = _active([cell])
    return "CSV19" in code or "det_s1iso" in code


def _is_18(cell: str) -> bool:
    """§18 (PHASE C §2M): `formB` and `ft` rerun on OPEN100-ISO v2 (its own data root)."""
    code = _active([cell])
    return not _is_19(cell) and ("ISO2" in code or "ROWS18" in code)


def _is_17(cell: str) -> bool:
    """§17 (PHASE C §2L): the fine-tuned DINOv2 (FT-triển khai) against `formB`, plus e2e."""
    code = _active([cell])
    return not _is_18(cell) and ("ENC_FT" in code or "ROWS17" in code)


def _is_16(cell: str) -> bool:
    """§16 (PHASE C §2K.6): form B of the butterfly and aggregate=max, against §15's base."""
    code = _active([cell])
    return not (_is_17(cell) or _is_18(cell)) and (
        "PROTO43" in code or "ROWS16" in code or "in K4:" in code
    )


def _is_15(cell: str) -> bool:
    """§15 (PHASE C §2G, Điểm 4): reruns §13's protocols with `pipeline` excluded."""
    code = _active([cell])
    return not (_is_16(cell) or _is_17(cell) or _is_18(cell) or _is_19(cell)) and (
        "NOPIPE" in code or "ROWS15" in code or "dinov2_nopipe" in code
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
    code_cells = _before_13(code_cells)
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
    code_cells = _before_13(code_cells)
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
    code_cells = _before_13(code_cells)
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
    code_cells = _before_13(code_cells)
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
    code_cells = _before_13(code_cells)
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


def test_section_13_runs_dinov2_with_a4_and_cl2n_through_every_protocol(code_cells):
    """PHASE C 2B: every command reading __dinov2__.pt carries the DINOv2 overrides and A4; the
    center comes from protocol=center (R1a groups) and every later run passes it; tau on
    val_sel; P1 with no subset; P4 for DINOv2 plus the two CL2N rows, tags matching their tau
    runs (chosen_taus(tag) reads eval_tau_<tag>)."""
    s13 = [
        c
        for c in code_cells
        if "__dinov2__" in _active([c]) and not (_is_14(c) or _is_15(c) or _is_16(c) or _is_17(c))
    ]
    assert len(s13) == 4  # 13a, 13b, 13c, 13e (13d only reads metrics)
    make, run, p4, e2e = (_active([c]) for c in s13)
    dino = "fgc.backbone=vit_small_patch14_dinov2.lvd142m fgc.imgsz=224 fgc.arcface.embed_dim=384"
    assert f"DINO = '{dino}'" in make
    assert "A4 = '++oneshot.input_norm=ink_letterbox'" in make
    assert (
        "scripts/make_imagenet_encoder.py --out $ENC_DIR/__dinov2__.pt" in make
        and "$DINO $A4" in make
    )
    assert "{DINO} {A4}" in run and "BASE_D" in run
    for p in ("center", "tau", "p1"):
        assert f"protocol={p}" in run
    assert run.index("protocol=center") < run.index("protocol=tau") < run.index("protocol=p1")
    assert run.count("++oneshot.center=$CENTER_D") == 2 and "++eval.image_subset=val_sel" in run
    assert "GROUPS = f'++eval.oneshot.groups_ckpt={ENC_DIR}/__global_lo__.pt'" in run
    assert "run_dir=$RUNS_DIR/eval_tau_dinov2" in run and "run_dir=$RUNS_DIR/eval_p1_dinov2" in run
    assert "latest('eval_tau_lo')['classes']" in run
    for tag in ("'dinov2'", "'imagenet_cl2n'", "'lo_cl2n'"):
        assert f"({tag}, " in p4
    assert "{DINO} {A4} ++oneshot.center={ENC_DIR}/center_dinov2_a4.pt" in p4
    assert "center_imagenet.pt" in p4 and "center_lo.pt" in p4 and "image_subset" not in p4
    assert "run_dir=$RUNS_DIR/eval_p4_$tag" in p4 and "data_root=$LOCAL_ISO_ROOT" in p4
    assert "{DINO} {A4} ++oneshot.center=" in e2e and "chosen_taus(tag)" in e2e
    assert not any(
        "protocol=tau" in _active([c]) and "LOCAL_ISO_ROOT" in _active([c]) for c in code_cells
    )
    idx = [next(i for i, c in enumerate(code_cells) if c is x) for x in s13]
    assert idx == sorted(idx)
    last_12 = max(
        i for i, c in enumerate(code_cells) if "eval_e2e_iso_" in _active([c]) and c not in s13
    )
    assert last_12 < idx[0]


def test_section_14_reruns_tau_p1_p4_with_each_template_align_and_keeps_the_csvs(code_cells):
    """PHASE C §2E.3: f0a / shift on §13's encoder, prototypes and center (nothing rebuilt);
    tau on val_sel with R1a groups, P1 with no subset and no groups (as 13b, same crops), P4 on
    OPEN100-ISO with groups (as 13c); every run carries the alignment, checked in metrics; the
    per-crop CSVs are copied under the names scripts/f0_colab_compare.py reads."""
    s14 = [c for c in code_cells if _is_14(c)]
    assert len(s14) == 3  # 14a tau + P1, 14b P4, 14c table
    run, p4cell, table = (_active([c]) for c in s14)
    both = run + "\n" + p4cell
    assert "make_imagenet_encoder" not in both and "protocol=center" not in both
    assert "prototypes run_dir" not in both
    assert "('f0a', 'domain_center', '')" in run
    assert "('shift', 'pair_shift', f'++oneshot.template_pairs={PAIRS}')" in run
    assert "++oneshot.center={CENTER_D}" in run and "{DINO} {A4}" in run
    lines = [ln for ln in both.splitlines() if "protocol=" in ln]
    tau, p1, p4 = (next(ln for ln in lines if f"protocol={p}" in ln) for p in ("tau", "p1", "p4"))
    assert all("$ENC_D $ALIGN" in ln for ln in (tau, p1, p4))
    assert "image_subset=val_sel" in tau and "$GROUPS" in tau and "LOCAL_DATA_ROOT" in tau
    assert "image_subset" not in p1 and "$GROUPS" not in p1 and "LOCAL_DATA_ROOT" in p1
    assert "LOCAL_ISO_ROOT" in p4 and "$GROUPS" in p4 and "protocol=p4" in p4cell
    assert run.index("protocol=tau") < run.index("protocol=p1")
    check = "o['template_align']['kind'] == kind"
    assert check in run and check in p4cell
    assert "f'{CSV_DIR}/oneshot_p1_crops_dinov2_{v}.csv'" in run
    assert "f'{CSV_DIR}/oneshot_p4_crops_dinov2_{v}.csv'" in p4cell
    assert "('2B', 'dinov2')" in table
    idx = [next(i for i, c in enumerate(code_cells) if c is x) for x in s14]
    last_13 = max(
        i
        for i, c in enumerate(code_cells)
        if "__dinov2__" in c and not (_is_14(c) or _is_15(c) or _is_16(c) or _is_17(c))
    )
    assert last_13 < idx[0] < idx[1] < idx[2]


def test_section_15_reruns_2b_without_pipeline_and_runs_p2_with_r1b_groups(code_cells):
    """PHASE C §2G (Điểm 4): §13's encoder, prototypes and center (nothing rebuilt) with
    `pipeline` excluded (quoted: the brackets reach Hydra, not the shell glob); tau on val_sel
    with R1a groups; P2 on val `test` with R1b groups for BOTH 42 and 41 (2B never ran P2); P1
    with no subset and no groups; P4 on OPEN100-ISO with R1a groups; 41 / 42 prototypes checked
    in metrics; CSVs copied under the names scripts/nopipe_compare.py reads."""
    s15 = [c for c in code_cells if _is_15(c)]
    assert len(s15) == 3  # 15a tau + P2 + P1, 15b P4, 15c table
    run, p4cell, table = (_active([c]) for c in s15)
    both = run + "\n" + p4cell
    assert "make_imagenet_encoder" not in both and "protocol=center" not in both
    assert "prototypes run_dir" not in both
    assert "NOPIPE = \"'++oneshot.exclude_prototypes=[pipeline]'\"" in run
    assert "++oneshot.center={CENTER_D}" in run and "{DINO} {A4}" in run
    assert "GROUPS = f'++eval.oneshot.groups_ckpt={ENC_DIR}/__global_lo__.pt'" in run
    assert "GROUPS_P2 = f'++eval.oneshot.groups_ckpt={ENC_DIR}/__global__.pt'" in run
    lines = [ln for ln in both.splitlines() if "protocol=" in ln]
    tau, p2, p1, p4 = (
        next(ln for ln in lines if f"protocol={p}" in ln) for p in ("tau", "p2", "p1", "p4")
    )
    assert "image_subset=val_sel" in tau and "$GROUPS $ENC_D $NOPIPE" in tau
    assert "image_subset=test" in p2 and "$GROUPS_P2 $ENC_D $extra" in p2
    assert "(('dinov2', ''), ('dinov2_nopipe', NOPIPE))" in run
    assert "image_subset" not in p1 and "GROUPS" not in p1 and "$ENC_D $NOPIPE" in p1
    assert "LOCAL_ISO_ROOT" in p4 and "$GROUPS $ENC_D $NOPIPE" in p4
    assert all("LOCAL_DATA_ROOT" in ln for ln in (tau, p2, p1))
    assert run.index("protocol=tau") < run.index("protocol=p2") < run.index("protocol=p1")
    assert "o['n_prototypes'] == n" in run and "o['n_prototypes'] == 41" in p4cell
    assert "f'{CSV_DIR}/oneshot_{p}_crops_{tag}.csv'" in run
    assert "f'{CSV_DIR}/oneshot_p4_crops_dinov2_nopipe.csv'" in p4cell
    assert "CSV_DIR = f'{RUNS_DIR}/nopipe_csv'" in run
    assert "('41', 'dinov2_nopipe')" in table and "nopipe_compare.py" in table
    idx = [next(i for i, c in enumerate(code_cells) if c is x) for x in s15]
    last_14 = max(i for i, c in enumerate(code_cells) if _is_14(c))
    assert last_14 < idx[0] < idx[1] < idx[2]


def test_section_16_checks_form_b_and_max_against_the_41_template_base(code_cells):
    """PHASE C §2K.6: §13's encoder and center (not rebuilt), a 43-prototype file built with
    ++oneshot.extra_forms_dir and checked against the 42-row file; three lines (formB / mean,
    max / 42 rows, formB_max), all without `pipeline`; per line tau (val_sel, R1a groups) ->
    P2 (val test, R1b groups) -> P1 -> P4 (R1a groups), each metrics checked for its prototype
    count and aggregate; CSVs copied under the names scripts/k4_colab_compare.py reads."""
    s16 = [c for c in code_cells if _is_16(c)]
    assert len(s16) == 4  # 16a build, 16b tau + P2 + P1, 16c P4, 16d table
    build, run, p4cell, table = (_active([c]) for c in s16)
    assert "make_imagenet_encoder" not in run + p4cell and "protocol=center" not in run + p4cell
    assert (
        "++oneshot.extra_forms_dir=$FORMS_DIR" in build and "oneshot.prototypes=$PROTO43" in build
    )
    assert "FORMS_DIR = f'{REPO_DATA}/iso_symbols_forms'" in build
    assert "d43.embeddings[:42] - d42.embeddings" in build and "diff <= 1e-5" in build
    assert "eval_p4_dinov2_nopipe" in build  # §15 first
    assert "NOPIPE = \"'++oneshot.exclude_prototypes=[pipeline]'\"" in run
    assert "++oneshot.center={CENTER_D} {NOPIPE}" in run
    assert (
        "K4 = (('formB', PROTO43, 'mean', 42), ('max', PROTO42, 'max', 41), "
        "('formB_max', PROTO43, 'max', 42))"
    ) in run
    assert "oneshot.aggregate={agg}" in run and "oneshot.aggregate={agg}" in p4cell
    lines = [ln for ln in (run + "\n" + p4cell).splitlines() if "protocol=" in ln]
    tau, p2, p1, p4 = (
        next(ln for ln in lines if f"protocol={p}" in ln) for p in ("tau", "p2", "p1", "p4")
    )
    assert "image_subset=val_sel" in tau and "$GROUPS $ENC_K" in tau
    assert "image_subset=test" in p2 and "$GROUPS_P2 $ENC_K" in p2
    assert "image_subset" not in p1 and "GROUPS" not in p1 and "$ENC_K" in p1
    assert "LOCAL_ISO_ROOT" in p4 and "$GROUPS $ENC_K" in p4
    assert run.index("protocol=tau") < run.index("protocol=p2") < run.index("protocol=p1")
    assert "o['n_prototypes'] == n and o['aggregate'] == agg" in run
    assert "o['n_prototypes'] == n and o['aggregate'] == agg" in p4cell
    assert "CSV_DIR = f'{RUNS_DIR}/k4_csv'" in run
    assert "f'{CSV_DIR}/oneshot_{p}_crops_dinov2_{tag}.csv'" in run
    assert "f'{CSV_DIR}/oneshot_p4_crops_dinov2_{tag}.csv'" in p4cell
    assert "('base41', 'dinov2_nopipe')" in table and "k4_colab_compare.py" in table
    idx = [next(i for i, c in enumerate(code_cells) if c is x) for x in s16]
    last_15 = max(i for i, c in enumerate(code_cells) if _is_15(c))
    assert last_15 < idx[0] < idx[1] < idx[2] < idx[3]


def test_section_17_checks_the_deployed_fine_tune_against_form_b_with_e2e(code_cells):
    """PHASE C §2L: the FT encoder trained locally is checked by sha256 and its meta (all
    templates, alpha 0.5) before use; 43 prototypes and a CL2N center rebuilt with it (center
    on §13's crops); tau (val_sel, R1a) -> P2 (test, R1b) -> P1 -> P4 (R1a) without `pipeline`,
    checked against formB's crops; e2e of formB and ft at each line's own tau, Stage 2 cache
    keyed by the checkpoint hash; CSVs / e2e JSON under the names scripts/ft_colab_compare.py
    reads."""
    s17 = [c for c in code_cells if _is_17(c)]
    assert len(s17) == 5  # 17a build, 17b tau + P2 + P1, 17c P4, 17d e2e, 17e table
    build, run, p4cell, e2e, table = (_active([c]) for c in s17)
    assert "ENC_FT = f'{ENC_DIR}/__dinov2_ft_all_a0.5__.pt'" in build
    assert "h.hexdigest() == FT_SHA" in build and "__FT_SHA__" not in build
    assert "m['fold'] == 'all' and m['held_out'] == [] and m['alpha'] == 0.5" in build
    assert "oneshot.encoder_ckpt=$ENC_FT oneshot.prototypes=$PROTO_FT" in build
    assert "++oneshot.extra_forms_dir=$FORMS_DIR" in build and "d.names == d43.names" in build
    assert "protocol=center" in build and "c['n_crops'] == c0['n_crops']" in build
    assert build.index("prototypes run_dir") < build.index("protocol=center")
    assert "make_imagenet_encoder" not in "".join((build, run, p4cell, e2e))
    assert "++oneshot.center={CENTER_FT} {NOPIPE}" in run
    lines = [ln for ln in (run + "\n" + p4cell).splitlines() if "protocol=" in ln]
    tau, p2, p1, p4 = (
        next(ln for ln in lines if f"protocol={p}" in ln) for p in ("tau", "p2", "p1", "p4")
    )
    assert "image_subset=val_sel" in tau and "$GROUPS $ENC_K" in tau
    assert "image_subset=test" in p2 and "$GROUPS_P2 $ENC_K" in p2
    assert "image_subset" not in p1 and "GROUPS" not in p1 and "$ENC_K" in p1
    assert "LOCAL_ISO_ROOT" in p4 and "$GROUPS $ENC_K" in p4
    assert run.index("protocol=tau") < run.index("protocol=p2") < run.index("protocol=p1")
    assert "eval_tau_dinov2_ft" in run and "eval_tau_dinov2_formB" in run
    assert "CSV_DIR = f'{RUNS_DIR}/ft_csv'" in build
    assert "f'{CSV_DIR}/oneshot_{p}_crops_dinov2_ft.csv'" in run
    assert "f'{CSV_DIR}/oneshot_p4_crops_dinov2_ft.csv'" in p4cell
    assert "chosen_taus(f'dinov2_{tag}')['rule_a']" in e2e and "fgc --infer" in e2e
    assert "eval.mode=det_plus_fgc" in e2e and e2e.count("++fgc.oneshot=true") == 2
    assert "f'ft_{FT_SHA[:8]}'" in e2e and "{DINO} {A4} {NOPIPE}" in e2e
    assert "restore(f'{KEEP_ISO}/detections_agnostic', ISO_DET, 864)" in e2e
    assert "eval_e2e_iso_" not in e2e  # §12 / §13 tests key on that run name
    assert "f'{CSV_DIR}/e2e_dinov2_{tag}.json'" in e2e
    assert "('formB', 'dinov2_formB'), ('ft', 'dinov2_ft')" in table
    assert "ft_colab_compare.py" in table
    idx = [next(i for i, c in enumerate(code_cells) if c is x) for x in s17]
    last_16 = max(i for i, c in enumerate(code_cells) if _is_16(c))
    assert last_16 < idx[0] < idx[1] < idx[2] < idx[3] < idx[4]


def test_section_18_reruns_form_b_and_ft_on_open100_iso_v2(code_cells):
    """PHASE C §2M: OPEN100-ISO v2 in its own root (zip checked: 864 images, the v2 annotation
    count, a `drawing_box` on every image); Stage 1 agnostic_merge once on v2, kept on Drive;
    P4 for formB and ft with the same arguments as §16 / §17 (tau not chosen again: read from
    §16 / §17's tau runs); e2e of both from v2's detections; v1 and v2 printed side by side; no
    write to v1's run names."""
    s18 = [c for c in code_cells if _is_18(c)]
    assert len(s18) == 5  # 18a unzip, 18b Stage 1, 18c P4, 18d e2e, 18e table
    unzip, det, p4, e2e, table = (_active([c]) for c in s18)
    assert (
        "iso_open100_v2_canonical.zip" in unzip
        and "LOCAL_ISO2_ROOT = '/content/iso_open100_v2'" in unzip
    )
    assert "n_sym == 93794" in unzip and "'drawing_box' in im" in unzip
    assert "++detector.agnostic_merge=true" in det and "data_root=$LOCAL_ISO2_ROOT" in det
    assert "KEEP_ISO2 = f'{RUNS_DIR}/stage1_iso_v2'" in det and "eval_det_iso2_agnostic_lo" in det
    assert "protocol=p4" in p4 and "data_root=$LOCAL_ISO2_ROOT $GROUPS $ENC_V2" in p4
    assert "{BASE_K4} oneshot.prototypes={PROTO43} oneshot.aggregate=mean" in p4
    assert "++oneshot.center={CENTER_FT} {NOPIPE}" in p4 and "eval_p4v2_dinov2_" in p4
    assert "protocol=tau" not in "".join((unzip, det, p4, e2e, table))  # nothing chosen on v2
    assert "chosen_taus(f'dinov2_{tag}')['rule_a']" in e2e and "in E2E17:" in e2e
    assert "data_root=$LOCAL_ISO2_ROOT" in e2e and "eval_e2e18_" in e2e
    assert "restore(f'{KEEP_ISO2}/detections_agnostic', ISO2_DET, 864)" in e2e
    assert "eval_p4_dinov2_{t}" in table and "eval_p4v2_dinov2_{t}" in table
    for cell in (det, p4, e2e):
        assert "data_root=$LOCAL_ISO_ROOT " not in cell and "KEEP_ISO}" not in cell
    idx = [next(i for i, c in enumerate(code_cells) if c is x) for x in s18]
    last_17 = max(i for i, c in enumerate(code_cells) if _is_17(c))
    assert last_17 < idx[0] < idx[1] < idx[2] < idx[3] < idx[4]


def test_section_19_runs_the_new_stage1_once_on_open100_iso_v2(code_cells):
    """PHASE C §2N rule 7: the adopted one-class detector (sha checked, `classes_symbol.yaml`)
    runs once on v2 with agnostic_merge and is scored class-agnostic like §18's old one; e2e of
    formB / ft with the same tau / arguments as 18d on the new boxes only; nothing chosen, P4 not
    rerun; the frame count covers old and new, Stage 1 and one-shot; §18's kept outputs are never
    overwritten."""
    s19 = [c for c in code_cells if _is_19(c)]
    assert len(s19) == 3  # 19a Stage 1, 19b e2e, 19c table
    det, e2e, table = (_active([c]) for c in s19)
    assert "DET_S1 = f'{ENC_DIR}/det_s1iso.pt'" in det and "S1_SHA = '57c4c4c61430'" in det
    assert "S1_CLASSES = '/content/pid/configs/classes_symbol.yaml'" in det
    assert "coarse_classes == ['symbol']" in det
    assert "detector.ckpt=$DET_S1" in det and "++detector.classes_yaml=$S1_CLASSES" in det
    assert "++detector.agnostic_merge=true" in det and "data_root=$LOCAL_ISO2_ROOT" in det
    assert "++eval.class_agnostic=true" in det and "__global_lo__.pt" in det
    assert "keep = f'{KEEP_ISO2}/detections_agnostic_s1iso'" in det
    assert "('old', 'eval_det_iso2_agnostic_lo'), ('new', 'eval_det_iso2_s1iso_lo')" in det
    assert "restore(f'{KEEP_ISO2}/detections_agnostic_s1iso', ISO2_DET, 864)" in e2e
    assert "chosen_taus(f'dinov2_{tag}')['rule_a']" in e2e and "in E2E17:" in e2e
    assert "E_ARGS = f'{enc} {DINO} {A4} {NOPIPE}'" in e2e
    assert "++detector.classes_yaml=$S1_CLASSES" in e2e and "eval_e2e19_" in e2e
    assert "detections_oneshot_s1iso_dinov2_{keep_tag}" in e2e
    every = "".join((det, e2e, table))
    assert "protocol=" not in every  # no tau, no P4
    assert "run_dir=$RUNS_DIR/eval_det_iso2_agnostic_lo" not in every
    assert "run_dir=$RUNS_DIR/eval_e2e18_" not in every
    assert "keep = f'{KEEP_ISO2}/detections_agnostic'" not in every
    assert "data_root=$LOCAL_ISO_ROOT " not in every and "KEEP_ISO}" not in every
    assert "scripts/frame_predictions.py" in table and "drawing_box" not in det
    for d in ("'detections_agnostic', 'detections_agnostic_s1iso'", "detections_oneshot_dinov2_{k}",
              "detections_oneshot_s1iso_dinov2_{k}"):  # fmt: skip
        assert d in table, d
    assert "e2e_v2_dinov2_{tag}.json" in table and "e2e_v2_s1iso_dinov2_{tag}.json" in table
    idx = [next(i for i, c in enumerate(code_cells) if c is x) for x in s19]
    last_18 = max(i for i, c in enumerate(code_cells) if _is_18(c))
    assert last_18 < idx[0] < idx[1] < idx[2]
