"""Tests for leave-classes-out (cfg.fgc.exclude_classes, docs/PLAN_STAGE2_ONESHOT.md B1b).

The one-shot claim is only tested if the encoder has NEVER seen the test classes
(protocol P1). That property is easy to break silently, in two ways these tests pin:

  LEAK      a held-out class still reaches training (a typo in the list, a dataset that
            filters but a head that doesn't) — the P1 number then measures recall of a
            seen class and the thesis claim is void without any error.
  MISALIGN  the head drops a class but the dataset doesn't (or the reverse) — labels end
            up one column off the head, which trains fine and is wrong everywhere.

Both are prevented by routing every consumer through utils.classmap.training_fine_classes;
these tests check that the consumers actually do.
"""

from __future__ import annotations

import pytest

from utils.classmap import GLOBAL_GROUP, ClassMap, training_fine_classes

_FINE = {
    "valve_gate": "valve",
    "valve_ball": "valve",
    "valve_globe": "valve",
    "bubble_field": "instrument",
    "bubble_logic": "instrument",
    "flange": "flange",
}


def _classmap() -> ClassMap:
    return ClassMap(
        coarse_classes=["valve", "instrument", "flange"],
        fine_to_coarse=dict(_FINE),
        fgc_groups=["valve", "instrument"],
        rare_fine=[],
    )


# ------------------------------------------------------------------ #
# training_fine_classes: the single decision point
# ------------------------------------------------------------------ #


def test_empty_exclusion_is_exactly_the_old_behaviour():
    classmap = _classmap()
    for group in (GLOBAL_GROUP, "valve", "instrument", "flange"):
        assert training_fine_classes(classmap, group) == classmap.fine_of_group(group)
        assert training_fine_classes(classmap, group, []) == classmap.fine_of_group(group)


def test_global_head_loses_exactly_the_held_out_classes_in_yaml_order():
    kept = training_fine_classes(_classmap(), GLOBAL_GROUP, ["valve_ball", "bubble_logic"])
    assert kept == ["valve_gate", "valve_globe", "bubble_field", "flange"]


def test_per_group_exclusion_only_touches_that_group():
    classmap = _classmap()
    exclude = ["valve_ball"]
    assert training_fine_classes(classmap, "valve", exclude) == ["valve_gate", "valve_globe"]
    assert training_fine_classes(classmap, "instrument", exclude) == [
        "bubble_field",
        "bubble_logic",
    ]


def test_an_unknown_name_is_an_error_not_a_silent_no_op():
    """A typo would leave the class in training — the LEAK failure — so it must fail."""
    with pytest.raises(ValueError, match="Symbol_99"):
        training_fine_classes(_classmap(), GLOBAL_GROUP, ["valve_gate", "Symbol_99"])


def test_holding_out_a_whole_group_is_an_error():
    with pytest.raises(ValueError, match="nothing is left to train"):
        training_fine_classes(_classmap(), "instrument", ["bubble_field", "bubble_logic"])


def test_holding_out_everything_from_the_global_head_is_an_error():
    with pytest.raises(ValueError, match="nothing is left to train"):
        training_fine_classes(_classmap(), GLOBAL_GROUP, list(_FINE))


# ------------------------------------------------------------------ #
# Pipelines: dataset, train loop, checkpoint, infer — end to end on tiny crops
# ------------------------------------------------------------------ #

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

import numpy as np  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from PIL import Image  # noqa: E402

from pipeline.fgc_infer import FGCInferPipeline  # noqa: E402
from pipeline.fgc_train import FGCTrainPipeline  # noqa: E402
from utils.dataset import build_dataset, excluded_classes  # noqa: E402
from utils.model import checkpoint_excluded_classes, load_arcface_encoder  # noqa: E402

_CLASSES_YAML = """
coarse: [valve, instrument, flange]
fine:
  valve_gate: valve
  valve_ball: valve
  valve_globe: valve
  bubble_field: instrument
  bubble_logic: instrument
  flange: flange
fgc_groups: [valve, instrument]
"""

_HELD_OUT = ["valve_ball", "bubble_logic"]


def _cfg(tmp_path, exclude, global_encoder: bool = True):
    data = tmp_path / "data"
    (data / "canonical").mkdir(parents=True, exist_ok=True)
    (data / "canonical" / "classes.yaml").write_text(_CLASSES_YAML)
    return OmegaConf.create(
        {
            "data_root": str(data),
            "run_dir": str(tmp_path / "runs"),
            "device": "cpu",
            "fgc": {
                "global_encoder": global_encoder,
                "exclude_classes": exclude,
                "metric_learning": True,
                "backbone": "resnet18",
                "pretrained": False,
                "imgsz": 32,
                "epochs": 1,
                "batch": 8,
                "loss": "ce",
                "balanced_sampler": True,
                "resume_from": None,
                "ckpt_dir": None,
                "arcface": {"m": 0.5, "s": 30.0, "embed_dim": 32},
                "route": {"min_score": 0.25, "other_min_cosine": 0.35},
                "aug": {"hflip": 0.0, "rotate": 0, "jitter": 0.0, "subpart": {"p": 0.0}},
            },
        }
    )


def _write_crops(tmp_path, per_class: int = 3) -> None:
    """A few crops for EVERY class in both splits — held-out ones included, exactly as
    pid-prep-crops leaves them on disk (the unseen-class test needs them)."""
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        for fine in _FINE:
            out = tmp_path / "data" / "derived" / "crops_fgc" / split / fine
            out.mkdir(parents=True, exist_ok=True)
            for i in range(per_class):
                pixels = rng.integers(0, 255, (24, 24, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(out / f"img_{i}.png")


def test_excluded_classes_reads_absent_null_and_lists():
    assert excluded_classes(OmegaConf.create({"fgc": {}})) == []
    assert excluded_classes(OmegaConf.create({"fgc": {"exclude_classes": None}})) == []
    assert excluded_classes(OmegaConf.create({"fgc": {"exclude_classes": ["a"]}})) == ["a"]


@pytest.mark.parametrize("split", ["train", "val"])
def test_dataset_holds_no_crop_of_a_held_out_class(tmp_path, split):
    _write_crops(tmp_path)
    ds = build_dataset(_cfg(tmp_path, _HELD_OUT), split, GLOBAL_GROUP)
    folders = {path.parent.name for path, _ in ds.samples}
    assert folders == set(_FINE) - set(_HELD_OUT)
    assert len(ds) == 3 * (len(_FINE) - len(_HELD_OUT))


def test_dataset_labels_index_the_training_list(tmp_path):
    """MISALIGN guard: label i must mean training_fine_classes(...)[i]."""
    _write_crops(tmp_path)
    ds = build_dataset(_cfg(tmp_path, _HELD_OUT), "train", GLOBAL_GROUP)
    for path, label in ds.samples:
        assert ds.fine_classes[label] == path.parent.name
    assert ds.fine_classes == training_fine_classes(_classmap(), GLOBAL_GROUP, _HELD_OUT)


def _train(tmp_path, exclude):
    _write_crops(tmp_path)
    FGCTrainPipeline(_cfg(tmp_path, exclude)).run()
    ckpt = tmp_path / "runs" / "fgc_checkpoints" / f"{GLOBAL_GROUP}.pt"
    assert ckpt.exists()
    return ckpt, torch.load(ckpt, map_location="cpu", weights_only=False)


def test_trained_checkpoint_has_no_column_for_a_held_out_class(tmp_path):
    ckpt, state = _train(tmp_path, _HELD_OUT)
    kept = [name for name in _FINE if name not in _HELD_OUT]

    assert state["model"]["head.weight"].shape[0] == len(_FINE) - len(_HELD_OUT)
    assert state["fine_classes"] == kept
    assert not set(_HELD_OUT) & set(state["fine_classes"])
    assert state["excluded_classes"] == _HELD_OUT
    assert state["class_centers"].shape[0] == len(kept)

    # The checkpoint's column order is the dataset's label order.
    ds = build_dataset(_cfg(tmp_path, _HELD_OUT), "train", GLOBAL_GROUP)
    assert ds.fine_classes == state["fine_classes"]

    encoder, fine_classes = load_arcface_encoder(_cfg(tmp_path, _HELD_OUT), ckpt)
    assert fine_classes == kept
    assert checkpoint_excluded_classes(ckpt) == _HELD_OUT


def test_no_exclusion_trains_the_full_vocabulary(tmp_path):
    _, state = _train(tmp_path, [])
    assert state["fine_classes"] == list(_FINE)
    assert state["excluded_classes"] == []


def test_a_typo_stops_training_before_it_starts(tmp_path):
    _write_crops(tmp_path)
    with pytest.raises(ValueError, match="valve_bal"):
        FGCTrainPipeline(_cfg(tmp_path, ["valve_bal"])).run()
    assert not (tmp_path / "runs" / "fgc_checkpoints" / f"{GLOBAL_GROUP}.pt").exists()


def test_a_checkpoint_from_before_exclusion_existed_reports_none(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"model": {}, "fine_classes": ["a"]}, path)
    assert checkpoint_excluded_classes(path) == []


def test_fgc_infer_sizes_per_group_heads_without_the_held_out_classes(tmp_path):
    """Per-group heads trained with an exclusion must be rebuilt at the same width, or
    load_state_dict fails on a head-size mismatch."""
    cfg = _cfg(tmp_path, ["valve_ball"], global_encoder=False)
    pipeline = FGCInferPipeline(cfg)
    assert pipeline._fine_classes_per_group["valve"] == ["valve_gate", "valve_globe"]
    assert pipeline._fine_classes_per_group["instrument"] == ["bubble_field", "bubble_logic"]


def test_resume_refuses_a_checkpoint_trained_on_other_classes(tmp_path):
    """Same number held out, different classes: the head shape matches, so without this
    check load_state_dict would succeed and every column would mean another class."""
    ckpt, _ = _train(tmp_path, ["valve_ball"])
    cfg = _cfg(tmp_path, ["valve_globe"])
    cfg.fgc.resume_from = str(ckpt.parent)
    with pytest.raises(ValueError, match="same fgc.exclude_classes"):
        FGCTrainPipeline(cfg).run()


def test_resume_with_the_same_exclusion_continues(tmp_path):
    ckpt, _ = _train(tmp_path, _HELD_OUT)
    cfg = _cfg(tmp_path, _HELD_OUT)
    cfg.fgc.resume_from = str(ckpt.parent)
    cfg.fgc.epochs = 2
    FGCTrainPipeline(cfg).run()
