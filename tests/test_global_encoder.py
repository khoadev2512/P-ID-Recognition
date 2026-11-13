"""Tests for the single-global-encoder configuration (cfg.fgc.global_encoder).

The default Stage-2 setup trains one head per fgc_group. That is right for closed-set
within-family classification but wrong for the one-shot ISO path, because per-group
heads learn SEPARATE embedding spaces and cosines from different spaces cannot be
compared — a single ISO prototype directory and a vocabulary-wide "Other" gate both
need every class in one space. `fgc.global_encoder=true` is that switch, and it has to
reach three stages consistently:

  pid-prep-crops   extract crops for EVERY symbol, not only fgc_group ones
  pid-fgc-train    train one head (GLOBAL_GROUP) over the whole fine vocabulary
  pid-fgc-infer    load __global__.pt (closed-set, PLAN B6c)

These tests pin the wiring, not the training: a class silently missing from the crop
tree, or a group loop that still iterates fgc_groups, would both yield an encoder whose
embedding space is quietly narrower than the vocabulary it is supposed to separate.
"""

from __future__ import annotations

import pytest

from utils.classmap import GLOBAL_GROUP, ClassMap


def _classmap() -> ClassMap:
    """A coarse map shaped like DigitizePID's: two multi-fine families that get heads,
    and two 1:1 families that the per-group path has no head for."""
    return ClassMap(
        coarse_classes=["valve", "instrument", "flange", "reducer"],
        fine_to_coarse={
            "valve_gate": "valve",
            "valve_ball": "valve",
            "bubble_field": "instrument",
            "bubble_logic": "instrument",
            "flange": "flange",
            "reducer": "reducer",
        },
        fgc_groups=["valve", "instrument"],
        rare_fine=[],
    )


# ------------------------------------------------------------------ #
# classmap: the GLOBAL_GROUP sentinel
# ------------------------------------------------------------------ #


def test_the_yaml_fixture_matches_the_in_memory_classmap(tmp_path):
    """Keeps the two fixtures from drifting: the on-disk classes.yaml the pipeline tests
    load must describe the same vocabulary as the in-memory map used above."""
    from utils.classmap import load_classmap

    path = tmp_path / "classes.yaml"
    path.write_text(_CLASSES_YAML)
    on_disk = load_classmap(path)
    assert on_disk.fine_classes == _classmap().fine_classes
    assert on_disk.fgc_groups == _classmap().fgc_groups
    assert on_disk.coarse_classes == _classmap().coarse_classes


def test_global_group_resolves_to_the_whole_fine_vocabulary():
    """Including the 1:1 families — the ISO vocabulary has plenty of those (pump,
    vessel, piping...), and leaving them out is exactly the gap this switch closes."""
    classmap = _classmap()
    assert classmap.fine_of_group(GLOBAL_GROUP) == classmap.fine_classes
    assert len(classmap.fine_of_group(GLOBAL_GROUP)) == 6
    assert "flange" in classmap.fine_of_group(GLOBAL_GROUP)


def test_global_group_preserves_classes_yaml_order():
    """Head column order comes from this list, so it must stay the file's order --
    a reordering would silently invalidate every saved checkpoint's label mapping."""
    assert _classmap().fine_of_group(GLOBAL_GROUP) == [
        "valve_gate",
        "valve_ball",
        "bubble_field",
        "bubble_logic",
        "flange",
        "reducer",
    ]


def test_per_group_lookup_is_unchanged_by_the_sentinel():
    classmap = _classmap()
    assert classmap.fine_of_group("valve") == ["valve_gate", "valve_ball"]
    assert classmap.fine_of_group("flange") == ["flange"]
    assert classmap.fine_of_group("not_a_class") == []


def test_global_group_name_cannot_collide_with_a_real_coarse_class():
    assert GLOBAL_GROUP not in _classmap().coarse_classes
    assert GLOBAL_GROUP.startswith("__") and GLOBAL_GROUP.endswith("__")


# ------------------------------------------------------------------ #
# fgc_train: which heads get trained
# ------------------------------------------------------------------ #

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from omegaconf import OmegaConf  # noqa: E402

from pipeline.extract_crops import ExtractCropsPipeline  # noqa: E402
from pipeline.fgc_infer import FGCInferPipeline  # noqa: E402
from pipeline.fgc_train import FGCTrainPipeline  # noqa: E402

# The same vocabulary as _classmap(), in classes.yaml form. Written to disk rather than
# injected, because BasePipeline (and FGCInferPipeline's constructor in particular) loads
# canonical/classes.yaml eagerly -- and because the fine: block's ORDER is what defines
# the global head's column order, which is worth exercising through the real loader.
_CLASSES_YAML = """
coarse: [valve, instrument, flange, reducer]
fine:
  valve_gate: valve
  valve_ball: valve
  bubble_field: instrument
  bubble_logic: instrument
  flange: flange
  reducer: reducer
fgc_groups: [valve, instrument]
"""


def _cfg(tmp_path, global_encoder: bool):
    canonical = tmp_path / "data" / "canonical"
    canonical.mkdir(parents=True, exist_ok=True)
    (canonical / "classes.yaml").write_text(_CLASSES_YAML)
    return OmegaConf.create(
        {
            "data_root": str(tmp_path / "data"),
            "run_dir": str(tmp_path / "runs"),
            "device": "cpu",
            "fgc": {
                "global_encoder": global_encoder,
                "metric_learning": True,
                "ckpt_dir": None,
                "route": {"min_score": 0.25, "other_min_cosine": 0.35},
                "arcface": {"m": 0.5, "s": 30.0, "embed_dim": 32},
                "backbone": "resnet18",
                "pretrained": False,
                "imgsz": 32,
            },
            "eval": {"split": "val"},
        }
    )


def _pipeline(cls, tmp_path, global_encoder: bool):
    """Build a pipeline over a tmp data_root holding just classes.yaml."""
    return cls(_cfg(tmp_path, global_encoder))


def test_train_iterates_one_head_per_group_by_default(tmp_path):
    pipeline = _pipeline(FGCTrainPipeline, tmp_path, global_encoder=False)
    assert pipeline.training_groups() == ["valve", "instrument"]


def test_train_iterates_a_single_global_head_when_enabled(tmp_path):
    pipeline = _pipeline(FGCTrainPipeline, tmp_path, global_encoder=True)
    assert pipeline.training_groups() == [GLOBAL_GROUP]


def test_the_global_head_is_sized_to_the_whole_vocabulary(tmp_path):
    """The number that matters: the global head must have one class center per fine
    class in the vocabulary (6 here), not per fine class of one family (2)."""
    pipeline = _pipeline(FGCTrainPipeline, tmp_path, global_encoder=True)
    (group,) = pipeline.training_groups()
    assert len(pipeline.classmap.fine_of_group(group)) == 6


# ------------------------------------------------------------------ #
# extract_crops: which symbols get cropped
# ------------------------------------------------------------------ #


def _categories() -> dict[int, dict]:
    classmap = _classmap()
    return {
        i: {"id": i, "name": fine, "supercategory": coarse}
        for i, (fine, coarse) in enumerate(classmap.fine_to_coarse.items())
    }


def test_crops_default_to_fgc_group_symbols_only(tmp_path):
    pipeline = _pipeline(ExtractCropsPipeline, tmp_path, global_encoder=False)
    categories = _categories()
    anns = [{"category_id": i} for i in categories]

    kept = pipeline._wanted_anns(anns, categories)
    kept_coarse = {categories[a["category_id"]]["supercategory"] for a in kept}
    assert kept_coarse == {"valve", "instrument"}
    assert len(kept) == 4


def test_crops_cover_every_symbol_under_a_global_encoder(tmp_path):
    """The 1:1 families (flange, reducer) must be cropped too -- otherwise they are
    absent from the embedding space the global encoder is meant to separate."""
    pipeline = _pipeline(ExtractCropsPipeline, tmp_path, global_encoder=True)
    categories = _categories()
    anns = [{"category_id": i} for i in categories]

    kept = pipeline._wanted_anns(anns, categories)
    assert len(kept) == len(anns) == 6
    kept_coarse = {categories[a["category_id"]]["supercategory"] for a in kept}
    assert kept_coarse == {"valve", "instrument", "flange", "reducer"}


def test_wanted_anns_on_an_image_with_no_symbols(tmp_path):
    for global_encoder in (False, True):
        pipeline = _pipeline(ExtractCropsPipeline, tmp_path, global_encoder=global_encoder)
        assert pipeline._wanted_anns([], _categories()) == []


# ------------------------------------------------------------------ #
# fgc_infer: refuses a global encoder with an actionable message
# ------------------------------------------------------------------ #


def test_fgc_infer_takes_a_global_encoder_and_asks_for_its_checkpoint(tmp_path):
    """Closed-set inference with __global__.pt (PLAN B6c; tests/test_closed_set_global.py).
    A ckpt_dir without it is reported by name, not as a missing valve.pt."""
    pipeline = _pipeline(FGCInferPipeline, tmp_path, global_encoder=True)
    with pytest.raises(ValueError, match="fgc.ckpt_dir"):
        pipeline.validate()
    (tmp_path / "ckpt").mkdir()
    pipeline.cfg.fgc.ckpt_dir = str(tmp_path / "ckpt")
    with pytest.raises(FileNotFoundError, match="__global__.pt"):
        pipeline.validate()


def test_fgc_infer_still_reports_a_missing_ckpt_dir_in_the_default_mode(tmp_path):
    """The new guard must not shadow the pre-existing precondition."""
    pipeline = _pipeline(FGCInferPipeline, tmp_path, global_encoder=False)
    with pytest.raises(ValueError, match="fgc.ckpt_dir"):
        pipeline.validate()
