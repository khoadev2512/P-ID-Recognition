"""One-shot inference in pid-fgc-infer (fgc.oneshot=true, PLAN B6a).

Logic tests inject a deterministic `_FakeEncoder` and a directory built with
`build_prototypes`; the loading/guard tests go through `load_arcface_encoder` with a tiny
real (untrained) ArcFace checkpoint, saved the way pid-fgc-train does.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from omegaconf import OmegaConf
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
from torchvision import transforms  # noqa: E402

from pipeline.fgc_infer import FGCInferPipeline  # noqa: E402
from pipeline.oneshot_infer import (  # noqa: E402
    UNREFINED_LABEL,
    OneShotClassifier,
    OneShotDetection,
    read_oneshot_detections,
    write_oneshot_detections,
)
from utils.classmap import GLOBAL_GROUP, OTHER_CLASS_NAME  # noqa: E402
from utils.dataset import build_eval_transform  # noqa: E402
from utils.model import build_model, checkpoint_sha256, load_arcface_encoder  # noqa: E402
from utils.oneshot import PrototypeDirectory, ViewSpec, build_prototypes  # noqa: E402
from utils.types import BBox  # noqa: E402

_CLASSES_YAML = """
coarse: [valve, instrument, flange]
fine:
  valve_gate: valve
  valve_ball: valve
  bubble_field: instrument
  bubble_logic: instrument
  flange: flange
fgc_groups: [valve, instrument]
"""
_NAMES = ["gate", "ball", "pump", "pipeline"]
_COARSE_OF = {"gate": "valve", "ball": "valve", "pump": "pump", "pipeline": "piping"}
_SIDE, _DIM = 8, 16


class _FakeEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.register_buffer("w", torch.randn(3 * _SIDE * _SIDE, _DIM, generator=g))

    def embed(self, images):
        return images.flatten(1) @ self.w


def _tf():
    return transforms.Compose([transforms.Resize((_SIDE, _SIDE)), transforms.ToTensor()])


def _glyph(seed, size=(30, 30)):
    g = torch.Generator().manual_seed(seed)
    px = (torch.rand(size[1], size[0], 3, generator=g) * 255).to(torch.uint8)
    return Image.fromarray(px.numpy(), mode="RGB")


def _templates():
    return {n: _glyph(i) for i, n in enumerate(_NAMES)}


def _directory(meta=None):
    return build_prototypes(_FakeEncoder(), _templates(), _tf(), coarse_of=_COARSE_OF, meta=meta)


def _cfg(tmp_path, **over):
    canonical = tmp_path / "data" / "canonical"
    canonical.mkdir(parents=True, exist_ok=True)
    (canonical / "classes.yaml").write_text(_CLASSES_YAML)
    cfg = OmegaConf.create(
        {
            "data_root": str(tmp_path / "data"),
            "run_dir": str(tmp_path / "runs"),
            "device": "cpu",
            "fgc": {
                "oneshot": True,
                "metric_learning": True,
                "route": {"min_score": 0.25},
                "arcface": {"m": 0.5, "s": 30.0, "embed_dim": 32},
                "backbone": "resnet18",
                "pretrained": False,
                "imgsz": 32,
                "exclude_classes": [],
                "aug": {"subpart": {"p": 0.0}},
            },
            "detector": {"label_level": "coarse"},
            "eval": {"split": "val"},
            "oneshot": {
                "prototypes": str(tmp_path / "protos.pt"),
                "encoder_ckpt": None,
                "background": [255, 255, 255],
                "query_views": {"scales": [1.0], "orientations": [0]},
                "aggregate": "vote",
                "other_min_cosine": -1.0,
            },
        }
    )
    return OmegaConf.merge(cfg, OmegaConf.create(over))


def _classifier(cfg, directory=None):
    return OneShotClassifier(_FakeEncoder(), directory or _directory(), cfg, _tf())


def _arr(img):
    return np.array(img)


# ------------------------------------------------------------------ #
# classification logic
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("aggregate", ["vote", "mean"])
def test_a_crop_equal_to_a_template_gets_that_name(tmp_path, aggregate):
    clf = _classifier(_cfg(tmp_path, oneshot={"aggregate": aggregate}))
    label, best, best_cos, second_cos = clf.classify(_arr(_glyph(2)), "valve")
    assert label == best == "pump"
    assert best_cos == pytest.approx(1.0, abs=1e-5)
    assert second_cos < best_cos


def test_an_unrelated_crop_is_other_but_keeps_name_and_cosines(tmp_path):
    clf = _classifier(_cfg(tmp_path, oneshot={"other_min_cosine": 0.999}))
    label, best, best_cos, second_cos = clf.classify(_arr(_glyph(99)), "valve")
    assert label == OTHER_CLASS_NAME
    assert best in _NAMES and best_cos < 0.999 and second_cos <= best_cos


def test_vote_second_cosine_is_the_margin_over_the_winning_views(tmp_path):
    cfg = _cfg(
        tmp_path,
        oneshot={"query_views": {"scales": [1.0, 1.25], "orientations": [0]}},
    )
    clf = _classifier(cfg)
    _, best, best_cos, second_cos = clf.classify(_arr(_glyph(1)), "valve")
    assert best == "ball" and second_cos < best_cos


def test_routing_is_off_by_default_and_uses_the_full_directory(tmp_path):
    cfg = _cfg(tmp_path, oneshot={"coarse_map": {"valve": ["piping"]}})
    # `valve` maps to piping, but routing is off: a pump crop is still a pump.
    label = _classifier(cfg).classify(_arr(_glyph(2)), "valve")[0]
    assert label == "pump"


def test_routing_restricts_the_candidates(tmp_path):
    cfg = _cfg(tmp_path, oneshot={"route_by_coarse": True, "coarse_map": {"valve": ["valve"]}})
    label, best, *_ = _classifier(cfg).classify(_arr(_glyph(2)), "valve")
    assert label == best and best in {"gate", "ball"}


def test_an_unmapped_coarse_class_falls_back_to_the_full_directory_and_warns_once(tmp_path, caplog):
    cfg = _cfg(tmp_path, oneshot={"route_by_coarse": True, "coarse_map": {"valve": ["valve"]}})
    clf = _classifier(cfg)
    with caplog.at_level("WARNING"):
        for _ in range(3):
            assert clf.classify(_arr(_glyph(2)), "flange")[0] == "pump"
    assert sum("flange" in r.message for r in caplog.records) == 1


def test_routing_without_a_coarse_map_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="coarse_map"):
        _classifier(_cfg(tmp_path, oneshot={"route_by_coarse": True}))


def test_exclude_prototypes_removes_rows_and_rejects_unknown_names(tmp_path):
    clf = _classifier(_cfg(tmp_path, oneshot={"exclude_prototypes": ["pump"]}))
    assert "pump" not in clf.directory.names and len(clf.directory) == 3
    assert clf.classify(_arr(_glyph(2)), "valve")[1] != "pump"
    with pytest.raises(ValueError, match="nope"):
        _classifier(_cfg(tmp_path, oneshot={"exclude_prototypes": ["nope"]}))


# ------------------------------------------------------------------ #
# output file
# ------------------------------------------------------------------ #


def test_write_read_round_trip(tmp_path):
    rows = [
        OneShotDetection("gate", "gate", BBox(0.5, 0.5, 0.2, 0.1), 0.9, 0.8, 0.3, 4),
        OneShotDetection(OTHER_CLASS_NAME, "ball", BBox(0.1, 0.2, 0.3, 0.4), 0.7, 0.2, 0.1, 0),
        OneShotDetection(
            UNREFINED_LABEL, UNREFINED_LABEL, BBox(0.1, 0.1, 0.1, 0.1), 0.1, math.nan, math.nan, 2
        ),
    ]
    path = tmp_path / "a.txt"
    write_oneshot_detections(path, rows)
    got = read_oneshot_detections(path)
    assert [(r.label, r.best_name, r.coarse_id) for r in got] == [
        (r.label, r.best_name, r.coarse_id) for r in rows
    ]
    assert got[0].box == BBox(0.5, 0.5, 0.2, 0.1)
    assert got[0].best_cos == pytest.approx(0.8) and got[1].second_cos == pytest.approx(0.1)
    assert math.isnan(got[2].best_cos)
    write_oneshot_detections(path, [])
    assert read_oneshot_detections(path) == []


def _stage(tmp_path):
    data = tmp_path / "data"
    (data / "canonical" / "images").mkdir(parents=True)
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    Image.fromarray(img).save(data / "canonical" / "images" / "a.png")
    (data / "canonical" / "manifest.csv").write_text("id,split\na,val\n")
    det = data / "derived" / "detections" / "val"
    det.mkdir(parents=True)
    # valve box above min_score, instrument box below it, flange box on the image edge
    (det / "a.txt").write_text(
        "0 0.5 0.5 0.2 0.2 0.9\n1 0.3 0.3 0.1 0.1 0.1\n2 1.0 1.0 0.0 0.0 0.9\n"
    )
    return data


def test_run_writes_detections_oneshot_and_leaves_detections_fgc_alone(tmp_path):
    data = _stage(tmp_path)
    cfg = _cfg(tmp_path, oneshot={"other_min_cosine": 0.35})
    pipe = FGCInferPipeline(cfg)
    pipe._oneshot = _classifier(cfg)
    pipe.run()
    out = data / "derived" / "detections_oneshot" / "val" / "a.txt"
    rows = read_oneshot_detections(out)
    assert len(rows) == 3
    assert rows[0].best_name in _NAMES and rows[0].coarse_id == 0
    assert rows[0].label in {rows[0].best_name, OTHER_CLASS_NAME}
    assert rows[0].box == BBox(0.5, 0.5, 0.2, 0.2) and rows[0].score == pytest.approx(0.9)
    assert rows[1].label == UNREFINED_LABEL and rows[1].coarse_id == 1
    assert rows[2].label == UNREFINED_LABEL  # empty crop
    assert len(out.read_text().split("\n")[0].split()) == 10
    assert not (data / "derived" / "detections_fgc").exists()


# ------------------------------------------------------------------ #
# loading and the encoder/prototype guard (tiny real checkpoint)
# ------------------------------------------------------------------ #

_FINE = ["valve_gate", "valve_ball", "bubble_field", "bubble_logic", "flange"]


def _save_ckpt(cfg, path, fine_classes=_FINE, seed=0):
    torch.manual_seed(seed)
    model = build_model(cfg, GLOBAL_GROUP, len(fine_classes))
    torch.save(
        {"model": model.state_dict(), "fine_classes": list(fine_classes), "excluded_classes": []},
        path,
    )


def _real_prototypes(cfg, ckpt, **meta_over):
    encoder, fine = load_arcface_encoder(cfg, ckpt, "cpu")
    meta = {
        "encoder_ckpt": str(ckpt),
        "backbone": "resnet18",
        "embed_dim": 32,
        "imgsz": 32,
        "encoder_train_classes": fine,
        "encoder_excluded_classes": [],
        "encoder_sha256": checkpoint_sha256(ckpt),
    }
    meta.update(meta_over)
    d = build_prototypes(
        encoder, _templates(), build_eval_transform(cfg), coarse_of=_COARSE_OF, meta=meta
    )
    d.save(cfg.oneshot.prototypes)
    return d


def test_validate_names_a_missing_encoder_and_missing_prototypes(tmp_path):
    _stage(tmp_path)
    with pytest.raises(ValueError, match="oneshot.encoder_ckpt"):
        FGCInferPipeline(_cfg(tmp_path)).validate()
    cfg = _cfg(tmp_path, oneshot={"encoder_ckpt": str(tmp_path / "gone.pt")})
    with pytest.raises(FileNotFoundError, match="encoder_ckpt"):
        FGCInferPipeline(cfg).validate()
    ckpt = tmp_path / "enc.pt"
    cfg = _cfg(tmp_path, oneshot={"encoder_ckpt": str(ckpt)})
    _save_ckpt(cfg, ckpt)
    with pytest.raises(FileNotFoundError, match="prototypes"):
        FGCInferPipeline(cfg).validate()


def test_a_matching_encoder_loads_and_runs_through_load_arcface_encoder(tmp_path):
    _stage(tmp_path)
    ckpt = tmp_path / "enc.pt"
    cfg = _cfg(tmp_path, oneshot={"encoder_ckpt": str(ckpt)})
    _save_ckpt(cfg, ckpt)
    _real_prototypes(cfg, ckpt)
    pipe = FGCInferPipeline(cfg)
    assert pipe.validate() is True
    pipe.run()
    rows = read_oneshot_detections(
        tmp_path / "data" / "derived" / "detections_oneshot" / "val" / "a.txt"
    )
    assert rows[0].best_name in _NAMES and not math.isnan(rows[0].best_cos)


@pytest.mark.parametrize(
    "meta_over,other_classes,other_seed",
    [
        ({"encoder_train_classes": _FINE[:3]}, None, 0),  # R1a prototypes, R1b encoder
        ({}, None, 1),  # same classes, different weights -> sha256
        ({"encoder_excluded_classes": ["flange"]}, None, 0),
    ],
)
def test_prototype_encoder_mismatch_is_refused(tmp_path, meta_over, other_classes, other_seed):
    _stage(tmp_path)
    ckpt = tmp_path / "enc.pt"
    cfg = _cfg(tmp_path, oneshot={"encoder_ckpt": str(ckpt)})
    _save_ckpt(cfg, ckpt)
    _real_prototypes(cfg, ckpt, **meta_over)
    if other_seed:
        _save_ckpt(cfg, ckpt, seed=other_seed)  # the file the prototypes were not built from
    with pytest.raises(ValueError, match="not built with"):
        FGCInferPipeline(cfg).validate()


def test_a_directory_without_provenance_still_loads_with_a_warning(tmp_path, caplog):
    _stage(tmp_path)
    ckpt = tmp_path / "enc.pt"
    cfg = _cfg(tmp_path, oneshot={"encoder_ckpt": str(ckpt)})
    _save_ckpt(cfg, ckpt)
    PrototypeDirectory(names=["a", "b"], embeddings=torch.eye(2, 32), coarse=["x", "y"]).save(
        cfg.oneshot.prototypes
    )
    with caplog.at_level("WARNING"):
        assert FGCInferPipeline(cfg).validate() is True
    assert any("provenance" in r.message for r in caplog.records)


def test_closed_set_path_is_untouched_when_oneshot_is_off(tmp_path):
    cfg = _cfg(tmp_path, fgc={"oneshot": False, "ckpt_dir": None})
    with pytest.raises(ValueError, match="fgc.ckpt_dir"):
        FGCInferPipeline(cfg).validate()
    assert ViewSpec().n_views == 1
