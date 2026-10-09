"""Unit tests for the ArcFace FGC path: the angular-margin head, the ArcFaceModel
wrapper, and the "Other" reject gate wiring.

torch/timm are heavy optional deps (the same ones fgc_train/fgc_infer need), so this
whole module is skipped when they're unavailable — matching test_eval_run_smoke.py's
importorskip guard. The head-level tests avoid pretrained-backbone downloads by
building ArcMarginProduct directly on synthetic embeddings; only the ArcFaceModel /
build_model tests touch timm, and they force pretrained=False.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from omegaconf import OmegaConf  # noqa: E402

from utils.classmap import OTHER_FINE_ID, UNREFINED_FINE_ID  # noqa: E402
from utils.model import ArcFaceModel, ArcMarginProduct, build_model  # noqa: E402


def _arcface_cfg(**overrides):
    """Minimal cfg with the fgc.* keys build_model/ArcFaceModel read. Uses the smallest
    timm model available offline-friendly; pretrained forced off so no weights download."""
    cfg = {
        "fgc": {
            "backbone": "resnet18",
            "pretrained": False,
            "metric_learning": True,
            "arcface": {"m": 0.5, "s": 30.0, "embed_dim": 64},
            "route": {"min_score": 0.25, "other_min_cosine": 0.35},
        }
    }
    base = OmegaConf.create(cfg)
    return OmegaConf.merge(base, OmegaConf.create(overrides)) if overrides else base


# ------------------------------------------------------------------ #
# ArcMarginProduct
# ------------------------------------------------------------------ #


def test_cosine_is_in_valid_range_and_right_shape():
    head = ArcMarginProduct(embed_dim=8, num_fine=4)
    embedding = torch.randn(5, 8)
    cos = head.cosine(embedding)
    assert cos.shape == (5, 4)
    # Cosine of an angle is always within [-1, 1] regardless of embedding magnitude.
    assert torch.all(cos <= 1.0 + 1e-5) and torch.all(cos >= -1.0 - 1e-5)


def test_forward_without_target_is_scaled_cosine():
    """Eval/infer path (target=None) must be exactly s * cosine — no margin — so
    downstream softmax-argmax / top-k ordering is preserved."""
    head = ArcMarginProduct(embed_dim=8, num_fine=4, s=30.0, m=0.5)
    embedding = torch.randn(3, 8)
    out = head(embedding, target=None)
    expected = head.cosine(embedding) * 30.0
    assert torch.allclose(out, expected, atol=1e-5)


def test_margin_only_lowers_the_target_logit():
    """With a margin, the target class's pre-scale value is cos(θ+m) < cosθ, so the
    target logit is strictly reduced vs. the no-margin forward; non-target columns are
    unchanged. This is the whole point of ArcFace (harder positive => tighter cone)."""
    head = ArcMarginProduct(embed_dim=8, num_fine=4, s=1.0, m=0.5)
    embedding = torch.randn(4, 8)
    target = torch.tensor([0, 1, 2, 3])

    no_margin = head(embedding, target=None)  # s=1 so this is plain cosine
    with_margin = head(embedding, target=target)

    for i, t in enumerate(target.tolist()):
        assert with_margin[i, t] < no_margin[i, t] + 1e-6
        # Non-target columns unchanged.
        for j in range(4):
            if j != t:
                assert torch.isclose(with_margin[i, j], no_margin[i, j], atol=1e-5)


# ------------------------------------------------------------------ #
# ArcFaceModel + build_model
# ------------------------------------------------------------------ #


def test_build_model_returns_arcface_when_metric_learning_on():
    model = build_model(_arcface_cfg(), group="valve", num_fine=3)
    assert isinstance(model, ArcFaceModel)


def test_build_model_returns_plain_head_when_metric_learning_off():
    cfg = _arcface_cfg()
    cfg.fgc.metric_learning = False
    model = build_model(cfg, group="valve", num_fine=3)
    assert not isinstance(model, ArcFaceModel)


def test_arcface_model_forward_shapes():
    model = ArcFaceModel(_arcface_cfg(), num_fine=3).eval()
    images = torch.randn(2, 3, 32, 32)
    # No target -> logits (scaled cosine), shape (N, num_fine).
    logits = model(images)
    assert logits.shape == (2, 3)
    # embed() width matches the configured embed_dim.
    assert model.embed(images).shape == (2, 64)
    # class_centers are L2-normalized rows.
    centers = model.class_centers()
    assert centers.shape == (3, 64)
    assert torch.allclose(centers.norm(dim=1), torch.ones(3), atol=1e-5)


def test_arcface_model_forward_with_target_runs():
    """Training path: passing labels must not error and must keep logit shape."""
    model = ArcFaceModel(_arcface_cfg(), num_fine=3).train()
    images = torch.randn(4, 3, 32, 32)
    target = torch.tensor([0, 1, 2, 0])
    logits = model(images, target)
    assert logits.shape == (4, 3)


# ------------------------------------------------------------------ #
# checkpoint round-trip (train saves class_centers; infer loads model)
# ------------------------------------------------------------------ #


def test_checkpoint_roundtrip_preserves_centers(tmp_path):
    cfg = _arcface_cfg()
    model = ArcFaceModel(cfg, num_fine=3)
    ckpt = {"model": model.state_dict(), "class_centers": model.class_centers().cpu()}
    path = tmp_path / "valve.pt"
    torch.save(ckpt, path)

    loaded = torch.load(path, map_location="cpu")
    fresh = ArcFaceModel(cfg, num_fine=3)
    fresh.load_state_dict(loaded["model"])
    # Reconstructed centers match what was saved (weights loaded correctly).
    assert torch.allclose(fresh.class_centers(), loaded["class_centers"], atol=1e-5)


# ------------------------------------------------------------------ #
# sentinel ids
# ------------------------------------------------------------------ #


def test_sentinel_ids_are_distinct_and_negative():
    # Both must be negative (never collide with a real fine_id >= 0) and distinct from
    # each other so eval can tell "Other" apart from "un-refined".
    assert OTHER_FINE_ID < 0 and UNREFINED_FINE_ID < 0
    assert OTHER_FINE_ID != UNREFINED_FINE_ID


# ------------------------------------------------------------------ #
# reject gate in fgc_infer._classify_crop
# ------------------------------------------------------------------ #


def _make_infer_pipeline(cfg, monkeypatch):
    """Construct FGCInferPipeline without touching the filesystem/classmap loader."""
    import numpy as np

    from pipeline import fgc_infer

    # Bypass BasePipeline.__init__ side effects: build the object and set just what
    # _classify_crop reads. A stub classmap with one fgc_group of two fine classes.
    class _StubClassmap:
        fgc_groups = ["valve"]

        def fine_of_group(self, group):
            return ["gate_valve", "globe_valve"]

    pipe = object.__new__(fgc_infer.FGCInferPipeline)
    pipe.device = torch.device("cpu")
    pipe._classmap = _StubClassmap()
    pipe.transform = lambda img: torch.zeros(3, 8, 8)  # deterministic dummy tensor
    pipe._fine_classes_per_group = {"valve": ["gate_valve", "globe_valve"]}
    pipe._reject_enabled = True
    pipe._other_min_cosine = float(cfg.fgc.route.other_min_cosine)
    return pipe, np


class _FakeArcFaceModel(ArcFaceModel):
    """ArcFaceModel with embed()/cosine forced to a fixed max cosine, so the gate's
    threshold decision can be tested without a real trained backbone."""

    def __init__(self, cfg, num_fine, forced_max_cosine):
        super().__init__(cfg, num_fine)
        self._forced = forced_max_cosine

    def embed(self, images):  # noqa: D102
        return torch.zeros(images.shape[0], int(self.head.weight.shape[1]))

    def forward(self, images, target=None):  # noqa: D102
        # Softmax-argmax should pick class 0 deterministically.
        n = images.shape[0]
        logits = torch.zeros(n, int(self.head.weight.shape[0]))
        logits[:, 0] = 10.0
        return logits


def _patch_cosine(model, value):
    n_classes = int(model.head.weight.shape[0])
    model.head.cosine = lambda emb: torch.full((emb.shape[0], n_classes), value)


def test_reject_gate_labels_other_below_threshold(monkeypatch):
    cfg = _arcface_cfg()
    cfg.fgc.route.other_min_cosine = 0.5
    pipe, np = _make_infer_pipeline(cfg, monkeypatch)

    from pipeline.fgc_infer import OTHER_CLASS_NAME

    model = _FakeArcFaceModel(cfg, num_fine=2, forced_max_cosine=0.1)
    _patch_cosine(model, 0.1)  # below 0.5 -> Other
    pipe._models = {"valve": model.eval()}

    crop = np.zeros((8, 8, 3), dtype=np.uint8)
    label, _score = pipe._classify_crop("valve", crop)
    assert label == OTHER_CLASS_NAME


def test_reject_gate_keeps_known_label_above_threshold(monkeypatch):
    cfg = _arcface_cfg()
    cfg.fgc.route.other_min_cosine = 0.5
    pipe, np = _make_infer_pipeline(cfg, monkeypatch)

    model = _FakeArcFaceModel(cfg, num_fine=2, forced_max_cosine=0.9)
    _patch_cosine(model, 0.9)  # above 0.5 -> keep argmax label (class 0)
    pipe._models = {"valve": model.eval()}

    crop = np.zeros((8, 8, 3), dtype=np.uint8)
    label, _score = pipe._classify_crop("valve", crop)
    assert label == "gate_valve"


def test_reject_gate_disabled_never_returns_other(monkeypatch):
    cfg = _arcface_cfg()
    cfg.fgc.route.other_min_cosine = 0.99  # would reject everything IF enabled
    pipe, np = _make_infer_pipeline(cfg, monkeypatch)
    pipe._reject_enabled = False  # gate off

    model = _FakeArcFaceModel(cfg, num_fine=2, forced_max_cosine=0.1)
    _patch_cosine(model, 0.1)
    pipe._models = {"valve": model.eval()}

    crop = np.zeros((8, 8, 3), dtype=np.uint8)
    label, _score = pipe._classify_crop("valve", crop)
    assert label == "gate_valve"  # never Other when the gate is disabled
