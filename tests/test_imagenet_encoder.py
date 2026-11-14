"""Tests for scripts/make_imagenet_encoder.py — the ImageNet-only encoder baseline (PLAN B4a).

No download: timm's pretrained=True is replaced by a torchvision resnet34 with seeded random
weights copied into the timm trunk (same parameter names), so "the ImageNet weights" here are
those, and the checkpoint must reproduce that torchvision trunk exactly through the real loader.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")
torchvision = pytest.importorskip("torchvision")

_SCRIPTS = str(Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import make_imagenet_encoder  # noqa: E402

import utils.model as model_mod  # noqa: E402
from utils.model import checkpoint_excluded_classes, load_arcface_encoder  # noqa: E402

_CLASSES_YAML = """
coarse: [valve, instrument]
fine:
  Symbol_1: valve
  Symbol_2: valve
  Symbol_3: instrument
fgc_groups: [valve]
"""


@pytest.fixture
def trunk(monkeypatch):
    """A torchvision resnet34 trunk (fc removed) whose weights stand in for ImageNet's."""
    torch.manual_seed(0)
    tv = torchvision.models.resnet34(weights=None)
    weights = {k: v for k, v in tv.state_dict().items() if not k.startswith("fc.")}
    real_create = timm.create_model

    def fake_create(name, pretrained=False, **kw):
        model = real_create(name, pretrained=False, **kw)
        if pretrained:
            model.load_state_dict(weights)  # strict: same names as the timm trunk
        return model

    monkeypatch.setattr(model_mod.timm, "create_model", fake_create)
    tv.fc = torch.nn.Identity()
    return tv.eval()


def _cfg(tmp_path, **fgc):
    (tmp_path / "data" / "canonical").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "canonical" / "classes.yaml").write_text(_CLASSES_YAML)
    base = {
        "backbone": "resnet34",
        "pretrained": True,
        "imgsz": 64,
        "arcface": {"m": 0.5, "s": 30.0, "embed_dim": 512},
    }
    return OmegaConf.create({"data_root": str(tmp_path / "data"), "fgc": {**base, **fgc}})


def test_checkpoint_loads_and_embeds_like_the_torchvision_trunk(tmp_path, trunk):
    cfg = _cfg(tmp_path)
    state = make_imagenet_encoder.build_imagenet_checkpoint(cfg)
    assert state["fine_classes"] == [] and state["imagenet_only"] is True
    assert state["excluded_classes"] == ["Symbol_1", "Symbol_2", "Symbol_3"]
    assert state["epoch"] is None and state["val_acc"] is None and state["best_val_acc"] is None
    assert tuple(state["model"]["head.weight"].shape) == (0, 512)
    assert not any(k.startswith("proj.") for k in state["model"])  # Identity: no parameters
    path = tmp_path / "__imagenet__.pt"
    torch.save(state, path)

    # The loader rebuilds without pretrained weights: they must come from the file.
    load_cfg = OmegaConf.merge(cfg, {"fgc": {"pretrained": False}})
    encoder, fine = load_arcface_encoder(load_cfg, path, "cpu")
    assert fine == [] and isinstance(encoder.proj, torch.nn.Identity)
    assert checkpoint_excluded_classes(path) == ["Symbol_1", "Symbol_2", "Symbol_3"]
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(encoder.embed(x), trunk(x), atol=1e-5)


def test_refuses_random_features_and_a_random_projection(tmp_path, trunk):
    with pytest.raises(ValueError, match="pretrained"):
        make_imagenet_encoder.build_imagenet_checkpoint(_cfg(tmp_path, pretrained=False))
    cfg = _cfg(tmp_path, arcface={"m": 0.5, "s": 30.0, "embed_dim": 128})
    with pytest.raises(ValueError, match="embed_dim=512"):
        make_imagenet_encoder.build_imagenet_checkpoint(cfg)


def test_main_composes_the_repo_config_with_overrides(tmp_path, trunk):
    _cfg(tmp_path)  # writes classes.yaml
    out = tmp_path / "enc" / "__imagenet__.pt"
    assert make_imagenet_encoder.main(["--out", str(out), f"data_root={tmp_path / 'data'}"]) == 0
    state = torch.load(out, map_location="cpu", weights_only=False)
    assert state["fine_classes"] == [] and len(state["excluded_classes"]) == 3
    assert state["pretrained_weights"].startswith("timm/resnet34")
