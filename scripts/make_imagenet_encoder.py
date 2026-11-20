"""B4a: the ImageNet-only encoder baseline, saved as a pid-fgc-train checkpoint.

docs/PLAN_STAGE2_ONESHOT.md B4a. The question it answers: does training on DigitizePID help
or hurt one-shot transfer to classes the encoder never saw, compared with the raw features
the training started from? So this is the backbone R1a / R1b were initialised with
(`fgc.backbone` with `fgc.pretrained=true`: timm's default ImageNet weights for that name),
wrapped in ArcFaceModel and saved WITHOUT any training step, in the format pid-fgc-train
writes, so pid-iso-prototypes and `pid-eval eval.mode=oneshot` read it unchanged.

What the checkpoint says about itself:
    fine_classes      []: it trained on no P&ID class. The ArcFace head therefore has zero
                      rows (head.weight is (0, embed_dim)); one-shot never uses the head.
    excluded_classes  every fine class of classes.yaml, so no class is "held out without
                      being declared" (utils.oneshot_metrics.assign_groups) and the P1 leak
                      guard passes because nothing was trained on, not because of a list.
    epoch, val_acc,   None: there was no epoch and no validation. Not -1 / 0.0, which would
    best_val_acc      read as a (bad) trained run.
    imagenet_only     True, with the timm weight tag in `pretrained_weights`.

The embedding is the backbone's global-average-pooled feature with nothing on top: with
resnet34 (512 features) and fgc.arcface.embed_dim=512, ArcFaceModel.proj is Identity. A
backbone whose width differs from embed_dim would get an untrained random Linear projection,
which is not "ImageNet features"; that is refused, naming the embed_dim to pass instead.

Also the frozen DINOv2 encoder of PLAN_PHASE_C Bước 2B (self-supervised ViT-S/14, LVD-142M, no
labels): the same checkpoint layout with
    fgc.backbone=vit_small_patch14_dinov2.lvd142m fgc.imgsz=224 fgc.arcface.embed_dim=384
(384 = its width, so proj stays Identity; utils.model.backbone_kwargs builds the ViT for 224 px).
Pass the same three overrides to every stage that reads that checkpoint. `pretrained_only` is
True for both; `imagenet_only` only for weights from ImageNet labels (not DINOv2).

Run from the repo root; Hydra overrides as for src/run_pipeline.py:
    .venv/bin/python scripts/make_imagenet_encoder.py --out runs/__imagenet__.pt
    python scripts/make_imagenet_encoder.py --out "$ENC_DIR/__imagenet__.pt" \
        data_root=$LOCAL_DATA_ROOT

`data_root` is read for classes.yaml only. Downloads the timm weights if they are not cached.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.classmap import load_classmap  # noqa: E402
from utils.cli import CONFIG_DIR  # noqa: E402
from utils.model import ArcFaceModel  # noqa: E402
from utils.paths import DataPaths  # noqa: E402

logger = logging.getLogger(__name__)


def build_imagenet_checkpoint(cfg: DictConfig) -> dict:
    """The untrained ArcFaceModel as a pid-fgc-train checkpoint dict (see module docstring)."""
    if not bool(cfg.fgc.pretrained):
        raise ValueError(
            "fgc.pretrained=false would save randomly initialised features, not the ImageNet "
            "baseline; leave fgc.pretrained=true"
        )
    model = ArcFaceModel(cfg, num_fine=0)
    if not isinstance(model.proj, nn.Identity):
        raise ValueError(
            f"{cfg.fgc.backbone} emits {model.backbone.num_features} features but "
            f"fgc.arcface.embed_dim={cfg.fgc.arcface.embed_dim}: the embedding would go "
            "through an untrained random projection. Pass "
            f"fgc.arcface.embed_dim={model.backbone.num_features} (here and to every stage "
            "that reads this checkpoint)"
        )
    model.eval()
    pretrained = getattr(model.backbone, "pretrained_cfg", None) or {}
    classes = load_classmap(DataPaths(root=Path(str(cfg.data_root))).classes_yaml).fine_classes
    return {
        "model": model.state_dict(),
        "epoch": None,
        "val_acc": None,
        "best_val_acc": None,
        "fine_classes": [],
        "excluded_classes": list(classes),
        "class_centers": model.class_centers().cpu(),
        # pretrained_only: no P&ID training step at all. imagenet_only (the B4a name): the
        # pretrained weights come from ImageNet labels; DINOv2 is self-supervised, not that.
        "pretrained_only": True,
        "imagenet_only": "dinov2" not in str(cfg.fgc.backbone),
        "pretrained_weights": str(pretrained.get("hf_hub_id") or cfg.fgc.backbone),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True, help="checkpoint to write (.pt)")
    args, overrides = ap.parse_known_args(argv)
    logging.basicConfig(level=logging.INFO)
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config", overrides=overrides)

    state = build_imagenet_checkpoint(cfg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.out)
    logger.info(
        "wrote %s: %s (%s), embed_dim=%d, proj=Identity, trained on 0 classes, %d declared "
        "excluded",
        args.out,
        cfg.fgc.backbone,
        state["pretrained_weights"],
        int(cfg.fgc.arcface.embed_dim),
        len(state["excluded_classes"]),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
