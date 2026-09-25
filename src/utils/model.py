"""ResNet-34 fine-grained classifier (report roadmap Phase 3).

Backbone from timm; one head per fgc_group sized to that group's fine-class count.
cfg.fgc.loss selects cross-entropy vs. Focal Loss (ref [1]) for rare classes.

cfg.fgc.metric_learning swaps the plain linear head for an ArcFace margin head (ref
[43], §4.2.3): the backbone learns an L2-normalized embedding, the head holds one
L2-normalized class-center vector per fine class, and the additive angular margin
pulls same-class embeddings into tight cones. Two payoffs for this project:

  1. sharper within-group separation on the long-tail valve/bubble families, and
  2. an "Other" gate at inference — a crop whose embedding is far (in cosine) from
     every class center is an out-of-vocabulary symbol, not a low-confidence known
     one. Plain softmax can't distinguish those (it's overconfident off-distribution);
     the angular geometry can. See pipeline.fgc_infer for the reject gate.

The classification decision at inference stays softmax-argmax over the head's logits
(unchanged output contract); the class centers are used ONLY as the reject signal.
"""

from __future__ import annotations

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class ArcMarginProduct(nn.Module):
    """Additive angular margin head (ArcFace, ref [43]).

    Holds a weight matrix W of shape (num_fine, embed_dim); each row is a class center.
    Both the input embedding and W are L2-normalized, so `embedding @ W.T` is the cosine
    of the angle to each class center. During TRAINING an additive margin `m` is added to
    the angle of the target class before scaling by `s` (this is what forces the cones to
    tighten). During EVAL/INFER no label is available, so the plain scaled cosine `s*cosθ`
    is returned — order-preserving, so downstream softmax-argmax / top-k are unaffected.

    The raw (un-scaled) cosine-to-nearest-center is what fgc_infer thresholds for "Other";
    `cosine(embedding)` exposes it directly so callers don't have to undo the `s` scaling.
    """

    def __init__(self, embed_dim: int, num_fine: int, s: float = 30.0, m: float = 0.5) -> None:
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.empty(num_fine, embed_dim))
        nn.init.xavier_uniform_(self.weight)
        # Precomputed margin constants (standard ArcFace numerically-stable form).
        self._cos_m = math.cos(m)
        self._sin_m = math.sin(m)
        # Beyond θ = π - m the cos(θ+m) curve is non-monotonic; clamp with the tangent
        # line past this threshold so the target logit never increases as θ grows.
        self._threshold = math.cos(math.pi - m)
        self._mm = math.sin(math.pi - m) * m

    def cosine(self, embedding: torch.Tensor) -> torch.Tensor:
        """Raw cos θ to every class center, (N, num_fine), in [-1, 1]. No scaling/margin.

        This is the quantity fgc_infer compares against fgc.route.other_min_cosine.
        """
        return F.linear(F.normalize(embedding), F.normalize(self.weight))

    def forward(self, embedding: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
        cosine = self.cosine(embedding)
        if target is None:
            # No label (eval/infer): plain scaled cosine, no margin.
            return cosine * self.s

        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=1e-9))
        phi = cosine * self._cos_m - sine * self._sin_m  # cos(θ + m)
        # Only apply the margin where it keeps the logit monotone in θ (see __init__).
        phi = torch.where(cosine > self._threshold, phi, cosine - self._mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, target.view(-1, 1), 1.0)
        output = one_hot * phi + (1.0 - one_hot) * cosine
        return output * self.s


class ArcFaceModel(nn.Module):
    """timm backbone -> embedding -> ArcMarginProduct head.

    The backbone is built with `num_classes=0` so timm returns a pooled feature vector
    (the embedding) rather than class logits; `embed_dim` is inferred from it and
    (optionally) projected to cfg.fgc.arcface.embed_dim so the class centers live in a
    fixed-width space regardless of backbone.
    """

    def __init__(self, cfg: DictConfig, num_fine: int) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            str(cfg.fgc.backbone), pretrained=bool(cfg.fgc.pretrained), num_classes=0
        )
        backbone_dim = int(self.backbone.num_features)
        embed_dim = int(cfg.fgc.arcface.embed_dim)
        # Identity when the backbone already emits the requested width, else a linear
        # projection — keeps the class-center space width fixed across backbones.
        self.proj = (
            nn.Identity() if backbone_dim == embed_dim else nn.Linear(backbone_dim, embed_dim)
        )
        self.head = ArcMarginProduct(
            embed_dim=embed_dim,
            num_fine=num_fine,
            s=float(cfg.fgc.arcface.s),
            m=float(cfg.fgc.arcface.m),
        )

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """Backbone -> projected embedding (pre-normalization). Used by the reject gate."""
        return self.proj(self.backbone(images))

    def forward(self, images: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.embed(images), target)

    @torch.no_grad()
    def class_centers(self) -> torch.Tensor:
        """L2-normalized class-center vectors (num_fine, embed_dim) for the reject gate."""
        return F.normalize(self.head.weight.detach())


def build_model(cfg: DictConfig, group: str, num_fine: int) -> nn.Module:
    """FGC model for one fgc_group.

    cfg.fgc.metric_learning=false -> plain linear-head timm classifier (the CE/focal
    baseline). true -> ArcFaceModel (angular-margin embedding + class centers), which
    also unlocks the "Other" reject gate in pipeline.fgc_infer.

    `group` is not consumed by timm itself — it exists so the caller (one model
    instance per fgc_group) documents intent and call sites can log/name checkpoints
    by group without threading a second variable around.
    """
    del group  # unused here; kept in the signature for caller-side clarity (see docstring)
    if bool(cfg.fgc.get("metric_learning", False)):
        return ArcFaceModel(cfg, num_fine)
    return timm.create_model(
        str(cfg.fgc.backbone), pretrained=bool(cfg.fgc.pretrained), num_classes=num_fine
    )


class FocalLoss(nn.Module):
    """Standard multi-class focal loss (ref [1]).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Computed via F.cross_entropy(..., reduction="none") to get per-sample -log(p_t),
    then p_t = exp(-ce_loss), then the (1 - p_t)^gamma focusing term down-weights
    already-easy (high p_t) samples so training doesn't get swamped by the frequent
    classes. `weight` (per-class alpha_t, e.g. inverse frequency) is deliberately
    applied as a per-sample multiplier on the finished focal term rather than passed
    into the F.cross_entropy(...) call: nn.CrossEntropyLoss/F.cross_entropy's own
    `weight=` kwarg multiplies the per-sample loss BEFORE we'd extract p_t, which
    would make p_t = exp(-ce_loss) recover p(y)^weight[y] instead of the true softmax
    probability and silently corrupt the (1 - p_t)^gamma focusing term.
    """

    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0) -> None:
        super().__init__()
        # Registered as a buffer (not a plain attribute) so criterion.to(device) moves
        # it along with the module, same as nn.CrossEntropyLoss does internally.
        self.register_buffer("weight", weight)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, target, reduction="none")  # -log(p_t), unweighted
        p_t = torch.exp(-ce_loss)
        focal = (1.0 - p_t) ** self.gamma * ce_loss
        if self.weight is not None:
            alpha_t = self.weight[target]  # per-sample class weight, i.e. alpha_t
            focal = alpha_t * focal
        return focal.mean()


def _class_weights(class_counts: list[int]) -> torch.Tensor:
    """Inverse-frequency per-class weights, normalized to average 1 across classes
    (so the overall loss scale doesn't shift with the number of fine classes in a
    group). Counts are clamped to >=1 to avoid a divide-by-zero for a fine class with
    no crops in this split.
    """
    counts = torch.tensor(class_counts, dtype=torch.float32).clamp(min=1.0)
    inv_freq = 1.0 / counts
    return inv_freq / inv_freq.sum() * len(counts)


def build_loss(cfg: DictConfig, class_counts: list[int]) -> nn.Module:
    """CE (nn.CrossEntropyLoss) or Focal Loss (cfg.fgc.loss), weighted by inverse
    class_counts frequency for either loss.
    """
    weight = _class_weights(class_counts)
    loss_name = str(cfg.fgc.loss)
    if loss_name == "focal":
        return FocalLoss(weight=weight, gamma=2.0)
    if loss_name == "ce":
        return nn.CrossEntropyLoss(weight=weight)
    raise ValueError(f"Unknown cfg.fgc.loss={loss_name!r}; expected 'ce' or 'focal'")
