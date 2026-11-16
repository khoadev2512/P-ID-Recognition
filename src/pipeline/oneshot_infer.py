"""One-shot inference over Stage-1 detections (PLAN B6a, GĐ4): crop -> ISO prototype.

Used by pipeline.fgc_infer when `fgc.oneshot=true`. Each detection crop is embedded by the
frozen ArcFace encoder and matched against the ISO prototype directory; the result goes to
a NEW file, derived/detections_oneshot/<split>/<image_id>.txt, so the 7-column
`detections_fgc` contract the closed-set path writes is left alone.

Output row (one detection per line, whitespace-separated; no field contains a space):

    label best_name cx cy w h det_score best_cos second_cos coarse_id

    label        the decision: an ISO fine_name, `__other__` (gated out by
                 oneshot.other_min_cosine) or `__unrefined__` (never classified: score below
                 fgc.route.min_score, or an empty crop)
    best_name    the UN-GATED best prototype (same as label unless label is `__other__`;
                 `__unrefined__` for an unrefined row)
    cx cy w h    Stage-1 box, YOLO-normalized
    det_score    Stage-1 detector confidence
    best_cos     cosine of best_name, un-gated (see below)
    second_cos   the same quantity for the best OTHER prototype (top-1/top-2 margin)
    coarse_id    Stage-1 coarse class id

Why un-gated scores are saved: the "Other" threshold tau is chosen LATER (B6b, rules in
PLAN B8) by sweeping saved scores, so inference must not bake one in. `label` carries the
gate at oneshot.other_min_cosine only so a run is readable on its own; B6b recomputes it.

What `best_cos` / `second_cos` are, per oneshot.aggregate:
  mean  the query views are averaged into one vector; best_cos is its cosine to the nearest
        prototype and second_cos its cosine to the runner-up. The gate is best_cos < tau, so
        sweeping tau over best_cos reproduces `label` exactly.
  vote  every view names its nearest prototype, the most frequent name wins (best_name);
        best_cos is the MEAN, over the winning views, of their cosine to best_name, and
        second_cos the mean over the same views of their best cosine to a different
        prototype (a per-view margin, averaged). Un-gated, so all views vote. Under the
        gate only views with cosine >= tau vote, so the gated winner can differ from
        best_name when views disagree; best_cos < tau => `__other__` is then an
        approximation of the real gated vote (exact whenever the views agree).
Rows labelled `__unrefined__` carry nan cosines: they are not decisions and must be dropped
(not counted as a tau-rejection) when sweeping.

`oneshot.input_norm` (optional, `++`; null = off): canonical input form of every query crop
(utils.input_norm, e.g. `ink_letterbox` = PLAN_PHASE_C A4); the prototypes and the CL2N center
must have been built with the same value (checked).

`oneshot.center` (optional, `++`): a CL2N center file (utils.oneshot.save_center, built by
eval.mode=oneshot ++eval.oneshot.protocol=center). Every query view and every prototype is
then centered and renormalized before the cosine, so the cosines (and the tau chosen on them)
live in a different space: re-pick tau with the same center.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image

from utils import bbox_utils
from utils.classmap import OTHER_CLASS_NAME
from utils.crop_box import cut
from utils.input_norm import input_norm_fn
from utils.model import checkpoint_excluded_classes, checkpoint_sha256, load_arcface_encoder
from utils.oneshot import (
    Encoder,
    PrototypeDirectory,
    ViewSpec,
    center_embeddings,
    embed_images,
    load_center,
    make_views,
)
from utils.types import BBox

logger = logging.getLogger(__name__)

# A detection that was never classified (see module docstring). Not a fine_name.
UNREFINED_LABEL = "__unrefined__"


@dataclass(frozen=True)
class OneShotDetection:
    """One row of derived/detections_oneshot/<split>/<image_id>.txt."""

    label: str
    best_name: str
    box: BBox
    score: float
    best_cos: float
    second_cos: float
    coarse_id: int


def write_oneshot_detections(path: Path, rows: list[OneShotDetection]) -> None:
    lines = [
        f"{r.label} {r.best_name} {r.box.x:.6f} {r.box.y:.6f} {r.box.w:.6f} {r.box.h:.6f} "
        f"{r.score:.6f} {r.best_cos:.6f} {r.second_cos:.6f} {r.coarse_id}"
        for r in rows
    ]
    Path(path).write_text("\n".join(lines) + "\n" if lines else "")


def read_oneshot_detections(path: Path) -> list[OneShotDetection]:
    """Parse a detections_oneshot file (column order in the module docstring)."""
    rows: list[OneShotDetection] = []
    with Path(path).open("r") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 10:
                raise ValueError(f"{path}: expected 10 columns, got {len(parts)}: {line!r}")
            cx, cy, w, h, score, best_cos, second_cos = (float(v) for v in parts[2:9])
            rows.append(
                OneShotDetection(
                    label=parts[0],
                    best_name=parts[1],
                    box=BBox(x=cx, y=cy, w=w, h=h),
                    score=score,
                    best_cos=best_cos,
                    second_cos=second_cos,
                    coarse_id=int(parts[9]),
                )
            )
    return rows


def check_encoder_matches_prototypes(
    directory: PrototypeDirectory, encoder_ckpt: Path, fine_classes: list[str], cfg: DictConfig
) -> None:
    """Raise if `directory` was not built with this encoder.

    Prototypes from one encoder matched against crops embedded by another are unrelated
    vectors: nothing crashes, the cosines are just noise. pid-iso-prototypes records the
    encoder's training classes, excluded classes, backbone, embed_dim, imgsz and (newer
    files) the checkpoint's sha256 in `meta`; every recorded field is compared. The file
    path is NOT compared (Colab and local paths differ). A directory with none of those
    fields (built before provenance existed) loads with a warning.
    """
    meta = directory.meta
    recorded = {
        "encoder_train_classes": list(fine_classes),
        "encoder_excluded_classes": checkpoint_excluded_classes(encoder_ckpt),
        "backbone": str(cfg.fgc.backbone),
        "embed_dim": int(cfg.fgc.arcface.embed_dim),
        "imgsz": int(cfg.fgc.imgsz),
    }
    problems = [
        f"{key}: prototypes={meta[key]!r} vs encoder={value!r}"
        for key, value in recorded.items()
        if key in meta and meta[key] != value
    ]
    norm = cfg.oneshot.get("input_norm", None) or None
    if (meta.get("input_norm") or None) != norm:  # absent = built before the key = off
        problems.append(
            f"input_norm: prototypes={meta.get('input_norm')!r} vs oneshot.input_norm={norm!r}"
        )
    sha = meta.get("encoder_sha256")
    if sha is not None and sha != checkpoint_sha256(encoder_ckpt):
        problems.append("encoder_sha256: the checkpoint file differs from the one used")
    if problems:
        raise ValueError(
            f"oneshot.prototypes was not built with oneshot.encoder_ckpt={encoder_ckpt} "
            f"(built from {meta.get('encoder_ckpt')!r}); rebuild it with pid-iso-prototypes "
            "or point at the matching encoder. Mismatch: " + "; ".join(problems)
        )
    if not any(k in meta for k in recorded) and sha is None:
        logger.warning(
            "[oneshot] prototype directory carries no encoder provenance (built by an older "
            "pid-iso-prototypes); cannot verify it matches %s",
            encoder_ckpt,
        )


class OneShotClassifier:
    """Classify detection crops against an ISO prototype directory.

    Routing by the Stage-1 coarse class is optional and OFF by default: on ISO drawings the
    DigitizePID detector files pumps etc. under whichever coarse class it knows, so
    restricting candidates by it would lock in its mistakes. With `coarse_map` given, a
    Stage-1 class restricts matching to the mapped ISO coarse classes; a class missing from
    the map (or whose mapped prototypes were all excluded) falls back to the whole
    directory, logged once per class. Example (HANDOFF §3, DigitizePID -> ISO):
    {"reducer": ["piping"], "flange": ["piping"], "flow_direction": ["piping"],
     "instrument": ["agitator"]}.
    """

    def __init__(
        self,
        encoder: Encoder,
        directory: PrototypeDirectory,
        cfg: DictConfig,
        transform,
        device: torch.device | str = "cpu",
    ) -> None:
        osh = cfg.oneshot
        self.encoder = encoder
        self.transform = transform
        self.device = device
        exclude = list(osh.get("exclude_prototypes", None) or [])
        self.directory = directory.without(exclude) if exclude else directory
        self.views = ViewSpec.from_cfg(osh.query_views)
        # Canonical input form (utils.input_norm), the same the prototypes were built with.
        self.input_norm = osh.get("input_norm", None) or None
        self._prep = input_norm_fn(self.input_norm)
        self.background = tuple(int(c) for c in osh.get("background", (255, 255, 255)))
        self.min_cosine = float(osh.other_min_cosine)
        self.aggregate = str(osh.get("aggregate", "vote"))
        self.route_by_coarse = bool(osh.get("route_by_coarse", False))
        coarse_map = osh.get("coarse_map", None)
        if self.route_by_coarse and not coarse_map:
            raise ValueError(
                "oneshot.route_by_coarse=true needs oneshot.coarse_map "
                "(DigitizePID coarse class -> list of ISO coarse classes)"
            )
        self.coarse_map: dict[str, list[str]] = {
            str(k): [str(c) for c in v] for k, v in (coarse_map or {}).items()
        }
        self._subsets: dict[str, PrototypeDirectory] = {}
        self._warned: set[str] = set()
        # The encoder checkpoint's training classes; set by load_oneshot_classifier.
        self.encoder_classes: list[str] = []
        # CL2N center (set_center); None = raw cosines.
        self.center: torch.Tensor | None = None

    def set_center(self, center: torch.Tensor) -> None:
        """Match in CL2N space from now on: queries and prototypes minus `center`,
        renormalized (utils.oneshot.center_embeddings). Once only."""
        if self.center is not None:
            raise ValueError("the classifier is already centered")
        self.directory = self.directory.centered(center)
        self._subsets.clear()
        self.center = center

    def _candidates(self, coarse_class: str) -> PrototypeDirectory:
        if not self.route_by_coarse:
            return self.directory
        if coarse_class in self._subsets:
            return self._subsets[coarse_class]
        subset = self.directory
        if coarse_class not in self.coarse_map:
            self._warn_once(coarse_class, "is not in oneshot.coarse_map")
        else:
            try:
                subset = self.directory.subset_by_coarse(self.coarse_map[coarse_class])
            except ValueError:
                self._warn_once(coarse_class, "maps to no remaining prototype")
        self._subsets[coarse_class] = subset
        return subset

    def _warn_once(self, coarse_class: str, why: str) -> None:
        if coarse_class not in self._warned:
            self._warned.add(coarse_class)
            logger.warning(
                "[oneshot] Stage-1 class %r %s: matching the full directory", coarse_class, why
            )

    def classify(self, crop: np.ndarray, coarse_class: str) -> tuple[str, str, float, float]:
        """(label, best_name, best_cos, second_cos) for one crop; see module docstring."""
        directory = self._candidates(coarse_class)
        emb = self.embed_crop(crop)
        gated = directory.match_views(emb, self.min_cosine, self.aggregate)
        free = directory.match_views(emb, None, self.aggregate)
        label = OTHER_CLASS_NAME if gated.is_other else gated.name
        return label, free.name, free.cosine, self._second_cosine(directory, emb, free.name)

    def embed_crop(self, crop: np.ndarray) -> torch.Tensor:
        """The (V, D) unit embeddings of a crop's query views (oneshot.query_views)."""
        img = Image.fromarray(crop)
        if self._prep is not None:
            img = self._prep(img)
        views = make_views(img, self.views, self.background)
        emb = embed_images(self.encoder, views, self.transform, self.device)
        return emb if self.center is None else center_embeddings(emb, self.center)

    def _second_cosine(
        self, directory: PrototypeDirectory, emb: torch.Tensor, best_name: str
    ) -> float:
        best = directory.names.index(best_name)
        if len(directory) < 2:
            return math.nan
        if self.aggregate == "mean":
            cos = directory.cosines(emb.mean(dim=0))
            cos[best] = -math.inf
            return float(cos.max())
        cos = directory.cosines(emb)  # (V, N)
        winners = cos.argmax(dim=1) == best
        rest = cos.clone()
        rest[:, best] = -math.inf
        return float(rest.max(dim=1).values[winners].mean())

    def resolve(
        self,
        coarse_id: int,
        coarse_class: str,
        box: BBox,
        score: float,
        image_np: np.ndarray,
        image_w: int,
        image_h: int,
        min_score: float,
    ) -> OneShotDetection:
        label, best_name, best_cos, second_cos = (
            UNREFINED_LABEL,
            UNREFINED_LABEL,
            math.nan,
            math.nan,
        )
        if score >= min_score:
            crop = cut(image_np, bbox_utils.yolo_to_xyxy(box, image_w, image_h))
            if crop.size > 0:
                label, best_name, best_cos, second_cos = self.classify(crop, coarse_class)
        return OneShotDetection(label, best_name, box, score, best_cos, second_cos, coarse_id)


def load_oneshot_classifier(cfg: DictConfig, transform, device) -> OneShotClassifier:
    """Encoder + prototype directory from cfg.oneshot.{encoder_ckpt,prototypes}, with the
    encoder/prototype mismatch guard."""
    ckpt = Path(str(cfg.oneshot.encoder_ckpt))
    encoder, fine_classes = load_arcface_encoder(cfg, ckpt, device)
    directory = PrototypeDirectory.load(Path(str(cfg.oneshot.prototypes)))
    check_encoder_matches_prototypes(directory, ckpt, fine_classes, cfg)
    classifier = OneShotClassifier(encoder, directory, cfg, transform, device)
    classifier.encoder_classes = fine_classes
    center_path = cfg.oneshot.get("center", None)
    if center_path:
        center, meta = load_center(Path(str(center_path)))
        if meta.get("encoder_sha256") != checkpoint_sha256(ckpt):
            raise ValueError(
                f"oneshot.center {center_path} was built with encoder "
                f"{meta.get('encoder_checkpoint')}, not {ckpt} (sha256 differs)"
            )
        norm = cfg.oneshot.get("input_norm", None) or None
        if (meta.get("input_norm") or None) != norm:
            raise ValueError(
                f"oneshot.center {center_path} was built with input_norm="
                f"{meta.get('input_norm')!r}, this run has oneshot.input_norm={norm!r}"
            )
        if center.shape[0] != directory.dim:
            raise ValueError(
                f"oneshot.center has dim {center.shape[0]}, the prototypes {directory.dim}"
            )
        classifier.set_center(center)
    return classifier


def require_oneshot_inputs(cfg: DictConfig) -> None:
    """Clear errors for an unset/missing encoder or prototype file (before any loading)."""
    osh = cfg.oneshot
    if osh.get("encoder_ckpt") is None:
        raise ValueError(
            "fgc.oneshot=true needs oneshot.encoder_ckpt, the ArcFace checkpoint the "
            "prototypes were built with, e.g.:\n"
            "    pid-fgc-infer fgc.oneshot=true oneshot.encoder_ckpt=<run>/__global__.pt"
        )
    if not Path(str(osh.encoder_ckpt)).exists():
        raise FileNotFoundError(f"oneshot.encoder_ckpt does not exist: {osh.encoder_ckpt}")
    if osh.get("prototypes") is None:
        raise ValueError("fgc.oneshot=true needs oneshot.prototypes (pid-iso-prototypes output)")
    if not Path(str(osh.prototypes)).exists():
        raise FileNotFoundError(
            f"oneshot.prototypes does not exist: {osh.prototypes} "
            "(build it with pid-iso-prototypes)"
        )
    if osh.get("center", None) and not Path(str(osh.center)).exists():
        raise FileNotFoundError(f"oneshot.center does not exist: {osh.center}")
    if bool(osh.get("route_by_coarse", False)) and not osh.get("coarse_map", None):
        raise ValueError("oneshot.route_by_coarse=true needs oneshot.coarse_map")


__all__ = [
    "OneShotClassifier",
    "OneShotDetection",
    "UNREFINED_LABEL",
    "read_oneshot_detections",
    "write_oneshot_detections",
]
