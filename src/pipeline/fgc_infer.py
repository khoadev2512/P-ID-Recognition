"""Apply the FGC head(s) to detector output, producing final fine labels.

Routing policy (report §4.2.3, an implementation decision): a detection is passed to
the fine-grained head only when its coarse_class is in fgc_groups AND its score meets
cfg.fgc.route.min_score; otherwise the detection is left un-refined:

    - if its coarse class maps to exactly one fine class, that's unambiguous — use it
      directly, no model needed.
    - otherwise (coarse class has multiple fine children but wasn't routed, e.g. low
      score) the label stays the coarse class name (which is not, in general, itself a
      valid fine class -- downstream consumers should treat this as "un-refined").

This is where the two stages compose into the final list[Detection].
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from PIL import Image

from pipeline.base import BasePipeline
from utils import bbox_utils
from utils.cli import CONFIG_DIR
from utils.dataset import build_eval_transform
from utils.model import build_model
from utils.types import BBox

logger = logging.getLogger(__name__)


def _read_stage1_detections(path: Path) -> list[tuple[int, BBox, float]]:
    """Parse a Stage-1 detections file: `<coarse_class_id> <cx> <cy> <w> <h> <score>`
    per line (YOLO-normalized bbox + detector confidence) -- this exact path/format is
    pipeline.detect_infer's output contract (derived/detections/<split>/<image_id>.txt).

    Deliberately NOT reusing bbox_utils.read_yolo_txt: that helper explicitly discards
    a trailing confidence column, but we need `score` to carry through to routing and
    to the final 7-column output.
    """
    rows: list[tuple[int, BBox, float]] = []
    with Path(path).open("r") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            class_id = int(float(parts[0]))
            cx, cy, w, h, score = (float(v) for v in parts[1:6])
            rows.append((class_id, BBox(x=cx, y=cy, w=w, h=h), score))
    return rows


class FGCInferPipeline(BasePipeline):
    """Loads per-group FGC checkpoints and fills in fine_class/fine_score for every
    Stage-1 detection, writing derived/detections_fgc/<split>/<image_id>.txt.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.device = torch.device(
            "cuda" if str(cfg.device) == "cuda" and torch.cuda.is_available() else "cpu"
        )
        self.transform = build_eval_transform(cfg)
        self._models: dict[str, torch.nn.Module] | None = None
        self._fine_classes_per_group: dict[str, list[str]] = {
            group: self.classmap.fine_of_group(group) for group in self.classmap.fgc_groups
        }

    @property
    def models(self) -> dict[str, torch.nn.Module]:
        """Per-group classifiers, lazily loaded from cfg.fgc.ckpt_dir on first use."""
        if self._models is None:
            self._models = self._load_models()
        return self._models

    def _load_models(self) -> dict[str, torch.nn.Module]:
        ckpt_dir = self.cfg.fgc.ckpt_dir
        if ckpt_dir is None:
            raise ValueError(
                "cfg.fgc.ckpt_dir is not set. Point it at a trained-checkpoint directory "
                "produced by pid-fgc-train, e.g.:\n"
                "    pid-fgc-infer fgc.ckpt_dir=runs/<train-run-dir>/fgc_checkpoints"
            )
        ckpt_dir = Path(str(ckpt_dir))
        models: dict[str, torch.nn.Module] = {}
        for group in self.classmap.fgc_groups:
            fine_classes = self._fine_classes_per_group[group]
            ckpt_path = ckpt_dir / f"{group}.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Missing FGC checkpoint for group={group!r}: {ckpt_path}")
            model = build_model(self.cfg, group, len(fine_classes)).to(self.device)
            state = torch.load(ckpt_path, map_location=self.device)
            model.load_state_dict(state["model"])
            model.eval()
            models[group] = model
        return models

    def validate(self) -> bool:
        if self.cfg.fgc.ckpt_dir is None:
            raise ValueError(
                "cfg.fgc.ckpt_dir must be set to run FGC inference, e.g.:\n"
                "    pid-fgc-infer fgc.ckpt_dir=runs/<train-run-dir>/fgc_checkpoints"
            )
        ckpt_dir = Path(str(self.cfg.fgc.ckpt_dir))
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"fgc.ckpt_dir does not exist: {ckpt_dir}")

        split = str(self.cfg.eval.split)
        det_dir = self.paths.derived / "detections" / split
        if not det_dir.is_dir() or not any(det_dir.iterdir()):
            raise FileNotFoundError(
                f"Expected Stage-1 detections at {det_dir} (produced by pid-detect-infer); "
                "found none."
            )
        return True

    def run(self) -> None:
        self.validate()
        split = str(self.cfg.eval.split)
        image_ids = self._image_ids_for_split(split)

        det_dir = self.paths.derived / "detections" / split
        out_dir = self.paths.derived / "detections_fgc" / split
        out_dir.mkdir(parents=True, exist_ok=True)

        for image_id in image_ids:
            det_path = det_dir / f"{image_id}.txt"
            if not det_path.exists():
                # No Stage-1 output for this image (e.g. detector found nothing) --
                # nothing to refine.
                continue
            self._process_image(image_id, det_path, out_dir / f"{image_id}.txt")

    def _image_ids_for_split(self, split: str) -> list[str]:
        manifest = pd.read_csv(self.paths.manifest_csv)
        return manifest.loc[manifest["split"] == split, "id"].astype(str).tolist()

    def _process_image(self, image_id: str, det_path: Path, out_path: Path) -> None:
        rows = _read_stage1_detections(det_path)
        if not rows:
            out_path.write_text("")
            return

        image_path = self.paths.find_image(image_id)
        img = Image.open(image_path).convert("RGB")
        image_w, image_h = img.size
        image_np = np.array(img)

        out_lines: list[str] = []
        for coarse_id, box, score in rows:
            coarse_class = self.classmap.coarse_classes[coarse_id]
            fine_class, fine_score = self._resolve_fine_label(
                coarse_class, box, score, image_np, image_w, image_h
            )
            fine_id = self._safe_fine_id(fine_class, coarse_class)
            out_lines.append(
                f"{fine_id} {box.x:.6f} {box.y:.6f} {box.w:.6f} {box.h:.6f} "
                f"{score:.6f} {fine_score:.6f}"
            )

        out_path.write_text("\n".join(out_lines) + "\n")

    def _safe_fine_id(self, fine_class: str, coarse_class: str) -> int:
        """classmap.fine_id(), guarded.

        Every _resolve_fine_label branch is designed to return a genuine fine-class
        name -- EXCEPT one theoretical edge case the routing design can't fully close:
        a coarse class in fgc_groups (so it has >1 fine children, no 1:1 fallback)
        whose detection score falls below fgc.route.min_score, so it's never routed to
        a classifier and the label falls back to the (invalid-as-fine-class) coarse
        name. Rather than let that crash the whole inference run on one low-confidence
        box, emit a sentinel id (-1) and log it -- callers/eval can filter these out
        (they'd typically also be below eval.score_thr and dropped anyway).
        """
        try:
            return self.classmap.fine_id(fine_class)
        except KeyError:
            logger.warning(
                "coarse_class=%r has >1 fine children and was not routed to FGC "
                "(score below fgc.route.min_score) -- no valid fine_class to report; "
                "writing fine_class_id=-1.",
                coarse_class,
            )
            return -1

    def _resolve_fine_label(
        self,
        coarse_class: str,
        box: BBox,
        score: float,
        image_np: np.ndarray,
        image_w: int,
        image_h: int,
    ) -> tuple[str, float]:
        """Routing + fallback resolution (see module docstring).

        Every branch returns a genuine fine-class name EXCEPT the final fallback,
        which can return a bare coarse name when an fgc_group detection wasn't routed
        (score below threshold) and has no 1:1 fine fallback -- see _safe_fine_id,
        which guards exactly that case at the call site.
        """
        if self.classmap.needs_fgc(coarse_class) and score >= float(self.cfg.fgc.route.min_score):
            xyxy = bbox_utils.yolo_to_xyxy(box, image_w, image_h)
            xyxy_int = bbox_utils.clip_xyxy(xyxy, image_w, image_h)
            crop = bbox_utils.crop_xyxy(image_np, xyxy_int)
            if crop.size > 0:
                return self._classify_crop(coarse_class, crop)
            # Degenerate box (zero-area after clipping, e.g. sits on the image edge)
            # -- can't classify, fall through to the same un-refined fallback used
            # when routing doesn't fire.

        candidates = self.classmap.fine_of_group(coarse_class)
        if len(candidates) == 1:
            # Unambiguous 1:1 coarse -> fine mapping, no model needed.
            return candidates[0], score
        # Coarse class has multiple fine children but wasn't routed (low score, or the
        # degenerate-crop case above) -- leave un-refined, label stays coarse.
        return coarse_class, score

    def _classify_crop(self, group: str, crop: np.ndarray) -> tuple[str, float]:
        tensor = self.transform(Image.fromarray(crop)).unsqueeze(0).to(self.device)
        model = self.models[group]
        fine_classes = self._fine_classes_per_group[group]
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)
            conf, idx = probs.max(dim=1)
        return fine_classes[int(idx.item())], float(conf.item())


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    FGCInferPipeline(cfg).run()


if __name__ == "__main__":
    main()
