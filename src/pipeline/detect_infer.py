"""Run the detector over full-resolution diagrams with SAHI tiling + WBF merge.

Pipeline (report §4.2.2):
    full image -> SAHI slicing (sahi.slicing.slice_image, overlap-preserving, ref [40])
               -> per-tile YOLOv8 detections
               -> Weighted Boxes Fusion across overlapping tiles (ref [41])
               -> diagram-level list[Detection] with coarse_class filled.

Deliberately different from PID_Symbol_Detection's yolo_predictor.py: that reference
relies on SAHI's own implicit merge (get_sliced_prediction). This project's config
already exposes explicit detector.wbf.* fields, so tiles are sliced with
sahi.slicing.slice_image (slicing only, no built-in merge) and fused explicitly with
ensemble_boxes.weighted_boxes_fusion instead.

Output is the coarse symbol inventory consumed by pipeline.fgc_infer / pipeline.evaluation:
one 6-column YOLO-format txt (class_id cx cy w h score) per image under
derived/detections/<split>/<image_id>.txt.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import hydra
import pandas as pd
from ensemble_boxes import weighted_boxes_fusion
from omegaconf import DictConfig
from sahi.slicing import slice_image
from ultralytics import YOLO

from pipeline.base import BasePipeline
from utils.bbox_utils import xyxy_to_yolo
from utils.classmap import ClassMap
from utils.cli import CONFIG_DIR
from utils.types import BBox, Detection

logger = logging.getLogger(__name__)


def _write_detections_txt(
    path: Path, detections: list[Detection], classmap: ClassMap, image_w: int, image_h: int
) -> None:
    """Write detections as 6-column YOLO rows: class_id cx cy w h score (normalized box).

    bbox_utils.write_yolo_txt only handles the 5-column (no-score) form used for ground
    truth, so this local helper covers the extra trailing confidence column detections
    need.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for det in detections:
            yolo_box = xyxy_to_yolo(det.bbox.to_xyxy(), image_w, image_h)
            class_id = classmap.coarse_id(det.coarse_class)
            f.write(
                f"{class_id} {yolo_box.x:.6f} {yolo_box.y:.6f} "
                f"{yolo_box.w:.6f} {yolo_box.h:.6f} {det.score:.6f}\n"
            )


class DetectInferPipeline(BasePipeline):
    """SAHI-tiled YOLOv8 inference + WBF merge over full-resolution P&ID diagrams."""

    def validate(self) -> bool:
        if self.cfg.detector.ckpt is None:
            raise ValueError(
                "cfg.detector.ckpt is not set. Training writes into its own fresh Hydra "
                "run dir each invocation, so inference needs an explicit pointer, e.g. "
                "`pid-detect-infer detector.ckpt=runs/2026-.../weights/best.pt`."
            )
        ckpt = Path(str(self.cfg.detector.ckpt))
        if not ckpt.exists():
            raise FileNotFoundError(f"detector.ckpt={ckpt} does not exist.")
        if not self.paths.manifest_csv.exists():
            raise FileNotFoundError(
                f"{self.paths.manifest_csv} not found. Run `pid-prep-manifest` first."
            )
        return True

    def _image_ids(self) -> list[str]:
        manifest = pd.read_csv(self.paths.manifest_csv)
        split = str(self.cfg.eval.split)
        subset = manifest[manifest["split"] == split]
        return [str(i) for i in subset["id"].tolist()]

    def _detect_image(self, model: YOLO, image, image_w: int, image_h: int) -> list[Detection]:
        """Slice `image` with SAHI, run YOLO per tile, fuse tile detections with WBF."""
        slice_result = slice_image(
            image=image,
            slice_height=int(self.cfg.detector.sahi.slice_size),
            slice_width=int(self.cfg.detector.sahi.slice_size),
            overlap_height_ratio=float(self.cfg.detector.sahi.overlap_ratio),
            overlap_width_ratio=float(self.cfg.detector.sahi.overlap_ratio),
        )

        boxes_list: list[list[list[float]]] = []
        scores_list: list[list[float]] = []
        labels_list: list[list[float]] = []

        for sliced_image, starting_pixel in zip(
            slice_result.images, slice_result.starting_pixels, strict=True
        ):
            sx, sy = starting_pixel
            # conf kept low here: WBF/skip_box_thr does the real filtering downstream.
            pred = model.predict(sliced_image, conf=0.001, verbose=False)[0]

            boxes: list[list[float]] = []
            scores: list[float] = []
            labels: list[float] = []
            for xyxy, conf, cls in zip(
                pred.boxes.xyxy.tolist(),
                pred.boxes.conf.tolist(),
                pred.boxes.cls.tolist(),
                strict=True,
            ):
                x1, y1, x2, y2 = xyxy
                # reproject slice-local pixel coords -> full-image normalized [0,1] xyxy
                boxes.append(
                    [
                        (x1 + sx) / image_w,
                        (y1 + sy) / image_h,
                        (x2 + sx) / image_w,
                        (y2 + sy) / image_h,
                    ]
                )
                scores.append(float(conf))
                labels.append(float(cls))

            boxes_list.append(boxes)
            scores_list.append(scores)
            labels_list.append(labels)

        # Gracefully handles the empty-detections case: weighted_boxes_fusion returns
        # empty arrays (rather than raising) when every tile contributed zero boxes.
        merged_boxes, merged_scores, merged_labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            iou_thr=float(self.cfg.detector.wbf.iou_thr),
            skip_box_thr=float(self.cfg.detector.wbf.skip_box_thr),
        )

        detections: list[Detection] = []
        for (bx1, by1, bx2, by2), score, label in zip(
            merged_boxes, merged_scores, merged_labels, strict=True
        ):
            detections.append(
                Detection(
                    bbox=BBox.from_xyxy(
                        bx1 * image_w, by1 * image_h, bx2 * image_w, by2 * image_h
                    ),
                    coarse_class=self.classmap.coarse_classes[int(label)],
                    score=float(score),
                )
            )
        return detections

    def run(self) -> None:
        self.validate()
        model = YOLO(str(self.cfg.detector.ckpt))
        split = str(self.cfg.eval.split)
        image_ids = self._image_ids()
        out_dir = self.paths.derived / "detections" / split
        logger.info(
            "Running detection over %d images (split=%s) -> %s", len(image_ids), split, out_dir
        )

        for image_id in image_ids:
            image_path = self.paths.find_image(image_id)
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning("Could not read image %s, skipping", image_path)
                continue
            h, w = image.shape[:2]

            detections = self._detect_image(model, image, w, h)
            out_path = out_dir / f"{image_id}.txt"
            _write_detections_txt(out_path, detections, self.classmap, w, h)
            logger.info("%s: %d detections -> %s", image_id, len(detections), out_path)


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    DetectInferPipeline(cfg).run()


if __name__ == "__main__":
    main()
