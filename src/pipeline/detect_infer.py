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

detector.label_level=fine (the one-stage 32-class baseline of the closed-set chapter): the
same files hold FINE ids (classes.yaml fine: order), scored with eval.mode=det_only. The
checkpoint's own class names must equal the label set named by detector.label_level, so a
coarse model is never read as a fine one or the other way round.

detector.agnostic_merge=true (B6e; default false, set with `++detector.agnostic_merge=true`):
YOLO's NMS and the WBF merge both ignore the label, so two overlapping boxes of different
classes on one symbol (e.g. `valve` + `heat_exchanger` on an unseen ISO symbol) become ONE
box instead of two; the fused box keeps the label of its highest-scoring member. Off, the
merge is per label, as before.

++detector.classes_yaml=<classes.yaml> (default: the data root's): the label set of the checkpoint,
for a detector run on another dataset — the DigitizePID coarse detector on OPEN100-ISO
(data_root=data/iso_open100, whose classes.yaml is ISO's) writes DigitizePID coarse ids, checked
against that file. Scored there class-agnostically (eval.class_agnostic) or through Stage 2.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import hydra
import numpy as np
import pandas as pd
from ensemble_boxes import weighted_boxes_fusion
from ensemble_boxes.ensemble_boxes_wbf import (
    find_matching_box_fast,
    get_weighted_box,
    prefilter_boxes,
)
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
    path: Path,
    detections: list[Detection],
    classmap: ClassMap,
    image_w: int,
    image_h: int,
    fine: bool = False,
) -> None:
    """Write detections as 6-column YOLO rows: class_id cx cy w h score (normalized box).

    bbox_utils.write_yolo_txt only handles the 5-column (no-score) form used for ground
    truth, so this local helper covers the extra trailing confidence column detections
    need. `fine`: write `det.fine_class`'s fine id instead of the coarse id.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for det in detections:
            yolo_box = xyxy_to_yolo(det.bbox.to_xyxy(), image_w, image_h)
            if fine:
                class_id = classmap.fine_id(det.fine_class)
            else:
                class_id = classmap.coarse_id(det.coarse_class)
            f.write(
                f"{class_id} {yolo_box.x:.6f} {yolo_box.y:.6f} "
                f"{yolo_box.w:.6f} {yolo_box.h:.6f} {det.score:.6f}\n"
            )


def fuse_boxes(
    boxes: list[list[float]],
    scores: list[float],
    labels: list[float],
    iou_thr: float,
    skip_box_thr: float,
    agnostic: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """WBF over ONE detector's boxes (all tiles in a single model list, see `_detect_image`):
    normalised xyxy boxes -> (boxes, scores, labels), best score first. Per label unless
    `agnostic`, where boxes of any label fuse and each fused box takes the label of its
    highest-scoring member."""
    if not agnostic:
        return weighted_boxes_fusion(
            [boxes], [scores], [labels], iou_thr=iou_thr, skip_box_thr=skip_box_thr
        )
    return _wbf_agnostic(boxes, scores, labels, iou_thr, skip_box_thr)


def _wbf_agnostic(
    boxes: list[list[float]],
    scores: list[float],
    labels: list[float],
    iou_thr: float,
    skip_box_thr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`weighted_boxes_fusion` of one model list (conf_type `avg`) with every label taken as one,
    built from the library's own steps so the boxes and scores are exactly what it gives for a
    single-label input; the loop is repeated here only to see each cluster's members, which the
    library does not return. Each box goes in as its own "model" list of weight 1, so its score
    is untouched and `prefilter_boxes` records its index (column 3) for the label lookup."""
    n = len(boxes)
    pools = prefilter_boxes(
        [[b] for b in boxes], [[s] for s in scores], [[0]] * n, np.ones(n), skip_box_thr
    )
    if not pools:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))
    members: list[list[np.ndarray]] = []
    fused = np.empty((0, 8))
    for b in pools[0]:  # best score first: a cluster's first member is its highest-scoring one
        index, _ = find_matching_box_fast(fused, b, iou_thr)
        if index != -1:
            members[index].append(b)
            fused[index] = get_weighted_box(members[index], "avg")
        else:
            members.append([b.copy()])
            fused = np.vstack((fused, b.copy()))
    # one model of weight 1: WBF's score rescale (x min(1, n) / 1) is the identity
    out_labels = np.array([float(labels[int(m[0][3])]) for m in members])
    order = fused[:, 1].argsort()[::-1]
    return fused[order, 4:], fused[order, 1], out_labels[order]


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

    def _label_level(self) -> str:
        level = str(self.cfg.detector.get("label_level", "coarse"))
        if level not in ("coarse", "fine"):
            raise ValueError(f"detector.label_level must be 'coarse' or 'fine', got {level!r}")
        return level

    def _check_model_names(self, model: YOLO) -> None:
        """The checkpoint's class names (from the data.yaml it was trained on) must be the
        label set of detector.label_level, in classes.yaml order: the label id is read
        as an index into that list."""
        level = self._label_level()
        classmap = self.detector_classmap
        want = classmap.coarse_classes if level == "coarse" else classmap.fine_classes
        names = model.names
        have = [names[i] for i in range(len(names))] if isinstance(names, dict) else list(names)
        if have != list(want):
            raise ValueError(
                f"detector.ckpt classes {have[:4]}... ({len(have)}) are not the "
                f"detector.label_level={level} classes of classes.yaml {list(want)[:4]}... "
                f"({len(want)}); set detector.label_level to match the checkpoint (a detector "
                "trained on another dataset: ++detector.classes_yaml=<its classes.yaml>)"
            )

    def _detection(self, box: BBox, label: int, score: float) -> Detection:
        classmap = self.detector_classmap
        if self._label_level() == "fine":
            fine = classmap.fine_classes[label]
            return Detection(
                bbox=box,
                coarse_class=classmap.fine_to_coarse[fine],
                score=score,
                fine_class=fine,
            )
        return Detection(bbox=box, coarse_class=classmap.coarse_classes[label], score=score)

    def _agnostic_merge(self) -> bool:
        return bool(self.cfg.detector.get("agnostic_merge", False))

    def _detect_image(self, model: YOLO, image, image_w: int, image_h: int) -> list[Detection]:
        """Slice `image` with SAHI, run YOLO per tile, fuse tile detections with WBF."""
        slice_result = slice_image(
            image=image,
            slice_height=int(self.cfg.detector.sahi.slice_size),
            slice_width=int(self.cfg.detector.sahi.slice_size),
            overlap_height_ratio=float(self.cfg.detector.sahi.overlap_ratio),
            overlap_width_ratio=float(self.cfg.detector.sahi.overlap_ratio),
        )

        # Collect EVERY tile's detections into a SINGLE box list (one "model" for WBF).
        #
        # weighted_boxes_fusion divides each fused box's score by the number of model
        # lists passed. All tiles come from the SAME detector, so they are one source —
        # passing one list per tile (len ~= 54 for a 7168x4561 sheet) makes WBF divide
        # every score by ~54, collapsing a genuine 0.9 detection to ~0.02 (a symbol
        # appears in only 1-2 overlapping tiles, not all 54). That silently sinks every
        # score below eval.score_thr / fgc.route.min_score downstream. Treating all tiles
        # as one model keeps scores intact while WBF still merges the duplicate boxes that
        # overlapping tiles produce at their seams.
        all_boxes: list[list[float]] = []
        all_scores: list[float] = []
        all_labels: list[float] = []
        agnostic = self._agnostic_merge()

        for sliced_image, starting_pixel in zip(
            slice_result.images, slice_result.starting_pixels, strict=True
        ):
            sx, sy = starting_pixel
            # conf kept low here: WBF/skip_box_thr does the real filtering downstream.
            pred = model.predict(sliced_image, conf=0.001, verbose=False, agnostic_nms=agnostic)[0]

            for xyxy, conf, cls in zip(
                pred.boxes.xyxy.tolist(),
                pred.boxes.conf.tolist(),
                pred.boxes.cls.tolist(),
                strict=True,
            ):
                x1, y1, x2, y2 = xyxy
                # reproject slice-local pixel coords -> full-image normalized [0,1] xyxy
                all_boxes.append(
                    [
                        (x1 + sx) / image_w,
                        (y1 + sy) / image_h,
                        (x2 + sx) / image_w,
                        (y2 + sy) / image_h,
                    ]
                )
                all_scores.append(float(conf))
                all_labels.append(float(cls))

        # One model list -> no score division. Empty-detections case: WBF returns empty
        # arrays (rather than raising) when no tile contributed a box.
        merged_boxes, merged_scores, merged_labels = fuse_boxes(
            all_boxes,
            all_scores,
            all_labels,
            iou_thr=float(self.cfg.detector.wbf.iou_thr),
            skip_box_thr=float(self.cfg.detector.wbf.skip_box_thr),
            agnostic=agnostic,
        )

        detections: list[Detection] = []
        for (bx1, by1, bx2, by2), score, label in zip(
            merged_boxes, merged_scores, merged_labels, strict=True
        ):
            detections.append(
                self._detection(
                    BBox.from_xyxy(bx1 * image_w, by1 * image_h, bx2 * image_w, by2 * image_h),
                    int(label),
                    float(score),
                )
            )
        return detections

    def run(self) -> None:
        self.validate()
        model = YOLO(str(self.cfg.detector.ckpt))
        self._check_model_names(model)
        fine = self._label_level() == "fine"
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
            _write_detections_txt(out_path, detections, self.detector_classmap, w, h, fine)
            logger.info("%s: %d detections -> %s", image_id, len(detections), out_path)


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    DetectInferPipeline(cfg).run()


if __name__ == "__main__":
    main()
