"""Evaluation driver — runs both metric families and writes a report.

Reproduces the two-family protocol (Table 1) and supports the ablation interface of
§4.2.1: evaluate detector-only vs. detector+FGC, and coarse-trained vs. fine-trained
detector, on the same inputs. Also drives the Phase-3 synthetic-vs-real comparison
by running the same protocol on split=test (real) and the synthetic test set.

eval.mode:
    det_only      detection metrics on derived/detections (Stage 1 alone). The GT label
                  level follows detector.label_level, so a one-stage YOLO trained on the
                  32 fine classes (detector.label_level=fine) is scored per fine class:
                  the baseline of the closed-set chapter (protocol C).
    det_plus_fgc  end-to-end on derived/detections_fgc (fine labels), plus the FGC metrics.
                  With ++fgc.oneshot=true (the flag pid-fgc-infer wrote them with): the one-shot
                  end-to-end score on derived/detections_oneshot instead, gated at
                  oneshot.other_min_cosine (+ ++eval.oneshot.margin_delta) here, plus the
                  detector's wrong-box rate (pipeline.detection_eval, PLAN B6b-2).
    fgc_only      the FGC metrics alone, on ground-truth crops: needs no detector output,
                  so a freshly trained `__global__.pt` can be scored on its own.
    oneshot       crop-level one-shot evaluation against the ISO prototypes (protocols p1 /
                  p2 / tau, eval.oneshot.protocol; pipeline.oneshot_eval, PLAN B6b).

The crop-level modes (fgc_only, oneshot) read crops_fgc/val through eval.image_subset
(all | val_sel | test, image ids from eval.splits_file): the 100 val images are split 50/50
by image so that checkpoint / tau selection (val_sel) and the exam (test) do not share crops.
The one-shot end-to-end score takes it too (the wrong-box rate is read on val_sel).

Detection metrics take the extended path of pipeline.detection_eval when the COCO has ignore
regions (`iscrowd: 1`, B6d.2), when ++eval.class_agnostic=true (one class "symbol", recall per
group from eval.oneshot.groups_ckpt / oneshot.encoder_ckpt), or when its images carry
`degraded` + `sheet` (OPEN100-ISO: clean and degraded never pooled, per source sheet + mean +-
std). Otherwise (DigitizePID with both switches off) the original mean_ap path runs unchanged.

With fgc.global_encoder=true the FGC metrics are those of the single head in
`<fgc.ckpt_dir>/__global__.pt` (protocol C, docs/PLAN_STAGE2_ONESHOT.md B6c): top-k,
macro-F1, per-class P/R/F1, the full confusion matrix and a per-family breakdown.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from pipeline.base import BasePipeline
from utils import bbox_utils, detection_metrics, fgc_metrics
from utils.cli import CONFIG_DIR
from utils.types import BBox

logger = logging.getLogger(__name__)

MODES = ("det_only", "det_plus_fgc", "fgc_only", "oneshot")


def _stem(file_name: str) -> str:
    """`"synthetic/0001.jpg"` -> `"0001"` — same convention as build_manifest/coco_to_yolo."""
    name = file_name.split("/")[-1]
    return name.rsplit(".", 1)[0]


def _to_jsonable(obj):
    """Recursively cast numpy scalars/arrays (and dict keys) to plain JSON-safe types."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


class EvaluationPipeline(BasePipeline):
    """Runs detection metrics and (optionally) FGC metrics, writes metrics.json.

    Mirrors PID_Symbol_Detection's `EvaluationPipeline` shape (`validate()` /
    per-metric-family compute methods / `run()`), adapted to this project's
    prediction-file layout and to real confidence-ranked COCO-style AP/mAP (the
    reference repo's `MetricsCalculator` only computes a single-IoU-threshold,
    unranked precision/recall/F1 — see `utils.detection_metrics` for why that's not
    reused here beyond its exact IoU formula).
    """

    def validate(self) -> bool:
        """Check required inputs exist, raising a clear, actionable error if not."""
        if not self.paths.manifest_csv.exists():
            raise FileNotFoundError(
                f"{self.paths.manifest_csv} not found; run `pid-prep-manifest` first."
            )
        if not self.paths.coco_json.exists():
            raise FileNotFoundError(
                f"{self.paths.coco_json} not found; run `pid-prep-classes`/"
                "`pid-prep-manifest` first."
            )

        mode = str(self.cfg.eval.mode)
        if mode not in MODES:
            raise ValueError(f"cfg.eval.mode must be one of {MODES}, got {mode!r}")
        subset = str(self.cfg.eval.get("image_subset", "all"))
        if mode in ("det_only", "det_plus_fgc") and subset != "all" and not self._oneshot_e2e():
            raise ValueError(
                f"eval.image_subset={subset!r} only applies to the crop-level modes "
                "(fgc_only, oneshot) and the one-shot end-to-end score, not to detection metrics"
            )
        if mode == "oneshot":
            from pipeline.oneshot_eval import PROTOCOLS
            from pipeline.oneshot_infer import require_oneshot_inputs
            from utils.dataset import image_subset_ids

            require_oneshot_inputs(self.cfg)
            protocol = (self.cfg.eval.get("oneshot", None) or {}).get("protocol", None)
            if protocol not in PROTOCOLS:
                raise ValueError(
                    f"eval.oneshot.protocol must be one of {PROTOCOLS}, got {protocol!r}"
                )
            image_subset_ids(self.cfg, "val")  # unknown subset name / unreadable splits file
            return True
        if mode == "fgc_only":
            from utils.dataset import image_subset_ids

            if self.cfg.fgc.get("ckpt_dir", None) is None:
                raise ValueError("eval.mode=fgc_only needs fgc.ckpt_dir (the checkpoints to score)")
            image_subset_ids(self.cfg, str(self.cfg.eval.split))  # subset != all needs split=val
            return True

        if self._oneshot_e2e():
            from utils.dataset import image_subset_ids

            image_subset_ids(self.cfg, str(self.cfg.eval.split))  # subset != all needs split=val
        det_dir = self._predictions_dir()
        infer_cmd = {"det_only": "pid-detect-infer", "det_plus_fgc": "pid-fgc-infer"}[mode]
        if self._oneshot_e2e():
            infer_cmd += " ++fgc.oneshot=true"
        if not det_dir.exists() or not any(det_dir.glob("*.txt")):
            raise FileNotFoundError(
                f"{det_dir} does not exist or has no prediction .txt files; run "
                f"`{infer_cmd}` for split={self.cfg.eval.split!r} first."
            )

        return True

    # ------------------------------------------------------------------ #
    # detection metrics
    # ------------------------------------------------------------------ #

    def _predictions_dir(self) -> Path:
        """`derived/detections/<split>` (det_only) or `derived/detections_fgc/<split>`
        (det_plus_fgc; `derived/detections_oneshot/<split>` with fgc.oneshot=true) — the output
        contract of `pipeline.detect_infer` / `pipeline.fgc_infer` respectively.
        """
        split = str(self.cfg.eval.split)
        mode = str(self.cfg.eval.mode)
        subdir = "detections" if mode == "det_only" else "detections_fgc"
        if self._oneshot_e2e():
            subdir = "detections_oneshot"
        return self.paths.derived / subdir / split

    def _oneshot_e2e(self) -> bool:
        """det_plus_fgc on the one-shot Stage-2 output (++fgc.oneshot=true, as pid-fgc-infer)."""
        fgc = self.cfg.get("fgc", None) or {}
        return str(self.cfg.eval.mode) == "det_plus_fgc" and bool(fgc.get("oneshot", False))

    def _class_agnostic(self) -> bool:
        return bool(self.cfg.eval.get("class_agnostic", False))

    def _eval_image_ids(self) -> set[str]:
        manifest = pd.read_csv(self.paths.manifest_csv, dtype={"id": str})
        split = str(self.cfg.eval.split)
        ids = set(manifest.loc[manifest["split"] == split, "id"])
        if self._oneshot_e2e():
            from utils.dataset import image_subset_ids

            subset = image_subset_ids(self.cfg, split)
            if subset is not None:
                ids &= subset
        return ids

    def _load_coco(self) -> dict:
        with self.paths.coco_json.open("r") as f:
            return json.load(f)

    def _image_sizes(self, coco: dict) -> tuple[dict[str, tuple[int, int]], dict[int, str]]:
        """image_id -> (width, height), and COCO numeric image id -> image_id string.

        Prefers width/height already stored on the COCO `images` entry (standard COCO
        schema); falls back to reading the actual file via `self.paths.find_image`
        only if that's missing.
        """
        image_id_to_wh: dict[str, tuple[int, int]] = {}
        coco_id_to_image_id: dict[int, str] = {}

        for img in coco.get("images", []):
            image_id = _stem(img["file_name"])
            coco_id_to_image_id[img["id"]] = image_id

            width, height = img.get("width"), img.get("height")
            if width is None or height is None:
                from PIL import Image

                with Image.open(self.paths.find_image(image_id)) as im:
                    width, height = im.size
            image_id_to_wh[image_id] = (int(width), int(height))

        return image_id_to_wh, coco_id_to_image_id

    def _load_gt(
        self, coco: dict, eval_ids: set[str], image_id_to_wh, coco_id_to_image_id, mode: str
    ) -> list[tuple[str, int, tuple[float, float, float, float]]]:
        category_id_to_category = {c["id"]: c for c in coco.get("categories", [])}

        gts: list[tuple[str, int, tuple[float, float, float, float]]] = []
        for ann in coco.get("annotations", []):
            image_id = coco_id_to_image_id.get(ann["image_id"])
            if image_id is None or image_id not in eval_ids:
                continue

            if ann.get("iscrowd"):
                continue  # an ignore region (B6d.2); DigitizePID has none
            category = category_id_to_category[ann["category_id"]]
            x, y, w, h = ann["bbox"]
            xyxy = (float(x), float(y), float(x + w), float(y + h))
            gts.append(
                (
                    image_id,
                    self._gt_class_id(category["name"], category["supercategory"], mode),
                    xyxy,
                )
            )

        return gts

    def _gt_class_id(self, name: str, supercategory: str, mode: str) -> int:
        if mode == "det_only" and self._detector_level() == "coarse":
            return self.classmap.coarse_id(supercategory)
        return self.classmap.fine_id(name)

    def _detector_level(self) -> str:
        """Label set of the Stage-1 detector whose files det_only scores."""
        level = str(self.cfg.get("detector", {}).get("label_level", "coarse"))
        if level not in ("coarse", "fine"):
            raise ValueError(f"detector.label_level must be 'coarse' or 'fine', got {level!r}")
        return level

    def _load_preds(
        self, eval_ids: set[str], image_id_to_wh, agnostic: bool = False
    ) -> list[tuple[str, int, float, tuple[float, float, float, float]]]:
        """Predictions with score >= eval.score_thr. `agnostic`: every box is class 0, the
        ones without a vocabulary label included (a box is a box)."""
        det_dir = self._predictions_dir()
        score_thr = float(self.cfg.eval.score_thr)

        preds: list[tuple[str, int, float, tuple[float, float, float, float]]] = []
        n_unlabelled = 0
        for image_id in eval_ids:
            txt_path = det_dir / f"{image_id}.txt"
            if not txt_path.exists():
                continue
            if image_id not in image_id_to_wh:
                logger.warning("No image size for %s; skipping its predictions.", image_id)
                continue
            width, height = image_id_to_wh[image_id]

            with txt_path.open("r") as f:
                for line in f:
                    parts = line.split()
                    if not parts:
                        continue
                    # 6 cols (det_only) or 7 cols (det_plus_fgc, trailing fine_score
                    # ignored) — both share the first 6 columns' meaning.
                    class_id = int(float(parts[0]))
                    cx, cy, w, h, score = (float(v) for v in parts[1:6])
                    if score < score_thr:
                        continue
                    if agnostic:
                        class_id = 0
                    elif class_id < 0:
                        # OTHER_FINE_ID / UNREFINED_FINE_ID: the box got no vocabulary
                        # label, so it predicts no class. Kept, each would be scored as a
                        # class of its own with AP 0 and drag the mean down.
                        n_unlabelled += 1
                        continue
                    xyxy = bbox_utils.yolo_to_xyxy(BBox(x=cx, y=cy, w=w, h=h), width, height)
                    preds.append((image_id, class_id, score, xyxy))

        if n_unlabelled:
            logger.info(
                "%d predictions (score>=%.2f) carry no vocabulary label (Other / un-refined) "
                "and are left out of mAP.",
                n_unlabelled,
                score_thr,
            )
        return preds

    def compute_detection_metrics(self) -> dict:
        """mAP@50, mAP@50:95, and (if cfg.eval.per_class_ap) per-class AP@50; the extended
        report of pipeline.detection_eval with ignore regions / class_agnostic / OPEN100-ISO
        views (module docstring)."""
        from pipeline import detection_eval

        mode = str(self.cfg.eval.mode)
        eval_ids = self._eval_image_ids()
        coco = self._load_coco()
        image_id_to_wh, coco_id_to_image_id = self._image_sizes(coco)

        boxes, ignores = detection_eval.load_boxes(coco, eval_ids, coco_id_to_image_id)
        views = detection_eval.image_views(coco, eval_ids)
        if self._class_agnostic() or ignores or views is not None:
            return self._extended_detection_metrics(
                mode, eval_ids, image_id_to_wh, boxes, ignores, views
            )

        gts = self._load_gt(coco, eval_ids, image_id_to_wh, coco_id_to_image_id, mode)
        preds = self._load_preds(eval_ids, image_id_to_wh)

        logger.info(
            "Detection metrics: %d GT boxes, %d predictions (score>=%.2f) over "
            "%d images (split=%s, mode=%s).",
            len(gts),
            len(preds),
            float(self.cfg.eval.score_thr),
            len(eval_ids),
            self.cfg.eval.split,
            mode,
        )

        iou_thrs = [float(t) for t in self.cfg.eval.iou_thrs]
        result = detection_metrics.mean_ap(preds, gts, iou_thrs)

        if not bool(self.cfg.eval.per_class_ap):
            result = {k: v for k, v in result.items() if k != "per_class_ap_50"}

        return result

    def _extended_detection_metrics(
        self, mode: str, eval_ids, image_id_to_wh, boxes, ignores, views
    ) -> dict:
        from pipeline import detection_eval

        agnostic = self._class_agnostic()
        preds = self._load_preds(eval_ids, image_id_to_wh, agnostic=agnostic)
        gts = [
            (
                b.image_id,
                0 if agnostic else self._gt_class_id(b.name, b.supercategory, mode),
                b.xyxy,
            )
            for b in boxes
        ]
        groups = None
        if agnostic:
            groups = detection_eval.gt_groups(self.cfg, {b.name for b in boxes})
            if groups.note:
                logger.warning("[detection] %s", groups.note)
        iou_thrs = [float(t) for t in self.cfg.eval.iou_thrs]
        logger.info(
            "Detection metrics (class_agnostic=%s): %d GT boxes, %d ignore regions, %d "
            "predictions (score>=%.2f) over %d images (split=%s, mode=%s%s).",
            agnostic,
            len(gts),
            len(ignores),
            len(preds),
            float(self.cfg.eval.score_thr),
            len(eval_ids),
            self.cfg.eval.split,
            mode,
            ", clean / degraded apart" if views is not None else "",
        )
        report = detection_eval.detection_report(
            eval_ids,
            views,
            preds,
            gts,
            boxes,
            ignores,
            iou_thrs,
            groups,
            bool(self.cfg.eval.per_class_ap),
        )
        out = {
            "class_agnostic": agnostic,
            "score_thr": float(self.cfg.eval.score_thr),
            "ignore_match": {
                "iou": detection_metrics.IGNORE_IOU,
                "inside_frac_of_prediction": detection_metrics.IGNORE_INSIDE,
            },
            **report,
        }
        if groups is not None:
            out["groups"] = groups.summary()
        for view in ("degraded", "clean") if views is not None else (None,):
            block = report if view is None else report[view]
            logger.info(
                "[detection%s] P=%s R=%s mAP50=%.4f mAP50:95=%.4f%s",
                "" if view is None else f"/{view}",
                block["precision"],
                block["recall"],
                block["map_50"],
                block["map_50_95"],
                "".join(
                    f"  {g} R={r['recall']} (n={r['n']})"
                    for g, r in block.get("recall_by_group", {}).items()
                ),
            )
        return out

    def compute_oneshot_e2e_metrics(self) -> dict:
        """End-to-end Stage 1 + one-shot Stage 2 on whole sheets, and the wrong-box rate
        (pipeline.detection_eval), from derived/detections_oneshot, gated here."""
        from pipeline import detection_eval
        from pipeline.oneshot_infer import UNREFINED_LABEL, read_oneshot_detections

        eval_ids = self._eval_image_ids()
        coco = self._load_coco()
        image_id_to_wh, coco_id_to_image_id = self._image_sizes(coco)
        boxes, ignores = detection_eval.load_boxes(coco, eval_ids, coco_id_to_image_id)
        views = detection_eval.image_views(coco, eval_ids)
        groups = detection_eval.gt_groups(self.cfg, {b.name for b in boxes}, require_vocab=True)
        if groups.note:
            logger.warning("[oneshot_e2e] %s", groups.note)
        tau = float(self.cfg.oneshot.other_min_cosine)
        node = self.cfg.eval.get("oneshot", None) or {}
        delta = node.get("margin_delta", None)
        delta = None if delta is None else float(delta)
        score_thr = float(self.cfg.eval.score_thr)

        det_dir = self._predictions_dir()
        preds: list[detection_eval.E2EPred] = []
        n_unrefined = 0
        for image_id in sorted(eval_ids):
            path = det_dir / f"{image_id}.txt"
            if not path.exists():
                continue
            width, height = image_id_to_wh[image_id]
            for r in read_oneshot_detections(path):
                if r.score < score_thr:
                    continue
                if r.label == UNREFINED_LABEL or not np.isfinite(r.best_cos):
                    n_unrefined += 1  # never classified: no decision (B6a)
                    continue
                preds.append(
                    detection_eval.E2EPred(
                        image_id,
                        r.best_name,
                        r.score,
                        r.best_cos,
                        r.second_cos,
                        bbox_utils.yolo_to_xyxy(r.box, width, height),
                    )
                )
        logger.info(
            "[oneshot_e2e] %d GT boxes, %d ignore regions, %d predictions (score>=%.2f; %d "
            "unrefined dropped) over %d images; tau=%.4f delta=%s",
            len(boxes),
            len(ignores),
            len(preds),
            score_thr,
            n_unrefined,
            len(eval_ids),
            tau,
            delta,
        )
        report = detection_eval.e2e_report(
            eval_ids, views, preds, boxes, ignores, groups, tau, delta
        )
        out = {
            "predictions": str(det_dir),
            "encoder_checkpoint": self.cfg.oneshot.get("encoder_ckpt", None),
            "groups": groups.summary(),
            "tau": tau,
            "margin_delta": delta,
            "score_thr": score_thr,
            "iou": detection_eval.E2E_IOU,
            "wrong_box_iou": detection_eval.WRONG_BOX_IOU,
            "n_unrefined_dropped": n_unrefined,
            **report,
        }
        for view in ("degraded", "clean") if views is not None else (None,):
            block = report if view is None else report[view]
            wrong = block["wrong_boxes"]
            logger.info(
                "[oneshot_e2e%s] P=%s R=%s box R=%s | wrong boxes %d (%s of predictions), "
                "rejected at tau %s%s",
                "" if view is None else f"/{view}",
                block["precision"],
                block["recall"],
                block["box_recall"],
                wrong["n"],
                wrong["frac_of_predictions"],
                wrong["rejected_tau"],
                "" if delta is None else f", tau or margin {wrong['rejected_tau_or_margin']}",
            )
        return out

    # ------------------------------------------------------------------ #
    # FGC metrics
    # ------------------------------------------------------------------ #

    def compute_fgc_metrics(self) -> dict | None:
        """Standalone FGC-head evaluation on GROUND-TRUTH crops (derived/crops_fgc),
        independent of detector recall/errors — a separable-stage ablation.

        Returns None (logging why) when cfg.fgc.ckpt_dir is not set, so evaluation
        stays runnable in det_only-style setups without any trained FGC checkpoints.
        """
        ckpt_dir = self.cfg.fgc.get("ckpt_dir", None)
        if ckpt_dir is None:
            logger.info("cfg.fgc.ckpt_dir is None; skipping FGC metrics.")
            return None
        ckpt_dir = Path(str(ckpt_dir))
        if bool(self.cfg.fgc.get("global_encoder", False)):
            return self._global_fgc_metrics(ckpt_dir)

        from utils.classmap import training_fine_classes
        from utils.dataset import excluded_classes

        group_metrics: dict[str, dict] = {}
        pooled_y_true: list[str] = []
        pooled_y_pred: list[str] = []

        for group in self.classmap.fgc_groups:
            ckpt_path = ckpt_dir / f"{group}.pt"
            if not ckpt_path.exists():
                logger.warning("No checkpoint at %s for group=%s; skipping.", ckpt_path, group)
                continue

            # Same list the head was trained on (held-out classes removed), so names
            # line up with the checkpoint's columns.
            fine_names = training_fine_classes(self.classmap, group, excluded_classes(self.cfg))
            scored = self._score_crops(ckpt_path, group, fine_names)
            if scored is None:
                continue
            _, logits_arr, y_true_local, max_cosine = scored
            y_pred_local = logits_arr.argmax(axis=1).tolist()

            y_true_names = [fine_names[i] for i in y_true_local]
            y_pred_names = [fine_names[i] for i in y_pred_local]

            confusion = fgc_metrics.within_group_confusion(y_true_names, y_pred_names, fine_names)
            accuracy = fgc_metrics.within_group_accuracy(confusion)
            topk = fgc_metrics.topk_accuracy(logits_arr, y_true_local, ks=self._ks())

            group_metrics[group] = {
                "within_group_accuracy": accuracy,
                "n_samples": len(y_true_local),
                **topk,
            }
            if max_cosine is not None:
                group_metrics[group]["other_gate"] = self._other_gate(max_cosine)

            self._plot_confusion(group, confusion, fine_names)

            pooled_y_true.extend(y_true_names)
            pooled_y_pred.extend(y_pred_names)

        rare = fgc_metrics.rare_macro_f1(pooled_y_true, pooled_y_pred, self.classmap.rare_fine)

        return {"groups": group_metrics, "rare_macro_f1": rare}

    def _global_fgc_metrics(self, ckpt_dir: Path) -> dict | None:
        """Protocol C: the single head of `__global__.pt` over its whole vocabulary.

        The checkpoint's own `fine_classes` give the column order (and which classes it
        was trained on), so R1b is scored on its 32 classes and a leave-out checkpoint on
        the classes it saw, whatever fgc.exclude_classes the command carries.
        """
        from utils.classmap import GLOBAL_GROUP

        ckpt_path = ckpt_dir / f"{GLOBAL_GROUP}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"fgc.global_encoder=true but {ckpt_path} does not exist; point "
                "fgc.ckpt_dir at the folder holding __global__.pt"
            )
        scored = self._score_crops(ckpt_path, GLOBAL_GROUP, None)
        if scored is None:
            return None
        names, logits_arr, y_true_local, max_cosine = scored
        y_true = [names[i] for i in y_true_local]
        y_pred = [names[i] for i in logits_arr.argmax(axis=1).tolist()]
        confusion = fgc_metrics.within_group_confusion(y_true, y_pred, names)
        families = fgc_metrics.family_breakdown(y_true, y_pred, self.classmap.fine_to_coarse)
        cross = sum(f["cross_family"] * f["n_samples"] for f in families.values()) / len(y_true)
        result = {
            "encoder": GLOBAL_GROUP,
            "checkpoint": str(ckpt_path),
            "n_classes": len(names),
            "n_samples": len(y_true),
            **fgc_metrics.topk_accuracy(logits_arr, y_true_local, ks=self._ks()),
            "macro_f1": fgc_metrics.macro_f1(y_true, y_pred, names),
            "rare_macro_f1": fgc_metrics.rare_macro_f1(y_true, y_pred, self.classmap.rare_fine),
            "cross_family_error_rate": cross,
            "families": families,
            "per_class": fgc_metrics.per_class_prf(y_true, y_pred, names),
            "confusion": {"labels": names, "matrix": confusion},
        }
        if max_cosine is not None:
            result["other_gate"] = self._other_gate(max_cosine)
        self._plot_confusion(GLOBAL_GROUP, confusion, names)
        logger.info(
            "[fgc/%s] %d crops, %d classes: top1=%.4f macro_f1=%.4f cross_family=%.4f",
            GLOBAL_GROUP,
            len(y_true),
            len(names),
            result["top1"],
            result["macro_f1"],
            cross,
        )
        return result

    def _ks(self) -> tuple[int, ...]:
        return tuple(int(k) for k in self.cfg.eval.topk)

    def _other_gate(self, max_cosine: list[float]) -> dict:
        # All these crops ARE in-vocabulary (GT), so any that fall below the threshold
        # are false rejects. Report the rate so the threshold can be tuned to trade
        # false-reject against unknown-symbol recall.
        thr = float(self.cfg.fgc.route.get("other_min_cosine", 0.0))
        cos_arr = np.asarray(max_cosine)
        return {
            "threshold": thr,
            "false_reject_rate": float((cos_arr < thr).mean()),
            "mean_max_cosine": float(cos_arr.mean()),
        }

    def _score_crops(
        self, ckpt_path: Path, group: str, fine_names: list[str] | None
    ) -> tuple[list[str], np.ndarray, list[int], list[float] | None] | None:
        """Run one checkpoint over the eval split's GT crops: (column names, logits
        (N, C), labels, max cosine to a class center per crop — ArcFace only). None when
        there are no crops. `fine_names` None takes the columns from the checkpoint."""
        import torch
        from torch.utils.data import DataLoader

        from utils.dataset import build_dataset, crop_image_id, image_subset_ids
        from utils.model import ArcFaceModel, build_model

        split = str(self.cfg.eval.split)
        # pipeline.fgc_train saves {"model": state_dict, "optimizer": ..., "epoch": ..., ...},
        # not a bare state_dict -- match pipeline.fgc_infer's loading convention.
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if fine_names is None:
            fine_names = list(checkpoint["fine_classes"])

        image_ids = image_subset_ids(self.cfg, split)
        dataset = build_dataset(
            self.cfg, split, group, fine_classes=fine_names, image_ids=image_ids
        )
        if image_ids is not None:
            logger.info(
                "eval.image_subset=%s: kept %d crops from %d images (group=%s)",
                self.cfg.eval.image_subset,
                len(dataset),
                len({crop_image_id(p.name) for p, _ in dataset.samples}),
                group,
            )
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

        model = build_model(self.cfg, group, len(fine_names))
        model.load_state_dict(checkpoint["model"])
        model.eval()

        # ArcFace model(batch_x) with no target returns scaled cosine — order-
        # preserving, so argmax/top-k are unaffected. For an ArcFace model we also
        # collect the max cosine-to-center per crop to measure how often the "Other"
        # gate would (wrongly) reject a KNOWN crop — the false-reject rate, the signal
        # for calibrating fgc.route.other_min_cosine.
        is_arcface = isinstance(model, ArcFaceModel)
        all_logits: list[np.ndarray] = []
        all_labels: list[int] = []
        all_max_cosine: list[float] = []
        with torch.no_grad():
            for batch_x, batch_y in loader:
                logits = model(batch_x)
                all_logits.append(logits.cpu().numpy())
                all_labels.extend(int(v) for v in batch_y.tolist())
                if is_arcface:
                    cosine = model.head.cosine(model.embed(batch_x))
                    all_max_cosine.extend(float(v) for v in cosine.max(dim=1).values.cpu().tolist())

        if not all_labels:
            logger.warning("No crops found for group=%s split=%s; skipping.", group, split)
            return None
        return (
            fine_names,
            np.concatenate(all_logits, axis=0),
            all_labels,
            all_max_cosine if is_arcface else None,
        )

    def _plot_confusion(self, group: str, confusion: np.ndarray, labels: list[str]) -> None:
        """Minimal confusion-matrix PNG per group (or the global head), saved next to
        metrics.json."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        side = max(4, len(labels) * (0.8 if len(labels) <= 12 else 0.45))
        fig, ax = plt.subplots(figsize=(side, side))
        im = ax.imshow(confusion, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        global_head = group.startswith("__")
        ax.set_title("Global confusion" if global_head else f"Within-group confusion: {group}")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.output_dir / f"confusion_{group}.png")
        plt.close(fig)

    # ------------------------------------------------------------------ #
    # one-shot metrics
    # ------------------------------------------------------------------ #

    def compute_oneshot_metrics(self) -> dict:
        """Crop-level one-shot evaluation (eval.oneshot.protocol = p1 | p2 | tau), see
        pipeline.oneshot_eval. Writes its per-crop CSV (and sweep PNG) next to metrics.json."""
        from pipeline.oneshot_eval import OneShotEvaluation

        return OneShotEvaluation(self.cfg, self.output_dir).run()

    # ------------------------------------------------------------------ #
    # driver
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        self.validate()

        mode = str(self.cfg.eval.mode)
        metrics: dict = {"mode": mode, "split": str(self.cfg.eval.split)}
        if mode in ("fgc_only", "oneshot"):
            metrics["image_subset"] = str(self.cfg.eval.get("image_subset", "all"))
        if mode == "oneshot":
            metrics["oneshot"] = self.compute_oneshot_metrics()
        elif self._oneshot_e2e():
            metrics["image_subset"] = str(self.cfg.eval.get("image_subset", "all"))
            metrics["oneshot_e2e"] = self.compute_oneshot_e2e_metrics()
        elif mode != "fgc_only":
            metrics["detection"] = self.compute_detection_metrics()
            if mode == "det_only":
                metrics["detection"]["label_level"] = self._detector_level()

        if mode in ("det_plus_fgc", "fgc_only") and not self._oneshot_e2e():
            fgc_result = self.compute_fgc_metrics()
            if fgc_result is not None:
                metrics["fgc"] = fgc_result

        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / "metrics.json"
        with out_path.open("w") as f:
            json.dump(_to_jsonable(metrics), f, indent=2)

        logger.info("Wrote evaluation metrics to %s", out_path)


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    EvaluationPipeline(cfg).run()


if __name__ == "__main__":
    main()
