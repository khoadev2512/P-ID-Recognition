"""Evaluation driver — runs both metric families and writes a report.

Reproduces the two-family protocol (Table 1) and supports the ablation interface of
§4.2.1: evaluate detector-only vs. detector+FGC, and coarse-trained vs. fine-trained
detector, on the same inputs. Also drives the Phase-3 synthetic-vs-real comparison
by running the same protocol on split=test (real) and the synthetic test set.
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
        if mode not in ("det_only", "det_plus_fgc"):
            raise ValueError(f"cfg.eval.mode must be 'det_only' or 'det_plus_fgc', got {mode!r}")

        det_dir = self._predictions_dir()
        infer_cmd = "pid-detect-infer" if mode == "det_only" else "pid-fgc-infer"
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
        (det_plus_fgc) — the output contract of `pipeline.detect_infer` / `pipeline.fgc_infer`
        respectively.
        """
        split = str(self.cfg.eval.split)
        mode = str(self.cfg.eval.mode)
        subdir = "detections" if mode == "det_only" else "detections_fgc"
        return self.paths.derived / subdir / split

    def _eval_image_ids(self) -> set[str]:
        manifest = pd.read_csv(self.paths.manifest_csv, dtype={"id": str})
        split = str(self.cfg.eval.split)
        return set(manifest.loc[manifest["split"] == split, "id"])

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

            category = category_id_to_category[ann["category_id"]]
            x, y, w, h = ann["bbox"]
            xyxy = (float(x), float(y), float(x + w), float(y + h))

            if mode == "det_only":
                class_id = self.classmap.coarse_id(category["supercategory"])
            else:
                class_id = self.classmap.fine_id(category["name"])

            gts.append((image_id, class_id, xyxy))

        return gts

    def _load_preds(
        self, eval_ids: set[str], image_id_to_wh
    ) -> list[tuple[str, int, float, tuple[float, float, float, float]]]:
        det_dir = self._predictions_dir()
        score_thr = float(self.cfg.eval.score_thr)

        preds: list[tuple[str, int, float, tuple[float, float, float, float]]] = []
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
                    xyxy = bbox_utils.yolo_to_xyxy(BBox(x=cx, y=cy, w=w, h=h), width, height)
                    preds.append((image_id, class_id, score, xyxy))

        return preds

    def compute_detection_metrics(self) -> dict:
        """mAP@50, mAP@50:95, and (if cfg.eval.per_class_ap) per-class AP@50."""
        mode = str(self.cfg.eval.mode)
        eval_ids = self._eval_image_ids()
        coco = self._load_coco()
        image_id_to_wh, coco_id_to_image_id = self._image_sizes(coco)

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

        import torch
        from torch.utils.data import DataLoader

        from utils.dataset import build_dataset
        from utils.model import build_model

        split = str(self.cfg.eval.split)
        ks = tuple(int(k) for k in self.cfg.eval.topk)

        group_metrics: dict[str, dict] = {}
        pooled_y_true: list[str] = []
        pooled_y_pred: list[str] = []

        for group in self.classmap.fgc_groups:
            ckpt_path = ckpt_dir / f"{group}.pt"
            if not ckpt_path.exists():
                logger.warning("No checkpoint at %s for group=%s; skipping.", ckpt_path, group)
                continue

            fine_names = self.classmap.fine_of_group(group)
            num_fine = len(fine_names)

            dataset = build_dataset(self.cfg, split, group)
            loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

            model = build_model(self.cfg, group, num_fine)
            # pipeline.fgc_train saves {"model": state_dict, "optimizer": ..., "epoch": ..., ...},
            # not a bare state_dict -- match pipeline.fgc_infer's loading convention.
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(checkpoint["model"])
            model.eval()

            all_logits: list[np.ndarray] = []
            all_labels: list[int] = []
            with torch.no_grad():
                for batch_x, batch_y in loader:
                    logits = model(batch_x)
                    all_logits.append(logits.cpu().numpy())
                    all_labels.extend(int(v) for v in batch_y.tolist())

            if not all_labels:
                logger.warning("No crops found for group=%s split=%s; skipping.", group, split)
                continue

            logits_arr = np.concatenate(all_logits, axis=0)
            y_true_local = all_labels
            y_pred_local = logits_arr.argmax(axis=1).tolist()

            y_true_names = [fine_names[i] for i in y_true_local]
            y_pred_names = [fine_names[i] for i in y_pred_local]

            confusion = fgc_metrics.within_group_confusion(y_true_names, y_pred_names, fine_names)
            accuracy = fgc_metrics.within_group_accuracy(confusion)
            topk = fgc_metrics.topk_accuracy(logits_arr, y_true_local, ks=ks)

            group_metrics[group] = {
                "within_group_accuracy": accuracy,
                "n_samples": len(y_true_local),
                **topk,
            }

            self._plot_confusion(group, confusion, fine_names)

            pooled_y_true.extend(y_true_names)
            pooled_y_pred.extend(y_pred_names)

        rare = fgc_metrics.rare_macro_f1(pooled_y_true, pooled_y_pred, self.classmap.rare_fine)

        return {"groups": group_metrics, "rare_macro_f1": rare}

    def _plot_confusion(self, group: str, confusion: np.ndarray, labels: list[str]) -> None:
        """Minimal confusion-matrix PNG per group, saved next to metrics.json."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(max(4, len(labels) * 0.8), max(4, len(labels) * 0.8)))
        im = ax.imshow(confusion, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Within-group confusion: {group}")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.output_dir / f"confusion_{group}.png")
        plt.close(fig)

    # ------------------------------------------------------------------ #
    # driver
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        self.validate()

        metrics: dict = {"mode": str(self.cfg.eval.mode), "split": str(self.cfg.eval.split)}
        metrics["detection"] = self.compute_detection_metrics()

        if str(self.cfg.eval.mode) == "det_plus_fgc":
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
