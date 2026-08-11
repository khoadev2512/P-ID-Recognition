"""Detection metrics (report §2.2.3, Table 1 top group).

    IoU (eq. 2), Precision/Recall (eq. 3), AP (eq. 4), mAP (eq. 5),
    mAP@50 and mAP@50:95, plus per-class AP for weak-class identification.

High recall is emphasised in the P&ID domain (missed symbols = record gaps).
"""

from __future__ import annotations


def iou(pred_xyxy, gt_xyxy) -> float:
    raise NotImplementedError  # TODO: eq. 2


def average_precision(preds, gts, iou_thr: float) -> float:
    raise NotImplementedError  # TODO: eq. 4 (area under PR curve)


def mean_ap(preds, gts, iou_thrs) -> dict:
    """TODO: mAP@50 and mAP@50:95 (eq. 5) + per-class AP breakdown."""
    raise NotImplementedError
