"""Detection metrics (report §2.2.3, Table 1 top group).

    IoU (eq. 2), Precision/Recall (eq. 3), AP (eq. 4), mAP (eq. 5),
    mAP@50 and mAP@50:95, plus per-class AP for weak-class identification.

High recall is emphasised in the P&ID domain (missed symbols = record gaps).

`average_precision`/`mean_ap` take `image_id` and `class_id` alongside each box (a
shape change from the original stub's TODO signature, which had neither): matching
must never pair a prediction from one image against a ground-truth box from another,
and AP is inherently per-class, so both keys have to travel with the boxes rather
than being assumed out-of-band.
"""

from __future__ import annotations


def iou(
    pred_xyxy: tuple[float, float, float, float],
    gt_xyxy: tuple[float, float, float, float],
) -> float:
    """Exact corner-based IoU (eq. 2).

    Ported from PID_Symbol_Detection's `MetricsCalculator._box_iou`, generalized to
    operate on already-xyxy boxes (this project converts YOLO center-xywh -> pixel
    xyxy via `BBox.to_xyxy()` upstream, so the center/width/height unpacking the
    reference implementation did inline is not needed here).
    """
    x1_min, y1_min, x1_max, y1_max = pred_xyxy
    x2_min, y2_min, x2_max, y2_max = gt_xyxy

    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)

    area1 = max(0.0, x1_max - x1_min) * max(0.0, y1_max - y1_min)
    area2 = max(0.0, x2_max - x2_min) * max(0.0, y2_max - y2_min)
    union = area1 + area2 - intersection
    if union <= 0.0:
        return 0.0

    return intersection / union


def average_precision(
    preds: list[tuple[str, int, float, tuple[float, float, float, float]]],
    gts: list[tuple[str, int, tuple[float, float, float, float]]],
    iou_thr: float,
    class_id: int,
) -> float:
    """AP for one class at one IoU threshold (eq. 4), confidence-ranked / all-points
    precision-envelope method (standard VOC 2010+ / COCO AP, NOT the reference
    repo's un-ranked greedy match-and-average-P/R/F1).

    Args:
        preds: (image_id, class_id, score, xyxy) predicted boxes, any image/class mix.
        gts:   (image_id, class_id, xyxy) ground-truth boxes, any image/class mix.
        iou_thr: IoU threshold a match must clear to count as a true positive.
        class_id: restrict both preds and gts to this class before scoring.

    Algorithm:
        1. Filter preds/gts to this class_id.
        2. Sort preds by score DESCENDING — this ranking step is exactly what the
           reference repo's greedy matcher skips, and skipping it silently turns
           "AP" into an unranked precision/recall/F1 at a single implicit threshold.
        3. Walk the sorted preds; for each, among UNMATCHED gts in the SAME image_id
           (a GT in image A can never match a pred in image B) with
           iou(pred, gt) >= iou_thr, take the max-IoU one. Found -> TP, mark that GT
           used. Not found (no unmatched GT clears the threshold, or no GT of this
           class in that image at all) -> FP.
        4. cum_tp, cum_fp = cumulative sums over the sorted pred list.
        5. recall = cum_tp / total_gt_count_for_class; precision = cum_tp / (cum_tp + cum_fp).
        6. Precision envelope: precision[i] = max(precision[i:]) (monotonic non-increasing
           from the right).
        7. AP = sum_i (recall[i] - recall[i-1]) * precision[i] (area under the envelope,
           "all-points interpolation").

    Returns 0.0 (not NaN) when there are no predictions for this class, or when there
    are no ground truths for this class (AP undefined by convention -> 0.0).
    """
    class_preds = [p for p in preds if p[1] == class_id]
    class_gts = [g for g in gts if g[1] == class_id]

    total_gt = len(class_gts)
    if not class_preds or total_gt == 0:
        return 0.0

    # Sort by score descending. Stable sort keeps input order for score ties, which
    # is an arbitrary-but-deterministic tie-break (matches numpy/sklearn convention).
    class_preds = sorted(class_preds, key=lambda p: p[2], reverse=True)

    # GTs available for matching, grouped by image, with a per-gt "used" flag.
    gts_by_image: dict[str, list[dict]] = {}
    for image_id, _cid, xyxy in class_gts:
        gts_by_image.setdefault(image_id, []).append({"xyxy": xyxy, "used": False})

    n = len(class_preds)
    tp = [0] * n
    fp = [0] * n

    for i, (image_id, _cid, _score, pred_xyxy) in enumerate(class_preds):
        candidates = gts_by_image.get(image_id, [])
        best_iou = 0.0
        best_gt = None
        for gt in candidates:
            if gt["used"]:
                continue
            cur_iou = iou(pred_xyxy, gt["xyxy"])
            if cur_iou >= iou_thr and cur_iou > best_iou:
                best_iou = cur_iou
                best_gt = gt
        if best_gt is not None:
            tp[i] = 1
            best_gt["used"] = True
        else:
            fp[i] = 1

    cum_tp = 0
    cum_fp = 0
    recalls = [0.0] * n
    precisions = [0.0] * n
    for i in range(n):
        cum_tp += tp[i]
        cum_fp += fp[i]
        recalls[i] = cum_tp / total_gt
        precisions[i] = cum_tp / (cum_tp + cum_fp) if (cum_tp + cum_fp) > 0 else 0.0

    # Precision envelope: make precision monotonically non-increasing from the right.
    for i in range(n - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Area under the recall/precision-envelope curve (all-points interpolation).
    ap = 0.0
    prev_recall = 0.0
    for i in range(n):
        ap += (recalls[i] - prev_recall) * precisions[i]
        prev_recall = recalls[i]

    return ap


def mean_ap(
    preds: list[tuple[str, int, float, tuple[float, float, float, float]]],
    gts: list[tuple[str, int, tuple[float, float, float, float]]],
    iou_thrs: list[float],
) -> dict:
    """mAP@iou_thrs[0] (assumed 0.5) and mAP averaged over ALL iou_thrs (eq. 5),
    plus per-class AP at iou_thrs[0] for weak-class identification.

    Classes are enumerated from the UNION of class_ids present in preds and gts (a
    class with predictions but no GT, or vice versa, still gets an AP of 0.0 via
    `average_precision`'s guards rather than being silently dropped).

    Returns:
        {
          "map_50": float,
          "map_50_95": float,
          "per_class_ap_50": {class_id: float, ...},
        }
    """
    class_ids = sorted({p[1] for p in preds} | {g[1] for g in gts})

    if not class_ids or not iou_thrs:
        return {"map_50": 0.0, "map_50_95": 0.0, "per_class_ap_50": {}}

    per_class_ap_50: dict[int, float] = {}
    all_aps: list[float] = []  # one entry per (class, iou_thr)

    for class_id in class_ids:
        aps_this_class: list[float] = []
        for thr in iou_thrs:
            ap = average_precision(preds, gts, thr, class_id)
            aps_this_class.append(ap)
        per_class_ap_50[class_id] = aps_this_class[0]
        all_aps.extend(aps_this_class)

    map_50 = sum(per_class_ap_50.values()) / len(per_class_ap_50)
    map_50_95 = sum(all_aps) / len(all_aps)

    return {
        "map_50": map_50,
        "map_50_95": map_50_95,
        "per_class_ap_50": per_class_ap_50,
    }
