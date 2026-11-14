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

from dataclasses import dataclass

import numpy as np


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


# ------------------------------------------------------------------ #
# Vectorised matching with ignore regions (PLAN B6b-2 / B6d.2)
# ------------------------------------------------------------------ #
#
# `mean_ap` above stays the default path (DigitizePID numbers are byte-identical to before).
# The functions below are used when a run needs more than mAP: ignore regions (COCO
# `iscrowd: 1`), class-agnostic scoring, per-GT hits for recall by group, or the end-to-end
# one-shot labels. `greedy_match` makes exactly the decisions of `average_precision` (same
# score order with the same tie-break, same IoU arithmetic, same max-IoU-first-index choice),
# so `ap_from_match` reproduces `mean_ap` bit for bit when there is no ignore region
# (tests/test_detection_eval.py checks it on random data).

# A prediction that matches no GT box falls in an ignore region when IoU(pred, region) >=
# IGNORE_IOU or when at least IGNORE_INSIDE of the PREDICTION's area lies inside the region
# ("mostly inside": the small box a detector puts on a leader-arrow head inside the arrow's
# region). IoU >= 0.5 implies an inside fraction >= 0.5, so with both at 0.5 the second test
# alone decides; both are kept so either can be changed. Fixed, not swept with the AP IoU
# threshold: an ignore region is not a symbol whose box quality is being graded.
IGNORE_IOU = 0.5
IGNORE_INSIDE = 0.5


def _as_xyxy(boxes) -> np.ndarray:
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4)


def iou_matrix(a, b) -> np.ndarray:
    """(len(a), len(b)) IoU of xyxy boxes, the same arithmetic as `iou` element by element."""
    a, b = _as_xyxy(a), _as_xyxy(b)
    x_left = np.maximum(a[:, None, 0], b[None, :, 0])
    y_top = np.maximum(a[:, None, 1], b[None, :, 1])
    x_right = np.minimum(a[:, None, 2], b[None, :, 2])
    y_bottom = np.minimum(a[:, None, 3], b[None, :, 3])
    disjoint = (x_right < x_left) | (y_bottom < y_top)
    inter = (x_right - x_left) * (y_bottom - y_top)
    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    ok = ~disjoint & (union > 0.0)
    out = np.zeros(inter.shape)
    np.divide(inter, union, out=out, where=ok)
    return out


def inside_fraction(a, b) -> np.ndarray:
    """(len(a), len(b)) fraction of each `a` box's area that lies inside each `b` box (0 for
    an `a` box of zero area)."""
    a, b = _as_xyxy(a), _as_xyxy(b)
    w = np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0])
    h = np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1])
    inter = np.maximum(0.0, w) * np.maximum(0.0, h)
    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    out = np.zeros(inter.shape)
    np.divide(inter, area_a[:, None], out=out, where=area_a[:, None] > 0.0)
    return out


def ignore_hits(
    preds: list[tuple[str, int, float, tuple[float, float, float, float]]],
    ignores: list[tuple[str, tuple[float, float, float, float]]],
) -> np.ndarray:
    """Per prediction, the index (into `ignores`) of the ignore region it falls in, -1 for
    none (see IGNORE_IOU / IGNORE_INSIDE; several regions: the one covering most of it).
    Regions have no class: they apply to a prediction of any label."""
    out = np.full(len(preds), -1, dtype=np.int64)
    if not preds or not ignores:
        return out
    by_image: dict[str, list[int]] = {}
    for i, (image_id, _box) in enumerate(ignores):
        by_image.setdefault(image_id, []).append(i)
    pred_by_image: dict[str, list[int]] = {}
    for i, p in enumerate(preds):
        if p[0] in by_image:
            pred_by_image.setdefault(p[0], []).append(i)
    for image_id, pis in pred_by_image.items():
        ris = by_image[image_id]
        pb = [preds[i][3] for i in pis]
        rb = [ignores[i][1] for i in ris]
        inside = inside_fraction(pb, rb)
        hit = (iou_matrix(pb, rb) >= IGNORE_IOU) | (inside >= IGNORE_INSIDE)
        inside = np.where(hit, inside, -1.0)
        best = inside.argmax(axis=1)
        for row, pi in enumerate(pis):
            if hit[row, best[row]]:
                out[pi] = ris[int(best[row])]
    return out


@dataclass(frozen=True)
class Match:
    """`greedy_match`'s result, parallel to its inputs.

    order        prediction indices by score, best first (stable: ties keep input order)
    pred_gt      per prediction, the GT index it matched, -1 for none
    pred_ignored per prediction, True when it matched no GT but falls in an ignore region:
                 neither TP nor FP, left out of the ranking
    gt_hit       per GT box, True when a prediction matched it
    """

    order: np.ndarray
    pred_gt: np.ndarray
    pred_ignored: np.ndarray
    gt_hit: np.ndarray

    @property
    def tp(self) -> np.ndarray:
        return self.pred_gt >= 0

    @property
    def fp(self) -> np.ndarray:
        return (self.pred_gt < 0) & ~self.pred_ignored


def greedy_match(
    preds: list[tuple[str, int, float, tuple[float, float, float, float]]],
    gts: list[tuple[str, int, tuple[float, float, float, float]]],
    iou_thr: float,
    ignore_of_pred: np.ndarray | None = None,
) -> Match:
    """`average_precision`'s matching for every class at once: predictions by score
    descending; each takes the unmatched GT box of its class in its image with the highest
    IoU >= iou_thr (first one on a tie). A prediction left without a GT is ignored when
    `ignore_of_pred` (from `ignore_hits`) gives it a region, else it is a false positive."""
    n = len(preds)
    scores = np.array([p[2] for p in preds], dtype=np.float64)
    order = np.argsort(-scores, kind="stable") if n else np.zeros(0, dtype=np.int64)
    pred_gt = np.full(n, -1, dtype=np.int64)
    gt_hit = np.zeros(len(gts), dtype=bool)

    gt_by_key: dict[tuple[str, int], list[int]] = {}
    for j, (image_id, class_id, _box) in enumerate(gts):
        gt_by_key.setdefault((image_id, class_id), []).append(j)
    pred_by_key: dict[tuple[str, int], list[int]] = {}
    for i in order.tolist():
        key = (preds[i][0], preds[i][1])
        if key in gt_by_key:
            pred_by_key.setdefault(key, []).append(i)

    for key, pis in pred_by_key.items():
        gis = gt_by_key[key]
        ious = iou_matrix([preds[i][3] for i in pis], [gts[j][2] for j in gis])
        used = np.zeros(len(gis), dtype=bool)
        for row, pi in enumerate(pis):
            cand = np.where(used, -1.0, ious[row])
            col = int(cand.argmax())
            if cand[col] >= iou_thr and cand[col] > 0.0:
                used[col] = True
                pred_gt[pi] = gis[col]
                gt_hit[gis[col]] = True

    ignored = np.zeros(n, dtype=bool)
    if ignore_of_pred is not None and n:
        ignored = (pred_gt < 0) & (np.asarray(ignore_of_pred) >= 0)
    return Match(order=order, pred_gt=pred_gt, pred_ignored=ignored, gt_hit=gt_hit)


def ap_from_match(
    match: Match, pred_class: np.ndarray, gt_class: np.ndarray, class_id: int
) -> float:
    """AP of one class from a `greedy_match` result: the same ranking, cumulative sums,
    precision envelope and all-points area as `average_precision` (ignored predictions
    are left out of the ranking). 0.0 without a prediction or without a GT of the class."""
    in_class = pred_class[match.order] == class_id
    total_gt = int((gt_class == class_id).sum())
    if not in_class.any() or total_gt == 0:
        return 0.0
    ranked = match.order[in_class]
    ranked = ranked[~match.pred_ignored[ranked]]
    if len(ranked) == 0:
        return 0.0
    cum_tp = np.cumsum(match.pred_gt[ranked] >= 0)
    cum_fp = np.cumsum(match.pred_gt[ranked] < 0)
    recalls = cum_tp / total_gt
    precisions = np.maximum.accumulate((cum_tp / (cum_tp + cum_fp))[::-1])[::-1]
    terms = (recalls - np.concatenate([[0.0], recalls[:-1]])) * precisions
    return float(np.cumsum(terms)[-1])


def match_report(
    preds: list[tuple[str, int, float, tuple[float, float, float, float]]],
    gts: list[tuple[str, int, tuple[float, float, float, float]]],
    iou_thrs: list[float],
    ignores: list[tuple[str, tuple[float, float, float, float]]] | None = None,
) -> tuple[dict, Match, np.ndarray]:
    """`mean_ap`'s dict plus precision / recall at iou_thrs[0] (over every prediction given:
    the caller applies the score threshold), with ignore regions. Returns (metrics, the
    iou_thrs[0] Match, per-prediction ignore region index) so the caller can break the hits
    down by GT group or the ignored predictions by region kind."""
    hits = ignore_hits(preds, ignores or [])
    pred_class = np.array([p[1] for p in preds], dtype=np.int64)
    gt_class = np.array([g[1] for g in gts], dtype=np.int64)
    class_ids = sorted({p[1] for p in preds} | {g[1] for g in gts})
    first = greedy_match(preds, gts, iou_thrs[0], hits) if iou_thrs else None
    if not class_ids or not iou_thrs:
        metrics: dict = {"map_50": 0.0, "map_50_95": 0.0, "per_class_ap_50": {}}
    else:
        per_class_ap_50: dict[int, float] = {}
        all_aps: list[float] = []
        aps = {c: [] for c in class_ids}
        for k, thr in enumerate(iou_thrs):
            match = first if k == 0 else greedy_match(preds, gts, thr, hits)
            for c in class_ids:
                aps[c].append(ap_from_match(match, pred_class, gt_class, c))
        for c in class_ids:
            per_class_ap_50[c] = aps[c][0]
            all_aps.extend(aps[c])
        metrics = {
            "map_50": sum(per_class_ap_50.values()) / len(per_class_ap_50),
            "map_50_95": sum(all_aps) / len(all_aps),
            "per_class_ap_50": per_class_ap_50,
        }
    if first is None:
        first = greedy_match(preds, gts, 0.5, hits)
    tp, fp = int(first.tp.sum()), int(first.fp.sum())
    metrics.update(
        tp=tp,
        fp=fp,
        n_gt=len(gts),
        precision=tp / (tp + fp) if tp + fp else None,
        recall=tp / len(gts) if gts else None,
    )
    return metrics, first, hits
