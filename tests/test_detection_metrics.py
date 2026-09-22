"""Unit tests for utils.detection_metrics: iou, average_precision, mean_ap.

All cases use small synthetic, hand-computable data — no real images/models.
"""

from __future__ import annotations

import pytest

from utils.detection_metrics import average_precision, iou, mean_ap

# --------------------------------------------------------------------------- #
# iou
# --------------------------------------------------------------------------- #


def test_iou_identical_boxes():
    box = (0.0, 0.0, 10.0, 10.0)
    assert iou(box, box) == pytest.approx(1.0)


def test_iou_disjoint_boxes():
    box1 = (0.0, 0.0, 10.0, 10.0)
    box2 = (20.0, 20.0, 30.0, 30.0)
    assert iou(box1, box2) == pytest.approx(0.0)


def test_iou_touching_edges_is_zero_area_overlap():
    box1 = (0.0, 0.0, 10.0, 10.0)
    box2 = (10.0, 0.0, 20.0, 10.0)  # shares only the x=10 edge
    assert iou(box1, box2) == pytest.approx(0.0)


def test_iou_partial_overlap_hand_computed():
    # Two unit-scale squares offset by 1 in both axes: box1 area=4, box2 area=4,
    # intersection is the [1,2]x[1,2] square = area 1. union = 4+4-1 = 7.
    box1 = (0.0, 0.0, 2.0, 2.0)
    box2 = (1.0, 1.0, 3.0, 3.0)
    assert iou(box1, box2) == pytest.approx(1.0 / 7.0)


def test_iou_nested_box():
    outer = (0.0, 0.0, 10.0, 10.0)
    inner = (2.0, 2.0, 8.0, 8.0)  # area 36, fully inside outer (area 100)
    expected = 36.0 / 100.0
    assert iou(outer, inner) == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# average_precision / mean_ap
# --------------------------------------------------------------------------- #


def test_average_precision_no_predictions_returns_zero():
    gts = [("img1", 0, (0.0, 0.0, 10.0, 10.0))]
    assert average_precision([], gts, iou_thr=0.5, class_id=0) == 0.0


def test_average_precision_no_ground_truth_returns_zero():
    preds = [("img1", 0, 0.9, (0.0, 0.0, 10.0, 10.0))]
    assert average_precision(preds, [], iou_thr=0.5, class_id=0) == 0.0


def test_average_precision_perfect_single_match():
    gts = [("img1", 0, (0.0, 0.0, 10.0, 10.0))]
    preds = [("img1", 0, 0.9, (0.0, 0.0, 10.0, 10.0))]
    assert average_precision(preds, gts, iou_thr=0.5, class_id=0) == pytest.approx(1.0)


def test_average_precision_gt_in_other_image_never_matched():
    # A prediction in img2 must never be matchable against a GT that lives in img1.
    gts = [("img1", 0, (0.0, 0.0, 10.0, 10.0))]
    preds = [("img2", 0, 0.9, (0.0, 0.0, 10.0, 10.0))]
    # pred is an automatic FP (no GT of this class in img2) -> AP is 0.
    assert average_precision(preds, gts, iou_thr=0.5, class_id=0) == pytest.approx(0.0)


def test_average_precision_requires_confidence_sorting():
    """A regression to the reference repo's unsorted greedy matcher (processes preds
    in input-list order rather than score-descending order) would steal the GT match
    for the low-score prediction and mark the high-score one as a false positive,
    yielding a lower AP than the correct, confidence-ranked computation.
    """
    gt_box = (0.0, 0.0, 10.0, 10.0)
    high_score_correct = (0.0, 0.0, 10.0, 10.0)  # IoU = 1.0
    low_score_also_matches = (0.0, 0.0, 9.0, 9.0)  # IoU = 81/100 = 0.81, still >= thr

    gts = [("img1", 0, gt_box)]
    # Deliberately listed in ASCENDING score order (low first) so an implementation
    # that forgets to sort by score would process the weaker box first and let it
    # claim the only GT.
    preds = [
        ("img1", 0, 0.1, low_score_also_matches),
        ("img1", 0, 0.9, high_score_correct),
    ]

    ap = average_precision(preds, gts, iou_thr=0.5, class_id=0)
    # Correct (score-sorted) result: rank1 (score .9) TP, rank2 (score .1) FP since
    # the single GT is already used -> recall=[1,1], precision=[1, 0.5] ->
    # envelope=[1,1] -> AP = 1.0.
    assert ap == pytest.approx(1.0)
    # The unsorted-matching bug would instead give AP = 0.5 (see module docstring's
    # derivation) — assert we are NOT reproducing that regression.
    assert ap != pytest.approx(0.5)


def test_average_precision_false_positive_only():
    gts = [("img1", 0, (0.0, 0.0, 10.0, 10.0))]
    preds = [("img1", 0, 0.9, (100.0, 100.0, 110.0, 110.0))]  # no overlap
    assert average_precision(preds, gts, iou_thr=0.5, class_id=0) == pytest.approx(0.0)


def test_average_precision_missed_detection_partial_recall():
    # Two GTs of the same class in different images; only one is detected.
    gts = [
        ("img1", 0, (0.0, 0.0, 10.0, 10.0)),
        ("img2", 0, (0.0, 0.0, 10.0, 10.0)),
    ]
    preds = [("img1", 0, 0.9, (0.0, 0.0, 10.0, 10.0))]
    # recall tops out at 0.5 with precision 1.0 the whole way -> AP = 0.5.
    assert average_precision(preds, gts, iou_thr=0.5, class_id=0) == pytest.approx(0.5)


def test_mean_ap_multi_image_multi_class():
    """Hand-computed 2-class, 3-image scenario with a mix of correct/missed/false
    detections (see derivation in the accompanying design notes / PR description)."""
    gts = [
        ("img1", 0, (0.0, 0.0, 10.0, 10.0)),  # G1
        ("img2", 0, (0.0, 0.0, 10.0, 10.0)),  # G2
        ("img2", 1, (20.0, 20.0, 30.0, 30.0)),  # G3
        ("img3", 1, (0.0, 0.0, 10.0, 10.0)),  # G4
    ]
    preds = [
        # class 0: two perfect TPs ranked ahead of one FP -> AP = 1.0
        ("img1", 0, 0.95, (0.0, 0.0, 10.0, 10.0)),
        ("img2", 0, 0.80, (0.0, 0.0, 10.0, 10.0)),
        ("img3", 0, 0.60, (50.0, 50.0, 60.0, 60.0)),  # no class-0 GT in img3 -> FP
        # class 1: highest-score pred is a FP, then two TPs -> AP = 2/3
        ("img3", 1, 0.99, (100.0, 100.0, 110.0, 110.0)),  # FP
        ("img3", 1, 0.90, (0.0, 0.0, 10.0, 10.0)),  # TP vs G4
        ("img2", 1, 0.40, (20.0, 20.0, 30.0, 30.0)),  # TP vs G3
    ]

    result = mean_ap(preds, gts, iou_thrs=[0.5])

    assert result["per_class_ap_50"][0] == pytest.approx(1.0)
    assert result["per_class_ap_50"][1] == pytest.approx(2.0 / 3.0)
    assert result["map_50"] == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)
    # Single-threshold list -> map_50_95 collapses to the same value as map_50.
    assert result["map_50_95"] == pytest.approx(result["map_50"])


def test_mean_ap_empty_inputs():
    result = mean_ap([], [], iou_thrs=[0.5, 0.75])
    assert result["map_50"] == pytest.approx(0.0)
    assert result["map_50_95"] == pytest.approx(0.0)
    assert result["per_class_ap_50"] == {}


def test_mean_ap_averages_over_all_iou_thresholds():
    # A single perfect match: AP is 1.0 at every threshold <= 1.0, so map_50_95
    # across any list of thresholds in (0, 1] should still be exactly 1.0.
    gts = [("img1", 0, (0.0, 0.0, 10.0, 10.0))]
    preds = [("img1", 0, 0.9, (0.0, 0.0, 10.0, 10.0))]

    result = mean_ap(preds, gts, iou_thrs=[0.5, 0.55, 0.6, 0.75, 0.95])
    assert result["map_50"] == pytest.approx(1.0)
    assert result["map_50_95"] == pytest.approx(1.0)
