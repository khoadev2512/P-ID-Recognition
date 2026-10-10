"""Regression tests for the WBF score-dilution bug in pipeline.detect_infer.

The bug: passing one box-list PER TILE to weighted_boxes_fusion made WBF treat each
tile as a separate "model" and divide every fused score by the tile count (~54 for a
7168x4561 sheet), collapsing genuine 0.9 detections to ~0.02 and sinking them below
eval.score_thr / fgc.route.min_score. The fix collects all tiles' boxes into ONE model
list so scores survive while duplicate boxes at tile seams still merge.

These tests pin the WBF call shape at the library level (the exact contract
_detect_image relies on), without needing to stand up a real YOLO model.
"""

from __future__ import annotations

import pytest

pytest.importorskip("ensemble_boxes")
from ensemble_boxes import weighted_boxes_fusion  # noqa: E402

BOX = [0.4, 0.4, 0.5, 0.5]  # a single symbol's normalized xyxy


def test_per_tile_lists_dilute_score_the_old_bug():
    """Documents the OLD behavior: one list per tile divides score by tile count."""
    n_tiles = 54
    boxes_list = [[] for _ in range(n_tiles)]
    scores_list = [[] for _ in range(n_tiles)]
    labels_list = [[] for _ in range(n_tiles)]
    # symbol seen in only 2 overlapping tiles at conf 0.9
    for i in (0, 1):
        boxes_list[i] = [BOX]
        scores_list[i] = [0.9]
        labels_list[i] = [0.0]

    _, scores, _ = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list, iou_thr=0.55, skip_box_thr=0.001
    )
    # score is divided by ~n_tiles -> far below any sane threshold
    assert scores[0] < 0.1


def test_single_model_list_preserves_score_the_fix():
    """The FIX: all tile boxes in ONE list -> score is preserved, not divided."""
    # same symbol seen twice (from 2 overlapping tiles), merged as one source
    all_boxes = [BOX, BOX]
    all_scores = [0.9, 0.9]
    all_labels = [0.0, 0.0]

    _, scores, _ = weighted_boxes_fusion(
        [all_boxes], [all_scores], [all_labels], iou_thr=0.55, skip_box_thr=0.001
    )
    assert len(scores) == 1  # the two duplicates merged into one box
    assert scores[0] == pytest.approx(0.9, abs=1e-3)  # score preserved


def test_single_model_keeps_distinct_boxes_separate():
    """Non-overlapping boxes must stay separate (WBF only merges IoU>=thr)."""
    far_box = [0.8, 0.8, 0.9, 0.9]
    all_boxes = [BOX, far_box]
    all_scores = [0.9, 0.7]
    all_labels = [0.0, 1.0]

    boxes, scores, labels = weighted_boxes_fusion(
        [all_boxes], [all_scores], [all_labels], iou_thr=0.55, skip_box_thr=0.001
    )
    assert len(boxes) == 2  # both survive, not merged
    assert set(round(s, 3) for s in scores) == {0.9, 0.7}


def test_single_model_passes_over_a_resolved_score_threshold():
    """The whole point: a preserved 0.9 score clears eval.score_thr=0.25; a diluted one
    (~0.03) would not — so this guards the downstream filtering stays meaningful."""
    all_boxes, all_scores, all_labels = [BOX, BOX], [0.9, 0.9], [0.0, 0.0]
    _, scores, _ = weighted_boxes_fusion(
        [all_boxes], [all_scores], [all_labels], iou_thr=0.55, skip_box_thr=0.001
    )
    assert scores[0] >= 0.25  # survives the default score_thr / route.min_score
