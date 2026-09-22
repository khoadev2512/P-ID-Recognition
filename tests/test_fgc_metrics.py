"""Unit tests for utils.fgc_metrics: confusion/accuracy, top-k, rare macro-F1.

All cases use small synthetic label lists / logit arrays — no real crops/models.
"""

from __future__ import annotations

import numpy as np
import pytest

from utils.fgc_metrics import (
    rare_macro_f1,
    topk_accuracy,
    within_group_accuracy,
    within_group_confusion,
)

GROUP = ["gate_valve", "globe_valve", "ball_valve"]
Y_TRUE = ["gate_valve", "gate_valve", "globe_valve", "ball_valve", "ball_valve"]
Y_PRED = ["gate_valve", "globe_valve", "globe_valve", "ball_valve", "gate_valve"]


def test_within_group_confusion_matrix_values():
    confusion = within_group_confusion(Y_TRUE, Y_PRED, GROUP)
    expected = np.array(
        [
            [1, 1, 0],  # true=gate_valve   -> pred gate,globe,ball
            [0, 1, 0],  # true=globe_valve
            [1, 0, 1],  # true=ball_valve
        ]
    )
    np.testing.assert_array_equal(confusion, expected)


def test_within_group_accuracy_hand_computed():
    confusion = within_group_confusion(Y_TRUE, Y_PRED, GROUP)
    # diag = 1 (gate/gate) + 1 (globe/globe) + 1 (ball/ball) = 3; total = 5.
    assert within_group_accuracy(confusion) == pytest.approx(3.0 / 5.0)


def test_within_group_accuracy_empty_matrix_is_zero():
    confusion = np.zeros((3, 3), dtype=int)
    assert within_group_accuracy(confusion) == pytest.approx(0.0)


def test_within_group_accuracy_perfect_predictions():
    y = ["gate_valve", "globe_valve", "ball_valve", "gate_valve"]
    confusion = within_group_confusion(y, y, GROUP)
    assert within_group_accuracy(confusion) == pytest.approx(1.0)


def test_rare_macro_f1_hand_computed():
    # Rare class = ball_valve only. Occurrences: true ball_valve at idx3, idx4.
    # idx3: pred=ball_valve (TP). idx4: pred=gate_valve (FN for ball_valve).
    # No false positives for ball_valve (pred==ball_valve only at idx3, correctly).
    # precision = 1/(1+0) = 1.0, recall = 1/(1+1) = 0.5, F1 = 2*1*0.5/1.5 = 2/3.
    f1 = rare_macro_f1(Y_TRUE, Y_PRED, rare_fine_classes=["ball_valve"])
    assert f1 == pytest.approx(2.0 / 3.0)


def test_rare_macro_f1_multiple_rare_classes_macro_averaged():
    # gate_valve: true at idx0,idx1; pred=gate_valve only at idx0 -> TP=1,FN=1.
    #   FP: pred==gate_valve at idx4 (true=ball_valve) -> FP=1.
    #   precision = 1/2 = 0.5, recall = 1/2 = 0.5, F1 = 0.5.
    # ball_valve F1 = 2/3 (from above).
    # macro average over {gate_valve, ball_valve} = (0.5 + 2/3) / 2.
    f1 = rare_macro_f1(Y_TRUE, Y_PRED, rare_fine_classes=["gate_valve", "ball_valve"])
    assert f1 == pytest.approx((0.5 + 2.0 / 3.0) / 2.0)


def test_rare_macro_f1_excludes_non_rare_classes():
    # gate_valve has F1=0.5 (precision=1/2, recall=1/2) but is NOT in this
    # rare_fine_classes set, so it must not pull the ball_valve-only average down.
    f1_with_gate = rare_macro_f1(Y_TRUE, Y_PRED, rare_fine_classes=["ball_valve", "gate_valve"])
    f1_ball_only = rare_macro_f1(Y_TRUE, Y_PRED, rare_fine_classes=["ball_valve"])
    assert f1_with_gate != pytest.approx(f1_ball_only)
    assert f1_with_gate == pytest.approx((0.5 + 2.0 / 3.0) / 2.0)


def test_rare_macro_f1_empty_rare_set_is_zero():
    assert rare_macro_f1(Y_TRUE, Y_PRED, rare_fine_classes=[]) == pytest.approx(0.0)


def test_topk_accuracy_top1_and_top3_differ():
    # 4 samples, 3 classes. Constructed so the true label is ranked 1st, 2nd, 1st,
    # and 3rd respectively -> top-1 only catches 2/4, top-3 catches all 4/4.
    logits = np.array(
        [
            [3.0, 2.0, 1.0],  # true=0 ranked 1st  (top1 hit)
            [3.0, 2.0, 1.0],  # true=1 ranked 2nd  (top1 miss, top2/3 hit)
            [1.0, 2.0, 3.0],  # true=2 ranked 1st  (top1 hit)
            [1.0, 3.0, 2.0],  # true=0 ranked 3rd  (top1/2 miss, top3 hit)
        ]
    )
    y_true = [0, 1, 2, 0]

    result = topk_accuracy(logits, y_true, ks=(1, 3))
    assert result["top1"] == pytest.approx(2.0 / 4.0)
    assert result["top3"] == pytest.approx(4.0 / 4.0)
    assert result["top1"] != result["top3"]


def test_topk_accuracy_k_clipped_to_num_classes():
    logits = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    y_true = [2, 0]
    # k=10 requested but only 3 classes exist -> clip to 3 -> every sample's true
    # label is trivially among the top-3 (all columns) -> accuracy 1.0.
    result = topk_accuracy(logits, y_true, ks=(10,))
    assert result["top10"] == pytest.approx(1.0)


def test_topk_accuracy_empty_input():
    logits = np.zeros((0, 3))
    result = topk_accuracy(logits, [], ks=(1, 3))
    assert result == {"top1": 0.0, "top3": 0.0}
