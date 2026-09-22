"""Fine-grained classification metrics — the thesis contribution (report §2.2.3).

    within_group_confusion : |G|x|G| matrix M^G per visually similar group (Fig. 10)
    within_group_accuracy  : Acc_G, per-family headline metric (eq. 6)
    topk_accuracy          : top-1 / top-3 on the FGC head over GT crops
    rare_macro_f1          : F1_macro,rare over long-tail fine classes (eq. 7)

These isolate within-group error that aggregate mAP collapses into one number.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


def within_group_confusion(
    y_true: list[str], y_pred: list[str], group_fine_classes: list[str]
) -> np.ndarray:
    """|G|x|G| confusion matrix M^G restricted to `group_fine_classes` (Fig. 10).

    Rows are true labels, columns are predicted labels, both ordered by
    `group_fine_classes`. Labels outside the group (shouldn't occur for a
    correctly-routed group dataset, but guarded by sklearn's `labels=` regardless)
    are simply not counted.
    """
    return confusion_matrix(y_true, y_pred, labels=group_fine_classes)


def within_group_accuracy(confusion: np.ndarray) -> float:
    """Acc_G = diagonal mass / total mass (eq. 6), guarded against an empty/zero matrix."""
    total = confusion.sum()
    if total == 0:
        return 0.0
    return float(np.diag(confusion).sum() / total)


def topk_accuracy(logits: np.ndarray, y_true: list[int], ks: tuple[int, ...] = (1, 3)) -> dict:
    """Top-k accuracy on the FGC head over GT crops, for each k in `ks`.

    Args:
        logits: (N, C) per-sample class scores (softmax probs or raw logits — only
            relative order within a row matters, so either is fine).
        y_true: (N,) int labels in the same C-class index space as logits' columns.
        ks: which k values to report.

    Returns {f"top{k}": float, ...}. Each k is clipped to the number of classes (a
    "top-3" request against a 2-class problem degenerates to top-2 = 1.0 by
    definition, rather than erroring).
    """
    logits = np.asarray(logits)
    y_true_arr = np.asarray(y_true)
    n_samples, n_classes = logits.shape

    result: dict[str, float] = {}
    if n_samples == 0:
        for k in ks:
            result[f"top{k}"] = 0.0
        return result

    # Rank columns by descending score for each row.
    order = np.argsort(-logits, axis=1)

    for k in ks:
        k_eff = max(1, min(k, n_classes))
        top_k_preds = order[:, :k_eff]
        hits = (top_k_preds == y_true_arr[:, None]).any(axis=1)
        result[f"top{k}"] = float(hits.mean())

    return result


def rare_macro_f1(y_true: list[str], y_pred: list[str], rare_fine_classes: list[str]) -> float:
    """F1_macro,rare (eq. 7): macro-averaged F1 over ONLY the rare fine classes.

    Classes not in `rare_fine_classes` are excluded from the average even if they
    appear in y_true/y_pred (sklearn's `labels=` restricts which classes are scored).
    """
    if not rare_fine_classes:
        return 0.0
    return float(
        f1_score(y_true, y_pred, labels=rare_fine_classes, average="macro", zero_division=0)
    )
