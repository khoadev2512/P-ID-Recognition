"""Fine-grained classification metrics — the thesis contribution (report §2.2.3).

    within_group_confusion : |G|x|G| matrix M^G per visually similar group (Fig. 10)
    within_group_accuracy  : Acc_G, per-family headline metric (eq. 6)
    topk_accuracy          : top-1 / top-3 on the FGC head over GT crops
    rare_macro_f1          : F1_macro,rare over long-tail fine classes (eq. 7)

These isolate within-group error that aggregate mAP collapses into one number.
"""

from __future__ import annotations


def within_group_confusion(y_true, y_pred, group_fine_classes):
    """TODO: restricted confusion matrix M^G (only GT+pred within group G)."""
    raise NotImplementedError


def within_group_accuracy(confusion) -> float:
    raise NotImplementedError  # TODO: eq. 6 (diagonal mass / total)


def topk_accuracy(logits, y_true, ks=(1, 3)) -> dict:
    raise NotImplementedError


def rare_macro_f1(y_true, y_pred, rare_fine_classes) -> float:
    raise NotImplementedError  # TODO: eq. 7 (macro-F1 over rare set)
