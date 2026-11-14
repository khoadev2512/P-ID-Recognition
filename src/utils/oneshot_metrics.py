"""Open-set metrics and class groups for the crop-level one-shot evaluation (PLAN B6b / B8).

Pure numpy (no torch), so the numbers can be recomputed from the per-crop CSV the evaluation
writes, without a GPU.

Vocabulary used throughout:
    known     a crop of a class that has an ISO template (the test crop SHOULD be named)
    unknown   a crop of an out-of-vocabulary class (the test crop SHOULD be rejected: "Other")
    cos       the crop's best cosine to ANY prototype, un-gated (a crop is accepted at tau
              when cos >= tau, rejected as "Other" when cos < tau)
    correct   a known crop whose top-1 prototype is its expected template

Every metric that needs both a known and an unknown set returns None, with a warning, when
either is empty (an encoder with no held-out OOV crops in the subset must not crash a run).
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Threshold-free: AUROC, OSCR
# ------------------------------------------------------------------ #


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).ravel()


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks of `x`, ties given the mean of the ranks they span."""
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    last = np.cumsum(counts)
    return ((last - counts + 1 + last) / 2.0)[inverse]


def auroc(pos_scores, neg_scores) -> float | None:
    """P(a positive outscores a negative), ties counting 1/2 (rank-sum / Mann-Whitney).

    A HIGHER score means "more positive". For "Other" detection the positives are the
    unknown crops and the score is -best_cos (a low cosine = more likely unknown): see
    `other_auroc`. None when either side is empty.
    """
    pos, neg = _arr(pos_scores), _arr(neg_scores)
    if len(pos) == 0 or len(neg) == 0:
        logger.warning("auroc needs both classes (%d positives, %d negatives)", len(pos), len(neg))
        return None
    ranks = _average_ranks(np.concatenate([pos, neg]))
    u = ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def other_auroc(known_cos, unknown_cos) -> float | None:
    """AUROC of "Other" detection on max-cosine: positives = unknown, score = -best_cos.
    1.0 when every unknown crop has a lower cosine than every known crop."""
    return auroc(-_arr(unknown_cos), -_arr(known_cos))


def oscr(known_correct, known_cos, unknown_cos) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Open-set classification rate curve: (FPR, CCR, area), sweeping tau over every score.

        CCR(tau) = fraction of KNOWN crops with cos >= tau AND correctly named
        FPR(tau) = fraction of UNKNOWN crops with cos >= tau

    The curve starts at (0, 0) (tau above every score) and ends at (1, closed-set accuracy)
    (tau below every score); `area` is the trapezoid area under CCR over FPR. 1.0 when the
    unknowns all score below every known crop and every known crop is named correctly. AUROC
    ignores naming; OSCR does not: high AUROC with low OSCR = known and unknown separate well
    but the known are named wrong. None when either set is empty.
    """
    correct = np.asarray(known_correct, dtype=bool).ravel()
    known, unknown = _arr(known_cos), _arr(unknown_cos)
    if len(correct) != len(known):
        raise ValueError(f"{len(correct)} known_correct flags vs {len(known)} known cosines")
    if len(known) == 0 or len(unknown) == 0:
        logger.warning(
            "oscr needs known and unknown crops (%d known, %d unknown)", len(known), len(unknown)
        )
        return None
    taus = np.unique(np.concatenate([known, unknown]))[::-1]  # descending: accepts more and more
    hits = np.sort(known[correct])
    unknown = np.sort(unknown)
    ccr = (len(hits) - np.searchsorted(hits, taus, side="left")) / len(known)
    fpr = (len(unknown) - np.searchsorted(unknown, taus, side="left")) / len(unknown)
    fpr, ccr = np.concatenate([[0.0], fpr]), np.concatenate([[0.0], ccr])
    area = float(np.sum((fpr[1:] - fpr[:-1]) * (ccr[1:] + ccr[:-1]) / 2.0))
    return fpr, ccr, area


# ------------------------------------------------------------------ #
# Threshold sweep and the two tau rules (PLAN B8 point 2)
# ------------------------------------------------------------------ #


def _accepted(cos: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """(len(taus),) number of scores >= each tau."""
    s = np.sort(cos)
    return len(s) - np.searchsorted(s, taus, side="left")


def tau_sweep(known_correct, known_cos, unknown_cos, taus) -> dict[str, list]:
    """Per tau (columns of the result, parallel to `taus`):

        known_acc            known crops accepted AND correctly named / all known crops (= CCR)
        known_acc_accepted   correct / accepted among the known crops that pass (None: none pass)
        known_reject_rate    known crops rejected as "Other" / all known crops
        unknown_reject_rate  unknown crops rejected / all unknown crops (None without unknowns)

    known_acc never rises and both reject rates never fall as tau grows. The report figure of
    B6b; it is NOT where tau is chosen (that is val_sel, rules A and B).
    """
    correct = np.asarray(known_correct, dtype=bool).ravel()
    known, unknown, taus = _arr(known_cos), _arr(unknown_cos), _arr(taus)
    if len(correct) != len(known):
        raise ValueError(f"{len(correct)} known_correct flags vs {len(known)} known cosines")
    if len(known) == 0:
        raise ValueError("tau_sweep needs at least one known crop")
    accepted = _accepted(known, taus)
    right = _accepted(known[correct], taus)
    acc_of_accepted = [float(r / a) if a else None for r, a in zip(right, accepted, strict=True)]
    out: dict[str, list] = {
        "tau": taus.tolist(),
        "known_acc": (right / len(known)).tolist(),
        "known_acc_accepted": acc_of_accepted,
        "known_reject_rate": (1.0 - accepted / len(known)).tolist(),
    }
    if len(unknown) == 0:
        logger.warning("tau_sweep: no unknown crops, unknown_reject_rate left out")
        out["unknown_reject_rate"] = [None] * len(taus)
    else:
        out["unknown_reject_rate"] = (1.0 - _accepted(unknown, taus) / len(unknown)).tolist()
    return out


def tau_rule_a(known_cos, tpr: float = 0.95) -> float:
    """Rule A (tau@TPR95): the largest tau that still accepts at least `tpr` of the known
    crops. Needs no unknown class. With ties in the cosines more than `tpr` may be accepted."""
    known = _arr(known_cos)
    if len(known) == 0:
        raise ValueError("tau_rule_a needs at least one known crop")
    if not 0.0 < tpr <= 1.0:
        raise ValueError(f"tpr must be in (0, 1], got {tpr}")
    k = int(np.ceil(tpr * len(known) - 1e-9))
    return float(np.sort(known)[::-1][k - 1])


def tau_rule_b(known_cos, pseudo_unknown_cos) -> tuple[float, float] | None:
    """Rule B (leave a prototype out): the Youden-optimal tau between the known crops and
    simulated unknown crops (a crop scored against the prototypes WITHOUT its own class's;
    the caller builds `pseudo_unknown_cos`). Returns (tau, J) with J = accepted-known rate
    minus accepted-pseudo-unknown rate at tau, maximal. Among tied optima the largest wins; tau
    is placed halfway down to the next lower observed cosine so that it is not a crop's own
    score. A simulated unknown is easier than a real one, so this tau is optimistic. None
    without pseudo-unknown crops.
    """
    known, pseudo = _arr(known_cos), _arr(pseudo_unknown_cos)
    if len(known) == 0:
        raise ValueError("tau_rule_b needs at least one known crop")
    if len(pseudo) == 0:
        logger.warning("tau_rule_b: no simulated-unknown crops")
        return None
    cands = np.unique(np.concatenate([known, pseudo]))
    j = _accepted(known, cands) / len(known) - _accepted(pseudo, cands) / len(pseudo)
    best = len(cands) - 1 - int(np.argmax(j[::-1]))  # last (largest tau) maximum
    tau = float(cands[best]) if best == 0 else float((cands[best] + cands[best - 1]) / 2.0)
    return tau, float(j[best])


def margins(best_cos, second_cos) -> np.ndarray:
    """Top-1 minus top-2 cosine per crop (PLAN B8, score form 2). A crop with no second
    prototype (second_cos NaN) gets +inf: the margin gate never rejects it."""
    best, second = _arr(best_cos), _arr(second_cos)
    return np.where(np.isnan(second), np.inf, best - second)


def margin_rule_a(known_margin, tpr: float = 0.95) -> float:
    """delta@TPR95: the largest delta that still accepts at least `tpr` of the known crops on
    the margin alone (the same rule A as tau, applied to `margins`). 0.0 (gate off) when that
    crop's margin is infinite (no second prototype)."""
    picked = tau_rule_a(known_margin, tpr)
    return picked if np.isfinite(picked) else 0.0


def margin_auroc(known_margin, unknown_margin) -> float | None:
    """AUROC of "Other" detection on the margin: positives = unknown, score = -margin."""
    return auroc(-_arr(unknown_margin), -_arr(known_margin))


def gated_other_prf(known_reject, unknown_reject) -> dict | None:
    """P/R/F1 of "Other" for any gate, from per-crop reject flags (unknown = positive)."""
    known, unknown = np.asarray(known_reject, bool), np.asarray(unknown_reject, bool)
    if len(unknown) == 0:
        logger.warning("gated_other_prf: no unknown crops")
        return None
    tp, fp = int(unknown.sum()), int(known.sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / len(unknown)
    f1 = None
    if precision is not None:
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": len(unknown) - tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def other_prf(known_cos, unknown_cos, tau: float) -> dict | None:
    """Precision / recall / F1 of the "Other" decision (cos < tau) with unknown = positive."""
    if len(_arr(unknown_cos)) == 0:
        logger.warning("other_prf: no unknown crops")
        return None
    prf = gated_other_prf(_arr(known_cos) < tau, _arr(unknown_cos) < tau)
    return {"tau": float(tau), **prf}


def open_set_summary(known_correct, known_cos, unknown_cos, other_tau: float, taus) -> dict:
    """The "Other" block of a report: counts, AUROC, OSCR area, P/R/F1 at `other_tau` and the
    tau sweep. Entries that need unknown crops are None when there are none."""
    result = oscr(known_correct, known_cos, unknown_cos)
    return {
        "n_known": int(len(_arr(known_cos))),
        "n_unknown": int(len(_arr(unknown_cos))),
        "auroc": other_auroc(known_cos, unknown_cos),
        "oscr_area": None if result is None else result[2],
        "other_at_tau": other_prf(known_cos, unknown_cos, other_tau),
        "sweep": tau_sweep(known_correct, known_cos, unknown_cos, taus),
    }


# ------------------------------------------------------------------ #
# Class groups (PLAN "Lớp ISO thuộc nhóm nào (S / V / H)")
# ------------------------------------------------------------------ #

GROUPS = ("S", "V", "V_of_S", "H_chon", "h_extra", "OOV_seen", "OOV_heldout", "UNSURE")
STATUSES = ("ISO", "VARIANT", "OOV", "UNSURE")


@dataclass(frozen=True)
class ClassInfo:
    """One DigitizePID class: its symbol_map status, the ISO template it should be named as
    (None for OOV / UNSURE) and its group relative to one encoder.

        S            ISO class the encoder trained on (its template is "seen")
        V            VARIANT class the encoder trained on whose template is NOT an S template
        V_of_S       VARIANT class the encoder trained on whose template is also an S class's
                     (a drawing convention of a seen symbol: S2/S12/S13); not pooled into S or V
        H_chon       ISO class held out of the encoder (R1a)
        h_extra      VARIANT class held out of the encoder (R1a, with its h_in symbol)
        OOV_seen     out-of-vocabulary class the encoder trained on ("seen Other")
        OOV_heldout  out-of-vocabulary class held out of the encoder (h_out)
        UNSURE       identity uncertain: excluded from every test
    """

    name: str
    status: str
    reg_number: str
    template: str | None
    group: str


# oneshot.symbol_map's default, relative to the repository root (docs/ ships with the code).
DEFAULT_SYMBOL_MAP = "docs/symbol_mapping/symbol_map.csv"
_REPO_ROOT = Path(__file__).resolve().parents[2]


def symbol_map_path(value: str | Path | None = None) -> Path:
    """oneshot.symbol_map (None: DEFAULT_SYMBOL_MAP); a relative path is repo-relative."""
    path = Path(str(value or DEFAULT_SYMBOL_MAP))
    return path if path.is_absolute() else _REPO_ROOT / path


def read_symbol_map(path: str | Path) -> dict[str, dict[str, str]]:
    """symbol_map.csv as {`Symbol_<n>`: row}."""
    with Path(path).open(newline="", encoding="utf-8") as f:
        return {f"Symbol_{r['symbol_n'].strip()}": r for r in csv.DictReader(f)}


def read_vocab(path: str | Path) -> dict[str, dict[str, str]]:
    """vocab.csv as {reg_number: row}."""
    with Path(path).open(newline="", encoding="utf-8") as f:
        return {r["reg_number"].strip(): r for r in csv.DictReader(f)}


def assign_groups(
    symbol_map: Mapping[str, Mapping[str, str]],
    vocab: Mapping[str, Mapping[str, str]],
    fine_classes: Iterable[str],
    excluded_classes: Iterable[str] = (),
    holdout=None,
) -> dict[str, ClassInfo]:
    """Group of every symbol_map class for the encoder whose training classes are
    `fine_classes` — the ONE place the grouping is computed, never a hand-typed list.

    The template of an ISO class is vocab.csv's fine_name for its reg_number (that is what
    the prototype is called), falling back to the symbol_map row's; a VARIANT's is the vocab
    row with its reg_number. `excluded_classes` (the checkpoint's) and `holdout` (anything
    with an `.excluded` list, i.e. utils.oneshot.Holdout) are only used to warn about a class
    that is missing from `fine_classes` without being declared held out: it is still treated
    as held out.
    """
    seen = set(fine_classes)
    declared = set(excluded_classes) | set(getattr(holdout, "excluded", None) or [])
    status: dict[str, str] = {}
    reg: dict[str, str] = {}
    for name, row in symbol_map.items():
        status[name] = (row.get("status") or "").strip().upper()
        reg[name] = (row.get("reg_number") or "").strip()
        if status[name] not in STATUSES:
            raise ValueError(f"{name}: status {status[name]!r} is not one of {STATUSES}")

    def template_of(name: str) -> str | None:
        if status[name] not in ("ISO", "VARIANT") or not reg[name]:
            return None
        row = vocab.get(reg[name])
        if row is None and status[name] == "ISO":
            row = symbol_map[name]
        return ((row or {}).get("fine_name") or "").strip() or None

    seen_iso_regs = {reg[n] for n in symbol_map if status[n] == "ISO" and n in seen and reg[n]}
    out: dict[str, ClassInfo] = {}
    for name in symbol_map:
        st, held = status[name], name not in seen
        if held and name not in declared and st != "UNSURE":
            logger.warning(
                "%s is not in the encoder's training classes but is not declared excluded "
                "(checkpoint excluded_classes / oneshot.holdout): treated as held out",
                name,
            )
        if st == "UNSURE":
            group = "UNSURE"
        elif st == "OOV":
            group = "OOV_heldout" if held else "OOV_seen"
        elif st == "ISO":
            group = "H_chon" if held else "S"
        else:  # VARIANT
            group = "h_extra" if held else ("V_of_S" if reg[name] in seen_iso_regs else "V")
        out[name] = ClassInfo(name, st, reg[name], template_of(name), group)
    return out


# Groups of an ISO template (a vocab fine_name) on OPEN100-ISO (P4), never pooled.
ISO_GROUPS = ("S", "V", "H_chon", "V_held", "H_tu_nhien")
# OPEN100-ISO's out-of-vocabulary categories: other_iso (ISO templates outside the vocabulary)
# is the "Other" positive; other_isa (ISA bubbles, "seen Other" for R1b) and other_generic are
# reported apart.
OTHER_NAMES = ("other_iso", "other_isa", "other_generic")


def assign_iso_groups(
    class_groups: Mapping[str, ClassInfo], templates: Iterable[str]
) -> dict[str, str]:
    """Group of every ISO template for one encoder, from the DigitizePID classes mapped to it
    in `class_groups` (`assign_groups`' result: one computation for both sides).

        S           an ISO-status class with this template is among the training classes
        V           not S; a VARIANT class with this template is
        H_chon      not S / V; its ISO-status class is held out of the encoder (R1a h_in;
                    every one for the ImageNet-only encoder)
        V_held      not S / V and no ISO-status class: only held-out VARIANT classes map to
                    it (the ImageNet-only encoder's V)
        H_tu_nhien  no DigitizePID class maps to it
    """
    mapped: dict[str, list[ClassInfo]] = {}
    for info in class_groups.values():
        if info.template is not None and info.status in ("ISO", "VARIANT"):
            mapped.setdefault(info.template, []).append(info)
    out: dict[str, str] = {}
    for name in templates:
        infos = mapped.get(name, [])
        if any(i.status == "ISO" and i.group == "S" for i in infos):
            out[name] = "S"
        elif any(i.status == "VARIANT" and i.group in ("V", "V_of_S") for i in infos):
            out[name] = "V"
        elif any(i.status == "ISO" for i in infos):
            out[name] = "H_chon"
        else:
            out[name] = "V_held" if infos else "H_tu_nhien"
    return out


# ------------------------------------------------------------------ #
# Per-class report
# ------------------------------------------------------------------ #


def class_stats(
    top_names: Sequence[Sequence[str]], expected: str | None, best_cos: Sequence[float]
) -> dict:
    """One class's row: crops, top-1 / top-3 against `expected` (None for an unknown class:
    both None), the most frequent top-1 name and the mean best cosine. `top_names` holds one
    ranked name list (best first) per crop."""
    n = len(top_names)
    top1 = [t[0] for t in top_names]
    most = max(set(top1), key=top1.count) if top1 else None
    row: dict = {
        "n": n,
        "top1": None,
        "top3": None,
        "most_predicted": most,
        "most_predicted_frac": (top1.count(most) / n) if n else None,
        "mean_best_cos": float(np.mean(best_cos)) if n else None,
    }
    if expected is not None and n:
        row["top1"] = float(np.mean([t[0] == expected for t in top_names]))
        row["top3"] = float(np.mean([expected in t[:3] for t in top_names]))
    return row


def macro(values: Iterable[float | None]) -> float | None:
    """Mean over the classes that have a value (None: no class does)."""
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def mean_std(values: Iterable[float | None]) -> dict:
    """Mean and sample standard deviation (ddof=1) over the units that have a value, e.g. one
    number per OPEN100-ISO source sheet; std None with fewer than two."""
    vals = [v for v in values if v is not None]
    return {
        "mean": float(np.mean(vals)) if vals else None,
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
        "n": len(vals),
    }
