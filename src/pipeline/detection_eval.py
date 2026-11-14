"""Detection-level reports beyond mAP (PLAN B6b-2, B6d.2): ignore regions, class-agnostic Stage 1
with recall per group, the one-shot end-to-end score on whole sheets and the detector's wrong-box
rate. Driven by pipeline.evaluation; pure numpy apart from reading a checkpoint's class lists.

Ignore regions (B6d.2). OPEN100-ISO exports carry leader arrows and off-page flags as COCO
`iscrowd: 1` annotations of category utils.classmap.IGNORE_CATEGORY (`ignore_kind` = leader_arrow
| flag). They never enter a recall denominator. A prediction that matches no GT box but falls in a
region (IoU >= 0.5, or >= 50% of the PREDICTION's area inside it: utils.detection_metrics
IGNORE_IOU / IGNORE_INSIDE) is neither TP nor FP; how many predictions did, per kind, is reported.

Class-agnostic (eval.class_agnostic=true): every GT and predicted box is one class "symbol"
(predictions without a vocabulary label included: a box is a box); P, R at iou_thrs[0] and
eval.score_thr, mAP@50 and mAP@50:95 as usual. Every GT box carries its group, computed by the
one grouping function of P4 (utils.oneshot_metrics.assign_groups / assign_iso_groups) from the
encoder checkpoint eval.oneshot.groups_ckpt (default oneshot.encoder_ckpt): S / V / H_chon /
V_held / H_tu_nhien for an ISO template, other_iso / other_isa / other_generic, or the
DigitizePID group of a `Symbol_N`. Without a checkpoint there is no group (S needs the encoder's
training classes): recall is reported per fine class only, and the report says so.

Reporting rule (2026-11-11): when every evaluated COCO image carries `degraded` and `sheet`
(OPEN100-ISO), nothing is pooled over clean and degraded images: one block per view (degraded =
primary, clean = reference), each with one block per source sheet and mean +- std over the sheets
(utils.oneshot_metrics.mean_std), with instance counts. DigitizePID images carry neither: one block.

End-to-end one-shot (eval.mode=det_plus_fgc ++fgc.oneshot=true, on derived/detections_oneshot):
the gate is applied HERE, from the un-gated best_cos / second_cos B6a saves, so tau can change
without re-running inference: a prediction is "Other" when best_cos < tau (oneshot.other_min_cosine)
or, with ++eval.oneshot.margin_delta given, when best_cos - second_cos < delta. A prediction is a
TP when IoU >= 0.5 with a GT box whose expected answer is its label: the GT's own name for an ISO
template, "Other" for other_iso / other_isa / other_generic (no vocabulary name is right for an
ISA bubble or a generic shape either; the three are reported in separate rows, never pooled with
other_iso), the mapped template / "Other" for a DigitizePID class (symbol_map.csv; an UNSURE
class is no test and becomes an ignore region). Matching is per label, like mAP; the class-agnostic
box match is reported next to it (box_recall), with what the label of the matched box was.
Rows labelled `__unrefined__` (never classified) are dropped and counted, as B6a prescribes.

Wrong-box rate (PLAN B6b "Box sai của detector"): predictions with IoU < 0.3 to every GT box;
the fraction the gate rejects at tau, and at tau OR margin < delta when delta is given; the same
rates on the predictions that sit on a vocabulary GT box (the false rejects) for reference. On
DigitizePID it is read on val_sel (++eval.image_subset=val_sel, allowed in this mode only).
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from utils import detection_metrics
from utils.classmap import IGNORE_CATEGORY, OTHER_CLASS_NAME
from utils.oneshot_metrics import (
    GROUPS,
    ISO_GROUPS,
    OTHER_NAMES,
    assign_groups,
    assign_iso_groups,
    margins,
    mean_std,
    read_symbol_map,
    read_vocab,
    symbol_map_path,
)

logger = logging.getLogger(__name__)

Box = tuple[float, float, float, float]
VIEWS = ("degraded", "clean")  # degraded = the primary number, clean = the reference
PRIMARY_VIEW = "degraded"
E2E_IOU = 0.5  # end-to-end TP: IoU >= 0.5 AND the right label (PLAN B6b)
WRONG_BOX_IOU = 0.3  # a "wrong box": IoU < 0.3 with every GT box (PLAN B6b)
OTHER_GROUP = "Other"  # the group of a prediction labelled Other
_GROUP_ORDER = (*ISO_GROUPS, *OTHER_NAMES, *GROUPS)


def _stem(file_name: str) -> str:
    return file_name.split("/")[-1].rsplit(".", 1)[0]


@dataclass(frozen=True)
class GtBox:
    image_id: str
    name: str  # COCO category name
    supercategory: str
    xyxy: Box


@dataclass(frozen=True)
class IgnoreRegion:
    image_id: str
    kind: str  # `ignore_kind` (leader_arrow / flag), else the category name
    xyxy: Box


def load_boxes(
    coco: Mapping, eval_ids: set[str], coco_id_to_image_id: Mapping[int, str]
) -> tuple[list[GtBox], list[IgnoreRegion]]:
    """GT boxes and ignore regions (`iscrowd: 1`) of the evaluated images, COCO order."""
    cats = {c["id"]: c for c in coco.get("categories", [])}
    gts: list[GtBox] = []
    ignores: list[IgnoreRegion] = []
    for ann in coco.get("annotations", []):
        image_id = coco_id_to_image_id.get(ann["image_id"])
        if image_id is None or image_id not in eval_ids:
            continue
        cat = cats[ann["category_id"]]
        x, y, w, h = ann["bbox"]
        xyxy = (float(x), float(y), float(x + w), float(y + h))
        if ann.get("iscrowd"):
            kind = str(ann.get("ignore_kind") or cat["name"])
            ignores.append(IgnoreRegion(image_id, kind, xyxy))
        elif cat["name"] == IGNORE_CATEGORY:
            raise ValueError(f"annotation {ann.get('id')}: {IGNORE_CATEGORY!r} without iscrowd")
        else:
            gts.append(GtBox(image_id, cat["name"], str(cat.get("supercategory", "")), xyxy))
    return gts, ignores


def image_views(coco: Mapping, eval_ids: set[str]) -> dict[str, dict[int, set[str]]] | None:
    """{view: {sheet: image ids}} when every evaluated image carries `degraded` and `sheet`
    (OPEN100-ISO); None when none does (DigitizePID). A mix is an error: it would pool."""
    imgs = [im for im in coco.get("images", []) if _stem(im["file_name"]) in eval_ids]
    tagged = [("degraded" in im and "sheet" in im) for im in imgs]
    if not imgs or not any(tagged):
        return None
    if not all(tagged):
        bare = [im["file_name"] for im, t in zip(imgs, tagged, strict=True) if not t][:3]
        raise ValueError(f"some COCO images carry `degraded` / `sheet` and some do not: {bare}")
    out: dict[str, dict[int, set[str]]] = {v: {} for v in VIEWS}
    for im in imgs:
        view = "degraded" if im["degraded"] else "clean"
        out[view].setdefault(int(im["sheet"]), set()).add(_stem(im["file_name"]))
    return out


# ------------------------------------------------------------------ #
# Groups of the GT names (one function for P4, class-agnostic and end-to-end)
# ------------------------------------------------------------------ #


@dataclass
class GtGroups:
    """What a GT category name is, for one encoder.

    group           name -> its group ({} without a checkpoint: groups need its classes)
    expected        name -> the right one-shot answer: an ISO template, OTHER_CLASS_NAME, or
                    None (no test: an UNSURE DigitizePID class, or a name nothing maps)
    template_group  ISO template -> its group, to group the predicted labels
    """

    group: dict[str, str] = field(default_factory=dict)
    expected: dict[str, str | None] = field(default_factory=dict)
    template_group: dict[str, str] = field(default_factory=dict)
    checkpoint: str | None = None
    note: str | None = None

    def answer_group(self, answer: str) -> str:
        if answer == OTHER_CLASS_NAME:
            return OTHER_GROUP
        return self.template_group.get(answer, "named")

    def summary(self) -> dict:
        return {"checkpoint": self.checkpoint, "note": self.note, "of_name": dict(self.group)}


def checkpoint_classes(path: str | Path) -> tuple[list[str], list[str]]:
    """(fine_classes, excluded_classes) of a pid-fgc-train / make_imagenet_encoder checkpoint."""
    import torch

    state = torch.load(Path(str(path)), map_location="cpu", weights_only=False)
    return list(state["fine_classes"]), list(state.get("excluded_classes", []))


def gt_groups(cfg, names: Iterable[str], require_vocab: bool = False) -> GtGroups:
    """GtGroups of `names` from oneshot.vocab_csv, oneshot.symbol_map and the checkpoint
    eval.oneshot.groups_ckpt (default oneshot.encoder_ckpt). Without vocab.csv (and not
    `require_vocab`) nothing is grouped and only per-class numbers can be given."""
    names = sorted(set(names))
    osh = cfg.get("oneshot", None) or {}
    node = (cfg.get("eval", None) or {}).get("oneshot", None) or {}
    ckpt = node.get("groups_ckpt", None) or osh.get("encoder_ckpt", None)
    vocab_path = osh.get("vocab_csv", None)
    if vocab_path is None or not Path(str(vocab_path)).exists():
        if require_vocab:
            raise FileNotFoundError(
                f"oneshot.vocab_csv={vocab_path!r} does not exist (on Colab: "
                "oneshot.vocab_csv=$REPO_DATA/iso_symbols/vocab.csv)"
            )
        return GtGroups(note=f"no oneshot.vocab_csv ({vocab_path}): per fine class only")
    vocab = read_vocab(Path(str(vocab_path)))
    templates = [r["fine_name"].strip() for r in vocab.values()]
    sm_path = symbol_map_path(osh.get("symbol_map", None))
    symbol_map = read_symbol_map(sm_path) if sm_path.exists() else {}
    note = None
    if ckpt is not None:
        fine, excluded = checkpoint_classes(ckpt)
        classes = assign_groups(symbol_map, vocab, fine, excluded)
        iso = assign_iso_groups(classes, templates)
    else:
        # templates only (a class's template does not depend on the encoder); every class
        # counted as seen so that nothing warns
        classes = assign_groups(symbol_map, vocab, list(symbol_map))
        iso = {}
        note = (
            "no oneshot.encoder_ckpt / eval.oneshot.groups_ckpt: a group (S / V / H ...) needs "
            "the encoder's training classes, so recall is reported per fine class only"
        )
    out = GtGroups(template_group=iso, checkpoint=None if ckpt is None else str(ckpt), note=note)
    unmapped = []
    for name in names:
        if ckpt is not None:
            if name in iso:
                out.group[name] = iso[name]
            elif name in OTHER_NAMES:
                out.group[name] = name
            elif name in classes:
                out.group[name] = classes[name].group
        if name in templates:
            out.expected[name] = name
        elif name in OTHER_NAMES:
            out.expected[name] = OTHER_CLASS_NAME
        elif name in classes:
            info = classes[name]
            if info.status in ("ISO", "VARIANT") and info.template:
                out.expected[name] = info.template
            else:
                out.expected[name] = OTHER_CLASS_NAME if info.status == "OOV" else None
        else:
            out.expected[name] = None
            unmapped.append(name)
    if unmapped:
        logger.warning(
            "GT categories %s are neither vocab templates, %s nor symbol_map classes: no "
            "expected one-shot answer",
            unmapped,
            OTHER_NAMES,
        )
    return out


# ------------------------------------------------------------------ #
# Shared pieces
# ------------------------------------------------------------------ #


def _ordered(keys: Iterable[str]) -> list[str]:
    keys = set(keys)
    return [k for k in _GROUP_ORDER if k in keys] + sorted(keys - set(_GROUP_ORDER))


def _rate(num, den):
    return None if not den else float(num / den)


def recall_rows(keys: list[str | None], hit: np.ndarray, extra: Mapping | None = None) -> dict:
    """{key: {n, tp, recall}} over the GT boxes (keys parallel to `hit`; None: left out)."""
    hit = np.asarray(hit, dtype=bool)
    arr = np.array([k if k is not None else "" for k in keys], dtype=object)
    out = {}
    for k in _ordered(k for k in keys if k is not None):
        mask = arr == k
        n, tp = int(mask.sum()), int(hit[mask].sum())
        out[k] = {"n": n, "tp": tp, "recall": _rate(tp, n), **((extra or {}).get(k, {}))}
    return out


def _dig(block: Mapping, path: tuple[str, ...]):
    for key in path:
        if not isinstance(block, Mapping) or key not in block:
            return None
        block = block[key]
    return block


def by_view(
    views: Mapping[str, Mapping[int, set[str]]],
    block_for: Callable[[set[str], bool], dict],
    summary_paths: Callable[[dict], list[tuple[str, ...]]],
) -> dict:
    """One block per view (never pooled), each with one block per source sheet and the
    mean +- std over the sheets of every `summary_paths(view_block)` number."""
    out: dict = {"primary": PRIMARY_VIEW}
    for view in VIEWS:
        sheets = views.get(view, {})
        ids = set().union(*sheets.values()) if sheets else set()
        block = block_for(ids, True)
        per_sheet = {
            f"iso{s:02d}": block_for(sheet_ids, False) for s, sheet_ids in sorted(sheets.items())
        }
        stats: dict = {}
        for path in summary_paths(block):
            node = stats
            for key in path[:-1]:
                node = node.setdefault(key, {})
            node[path[-1]] = mean_std(_dig(b, path) for b in per_sheet.values())
        block["per_sheet"] = per_sheet
        block["sheet_mean_std"] = stats
        out[view] = block
    return out


def _max_iou(preds_xyxy: list[Box], pred_images: list[str], gts: list[GtBox]) -> np.ndarray:
    """Per prediction, its highest IoU with any GT box of its image (0 without one)."""
    out = np.zeros(len(preds_xyxy))
    by_image: dict[str, list[int]] = {}
    for j, g in enumerate(gts):
        by_image.setdefault(g.image_id, []).append(j)
    pred_by_image: dict[str, list[int]] = {}
    for i, image_id in enumerate(pred_images):
        if image_id in by_image:
            pred_by_image.setdefault(image_id, []).append(i)
    for image_id, pis in pred_by_image.items():
        ious = detection_metrics.iou_matrix(
            [preds_xyxy[i] for i in pis], [gts[j].xyxy for j in by_image[image_id]]
        )
        out[pis] = ious.max(axis=1)
    return out


# ------------------------------------------------------------------ #
# Stage 1: detection block (ignore regions, class-agnostic, recall by group)
# ------------------------------------------------------------------ #


def detection_block(
    ids: set[str],
    preds: list[tuple[str, int, float, Box]],
    gts: list[tuple[str, int, Box]],
    boxes: list[GtBox],
    ignores: list[IgnoreRegion],
    iou_thrs: list[float],
    groups: GtGroups | None,
    per_class_ap: bool,
    detail: bool,
) -> dict:
    """Detection numbers of the images `ids`: mAP@50, mAP@50:95, P / R / TP / FP at
    iou_thrs[0], predictions in ignore regions (by kind), recall per group (with a
    checkpoint) and, with `detail`, per fine class. `gts` and `boxes` are parallel."""
    p = [x for x in preds if x[0] in ids]
    keep = [j for j, g in enumerate(gts) if g[0] in ids]
    g = [gts[j] for j in keep]
    b = [boxes[j] for j in keep]
    regions = [r for r in ignores if r.image_id in ids]
    metrics, first, hits = detection_metrics.match_report(
        p, g, iou_thrs, [(r.image_id, r.xyxy) for r in regions]
    )
    if not (per_class_ap and detail):
        metrics.pop("per_class_ap_50")
    block = {"n_images": len(ids), "n_predictions": len(p), **metrics, "iou_pr": iou_thrs[0]}
    if ignores:  # the dataset has ignore regions: report them (also 0) for every block
        kinds = [regions[h].kind for h in hits[first.pred_ignored]]
        block["ignore"] = {
            "n_regions": len(regions),
            "regions_by_kind": dict(Counter(r.kind for r in regions)),
            "n_predictions": int(first.pred_ignored.sum()),
            "predictions_by_kind": dict(Counter(kinds)),
        }
    if groups is not None and groups.group:
        block["recall_by_group"] = recall_rows([groups.group.get(x.name) for x in b], first.gt_hit)
    if detail:
        extra = {n: {"group": groups.group.get(n)} for n in {x.name for x in b}} if groups else None
        block["recall_by_class"] = recall_rows([x.name for x in b], first.gt_hit, extra)
    return block


def detection_summary_paths(block: dict) -> list[tuple[str, ...]]:
    paths = [("precision",), ("recall",), ("map_50",), ("map_50_95",)]
    paths += [("recall_by_group", k, "recall") for k in block.get("recall_by_group", {})]
    if "ignore" in block:
        paths.append(("ignore", "n_predictions"))
    return paths


def detection_report(
    eval_ids: set[str],
    views,
    preds: list[tuple[str, int, float, Box]],
    gts: list[tuple[str, int, Box]],
    boxes: list[GtBox],
    ignores: list[IgnoreRegion],
    iou_thrs: list[float],
    groups: GtGroups | None,
    per_class_ap: bool,
) -> dict:
    def block_for(ids: set[str], detail: bool) -> dict:
        return detection_block(
            ids, preds, gts, boxes, ignores, iou_thrs, groups, per_class_ap, detail
        )

    if views is None:
        return block_for(eval_ids, True)
    return by_view(views, block_for, detection_summary_paths)


# ------------------------------------------------------------------ #
# End-to-end one-shot + wrong-box rate
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class E2EPred:
    """One detections_oneshot row (score >= eval.score_thr, classified), box in pixels."""

    image_id: str
    best_name: str
    score: float
    best_cos: float
    second_cos: float
    xyxy: Box


def reject(best_cos, second_cos, tau: float, delta: float | None = None) -> np.ndarray:
    """The Other gate: best_cos < tau, OR (delta given) top-1 minus top-2 margin < delta."""
    best = np.asarray(best_cos, dtype=np.float64)
    out = best < tau
    if delta is not None:
        out |= margins(best, second_cos) < delta
    return out


def _reject_rates(preds: list[E2EPred], mask: np.ndarray, tau: float, delta) -> dict:
    cos = np.array([p.best_cos for p in preds])[mask] if preds else np.zeros(0)
    second = np.array([p.second_cos for p in preds])[mask] if preds else np.zeros(0)
    n = int(mask.sum()) if len(mask) else 0
    out = {"n": n, "rejected_tau": _rate(int(reject(cos, second, tau).sum()), n)}
    if delta is not None:
        out["rejected_tau_or_margin"] = _rate(int(reject(cos, second, tau, delta).sum()), n)
    return out


def e2e_block(
    ids: set[str],
    preds: list[E2EPred],
    gts: list[GtBox],
    ignores: list[IgnoreRegion],
    groups: GtGroups,
    tau: float,
    delta: float | None,
    detail: bool,
) -> dict:
    """End-to-end numbers of the images `ids` (see module docstring)."""
    preds = [p for p in preds if p.image_id in ids]
    gts = [g for g in gts if g.image_id in ids]
    expected = [groups.expected.get(g.name) for g in gts]
    scored = [g for g, e in zip(gts, expected, strict=True) if e is not None]
    answers_exp = [e for e in expected if e is not None]
    # a GT box with no expected answer is no test: an ignore region, like a leader arrow
    regions = [r for r in ignores if r.image_id in ids] + [
        IgnoreRegion(g.image_id, "no_answer", g.xyxy)
        for g, e in zip(gts, expected, strict=True)
        if e is None
    ]
    gate = reject([p.best_cos for p in preds], [p.second_cos for p in preds], tau, delta)
    answers = [OTHER_CLASS_NAME if r else p.best_name for p, r in zip(preds, gate, strict=True)]
    label_id = {a: k for k, a in enumerate(sorted(set(answers) | set(answers_exp)))}

    lab_preds = [
        (p.image_id, label_id[a], p.score, p.xyxy) for p, a in zip(preds, answers, strict=True)
    ]
    lab_gts = [(g.image_id, label_id[e], g.xyxy) for g, e in zip(scored, answers_exp, strict=True)]
    hits = detection_metrics.ignore_hits(lab_preds, [(r.image_id, r.xyxy) for r in regions])
    match = detection_metrics.greedy_match(lab_preds, lab_gts, E2E_IOU, hits)
    box = detection_metrics.greedy_match(
        [(i, 0, s, b) for i, _c, s, b in lab_preds],
        [(i, 0, b) for i, _c, b in lab_gts],
        E2E_IOU,
        hits,
    )

    # what label the box-matched prediction of each GT carried
    taken_by = np.full(len(scored), -1, dtype=np.int64)
    for i, j in enumerate(box.pred_gt.tolist()):
        if j >= 0:
            taken_by[j] = i
    outcome: list[str | None] = []
    for j, exp in enumerate(answers_exp):
        i = int(taken_by[j])
        if i < 0:
            outcome.append(None)
        elif answers[i] == exp:
            outcome.append("right")
        elif answers[i] == OTHER_CLASS_NAME:
            outcome.append("rejected")
        else:
            outcome.append("wrong_name")

    def gt_rows(keys: list[str | None]) -> dict:
        rows = recall_rows(keys, match.gt_hit)
        box_rows = recall_rows(keys, box.gt_hit)
        arr = np.array([k if k is not None else "" for k in keys], dtype=object)
        for k, row in rows.items():
            mine = [o for o, kk in zip(outcome, arr, strict=True) if kk == k and o is not None]
            row["box_recall"] = box_rows[k]["recall"]
            row["box_matched"] = {o: mine.count(o) for o in ("right", "rejected", "wrong_name")}
        return rows

    tp, fp = int(match.tp.sum()), int(match.fp.sum())
    scored_in = ~match.pred_ignored
    pred_groups = [groups.answer_group(a) for a in answers]
    by_pred_group = {}
    for k in _ordered(set(pred_groups)):
        mask = np.array([pg == k for pg in pred_groups]) & scored_in
        n, t = int(mask.sum()), int((match.tp & mask).sum())
        by_pred_group[k] = {"n": n, "tp": t, "precision": _rate(t, n)}

    max_iou = _max_iou([p.xyxy for p in preds], [p.image_id for p in preds], gts)
    wrong = max_iou < WRONG_BOX_IOU
    true_vocab = np.zeros(len(preds), dtype=bool)
    for i, j in enumerate(box.pred_gt.tolist()):
        if j >= 0 and answers_exp[j] != OTHER_CLASS_NAME:
            true_vocab[i] = True

    block: dict = {
        "n_images": len(ids),
        "n_predictions": len(preds),
        "n_gt": len(scored),
        "n_gt_no_answer": len(gts) - len(scored),
        "tp": tp,
        "fp": fp,
        "precision": _rate(tp, tp + fp),
        "recall": _rate(tp, len(scored)),
        "box_recall": _rate(int(box.gt_hit.sum()), len(scored)),
        "fp_box": int((match.fp & (max_iou < E2E_IOU)).sum()),
        "fp_label": int((match.fp & (max_iou >= E2E_IOU)).sum()),
        "n_ignored_predictions": int(match.pred_ignored.sum()),
        "n_rejected_other": int(gate.sum()),
        "precision_by_predicted_group": by_pred_group,
        "wrong_boxes": {
            "iou_below": WRONG_BOX_IOU,
            "frac_of_predictions": _rate(int(wrong.sum()), len(preds)),
            "in_ignore_region": int((wrong & (hits >= 0)).sum()),
            **_reject_rates(preds, wrong, tau, delta),
        },
        "true_vocab_boxes": _reject_rates(preds, true_vocab, tau, delta),
    }
    if groups.group:
        block["by_group"] = gt_rows([groups.group.get(g.name) for g in scored])
    if detail:
        block["by_class"] = gt_rows([g.name for g in scored])
        for name, row in block["by_class"].items():
            row["group"] = groups.group.get(name)
            row["expected"] = groups.expected.get(name)
    return block


def e2e_summary_paths(block: dict) -> list[tuple[str, ...]]:
    paths = [("precision",), ("recall",), ("box_recall",)]
    for k in block.get("by_group", {}):
        paths += [("by_group", k, "recall"), ("by_group", k, "box_recall")]
    paths += [("wrong_boxes", "rejected_tau"), ("wrong_boxes", "frac_of_predictions")]
    if "rejected_tau_or_margin" in block["wrong_boxes"]:
        paths.append(("wrong_boxes", "rejected_tau_or_margin"))
    return paths


def e2e_report(
    eval_ids: set[str],
    views,
    preds: list[E2EPred],
    gts: list[GtBox],
    ignores: list[IgnoreRegion],
    groups: GtGroups,
    tau: float,
    delta: float | None,
) -> dict:
    def block_for(ids: set[str], detail: bool) -> dict:
        return e2e_block(ids, preds, gts, ignores, groups, tau, delta, detail)

    if views is None:
        return block_for(eval_ids, True)
    return by_view(views, block_for, e2e_summary_paths)
