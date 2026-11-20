"""Crop-level one-shot evaluation (PLAN B6b-1): eval.mode=oneshot, on DigitizePID (p1 / p2 /
tau) or OPEN100-ISO (p4).

Every ground-truth crop is embedded through its inference-time view (the stored context crop
cut to box + 10%, utils.crop_box.view), expanded into oneshot.query_views, aggregated by
oneshot.aggregate and matched against ALL prototypes (no coarse routing: this isolates the
Stage-2 error from detection and from Stage-1 mistakes). Nothing is gated: the per-crop result
keeps the top-3 names with cosines, best_cos and second_cos, so every threshold-dependent
number is recomputed from the CSV written next to metrics.json (utils.oneshot_metrics).

eval.oneshot.protocol:
    p1   leave-out encoder (R1a): h_in (ISO, expected template = its fine_name) vs h_out
         (unknown). Crops of train AND val: the encoder never trained on them. Raises if a
         tested class is among the checkpoint's training classes.
    p2   seen-class encoder (R1b): S / V classes vs OOV_seen ("seen Other"), crops of val
         filtered by eval.image_subset (use `test`).
    tau  tau selection on val_sel (PLAN B8 point 2): rule A (tau@TPR95) and rule B (leave the
         crop's own prototype out; once more also leaving out its near-twin prototypes), only
         on group-S classes the encoder trained on; plus delta@TPR95 on the top-1 minus top-2
         margin (score form 2, `margin_rule_a`): a crop is Other when cos < tau OR margin <
         delta. Prints tau and delta; writes nothing to config.
    p4   OPEN100-ISO (data_root=data/iso_open100, PLAN B6b "P4"): every GT box of the COCO
         export cut from its sheet with the inference-time margin (utils.crop_box.cut), each
         sheet read once. Templates grouped S / V / H_chon / V_held / H_tu_nhien
         (utils.oneshot_metrics.assign_iso_groups), one row per group, never pooled; "Other"
         with other_iso as the positive and every vocab instance as the negative, other_isa /
         other_generic apart. Every number separately for the degraded sheets (primary) and
         the clean ones (reference), and per source sheet (iso00..iso11) with mean +- std over
         the sheets. Selects nothing: tau is oneshot.other_min_cosine, recorded.
         eval.oneshot.max_per_class (null = all) samples that many boxes per (category,
         degraded) with eval.oneshot.seed, for cheap CPU runs.
    center  the CL2N center (SimpleShot; utils.oneshot "CL2N centering") of this encoder: the
         mean unit embedding of every query view of crops_fgc/train crops of the classes the
         groups checkpoint trained on, minus every oneshot.holdout class (so no class P1
         tests, and the same class set for each encoder pointed at the same groups_ckpt).
         eval.oneshot.max_per_class / seed sample crops per class. Writes
         oneshot_center.pt next to metrics.json; pass it as ++oneshot.center=<file> to
         every later run of this encoder (tau first: the cosines change). Refuses an
         already-centered classifier.

p1 / p2 also report `margin_auroc` (threshold-free, like `auroc`); the tau + delta gate
itself is recomputed from the CSV (best_cos, second_cos), as every other threshold.

eval.oneshot.groups_ckpt (default: oneshot.encoder_ckpt) is the checkpoint whose training
classes define the groups (S / H_chon / OOV_heldout ...). For the ImageNet-only baseline (B4a,
scripts/make_imagenet_encoder.py), which trained on no class and so has no group S: pointed at
the leave-out encoder, tau reads the same classes as R1a's own tau run. The P1 leak guard keeps
checking the encoder actually loaded, and tau with groups_ckpt also refuses every
oneshot.holdout class.

With oneshot.aggregate=vote best_cos is the mean cosine of the winning views, so a sweep of
tau over best_cos only approximates the gated vote (exact for `mean`).
"""

from __future__ import annotations

import csv
import json
import logging
import random
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image

from pipeline.oneshot_infer import OneShotClassifier, load_oneshot_classifier
from utils.classmap import IGNORE_CATEGORY
from utils.crop_box import cut, view
from utils.dataset import (
    build_eval_transform,
    crop_image_id,
    image_subset_ids,
    scan_crops,
)
from utils.model import checkpoint_excluded_classes, checkpoint_sha256
from utils.oneshot import (
    AGGREGATES,
    PrototypeDirectory,
    aggregate_views,
    holdout_classes,
    save_center,
)
from utils.oneshot_metrics import (
    DEFAULT_SYMBOL_MAP,  # noqa: F401  (re-exported: oneshot.symbol_map's default)
    ISO_GROUPS,
    OTHER_NAMES,
    ClassInfo,
    assign_groups,
    assign_iso_groups,
    class_stats,
    macro,
    margin_auroc,
    margin_rule_a,
    margins,
    mean_std,
    open_set_summary,
    other_auroc,
    read_symbol_map,
    read_vocab,
    symbol_map_path,
    tau_rule_a,
    tau_rule_b,
)
from utils.paths import DataPaths

logger = logging.getLogger(__name__)

PROTOCOLS = ("p1", "p2", "tau", "p4", "center")
_SWEEP_TAUS = np.round(np.linspace(0.0, 1.0, 101), 2)
_GROUP_SWEEP_TAUS = np.round(np.linspace(0.0, 1.0, 21), 2)
_CSV_FIELDS = (
    "split",
    "path",
    "image_id",
    "class",
    "group",
    "expected",
    "top1",
    "cos1",
    "top2",
    "cos2",
    "top3",
    "cos3",
    "best_cos",
    "second_cos",
)
_P4_FIELDS = ("sheet", "variant", "degraded", "ann_id")  # ann_id: the COCO annotation id
_P4_VIEWS = ("degraded", "clean")  # degraded = the primary number, clean = the reference


def rank_names(
    directory: PrototypeDirectory, emb: torch.Tensor, aggregate: str, k: int = 3
) -> list[tuple[str, float]]:
    """The k best (name, cosine) for one crop's (V, D) view embeddings, best first, un-gated.

    mean  the aggregated embedding's k nearest classes.
    max   classes ranked by their best cosine over the views.
    vote  names ranked by votes (ties: summed cosine), the cosine of a name being the mean
          over the views that voted for it (as OneShotClassifier's best_cos); names no view
          voted for fill the remaining places by their mean cosine over all views.
    Names are CLASSES: a class drawn in several forms (`name@form`) is one candidate scoring
    the best of its forms (PrototypeDirectory.class_cosines).
    """
    if aggregate == "mean":
        return directory.topk(aggregate_views(emb), k)
    if aggregate not in AGGREGATES:
        raise ValueError(f"unknown aggregate={aggregate!r}; expected one of {AGGREGATES}")
    classes = directory.classes
    cos = directory.class_cosines(emb)  # (V, C)
    if aggregate == "max":
        per_class = cos.max(dim=0).values
        values, idx = torch.topk(per_class, min(k, len(classes)))
        return [(classes[int(i)], float(v)) for v, i in zip(values, idx, strict=True)]
    best_cos, best_idx = cos.max(dim=1)
    voted: dict[int, list[float]] = defaultdict(list)
    for v in range(len(best_idx)):
        voted[int(best_idx[v])].append(float(best_cos[v]))
    order = sorted(voted.items(), key=lambda kv: (len(kv[1]), sum(kv[1])), reverse=True)
    ranked = [(classes[i], sum(c) / len(c)) for i, c in order]
    if len(ranked) < k:
        mean_cos = cos.mean(dim=0)
        rest = [i for i in torch.argsort(mean_cos, descending=True).tolist() if i not in voted]
        ranked += [(classes[i], float(mean_cos[i])) for i in rest]
    return ranked[:k]


def check_no_leak(tested: list[str], encoder_classes: list[str]) -> None:
    """P1 guard: a tested class the encoder trained on makes the number measure a seen class."""
    leaked = sorted(set(tested) & set(encoder_classes))
    if leaked:
        raise ValueError(
            f"eval.oneshot.protocol=p1 tests {leaked}, but oneshot.encoder_ckpt trained on "
            "them: P1 needs the leave-out encoder (__global_lo__.pt, built with "
            "fgc.exclude_classes = oneshot.holdout h_in + h_out + h_extra)"
        )


def validate_tau_classes(classes: list[str], encoder_classes: list[str], holdout) -> None:
    """tau guard: only classes the encoder trained on. For the leave-out encoder that rules
    out h_in / h_out / h_extra (its training classes exclude them); for the full encoder they
    are seen classes like any other. Raises naming the held-out ones when it trips."""
    unseen = sorted(set(classes) - set(encoder_classes))
    if unseen:
        held = sorted(set(unseen) & set(holdout.excluded))
        raise ValueError(
            f"tau selection may only read classes the encoder trained on; {unseen} are not"
            + (f" (held out: {held})" if held else "")
        )


def validate_p4_coco(coco: Mapping, templates: list[str]) -> None:
    """p4 guard: the COCO is an OPEN100-ISO export. Every category is a vocab template, an
    OPEN100-ISO Other name or the ignore-region category (a DigitizePID `Symbol_N`, or a
    category the groups cannot place, raises), and every image says its source sheet and
    whether it is degraded."""
    names = [c["name"] for c in coco.get("categories", [])]
    unknown = sorted(set(names) - set(templates) - set(OTHER_NAMES) - {IGNORE_CATEGORY})
    if unknown:
        raise ValueError(
            f"eval.oneshot.protocol=p4 reads OPEN100-ISO (data_root=data/iso_open100): COCO "
            f"categories {unknown} are neither oneshot.vocab_csv fine_names nor {OTHER_NAMES}"
        )
    need = {"sheet", "degraded"}
    bare = [im.get("file_name") for im in coco.get("images", []) if not need <= set(im)]
    if bare or not coco.get("images"):
        raise ValueError(
            "eval.oneshot.protocol=p4 needs every COCO image to carry `sheet` and `degraded` "
            f"(an OPEN100-ISO export); missing on {bare[:3]}"
        )


def sample_per_class(
    ids: Mapping[tuple[str, bool], list[int]], max_per_class: int | None, seed: int
) -> dict[tuple[str, bool], list[int]]:
    """At most `max_per_class` annotation ids per (category, degraded) key, sorted. The draw of
    a key depends only on (seed, key, its ids), not on the other keys or their order."""
    out = {}
    for key, values in ids.items():
        values = sorted(values)
        if max_per_class is not None and len(values) > max_per_class:
            rng = random.Random(f"{seed}/{key[0]}/{int(key[1])}")
            values = sorted(rng.sample(values, max_per_class))
        out[key] = values
    return out


class OneShotEvaluation:
    """eval.mode=oneshot for one protocol; `run()` returns the `oneshot` block of metrics.json."""

    def __init__(
        self,
        cfg: DictConfig,
        output_dir: Path,
        classifier: OneShotClassifier | None = None,
    ) -> None:
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.paths = DataPaths(root=Path(str(cfg.data_root)))
        node = cfg.eval.get("oneshot", None) or {}
        self.protocol = str(node.get("protocol", "") or "")
        if self.protocol not in PROTOCOLS:
            raise ValueError(
                f"eval.oneshot.protocol must be one of {PROTOCOLS}, got {self.protocol!r}"
            )
        self.tau_tpr = float(node.get("tau_tpr", 0.95))
        self.twin_cos = float(node.get("twin_cos", 0.95))
        self.groups_ckpt = node.get("groups_ckpt", None)
        self.max_per_class = node.get("max_per_class", None)
        if self.max_per_class is not None:
            self.max_per_class = int(self.max_per_class)
            if self.max_per_class < 1:
                raise ValueError(
                    f"eval.oneshot.max_per_class must be >= 1 or null, got {self.max_per_class}"
                )
        self.seed = int(node.get("seed", 0))
        self.subset = str(cfg.eval.get("image_subset", "all"))
        self.other_tau = float(cfg.oneshot.other_min_cosine)
        self.aggregate = str(cfg.oneshot.get("aggregate", "vote"))
        self._classifier = classifier
        self._seen: tuple[list[str], list[str]] | None = None

    # ------------------------------------------------------------------ #

    def _load(self) -> OneShotClassifier:
        if self._classifier is None:
            device = torch.device(
                "cuda"
                if str(self.cfg.get("device", "cpu")) == "cuda" and torch.cuda.is_available()
                else "cpu"
            )
            self._classifier = load_oneshot_classifier(
                self.cfg, build_eval_transform(self.cfg), device
            )
        return self._classifier

    def _seen_classes(self, clf: OneShotClassifier) -> tuple[list[str], list[str]]:
        """(training, excluded) classes that define the groups: the loaded encoder's, or
        eval.oneshot.groups_ckpt's when set."""
        if self._seen is None:
            if self.groups_ckpt is None:
                ckpt = Path(str(self.cfg.oneshot.encoder_ckpt))
                self._seen = list(clf.encoder_classes), checkpoint_excluded_classes(ckpt)
            else:
                state = torch.load(
                    Path(str(self.groups_ckpt)), map_location="cpu", weights_only=False
                )
                self._seen = list(state["fine_classes"]), list(state.get("excluded_classes", []))
        return self._seen

    def _groups(self, clf: OneShotClassifier, holdout) -> dict[str, ClassInfo]:
        symbol_map = symbol_map_path(self.cfg.oneshot.get("symbol_map", None))
        seen, excluded = self._seen_classes(clf)
        return assign_groups(
            read_symbol_map(symbol_map),
            read_vocab(Path(str(self.cfg.oneshot.vocab_csv))),
            seen,
            excluded,
            holdout,
        )

    def _score(
        self,
        clf: OneShotClassifier,
        splits: list[str],
        classes: list[str],
        groups: dict[str, ClassInfo],
        image_ids: set[str] | None,
        keep_emb: bool = False,
    ) -> list[dict]:
        """One row per crop of `classes` in `splits` (see _CSV_FIELDS; `emb` kept on request)."""
        rows: list[dict] = []
        for split in splits:
            root = self.paths.crops_fgc / split
            samples, stored = scan_crops(root, classes, image_ids)
            logger.info(
                "[oneshot] %s: %d crops from %d images, %d classes",
                split,
                len(samples),
                len({crop_image_id(p.name) for p, _ in samples}),
                len({i for _, i in samples}),
            )
            for n, (path, idx) in enumerate(samples, 1):
                img = Image.open(path).convert("RGB")
                if path in stored:
                    img = view(img, stored[path])
                emb = clf.embed_crop(np.array(img))
                info = groups[classes[idx]]
                row = {
                    "split": split,
                    "path": str(path.relative_to(self.paths.crops_fgc)),
                    "image_id": crop_image_id(path.name),
                    "class": info.name,
                    "group": info.group,
                    "expected": info.template or "",
                    **self._ranked(clf, emb),
                }
                if keep_emb:
                    row["emb"] = emb
                rows.append(row)
                if n % 500 == 0:
                    logger.info("[oneshot] %s: %d / %d crops", split, n, len(samples))
        return rows

    def _ranked(self, clf: OneShotClassifier, emb: torch.Tensor) -> dict:
        """The un-gated part of a crop's row: top-3 names / cosines, best_cos, second_cos."""
        top = rank_names(clf.directory, emb, self.aggregate, 3)
        row = {
            "names": [n for n, _ in top],
            "best_cos": top[0][1],
            "second_cos": clf._second_cosine(clf.directory, emb, top[0][0]),
        }
        for j, (name, cos) in enumerate(top, 1):
            row[f"top{j}"], row[f"cos{j}"] = name, cos
        return row

    def _write_csv(self, rows: list[dict], extra: tuple[str, ...] = ()) -> str:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"oneshot_{self.protocol}_crops.csv"
        with path.open("w", newline="") as f:
            out = csv.DictWriter(f, fieldnames=[*_CSV_FIELDS, *extra], extrasaction="ignore")
            out.writeheader()
            for r in rows:
                out.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
        return str(path)

    def _plot_sweep(self, sweep: dict, tag: str = "") -> str:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        name = f"{self.protocol}_{tag}" if tag else self.protocol
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(sweep["tau"], sweep["known_acc"], label="known: accepted and correct")
        ax.plot(sweep["tau"], sweep["known_reject_rate"], label="known: rejected")
        if sweep["unknown_reject_rate"][0] is not None:
            ax.plot(sweep["tau"], sweep["unknown_reject_rate"], label="unknown: rejected")
        ax.axvline(self.other_tau, color="grey", linestyle=":", label="other_min_cosine")
        ax.set_xlabel("tau (best cosine)")
        ax.set_ylabel("fraction of crops")
        ax.set_title(f"One-shot {name}: tau sweep")
        ax.legend(fontsize=8)
        fig.tight_layout()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"oneshot_{name}_sweep.png"
        fig.savefig(path)
        plt.close(fig)
        return str(path)

    # ------------------------------------------------------------------ #

    def run(self) -> dict:
        clf = self._load()
        holdout = holdout_classes(self.cfg)
        groups = self._groups(clf, holdout)
        ckpt = Path(str(self.cfg.oneshot.encoder_ckpt))
        if self.aggregate == "vote":
            logger.warning(
                "[oneshot] aggregate=vote: best_cos is the mean cosine of the winning views, so "
                "the tau sweep over best_cos is approximate (aggregate=mean is exact)"
            )
        result: dict = {
            "protocol": self.protocol,
            "encoder_checkpoint": str(ckpt),
            "encoder_sha256": checkpoint_sha256(ckpt),
            "encoder_classes": len(clf.encoder_classes),
            "groups_checkpoint": None if self.groups_ckpt is None else str(self.groups_ckpt),
            "prototypes": str(self.cfg.oneshot.prototypes),
            "n_prototypes": len(clf.directory),
            "aggregate": self.aggregate,
            "query_views": clf.views.as_dict(),
            "image_subset": self.subset,
            "other_min_cosine": self.other_tau,
            "center": self.cfg.oneshot.get("center", None) or None,
            "input_norm": clf.input_norm,
            "template_align": clf.template_align,
        }
        run = {
            "p1": self._p1, "p2": self._p2, "tau": self._tau, "p4": self._p4,
            "center": self._center,
        }[self.protocol]  # fmt: skip
        result.update(run(clf, groups, holdout))
        return result

    def _unsure_counts(self, groups: dict[str, ClassInfo], split: str, ids) -> dict[str, int]:
        unsure = [n for n, g in groups.items() if g.group == "UNSURE"]
        samples, _ = scan_crops(self.paths.crops_fgc / split, unsure, ids)
        counts = {n: 0 for n in unsure}
        for _, i in samples:
            counts[unsure[i]] += 1
        return counts

    def _known_unknown(self, rows, known_groups, unknown_groups):
        known = [r for r in rows if r["group"] in known_groups]
        unknown = [r for r in rows if r["group"] in unknown_groups]
        if not known:
            raise ValueError(f"no crops of in-vocabulary classes (groups {sorted(known_groups)})")
        return known, unknown

    def _summary(self, known, unknown, taus=_GROUP_SWEEP_TAUS) -> dict:
        summary = open_set_summary(
            [r["top1"] == r["expected"] for r in known],
            [r["best_cos"] for r in known],
            [r["best_cos"] for r in unknown],
            self.other_tau,
            taus,
        )

        def margin(rs):
            return margins([r["best_cos"] for r in rs], [r["second_cos"] for r in rs])

        summary["margin_auroc"] = margin_auroc(margin(known), margin(unknown))
        return summary

    def _per_class(self, rows: list[dict], groups: dict[str, ClassInfo], names: list[str]) -> dict:
        out = {}
        for name in names:
            mine = [r for r in rows if r["class"] == name]
            info = groups[name]
            stats = class_stats(
                [r["names"] for r in mine], info.template, [r["best_cos"] for r in mine]
            )
            stats.update(group=info.group, status=info.status, expected=info.template)
            if info.template is None and mine:
                stats["reject_rate_at_tau"] = float(
                    np.mean([r["best_cos"] < self.other_tau for r in mine])
                )
            out[name] = stats
        return out

    def _group_block(self, per_class: dict, group: str) -> dict:
        rows = [v for v in per_class.values() if v["group"] == group]
        return {
            "n_classes": len(rows),
            "n_crops": sum(v["n"] for v in rows),
            "macro_top1": macro(v["top1"] for v in rows),
            "macro_top3": macro(v["top3"] for v in rows),
        }

    # -- p1 -------------------------------------------------------------- #

    def _p1(self, clf, groups, holdout) -> dict:
        if self.subset != "all":
            raise ValueError(
                "eval.oneshot.protocol=p1 reads crops_fgc/train and crops_fgc/val (never "
                f"trained on), so eval.image_subset={self.subset!r} cannot apply; leave it `all`"
            )
        tested = holdout.tested
        if not holdout.h_in or not tested:
            raise ValueError("oneshot.holdout.h_in / h_out are empty: nothing to test")
        check_no_leak(tested, clf.encoder_classes)
        for name in holdout.h_in:
            if groups[name].group != "H_chon" or groups[name].template is None:
                raise ValueError(
                    f"h_in {name} must be an ISO class with a template: {groups[name]}"
                )
        for name in holdout.h_out:
            if groups[name].group != "OOV_heldout":
                raise ValueError(f"h_out {name} must be an OOV class: {groups[name]}")
        rows = self._score(clf, ["train", "val"], tested, groups, None)
        known, unknown = self._known_unknown(rows, {"H_chon"}, {"OOV_heldout"})
        per_class = self._per_class(rows, groups, tested)
        summary = self._summary(known, unknown, _SWEEP_TAUS)
        out = {
            "splits": ["train", "val"],
            "n_crops": len(rows),
            "h_in": self._group_block(per_class, "H_chon"),
            "h_out": self._group_block(per_class, "OOV_heldout"),
            "per_class": per_class,
            "open_set": summary,
            "crops_csv": self._write_csv(rows),
            "sweep_png": self._plot_sweep(summary["sweep"]),
        }
        logger.info(
            "[oneshot/p1] h_in macro top1=%s top3=%s  AUROC=%s (margin %s) OSCR=%s",
            out["h_in"]["macro_top1"],
            out["h_in"]["macro_top3"],
            summary["auroc"],
            summary["margin_auroc"],
            summary["oscr_area"],
        )
        return out

    # -- center ---------------------------------------------------------- #

    def _center(self, clf, groups, holdout) -> dict:
        if clf.center is not None:
            raise ValueError(
                "eval.oneshot.protocol=center builds the center of the RAW embeddings: drop "
                "oneshot.center"
            )
        if self.subset != "all":
            raise ValueError(
                "eval.oneshot.protocol=center reads crops_fgc/train only, so "
                f"eval.image_subset={self.subset!r} cannot apply; leave it `all`"
            )
        seen = self._seen_classes(clf)[0]
        held = sorted(set(seen) & set(holdout.excluded))
        classes = sorted(set(seen) - set(holdout.excluded))
        if not classes:
            raise ValueError(
                "no trained class outside oneshot.holdout to center on: an encoder trained on "
                "no class needs ++eval.oneshot.groups_ckpt (the leave-out encoder)"
            )
        if held:
            logger.info("[oneshot/center] held-out classes left out of the center: %s", held)
        root = self.paths.crops_fgc / "train"
        samples, stored = scan_crops(root, classes, None)
        by_class: dict[tuple[str, bool], list[int]] = defaultdict(list)
        for n, (_, idx) in enumerate(samples):
            by_class[(classes[idx], False)].append(n)
        picked = sample_per_class(by_class, self.max_per_class, self.seed)
        order = sorted(n for ns in picked.values() for n in ns)
        if not order:
            raise ValueError(f"no crop of {classes} under {root}")
        total, n_views = None, 0
        for k, n in enumerate(order, 1):
            path = samples[n][0]
            img = Image.open(path).convert("RGB")
            if path in stored:
                img = view(img, stored[path])
            emb = clf.embed_crop(np.array(img))
            total = emb.sum(0) if total is None else total + emb.sum(0)
            n_views += emb.shape[0]
            if k % 500 == 0:
                logger.info("[oneshot/center] %d / %d crops", k, len(order))
        center = total / n_views
        ckpt = Path(str(self.cfg.oneshot.encoder_ckpt))
        per_class = {name: len(ns) for (name, _), ns in sorted(picked.items())}
        meta = {
            "encoder_checkpoint": str(ckpt),
            "encoder_sha256": checkpoint_sha256(ckpt),
            "groups_checkpoint": None if self.groups_ckpt is None else str(self.groups_ckpt),
            "split": "train",
            "classes": classes,
            "per_class": per_class,
            "n_crops": len(order),
            "n_views": n_views,
            "query_views": clf.views.as_dict(),
            "max_per_class": self.max_per_class,
            "seed": self.seed,
            "input_norm": clf.input_norm,
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = save_center(self.output_dir / "oneshot_center.pt", center, meta)
        logger.info(
            "[oneshot/center] %d crops (%d views) of %d classes -> %s (|center|=%.3f)",
            len(order), n_views, len(classes), path, float(center.norm()),
        )  # fmt: skip
        return {
            "center_file": str(path),
            "center_norm": float(center.norm()),
            "held_out_dropped": held,
            **{k: meta[k] for k in ("split", "classes", "per_class", "n_crops", "n_views")},
        }

    # -- p2 -------------------------------------------------------------- #

    def _p2(self, clf, groups, holdout) -> dict:
        ids = image_subset_ids(self.cfg, "val")
        if ids is None:
            logger.warning(
                "[oneshot/p2] eval.image_subset=all: the val_sel half chose the checkpoint and "
                "tau; report p2 with eval.image_subset=test"
            )
        known_groups = ("S", "V", "V_of_S")
        classes = [n for n, g in groups.items() if g.group in (*known_groups, "OOV_seen")]
        rows = self._score(clf, ["val"], classes, groups, ids)
        known, unknown = self._known_unknown(rows, {"S", "V"}, {"OOV_seen"})
        per_class = self._per_class(rows, groups, classes)
        summary = self._summary(known, unknown, _SWEEP_TAUS)
        by_group = {
            g: self._summary([r for r in known if r["group"] == g], unknown)
            for g in ("S", "V")
            if any(r["group"] == g for r in known)
        }
        out = {
            "split": "val",
            "n_crops": len(rows),
            "groups": {g: self._group_block(per_class, g) for g in (*known_groups, "OOV_seen")},
            "per_class": per_class,
            "open_set": summary,
            "open_set_by_group": by_group,
            "unsure_crops_excluded": self._unsure_counts(groups, "val", ids),
            "crops_csv": self._write_csv(rows),
            "sweep_png": self._plot_sweep(summary["sweep"]),
        }
        logger.info(
            "[oneshot/p2] S macro top1=%s  V macro top1=%s  AUROC=%s (margin %s) OSCR=%s",
            out["groups"]["S"]["macro_top1"],
            out["groups"]["V"]["macro_top1"],
            summary["auroc"],
            summary["margin_auroc"],
            summary["oscr_area"],
        )
        return out

    # -- tau ------------------------------------------------------------- #

    def _tau(self, clf, groups, holdout) -> dict:
        if self.subset != "val_sel":
            raise ValueError(
                "tau is chosen on val_sel only (PLAN B8 point 2), "
                f"got eval.image_subset={self.subset!r}"
            )
        ids = image_subset_ids(self.cfg, "val")
        directory = clf.directory
        classes, skipped = [], []
        for name, info in groups.items():
            if info.group != "S":
                continue
            (classes if info.template in directory.classes else skipped).append(name)
        validate_tau_classes(classes, self._seen_classes(clf)[0], holdout)
        held = sorted(set(classes) & set(holdout.excluded))
        if self.groups_ckpt is not None and held:
            # Another encoder's group S: still never a class P1 tests (nor h_extra).
            raise ValueError(
                f"tau with eval.oneshot.groups_ckpt would read {held}, held out by "
                "oneshot.holdout; point groups_ckpt at the leave-out encoder"
            )
        if skipped:
            logger.warning(
                "[oneshot/tau] no prototype for %s (oneshot.exclude_prototypes?): skipped", skipped
            )
        if not classes:
            raise ValueError("no group-S class with a prototype: nothing to select tau on")
        rows = self._score(clf, ["val"], classes, groups, ids, keep_emb=True)
        if not rows:
            raise ValueError("no val_sel crop of a group-S class: nothing to select tau on")

        names = directory.names
        proto = directory.embeddings
        twins: dict[str, list[tuple[str, float]]] = {}
        reduced: dict[tuple[str, bool], PrototypeDirectory] = {}

        def without(cls: str, drop_twins: bool) -> PrototypeDirectory:
            if (cls, drop_twins) not in reduced:
                template = groups[cls].template
                drop = {template}
                if drop_twins:
                    cos = proto @ proto[names.index(template)]
                    found = [
                        (n, float(c))
                        for n, c in zip(names, cos, strict=True)
                        if n != template and c > self.twin_cos
                    ]
                    twins[cls] = found
                    drop |= {n for n, _ in found}
                reduced[(cls, drop_twins)] = directory.without(drop)
            return reduced[(cls, drop_twins)]

        for r in rows:
            for key, twins_too in (("pseudo_cos", False), ("pseudo_cos_no_twins", True)):
                r[key] = (
                    without(r["class"], twins_too)
                    .match_views(r["emb"], None, self.aggregate)
                    .cosine
                )
        known = [r["best_cos"] for r in rows]
        count = defaultdict(int)
        for r in rows:
            count[r["class"]] += 1
        base = {"n_crops": len(rows), "n_classes": len(classes), "crops_per_class": dict(count)}

        tau_a = tau_rule_a(known, self.tau_tpr)
        out: dict = {
            "split": "val",
            "classes": classes,
            "skipped_classes": skipped,
            "rule_a": {
                "tau": tau_a,
                "tpr": self.tau_tpr,
                "known_accepted": float(np.mean(np.asarray(known) >= tau_a)),
                **base,
            },
        }
        margin = margins(known, [r["second_cos"] for r in rows])
        delta = margin_rule_a(margin, self.tau_tpr)
        out["margin_rule_a"] = {
            "delta": delta,
            "tpr": self.tau_tpr,
            "known_accepted": float(np.mean(margin >= delta)),
            "known_accepted_with_rule_a_tau": float(
                np.mean((margin >= delta) & (np.asarray(known) >= tau_a))
            ),
            **base,
        }
        for rule, key in (("rule_b", "pseudo_cos"), ("rule_b_no_twins", "pseudo_cos_no_twins")):
            pseudo = [r[key] for r in rows]
            picked = tau_rule_b(known, pseudo)
            out[rule] = {
                "tau": picked[0],
                "youden_j": picked[1],
                "n_pseudo_unknown": len(pseudo),
                **base,
            }
        out["rule_b_no_twins"]["twin_cos"] = self.twin_cos
        out["rule_b_no_twins"]["twin_pairs"] = [
            {"class": c, "template": groups[c].template, "twin": n, "cos": cos}
            for c, found in twins.items()
            for n, cos in found
        ]
        out["note"] = (
            "tau is NOT written to config; copy the chosen rule's value into "
            "oneshot.other_min_cosine by hand (a simulated unknown is easier than a real one, "
            "so rule B is optimistic)"
        )
        out["crops_csv"] = self._write_csv(rows, ("pseudo_cos", "pseudo_cos_no_twins"))
        logger.info(
            "[oneshot/tau] %d crops, %d classes: rule A tau=%.4f (margin delta=%.4f), "
            "rule B tau=%s, rule B (no twins) tau=%s",
            len(rows),
            len(classes),
            tau_a,
            delta,
            out["rule_b"]["tau"],
            out["rule_b_no_twins"]["tau"],
        )
        return out

    # -- p4 -------------------------------------------------------------- #

    def _p4(self, clf, groups, holdout) -> dict:
        if self.subset != "all":
            raise ValueError(
                "eval.oneshot.protocol=p4 scores OPEN100-ISO, which has no val_sel / test "
                f"halves, so eval.image_subset={self.subset!r} cannot apply; leave it `all`"
            )
        vocab = read_vocab(Path(str(self.cfg.oneshot.vocab_csv)))
        templates = [r["fine_name"].strip() for r in vocab.values()]
        with self.paths.coco_json.open() as f:
            coco = json.load(f)
        validate_p4_coco(coco, templates)
        iso_groups = assign_iso_groups(groups, templates)
        group_of = {**iso_groups, **{n: n for n in OTHER_NAMES}}
        cat = {c["id"]: c["name"] for c in coco["categories"]}
        images = {im["id"]: im for im in coco["images"]}

        # Ignore regions (iscrowd, utils.classmap.IGNORE_CATEGORY) are no symbol: never a crop,
        # never known, never Other.
        anns = [a for a in coco["annotations"] if not a.get("iscrowd")]
        if any(cat[a["category_id"]] == IGNORE_CATEGORY for a in anns):
            raise ValueError(f"an {IGNORE_CATEGORY!r} annotation without iscrowd: not a symbol")
        ids: dict[tuple[str, bool], list[int]] = defaultdict(list)
        for a in anns:
            ids[(cat[a["category_id"]], bool(images[a["image_id"]]["degraded"]))].append(a["id"])
        available = {
            v: {name: len(i) for (name, deg), i in sorted(ids.items()) if deg == (v == "degraded")}
            for v in _P4_VIEWS
        }
        keep = {i for kept in sample_per_class(ids, self.max_per_class, self.seed).values()
                for i in kept}  # fmt: skip
        present = sorted({name for name, _ in ids if name in iso_groups})
        missing = sorted(set(present) - set(clf.directory.classes))
        if missing:
            logger.warning(
                "[oneshot/p4] no prototype for %s (oneshot.exclude_prototypes?): their crops "
                "can never be named right",
                missing,
            )
        rows = self._score_coco(clf, coco, cat, group_of, keep)  # keep holds no ignore region
        out: dict = {
            "dataset": str(self.paths.root),
            "primary": "degraded",
            "max_per_class": self.max_per_class,
            "seed": self.seed,
            "template_groups": iso_groups,
            "group_classes": {g: [n for n in present if iso_groups[n] == g] for g in ISO_GROUPS},
            "templates_without_prototype": missing,
            "n_annotations": len(anns),
            "n_ignored": len(coco["annotations"]) - len(anns),
            "n_crops": len(rows),
        }
        for v in _P4_VIEWS:
            out[v] = self._p4_view([r for r in rows if r["split"] == v], available[v], v)
        out["crops_csv"] = self._write_csv(rows, _P4_FIELDS)
        for v in _P4_VIEWS:
            block = out[v]
            logger.info(
                "[oneshot/p4] %s: %s  AUROC=%s (margin %s)",
                v,
                "  ".join(
                    f"{g} top1={b['macro_top1']} (n={b['n_crops']})"
                    for g, b in block["groups"].items()
                    if b["n_crops"]
                ),
                (block["open_set"] or {}).get("auroc"),
                (block["open_set"] or {}).get("margin_auroc"),
            )
        return out

    def _score_coco(self, clf, coco, cat, group_of, keep: set[int]) -> list[dict]:
        """One row per kept COCO annotation (see _CSV_FIELDS + _P4_FIELDS), the view cut from
        its sheet with the inference-time margin; each sheet image is read once."""
        by_image: dict[int, list[dict]] = defaultdict(list)
        for a in coco["annotations"]:
            if a["id"] in keep:
                by_image[a["image_id"]].append(a)
        images = {im["id"]: im for im in coco["images"]}
        logger.info("[oneshot/p4] %d crops from %d images", len(keep), len(by_image))
        rows: list[dict] = []
        for n, image_id in enumerate(sorted(by_image), 1):
            im = images[image_id]
            stem = Path(im["file_name"]).stem
            pixels = np.array(Image.open(self.paths.find_image(stem)).convert("RGB"))
            degraded = bool(im["degraded"])
            for a in sorted(by_image[image_id], key=lambda a: a["id"]):
                x, y, w, h = a["bbox"]
                emb = clf.embed_crop(np.ascontiguousarray(cut(pixels, (x, y, x + w, y + h))))
                name = cat[a["category_id"]]
                rows.append(
                    {
                        "split": "degraded" if degraded else "clean",
                        "path": im["file_name"],
                        "image_id": stem,
                        "class": name,
                        "group": group_of[name],
                        "expected": "" if name in OTHER_NAMES else name,
                        **self._ranked(clf, emb),
                        "sheet": int(im["sheet"]),
                        "variant": int(im["variant"]),
                        "degraded": int(degraded),
                        "ann_id": int(a["id"]),
                    }
                )
            if n % 50 == 0:
                logger.info("[oneshot/p4] %d / %d images", n, len(by_image))
        return rows

    @staticmethod
    def _p4_groups(rows: list[dict]) -> dict:
        """One row per ISO group (never pooled): classes, crops, macro and instance top-1/3."""

        def mean(flags):
            return float(np.mean(flags)) if flags else None

        out = {}
        for g in ISO_GROUPS:
            mine = [r for r in rows if r["group"] == g]
            stats = [
                class_stats(
                    [r["names"] for r in mine if r["class"] == c],
                    c,
                    [r["best_cos"] for r in mine if r["class"] == c],
                )
                for c in sorted({r["class"] for r in mine})
            ]
            out[g] = {
                "n_classes": len(stats),
                "n_crops": len(mine),
                "macro_top1": macro(s["top1"] for s in stats),
                "macro_top3": macro(s["top3"] for s in stats),
                "top1": mean([r["top1"] == r["expected"] for r in mine]),
                "top3": mean([r["expected"] in r["names"] for r in mine]),
            }
        return out

    def _p4_other(self, known: list[dict], other: list[dict]) -> dict:
        """An Other category against every vocab crop: reject rate at tau, AUROC (cos, margin)."""
        cos = [r["best_cos"] for r in other]
        both = bool(other) and bool(known)
        rejected = float(np.mean([c < self.other_tau for c in cos])) if cos else None

        def margin(rs):
            return margins([r["best_cos"] for r in rs], [r["second_cos"] for r in rs])

        stats = class_stats([r["names"] for r in other], None, cos)
        return {
            "n": len(other),
            "reject_rate_at_tau": rejected,
            "auroc": other_auroc([r["best_cos"] for r in known], cos) if both else None,
            "margin_auroc": margin_auroc(margin(known), margin(other)) if both else None,
            "most_predicted": stats["most_predicted"],
            "most_predicted_frac": stats["most_predicted_frac"],
            "mean_best_cos": stats["mean_best_cos"],
        }

    def _p4_view(self, rows: list[dict], available: dict[str, int], tag: str) -> dict:
        """Every P4 number for one view (degraded or clean), never mixing the two."""
        known = [r for r in rows if r["group"] in ISO_GROUPS]
        unknown = [r for r in rows if r["group"] == "other_iso"]
        per_class = {}
        for name in sorted({r["class"] for r in rows}):
            mine = [r for r in rows if r["class"] == name]
            expected = None if name in OTHER_NAMES else name
            stats = class_stats([r["names"] for r in mine], expected, [r["best_cos"] for r in mine])
            stats.update(group=mine[0]["group"], n_available=available.get(name, 0))
            if expected is None:
                stats["reject_rate_at_tau"] = float(
                    np.mean([r["best_cos"] < self.other_tau for r in mine])
                )
            per_class[name] = stats
        sheets = {}
        for sheet in sorted({r["sheet"] for r in rows}):
            mine = [r for r in rows if r["sheet"] == sheet]
            other = self._p4_other(
                [r for r in mine if r["group"] in ISO_GROUPS],
                [r for r in mine if r["group"] == "other_iso"],
            )
            sheets[f"iso{sheet:02d}"] = {
                "n_crops": len(mine),
                "groups": self._p4_groups(mine),
                "auroc": other["auroc"],
                "margin_auroc": other["margin_auroc"],
            }
        out: dict = {
            "n_crops": len(rows),
            "n_images": len({r["image_id"] for r in rows}),
            "n_available": available,
            "groups": self._p4_groups(rows),
            "per_class": per_class,
            "other": {
                name: self._p4_other(known, [r for r in rows if r["group"] == name])
                for name in OTHER_NAMES
            },
            "per_sheet": sheets,
            "sheet_mean_std": {
                "groups": {
                    g: {
                        m: mean_std(s["groups"][g][m] for s in sheets.values())
                        for m in ("macro_top1", "macro_top3")
                    }
                    for g in ISO_GROUPS
                },
                "auroc": mean_std(s["auroc"] for s in sheets.values()),
                "margin_auroc": mean_std(s["margin_auroc"] for s in sheets.values()),
            },
            "open_set": None,
            "open_set_by_group": {},
            "sweep_png": None,
        }
        if known:
            out["open_set"] = self._summary(known, unknown, _SWEEP_TAUS)
            out["open_set_by_group"] = {
                g: self._summary([r for r in known if r["group"] == g], unknown)
                for g in ISO_GROUPS
                if any(r["group"] == g for r in known)
            }
            out["sweep_png"] = self._plot_sweep(out["open_set"]["sweep"], tag)
        else:
            logger.warning("[oneshot/p4] %s: no vocab crop, open-set metrics left out", tag)
        return out
