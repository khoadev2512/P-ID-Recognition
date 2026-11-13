"""Crop-level one-shot evaluation on DigitizePID (PLAN B6b-1): eval.mode=oneshot.

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
         on group-S classes the encoder trained on. Prints tau; writes nothing to config.

With oneshot.aggregate=vote best_cos is the mean cosine of the winning views, so a sweep of
tau over best_cos only approximates the gated vote (exact for `mean`).
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image

from pipeline.oneshot_infer import OneShotClassifier, load_oneshot_classifier
from utils.crop_box import view
from utils.dataset import (
    build_eval_transform,
    crop_image_id,
    image_subset_ids,
    scan_crops,
)
from utils.model import checkpoint_excluded_classes, checkpoint_sha256
from utils.oneshot import PrototypeDirectory, aggregate_views, holdout_classes
from utils.oneshot_metrics import (
    ClassInfo,
    assign_groups,
    class_stats,
    macro,
    open_set_summary,
    read_symbol_map,
    read_vocab,
    tau_rule_a,
    tau_rule_b,
)
from utils.paths import DataPaths

logger = logging.getLogger(__name__)

PROTOCOLS = ("p1", "p2", "tau")
DEFAULT_SYMBOL_MAP = "docs/symbol_mapping/symbol_map.csv"  # oneshot.symbol_map, repo-relative
_REPO_ROOT = Path(__file__).resolve().parents[2]
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


def rank_names(
    directory: PrototypeDirectory, emb: torch.Tensor, aggregate: str, k: int = 3
) -> list[tuple[str, float]]:
    """The k best (name, cosine) for one crop's (V, D) view embeddings, best first, un-gated.

    mean  the aggregated embedding's k nearest prototypes.
    vote  names ranked by votes (ties: summed cosine), the cosine of a name being the mean
          over the views that voted for it (as OneShotClassifier's best_cos); names no view
          voted for fill the remaining places by their mean cosine over all views.
    """
    if aggregate == "mean":
        return directory.topk(aggregate_views(emb), k)
    if aggregate != "vote":
        raise ValueError(f"unknown aggregate={aggregate!r}; expected 'vote' or 'mean'")
    cos = directory.cosines(emb)  # (V, N)
    best_cos, best_idx = cos.max(dim=1)
    voted: dict[int, list[float]] = defaultdict(list)
    for v in range(len(best_idx)):
        voted[int(best_idx[v])].append(float(best_cos[v]))
    order = sorted(voted.items(), key=lambda kv: (len(kv[1]), sum(kv[1])), reverse=True)
    ranked = [(directory.names[i], sum(c) / len(c)) for i, c in order]
    if len(ranked) < k:
        mean_cos = cos.mean(dim=0)
        rest = [i for i in torch.argsort(mean_cos, descending=True).tolist() if i not in voted]
        ranked += [(directory.names[i], float(mean_cos[i])) for i in rest]
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
        self.subset = str(cfg.eval.get("image_subset", "all"))
        self.other_tau = float(cfg.oneshot.other_min_cosine)
        self.aggregate = str(cfg.oneshot.get("aggregate", "vote"))
        self._classifier = classifier

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

    def _groups(self, clf: OneShotClassifier, holdout) -> dict[str, ClassInfo]:
        symbol_map = Path(str(self.cfg.oneshot.get("symbol_map", None) or DEFAULT_SYMBOL_MAP))
        if not symbol_map.is_absolute():
            symbol_map = _REPO_ROOT / symbol_map
        ckpt = Path(str(self.cfg.oneshot.encoder_ckpt))
        return assign_groups(
            read_symbol_map(symbol_map),
            read_vocab(Path(str(self.cfg.oneshot.vocab_csv))),
            clf.encoder_classes,
            checkpoint_excluded_classes(ckpt),
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
                top = rank_names(clf.directory, emb, self.aggregate, 3)
                info = groups[classes[idx]]
                row = {
                    "split": split,
                    "path": str(path.relative_to(self.paths.crops_fgc)),
                    "image_id": crop_image_id(path.name),
                    "class": info.name,
                    "group": info.group,
                    "expected": info.template or "",
                    "names": [n_ for n_, _ in top],
                    "best_cos": top[0][1],
                    "second_cos": clf._second_cosine(clf.directory, emb, top[0][0]),
                }
                for j, (name, cos) in enumerate(top, 1):
                    row[f"top{j}"], row[f"cos{j}"] = name, cos
                if keep_emb:
                    row["emb"] = emb
                rows.append(row)
                if n % 500 == 0:
                    logger.info("[oneshot] %s: %d / %d crops", split, n, len(samples))
        return rows

    def _write_csv(self, rows: list[dict], extra: tuple[str, ...] = ()) -> str:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"oneshot_{self.protocol}_crops.csv"
        with path.open("w", newline="") as f:
            out = csv.DictWriter(f, fieldnames=[*_CSV_FIELDS, *extra], extrasaction="ignore")
            out.writeheader()
            for r in rows:
                out.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
        return str(path)

    def _plot_sweep(self, sweep: dict) -> str:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(sweep["tau"], sweep["known_acc"], label="known: accepted and correct")
        ax.plot(sweep["tau"], sweep["known_reject_rate"], label="known: rejected")
        if sweep["unknown_reject_rate"][0] is not None:
            ax.plot(sweep["tau"], sweep["unknown_reject_rate"], label="unknown: rejected")
        ax.axvline(self.other_tau, color="grey", linestyle=":", label="other_min_cosine")
        ax.set_xlabel("tau (best cosine)")
        ax.set_ylabel("fraction of crops")
        ax.set_title(f"One-shot {self.protocol}: tau sweep")
        ax.legend(fontsize=8)
        fig.tight_layout()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"oneshot_{self.protocol}_sweep.png"
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
            "prototypes": str(self.cfg.oneshot.prototypes),
            "n_prototypes": len(clf.directory),
            "aggregate": self.aggregate,
            "query_views": clf.views.as_dict(),
            "image_subset": self.subset,
            "other_min_cosine": self.other_tau,
        }
        run = {"p1": self._p1, "p2": self._p2, "tau": self._tau}[self.protocol]
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
        return open_set_summary(
            [r["top1"] == r["expected"] for r in known],
            [r["best_cos"] for r in known],
            [r["best_cos"] for r in unknown],
            self.other_tau,
            taus,
        )

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
            "[oneshot/p1] h_in macro top1=%s top3=%s  AUROC=%s OSCR=%s",
            out["h_in"]["macro_top1"],
            out["h_in"]["macro_top3"],
            summary["auroc"],
            summary["oscr_area"],
        )
        return out

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
            "[oneshot/p2] S macro top1=%s  V macro top1=%s  AUROC=%s OSCR=%s",
            out["groups"]["S"]["macro_top1"],
            out["groups"]["V"]["macro_top1"],
            summary["auroc"],
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
            (classes if info.template in directory.names else skipped).append(name)
        validate_tau_classes(classes, clf.encoder_classes, holdout)
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
            "[oneshot/tau] %d crops, %d classes: rule A tau=%.4f, rule B tau=%s, "
            "rule B (no twins) tau=%s",
            len(rows),
            len(classes),
            tau_a,
            out["rule_b"]["tau"],
            out["rule_b_no_twins"]["tau"],
        )
        return out
