"""Build the ISO 10628-2 prototype directory (one-shot plan GĐ2, docs/HANDOFF.md §6).

This is the stage that converts a model trained on DigitizePID into a recognizer for
the ISO vocabulary. It never trains anything: it pushes each of the 39 ISO symbol
templates through a FROZEN ArcFace encoder (a pid-fgc-train checkpoint) and stores the
resulting unit vectors as {fine_name -> embedding}.

    pid-fgc-train fgc.metric_learning=true          # GĐ1: learn the embedding space
    pid-iso-prototypes oneshot.encoder_ckpt=runs/<run>/fgc_checkpoints/<group>.pt
                                                    # GĐ2: this stage
    pid-fgc-infer fgc.oneshot=true                  # GĐ4: match crops against it

Inputs (both tracked in git, unlike the rest of data/):
  - oneshot.symbols_dir  one PNG per ISO reg_number (data/raw/iso_symbols_clean/),
                         cut from the standard's PDF by scripts/extract_iso_symbols.py
                         and hand-checked by the author.
  - oneshot.vocab_csv    the vocabulary: reg_number -> fine_name + coarse + fgc_group
                         (data/raw/iso_symbols/vocab.csv). This file, not the PNG
                         filenames, defines the label names and their row order.

The output is a single .pt file. Adding a symbol to the recognizable vocabulary is
re-running this stage with one more row in the CSV and one more PNG — no retraining,
which is the ISO-extensibility claim in HANDOFF §1.3 and the concrete difference from
OSSR-PID's conclusion (new class => retrain).
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from PIL import Image

from pipeline.base import BasePipeline
from utils.cli import CONFIG_DIR
from utils.dataset import build_eval_transform
from utils.model import checkpoint_excluded_classes, load_arcface_encoder
from utils.oneshot import ViewSpec, build_prototypes

logger = logging.getLogger(__name__)

# Columns read from vocab.csv. reg_number names the PNG; fine_name is the label the
# prototype is keyed by; coarse lets fgc_infer restrict matching to the coarse class
# Stage 1 predicted; fgc_group is the author's manual "needs fine-grained distinction"
# flag (see HANDOFF §2 on why "coarse has >1 fine" is the wrong rule for ISO).
_REQUIRED_COLUMNS = {"reg_number", "fine_name", "coarse", "fgc_group"}


def _cell(row: dict[str, str], column: str) -> str:
    """Stripped cell value, with a missing/None cell read as the empty string."""
    return str(row.get(column) or "").strip()


def read_vocab(
    vocab_csv: Path,
    fgc_groups_only: bool = False,
    coarse_filter: list[str] | None = None,
) -> list[dict[str, str]]:
    """Parse vocab.csv into row dicts, preserving file order (-> prototype row order).

    Validates the things that would otherwise produce a quietly wrong directory: a
    missing column, a blank fine_name (an unnamed prototype can never be matched to a
    ground-truth label), and duplicate fine_names (two rows would collide into one
    prototype key and one would be silently dropped). Those checks run on the WHOLE
    file, before any filtering — a malformed row is a vocab bug whether or not this
    particular run would have used it.

    `fgc_groups_only` keeps the fgc_group=yes rows (the visually-confusable families);
    `coarse_filter` keeps only the named ISO coarse classes. The latter exists because
    pid-fgc-train writes one checkpoint PER fgc_group: a directory built with the
    valve-group encoder should hold valve-family prototypes, not all 39, since an
    embedding space trained to separate valves says nothing useful about vessels. Both
    filters apply, in that order.
    """
    with Path(vocab_csv).open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{vocab_csv}: no rows")

    missing = _REQUIRED_COLUMNS - set(rows[0].keys())
    if missing:
        raise ValueError(f"{vocab_csv}: missing column(s) {sorted(missing)}")

    # csv.DictReader fills short rows with None, and str(None) would sail through a
    # naive blank check and become a prototype literally named "None" — vocab.csv is
    # hand-edited, so coerce None to "" before testing.
    blank = [r["reg_number"] for r in rows if not _cell(r, "fine_name")]
    if blank:
        raise ValueError(f"{vocab_csv}: rows with an empty fine_name: {blank}")

    names = [_cell(r, "fine_name") for r in rows]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"{vocab_csv}: duplicate fine_name(s) {duplicates}")

    blank_coarse = [r["reg_number"] for r in rows if not _cell(r, "coarse")]
    if blank_coarse:
        raise ValueError(f"{vocab_csv}: rows with an empty coarse: {blank_coarse}")

    available_coarse = sorted({_cell(r, "coarse") for r in rows})
    if fgc_groups_only:
        rows = [r for r in rows if _cell(r, "fgc_group").lower() == "yes"]
        if not rows:
            raise ValueError(f"{vocab_csv}: oneshot.fgc_groups_only=true left no rows")
    if coarse_filter:
        wanted = {str(c).strip() for c in coarse_filter}
        rows = [r for r in rows if _cell(r, "coarse") in wanted]
        if not rows:
            raise ValueError(
                f"{vocab_csv}: oneshot.coarse_filter={sorted(wanted)} left no rows; "
                f"coarse classes in this vocab: {available_coarse}"
            )
    return rows


def load_templates(
    symbols_dir: Path, rows: list[dict[str, str]]
) -> tuple[dict[str, Image.Image], dict[str, str]]:
    """Open <symbols_dir>/<reg_number>.png for every vocab row.

    Returns (fine_name -> image, fine_name -> coarse). A vocab row with no PNG is a
    hard error rather than a skip: a silently absent prototype means that ISO symbol
    can only ever be classified as something else or as "Other", and the eval would
    report that as a model failure instead of a missing input.
    """
    templates: dict[str, Image.Image] = {}
    coarse_of: dict[str, str] = {}
    missing: list[str] = []
    for row in rows:
        reg = _cell(row, "reg_number")
        name = _cell(row, "fine_name")
        png = Path(symbols_dir) / f"{reg}.png"
        if not png.exists():
            missing.append(reg)
            continue
        templates[name] = Image.open(png).convert("RGB")
        coarse_of[name] = _cell(row, "coarse")
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} vocab row(s) have no template PNG under {symbols_dir}: {missing}"
        )
    return templates, coarse_of


class IsoPrototypePipeline(BasePipeline):
    """Embed the ISO symbol library with a frozen ArcFace encoder -> prototype directory."""

    def validate(self) -> bool:
        cfg = self.cfg
        if cfg.oneshot.get("encoder_ckpt") is None:
            raise ValueError(
                "cfg.oneshot.encoder_ckpt must point at an ArcFace checkpoint from "
                "pid-fgc-train, e.g.:\n"
                "    pid-iso-prototypes oneshot.encoder_ckpt="
                "runs/<train-run>/fgc_checkpoints/<group>.pt"
            )
        ckpt = Path(str(cfg.oneshot.encoder_ckpt))
        if not ckpt.exists():
            raise FileNotFoundError(f"oneshot.encoder_ckpt does not exist: {ckpt}")

        symbols_dir = Path(str(cfg.oneshot.symbols_dir))
        if not symbols_dir.is_dir():
            raise FileNotFoundError(
                f"oneshot.symbols_dir does not exist: {symbols_dir} "
                "(run scripts/extract_iso_symbols.py to cut the ISO library)"
            )
        vocab_csv = Path(str(cfg.oneshot.vocab_csv))
        if not vocab_csv.exists():
            raise FileNotFoundError(f"oneshot.vocab_csv does not exist: {vocab_csv}")
        return True

    def run(self) -> None:
        self.validate()
        cfg = self.cfg
        device = torch.device(
            "cuda" if str(cfg.device) == "cuda" and torch.cuda.is_available() else "cpu"
        )

        ckpt = Path(str(cfg.oneshot.encoder_ckpt))
        encoder, train_classes = load_arcface_encoder(cfg, ckpt, device)
        logger.info(
            "[oneshot] encoder %s (%s, %d training classes, embed_dim=%d) on %s",
            ckpt,
            cfg.fgc.backbone,
            len(train_classes),
            int(cfg.fgc.arcface.embed_dim),
            device,
        )

        coarse_filter = cfg.oneshot.get("coarse_filter", None)
        coarse_filter = list(coarse_filter) if coarse_filter else None
        rows = read_vocab(
            Path(str(cfg.oneshot.vocab_csv)),
            fgc_groups_only=bool(cfg.oneshot.get("fgc_groups_only", False)),
            coarse_filter=coarse_filter,
        )
        templates, coarse_of = load_templates(Path(str(cfg.oneshot.symbols_dir)), rows)
        spec = ViewSpec.from_cfg(cfg.oneshot.template_views)
        logger.info(
            "[oneshot] %d ISO templates x %d view(s) (scales=%s, orientations=%s)",
            len(templates),
            spec.n_views,
            list(spec.scales),
            list(spec.orientations),
        )

        directory = build_prototypes(
            encoder=encoder,
            templates=templates,
            transform=build_eval_transform(cfg),
            spec=spec,
            coarse_of=coarse_of,
            device=device,
            background=tuple(int(c) for c in cfg.oneshot.background),
            # Provenance: a prototype directory is only meaningful next to the encoder
            # that produced it, so the checkpoint, backbone and preprocessing size are
            # recorded in the file itself rather than left to the run log.
            meta={
                "encoder_ckpt": str(ckpt),
                "backbone": str(cfg.fgc.backbone),
                "embed_dim": int(cfg.fgc.arcface.embed_dim),
                "imgsz": int(cfg.fgc.imgsz),
                "encoder_train_classes": train_classes,
                # Leave-classes-out encoders (protocol P1) must be told apart from
                # full-vocabulary ones by the file alone, not by its name on Drive.
                "encoder_excluded_classes": checkpoint_excluded_classes(ckpt),
                "vocab_csv": str(cfg.oneshot.vocab_csv),
                "symbols_dir": str(cfg.oneshot.symbols_dir),
                "fgc_groups_only": bool(cfg.oneshot.get("fgc_groups_only", False)),
                "coarse_filter": coarse_filter,
            },
        )

        out_path = directory.save(Path(str(cfg.oneshot.prototypes)))
        logger.info(
            "[oneshot] wrote %d prototypes (dim=%d) -> %s", len(directory), directory.dim, out_path
        )
        self._log_separation(directory)

    def _log_separation(self, directory) -> None:
        """Report how well-separated the prototypes are, as a sanity check on GĐ2.

        Two numbers, straight off the Gram matrix of the prototype set:
          - mean/max off-diagonal cosine: how close the closest two ISO symbols sit in
            this embedding space. A max near 1.0 means two templates are effectively
            indistinguishable and one-shot accuracy between them will be a coin flip.
          - the single most-confusable pair, named, so the author can look at those two
            PNGs directly.

        This is the cheapest possible check that GĐ1's encoder transferred to ISO at
        all — it needs no labelled ISO data, and it runs before any inference. It also
        sets expectations for oneshot.other_min_cosine: the gate threshold has to sit
        ABOVE this inter-prototype similarity to mean anything.
        """
        if len(directory) < 2:
            return
        gram = directory.embeddings @ directory.embeddings.T
        off_diag = gram - torch.eye(len(directory)) * 2.0  # push the diagonal out of argmax
        flat_idx = int(torch.argmax(off_diag))
        i, j = divmod(flat_idx, len(directory))
        mask = ~torch.eye(len(directory), dtype=torch.bool)
        logger.info(
            "[oneshot] prototype separation: mean off-diagonal cos=%.3f, max=%.3f "
            "(closest pair: %s vs %s)",
            float(gram[mask].mean()),
            float(off_diag[i, j]),
            directory.names[i],
            directory.names[j],
        )


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    IsoPrototypePipeline(cfg).run()


if __name__ == "__main__":
    main()
