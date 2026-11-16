"""Tests for the ISO prototype builder (pipeline.iso_prototypes) and the frozen-encoder
loader it depends on (utils.model.load_arcface_encoder).

Two layers:
  - vocab/template parsing, which needs no torch and guards the failure modes that
    would otherwise produce a quietly incomplete directory (missing column, blank or
    duplicated fine_name, a vocab row with no PNG);
  - an end-to-end build over the REAL ISO library in data/raw/ (42 symbols, tracked in
    git) using a tiny untrained ArcFace model as the encoder. Untrained weights make
    the embeddings meaningless but the wiring real: the stage either produces 42 named,
    unit-norm, coarse-tagged prototypes from the actual files on disk, or it doesn't.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

# pipeline.iso_prototypes pulls in torch/timm/hydra transitively, so the guard has to
# come before the import (same pattern as test_arcface.py).
torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from omegaconf import OmegaConf  # noqa: E402
from PIL import Image  # noqa: E402

from pipeline.iso_prototypes import (  # noqa: E402
    IsoPrototypePipeline,
    load_templates,
    read_vocab,
)
from utils.model import ArcFaceModel, load_arcface_encoder  # noqa: E402
from utils.oneshot import PrototypeDirectory  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
ISO_SYMBOLS_DIR = REPO_ROOT / "data" / "raw" / "iso_symbols_clean"
ISO_VOCAB_CSV = REPO_ROOT / "data" / "raw" / "iso_symbols" / "vocab.csv"

_has_iso_library = ISO_SYMBOLS_DIR.is_dir() and ISO_VOCAB_CSV.exists()
requires_iso_library = pytest.mark.skipif(
    not _has_iso_library, reason="ISO symbol library not present under data/raw/"
)

_COLUMNS = ["reg_number", "coarse", "fine_name", "pid_group", "fgc_group", "desc"]


def _write_vocab(path: Path, rows: list[dict[str, str]], columns: list[str] | None = None) -> Path:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns or _COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _row(reg: str, fine: str, coarse: str = "valve", fgc: str = "yes") -> dict[str, str]:
    return {
        "reg_number": reg,
        "coarse": coarse,
        "fine_name": fine,
        "pid_group": "valve",
        "fgc_group": fgc,
        "desc": f"{fine} description",
    }


# ------------------------------------------------------------------ #
# read_vocab
# ------------------------------------------------------------------ #


def test_read_vocab_preserves_file_order(tmp_path):
    """Row order is the prototype row order, so it must come from the file and not from
    any incidental sorting -- a reordered directory would invalidate a saved one."""
    path = _write_vocab(
        tmp_path / "v.csv", [_row("X2", "zebra_valve"), _row("X1", "alpha_valve")]
    )
    assert [r["fine_name"] for r in read_vocab(path)] == ["zebra_valve", "alpha_valve"]


def test_read_vocab_rejects_a_missing_column(tmp_path):
    path = tmp_path / "v.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["reg_number", "fine_name"])
        writer.writeheader()
        writer.writerow({"reg_number": "X1", "fine_name": "valve_gate"})
    with pytest.raises(ValueError, match="missing column"):
        read_vocab(path)


def test_read_vocab_rejects_an_empty_file(tmp_path):
    with pytest.raises(ValueError, match="no rows"):
        read_vocab(_write_vocab(tmp_path / "v.csv", []))


def test_read_vocab_rejects_a_blank_fine_name(tmp_path):
    """An unnamed prototype could never be compared against a ground-truth label, so a
    half-filled vocab row (docs/symbol_mapping has plenty of TODO rows) is an error."""
    path = _write_vocab(tmp_path / "v.csv", [_row("X1", "valve_gate"), _row("X2", "   ")])
    with pytest.raises(ValueError, match="empty fine_name"):
        read_vocab(path)


def test_read_vocab_rejects_a_short_row_whose_cells_are_missing(tmp_path):
    """csv.DictReader fills a short row's trailing cells with None, and str(None) would
    sail through a naive check and become a prototype literally named "None"."""
    path = tmp_path / "v.csv"
    path.write_text(",".join(_COLUMNS) + "\nX1,valve\n")
    with pytest.raises(ValueError, match="empty fine_name"):
        read_vocab(path)


def test_read_vocab_rejects_a_blank_coarse(tmp_path):
    """coarse tags every prototype and is what fgc_infer restricts matching by, so an
    untagged row would quietly become unroutable."""
    path = _write_vocab(tmp_path / "v.csv", [_row("X1", "valve_gate", coarse="  ")])
    with pytest.raises(ValueError, match="empty coarse"):
        read_vocab(path)


def test_read_vocab_rejects_duplicate_fine_names(tmp_path):
    path = _write_vocab(tmp_path / "v.csv", [_row("X1", "valve_gate"), _row("X2", "valve_gate")])
    with pytest.raises(ValueError, match="duplicate"):
        read_vocab(path)


def test_read_vocab_detects_duplicates_even_outside_the_fgc_filter(tmp_path):
    """The duplicate check runs on the whole file before filtering: a name collision is
    a vocab bug regardless of whether this particular run would have used both rows."""
    path = _write_vocab(
        tmp_path / "v.csv",
        [_row("X1", "dup", fgc="no"), _row("X2", "dup", fgc="no"), _row("X3", "ok", fgc="yes")],
    )
    with pytest.raises(ValueError, match="duplicate"):
        read_vocab(path, fgc_groups_only=True)


def test_read_vocab_fgc_groups_only_keeps_the_flagged_rows(tmp_path):
    path = _write_vocab(
        tmp_path / "v.csv",
        [_row("X1", "valve_gate", fgc="yes"), _row("X2", "pipeline", "piping", fgc="no")],
    )
    assert len(read_vocab(path)) == 2
    assert [r["fine_name"] for r in read_vocab(path, fgc_groups_only=True)] == ["valve_gate"]


def test_read_vocab_coarse_filter_keeps_only_the_named_families(tmp_path):
    path = _write_vocab(
        tmp_path / "v.csv",
        [
            _row("X1", "valve_gate", "valve"),
            _row("X2", "check_valve_swing", "check_valve"),
            _row("X3", "tank", "vessel"),
        ],
    )
    kept = read_vocab(path, coarse_filter=["valve", "check_valve"])
    assert [r["fine_name"] for r in kept] == ["valve_gate", "check_valve_swing"]


def test_read_vocab_coarse_filter_with_no_match_lists_what_exists(tmp_path):
    """A typo'd coarse name must not silently yield an empty or unfiltered directory."""
    path = _write_vocab(tmp_path / "v.csv", [_row("X1", "valve_gate", "valve")])
    with pytest.raises(ValueError, match="coarse classes in this vocab"):
        read_vocab(path, coarse_filter=["valves"])


def test_read_vocab_filters_compose(tmp_path):
    path = _write_vocab(
        tmp_path / "v.csv",
        [
            _row("X1", "valve_gate", "valve", fgc="yes"),
            _row("X2", "safety_valve_spring", "safety_valve", fgc="no"),
            _row("X3", "tank", "vessel", fgc="no"),
        ],
    )
    kept = read_vocab(path, fgc_groups_only=True, coarse_filter=["valve", "vessel"])
    assert [r["fine_name"] for r in kept] == ["valve_gate"]


def test_read_vocab_fgc_groups_only_with_nothing_flagged_is_an_error(tmp_path):
    path = _write_vocab(tmp_path / "v.csv", [_row("X1", "pipeline", "piping", fgc="no")])
    with pytest.raises(ValueError, match="fgc_groups_only"):
        read_vocab(path, fgc_groups_only=True)


# ------------------------------------------------------------------ #
# load_templates
# ------------------------------------------------------------------ #


def test_load_templates_keys_by_fine_name_and_maps_coarse(tmp_path):
    symbols = tmp_path / "symbols"
    symbols.mkdir()
    for reg in ("X1", "X2"):
        Image.new("RGB", (12, 8), (255, 255, 255)).save(symbols / f"{reg}.png")

    rows = [_row("X1", "valve_gate", "valve"), _row("X2", "pump_gear", "pump")]
    templates, coarse_of = load_templates(symbols, rows)

    assert list(templates.keys()) == ["valve_gate", "pump_gear"]
    assert coarse_of == {"valve_gate": "valve", "pump_gear": "pump"}
    assert templates["valve_gate"].mode == "RGB"


def test_load_templates_lists_every_missing_png(tmp_path):
    """Missing templates are reported together (and the run fails) rather than skipped:
    a silently absent prototype shows up in the eval as a model error, not an input one."""
    symbols = tmp_path / "symbols"
    symbols.mkdir()
    Image.new("RGB", (12, 8)).save(symbols / "X1.png")

    rows = [_row("X1", "a"), _row("X2", "b"), _row("X3", "c")]
    with pytest.raises(FileNotFoundError) as excinfo:
        load_templates(symbols, rows)
    message = str(excinfo.value)
    assert "X2" in message and "X3" in message and "X1" not in message


@requires_iso_library
def test_the_real_iso_library_is_complete():
    """Guards the project's own data: every vocab row has a PNG and every fine_name is
    unique. If this fails, the ISO library on disk drifted from vocab.csv."""
    rows = read_vocab(ISO_VOCAB_CSV)
    templates, coarse_of = load_templates(ISO_SYMBOLS_DIR, rows)
    assert len(rows) == len(templates) == len(coarse_of) == 42


# ------------------------------------------------------------------ #
# load_arcface_encoder + the full build, end to end
# ------------------------------------------------------------------ #

def _cfg(tmp_path: Path, **oneshot_overrides):
    """Config with only the fields this stage reads. imgsz is deliberately tiny and the
    backbone is the smallest resnet with pretrained off, so the test needs no download
    and no GPU."""
    cfg = OmegaConf.create(
        {
            "data_root": str(tmp_path / "data"),
            "run_dir": str(tmp_path / "runs"),
            "device": "cpu",
            "fgc": {
                "backbone": "resnet18",
                "pretrained": False,
                "imgsz": 32,
                "metric_learning": True,
                "arcface": {"m": 0.5, "s": 30.0, "embed_dim": 32},
            },
            "oneshot": {
                "symbols_dir": str(ISO_SYMBOLS_DIR),
                "vocab_csv": str(ISO_VOCAB_CSV),
                "encoder_ckpt": None,
                "prototypes": str(tmp_path / "iso_prototypes.pt"),
                "fgc_groups_only": False,
                "coarse_filter": None,
                "background": [255, 255, 255],
                "template_views": {"scales": [1.0], "orientations": [0]},
                "query_views": {"scales": [1.0], "orientations": [0, 90, 180, 270]},
                "aggregate": "vote",
                "other_min_cosine": 0.35,
            },
        }
    )
    for key, value in oneshot_overrides.items():
        cfg.oneshot[key] = value
    return cfg


def _write_arcface_ckpt(cfg, path: Path, num_fine: int = 6) -> Path:
    """Save an UNTRAINED ArcFaceModel in pid-fgc-train's checkpoint format."""
    model = ArcFaceModel(cfg, num_fine)
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": 0,
            "val_acc": 0.0,
            "fine_classes": [f"dpid_{i}" for i in range(num_fine)],
            "class_centers": model.class_centers(),
        },
        path,
    )
    return path


def test_load_arcface_encoder_recovers_class_count_from_the_checkpoint(tmp_path):
    """num_fine is read off head.weight, not from config -- so it can't drift out of
    sync with the file being loaded."""
    cfg = _cfg(tmp_path)
    ckpt = _write_arcface_ckpt(cfg, tmp_path / "enc.pt", num_fine=7)
    encoder, fine_classes = load_arcface_encoder(cfg, ckpt)
    assert len(fine_classes) == 7
    assert encoder.head.weight.shape == (7, 32)
    assert not encoder.training, "the encoder must come back frozen, in eval mode"
    assert encoder.embed(torch.zeros(2, 3, 32, 32)).shape == (2, 32)


def test_load_arcface_encoder_rejects_a_plain_linear_checkpoint(tmp_path):
    """fgc.metric_learning=false trains a model with no embedding space to put ISO
    prototypes in; say so instead of failing on a missing state_dict key."""
    import timm

    cfg = _cfg(tmp_path)
    plain = timm.create_model("resnet18", pretrained=False, num_classes=5)
    path = tmp_path / "plain.pt"
    torch.save({"model": plain.state_dict(), "fine_classes": ["a"]}, path)
    with pytest.raises(ValueError, match="metric_learning"):
        load_arcface_encoder(cfg, path)


def test_load_arcface_encoder_reports_an_embed_dim_mismatch_with_both_numbers(tmp_path):
    cfg = _cfg(tmp_path)
    ckpt = _write_arcface_ckpt(cfg, tmp_path / "enc.pt")
    mismatched = OmegaConf.merge(cfg, OmegaConf.create({"fgc": {"arcface": {"embed_dim": 64}}}))
    with pytest.raises(ValueError, match="embed_dim=32"):
        load_arcface_encoder(mismatched, ckpt)


def test_load_arcface_encoder_on_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_arcface_encoder(_cfg(tmp_path), tmp_path / "nope.pt")


def test_load_arcface_encoder_rejects_a_non_checkpoint(tmp_path):
    path = tmp_path / "junk.pt"
    torch.save({"something_else": 1}, path)
    with pytest.raises(ValueError, match="pid-fgc-train checkpoint"):
        load_arcface_encoder(_cfg(tmp_path), path)


@requires_iso_library
def test_build_the_real_iso_prototype_directory(tmp_path):
    """GĐ2 end to end on the real 42-symbol ISO library.

    The encoder is untrained, so nothing here asserts recognition QUALITY -- it asserts
    that the stage turns the files on disk into a well-formed, self-describing
    directory: 42 unit-norm rows named by vocab.csv's fine_name, tagged with their
    coarse class, and carrying the encoder provenance needed to interpret them later.
    """
    cfg = _cfg(tmp_path)
    cfg.oneshot.encoder_ckpt = str(_write_arcface_ckpt(cfg, tmp_path / "enc.pt"))

    IsoPrototypePipeline(cfg).run()

    directory = PrototypeDirectory.load(cfg.oneshot.prototypes)
    expected = [r["fine_name"] for r in read_vocab(ISO_VOCAB_CSV)]
    assert directory.names == expected
    assert len(directory) == 42
    assert directory.dim == 32
    assert torch.allclose(directory.embeddings.norm(dim=1), torch.ones(42), atol=1e-5)
    # Every prototype knows its coarse family, so fgc_infer can restrict matching.
    assert set(directory.coarse) >= {"valve", "pump", "heat_exchanger", "blind_disc"}
    # Provenance travels with the file, not just the run log.
    assert directory.meta["encoder_ckpt"] == cfg.oneshot.encoder_ckpt
    assert directory.meta["backbone"] == "resnet18"
    assert directory.meta["imgsz"] == 32
    assert len(directory.meta["encoder_train_classes"]) == 6
    # _write_arcface_ckpt predates leave-classes-out (no excluded_classes key).
    assert directory.meta["encoder_excluded_classes"] == []
    assert directory.meta["template_views"] == {"scales": [1.0], "orientations": [0]}


@requires_iso_library
def test_fgc_groups_only_builds_the_confusable_families_subset(tmp_path):
    """The ablation knob: 22 of the 42 ISO symbols are flagged as needing fine-grained
    distinction (valve 9, check_valve 3, heat_exchanger 3, agitator 3, blind_disc 4)."""
    cfg = _cfg(tmp_path, fgc_groups_only=True)
    cfg.oneshot.encoder_ckpt = str(_write_arcface_ckpt(cfg, tmp_path / "enc.pt"))

    IsoPrototypePipeline(cfg).run()

    directory = PrototypeDirectory.load(cfg.oneshot.prototypes)
    assert len(directory) == 22
    assert set(directory.coarse) == {
        "valve",
        "check_valve",
        "heat_exchanger",
        "agitator",
        "blind_disc",
    }
    assert directory.meta["fgc_groups_only"] is True


@requires_iso_library
def test_coarse_filter_builds_a_per_group_directory(tmp_path):
    """The per-group route: the valve-group encoder backs only the valve-family
    prototypes (9 valve + 3 check_valve rows in the real vocab)."""
    cfg = _cfg(tmp_path, coarse_filter=["valve", "check_valve"])
    cfg.oneshot.encoder_ckpt = str(_write_arcface_ckpt(cfg, tmp_path / "enc.pt"))

    IsoPrototypePipeline(cfg).run()

    directory = PrototypeDirectory.load(cfg.oneshot.prototypes)
    assert set(directory.coarse) == {"valve", "check_valve"}
    assert len(directory) == 12
    assert directory.meta["coarse_filter"] == ["valve", "check_valve"]


@requires_iso_library
def test_multi_view_templates_are_recorded_and_still_unit_norm(tmp_path):
    """Widening the support-side views must change the recorded spec (so a directory
    always says how it was built) without breaking the unit-norm invariant that makes
    the cosine threshold comparable across configurations."""
    cfg = _cfg(tmp_path, template_views={"scales": [1.0, 1.25], "orientations": [0]})
    cfg.oneshot.encoder_ckpt = str(_write_arcface_ckpt(cfg, tmp_path / "enc.pt"))

    IsoPrototypePipeline(cfg).run()

    directory = PrototypeDirectory.load(cfg.oneshot.prototypes)
    assert directory.meta["template_views"] == {"scales": [1.0, 1.25], "orientations": [0]}
    assert torch.allclose(directory.embeddings.norm(dim=1), torch.ones(len(directory)), atol=1e-5)


def test_validate_requires_an_encoder_checkpoint(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(ValueError, match="oneshot.encoder_ckpt"):
        IsoPrototypePipeline(cfg).validate()


def test_validate_reports_a_missing_symbols_dir(tmp_path):
    cfg = _cfg(tmp_path, symbols_dir=str(tmp_path / "absent"))
    cfg.oneshot.encoder_ckpt = str(_write_arcface_ckpt(cfg, tmp_path / "enc.pt"))
    with pytest.raises(FileNotFoundError, match="symbols_dir"):
        IsoPrototypePipeline(cfg).validate()


@requires_iso_library
def test_input_norm_normalizes_the_templates_and_is_recorded(tmp_path):
    """oneshot.input_norm (PLAN_PHASE_C A4): every template goes through the same
    normalization the query crops get, and the directory says so."""
    from utils.dataset import build_eval_transform
    from utils.input_norm import ink_letterbox
    from utils.oneshot import ViewSpec, build_prototypes

    cfg = _cfg(tmp_path, input_norm="ink_letterbox", coarse_filter=["valve"])
    cfg.oneshot.encoder_ckpt = str(_write_arcface_ckpt(cfg, tmp_path / "enc.pt"))
    IsoPrototypePipeline(cfg).run()
    directory = PrototypeDirectory.load(cfg.oneshot.prototypes)
    assert directory.meta["input_norm"] == "ink_letterbox"

    encoder, _ = load_arcface_encoder(cfg, cfg.oneshot.encoder_ckpt)
    rows = read_vocab(ISO_VOCAB_CSV, coarse_filter=["valve"])
    templates, _ = load_templates(ISO_SYMBOLS_DIR, rows)
    tf = build_eval_transform(cfg)
    norm = build_prototypes(
        encoder, {n: ink_letterbox(t) for n, t in templates.items()}, tf, ViewSpec()
    )
    raw = build_prototypes(encoder, templates, tf, ViewSpec())
    assert torch.allclose(directory.embeddings, norm.embeddings, atol=1e-5)
    assert not torch.allclose(directory.embeddings, raw.embeddings, atol=1e-3)

    cfg_off = _cfg(tmp_path, coarse_filter=["valve"])
    cfg_off.oneshot.encoder_ckpt = cfg.oneshot.encoder_ckpt
    IsoPrototypePipeline(cfg_off).run()
    assert PrototypeDirectory.load(cfg_off.oneshot.prototypes).meta["input_norm"] is None
