"""eval.mode=oneshot end to end on a tiny world (PLAN B6b-1): protocols p1 / p2 / tau / p4.

The world: 8 DigitizePID-like classes whose crops are EXACT copies of the ISO template glyph
of their class (so a one-shot match is a cosine of ~1 and the expected name is knowable) or
random glyphs (OOV), a real untrained ArcFace encoder saved the way pid-fgc-train does, and
prototypes built from it with the provenance pid-iso-prototypes records. P4 adds a tiny
OPEN100-ISO-like COCO export: sheets with the template glyphs pasted at known boxes.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from pipeline.evaluation import EvaluationPipeline  # noqa: E402
from pipeline.oneshot_eval import (  # noqa: E402
    OneShotEvaluation,
    check_no_leak,
    rank_names,
    sample_per_class,
    validate_tau_classes,
)
from utils.classmap import GLOBAL_GROUP, IGNORE_CATEGORY  # noqa: E402
from utils.dataset import build_eval_transform  # noqa: E402
from utils.model import build_model, checkpoint_sha256, load_arcface_encoder  # noqa: E402
from utils.oneshot import Holdout, PrototypeDirectory, build_prototypes  # noqa: E402
from utils.oneshot_metrics import (  # noqa: E402
    assign_groups,
    assign_iso_groups,
    read_symbol_map,
    read_vocab,
)

_SYMBOL_MAP = """symbol_n,class_id,coarse,fgc_group,status,fine_name,reg_number
1,0,valve,yes,ISO,gate,A1
2,1,valve,yes,ISO,ball,A2
3,2,valve,yes,ISO,globe,A3
4,3,valve,yes,VARIANT,ball_variant,A2
5,4,valve,yes,VARIANT,butterfly_variant,A4
6,5,instrument,yes,OOV,bubble_a,
7,6,instrument,yes,OOV,bubble_b,
8,7,instrument,yes,UNSURE,odd,
"""
_VOCAB = """reg_number,coarse,fine_name,pid_group,fgc_group,desc
A1,valve,gate,valve,yes,
A2,valve,ball,valve,yes,
A3,valve,globe,valve,yes,
A4,valve,butterfly,valve,yes,
"""
_CLASSES_YAML = """
coarse: [valve, instrument]
fine:
  Symbol_1: valve
  Symbol_2: valve
  Symbol_3: valve
  Symbol_4: valve
  Symbol_5: valve
  Symbol_6: instrument
  Symbol_7: instrument
  Symbol_8: instrument
fgc_groups: [valve, instrument]
"""
_FULL = [f"Symbol_{i}" for i in range(1, 9)]
_LO = ["Symbol_1", "Symbol_2", "Symbol_4", "Symbol_5", "Symbol_6", "Symbol_8"]  # no h_in/h_out
_TEMPLATE_SEED = {"gate": 1, "ball": 2, "globe": 3, "butterfly": 4}
_CLASS_TEMPLATE = {"Symbol_1": "gate", "Symbol_2": "ball", "Symbol_3": "globe",
                   "Symbol_4": "ball", "Symbol_5": "butterfly"}  # fmt: skip
_OOV_SEED = {"Symbol_6": 61, "Symbol_7": 71, "Symbol_8": 81}
_VAL_SEL, _TEST = ["10", "11"], ["12", "13"]
_DIM = 32


def _glyph(seed):
    g = torch.Generator().manual_seed(seed)
    px = (torch.rand(30, 30, 3, generator=g) * 255).to(torch.uint8)
    return Image.fromarray(px.numpy(), mode="RGB")


def _cfg(tmp_path, protocol, **over):
    data = tmp_path / "data"
    (data / "canonical").mkdir(parents=True, exist_ok=True)
    (data / "canonical" / "classes.yaml").write_text(_CLASSES_YAML)
    (data / "canonical" / "manifest.csv").write_text("id,split\n10,val\n")
    (data / "canonical" / "annotations.coco.json").write_text("{}")
    (tmp_path / "symbol_map.csv").write_text(_SYMBOL_MAP)
    (tmp_path / "vocab.csv").write_text(_VOCAB)
    (tmp_path / "splits.json").write_text(json.dumps({"val_sel": _VAL_SEL, "test": _TEST}))
    cfg = OmegaConf.create(
        {
            "data_root": str(data),
            "run_dir": str(tmp_path / "runs"),
            "device": "cpu",
            "fgc": {
                "metric_learning": True,
                "arcface": {"m": 0.5, "s": 30.0, "embed_dim": _DIM},
                "backbone": "resnet18",
                "pretrained": False,
                "imgsz": 32,
                "exclude_classes": [],
                "aug": {"subpart": {"p": 0.0}},
            },
            "eval": {
                "mode": "oneshot",
                "split": "test",
                "image_subset": "all",
                "splits_file": str(tmp_path / "splits.json"),
                "oneshot": {"protocol": protocol},
            },
            "oneshot": {
                "encoder_ckpt": str(tmp_path / "enc.pt"),
                "prototypes": str(tmp_path / "protos.pt"),
                "vocab_csv": str(tmp_path / "vocab.csv"),
                "symbol_map": str(tmp_path / "symbol_map.csv"),
                "background": [255, 255, 255],
                "query_views": {"scales": [1.0], "orientations": [0]},
                "aggregate": "mean",
                "other_min_cosine": 0.5,
                "holdout": {"h_in": ["Symbol_3"], "h_out": ["Symbol_7"], "h_extra": []},
            },
        }
    )
    return OmegaConf.merge(cfg, OmegaConf.create(over))


def _encoder(cfg, fine, excluded=(), template_seed=_TEMPLATE_SEED):
    """Save an untrained ArcFace checkpoint and the prototypes built from it."""
    ckpt = cfg.oneshot.encoder_ckpt
    torch.manual_seed(0)
    model = build_model(cfg, GLOBAL_GROUP, len(fine))
    torch.save(
        {
            "model": model.state_dict(),
            "fine_classes": list(fine),
            "excluded_classes": list(excluded),
        },
        ckpt,
    )
    encoder, _ = load_arcface_encoder(cfg, ckpt, "cpu")
    meta = {
        "encoder_ckpt": ckpt, "backbone": "resnet18", "embed_dim": _DIM, "imgsz": 32,
        "encoder_train_classes": list(fine), "encoder_excluded_classes": list(excluded),
        "encoder_sha256": checkpoint_sha256(ckpt),
    }  # fmt: skip
    templates = {n: _glyph(s) for n, s in template_seed.items()}
    build_prototypes(encoder, templates, build_eval_transform(cfg), meta=meta).save(
        cfg.oneshot.prototypes
    )


def _crops(tmp_path, images=("10", "11", "12", "13"), splits=("val", "train")):
    """Crops named <image>_<n>.png: an exact template copy (or an OOV glyph) per class."""
    for split in splits:
        for cls in _FULL:
            d = tmp_path / "data" / "derived" / "crops_fgc" / split / cls
            d.mkdir(parents=True, exist_ok=True)
            seed = (
                _TEMPLATE_SEED[_CLASS_TEMPLATE[cls]] if cls in _CLASS_TEMPLATE else _OOV_SEED[cls]
            )
            for image in images if split == "val" else ("1", "2", "3"):
                _glyph(seed).save(d / f"{image}_0.png")


def _run(tmp_path, protocol, fine=_FULL, excluded=(), **over):
    cfg = _cfg(tmp_path, protocol, **over)
    _encoder(cfg, fine, excluded)
    _crops(tmp_path)
    EvaluationPipeline(cfg).run()
    out = json.loads((tmp_path / "runs" / "metrics.json").read_text())
    return cfg, out


def _csv(tmp_path, protocol):
    with (tmp_path / "runs" / f"oneshot_{protocol}_crops.csv").open() as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------ #
# p2
# ------------------------------------------------------------------ #


def test_p2_scores_s_and_v_classes_on_the_test_images_only(tmp_path):
    cfg, out = _run(tmp_path, "p2", eval={"image_subset": "test"})
    assert out["mode"] == "oneshot" and out["image_subset"] == "test"
    o = out["oneshot"]
    assert o["protocol"] == "p2" and o["image_subset"] == "test" and o["aggregate"] == "mean"
    assert o["encoder_sha256"] == checkpoint_sha256(cfg.oneshot.encoder_ckpt)
    assert o["prototypes"] == cfg.oneshot.prototypes and o["n_prototypes"] == 4
    assert o["query_views"]["scales"] == [1.0]

    rows = _csv(tmp_path, "p2")
    assert {r["image_id"] for r in rows} == set(_TEST) and {r["split"] for r in rows} == {"val"}
    # S1..3, V (Symbol_5), V_of_S (Symbol_4), OOV_seen 6 and 7; UNSURE Symbol_8 never scored
    assert len(rows) == 7 * 2 and o["n_crops"] == 14
    assert {r["class"] for r in rows} == {f"Symbol_{i}" for i in range(1, 8)}
    assert set(rows[0]) >= {"path", "image_id", "class", "group", "expected", "top1", "cos1",
                            "top2", "cos2", "top3", "cos3", "best_cos", "second_cos"}  # fmt: skip
    assert o["unsure_crops_excluded"] == {"Symbol_8": 2}

    pc = o["per_class"]
    assert all(
        pc[c]["top1"] == 1.0 and pc[c]["top3"] == 1.0 for c in ("Symbol_1", "Symbol_2", "Symbol_3")
    )
    assert pc["Symbol_5"]["group"] == "V" and pc["Symbol_5"]["expected"] == "butterfly"
    assert pc["Symbol_4"]["group"] == "V_of_S" and pc["Symbol_4"]["top1"] == 1.0
    assert pc["Symbol_6"]["group"] == "OOV_seen" and pc["Symbol_6"]["top1"] is None
    assert pc["Symbol_6"]["reject_rate_at_tau"] is not None
    assert o["groups"]["S"]["macro_top1"] == 1.0 and o["groups"]["V"]["macro_top1"] == 1.0
    assert o["groups"]["S"]["n_classes"] == 3 and o["groups"]["OOV_seen"]["n_classes"] == 2

    s = o["open_set"]
    assert s["n_known"] == 8 and s["n_unknown"] == 4  # S + V; V_of_S is not pooled
    assert 0.0 <= s["auroc"] <= 1.0 and 0.0 <= s["oscr_area"] <= 1.0
    assert set(o["open_set_by_group"]) == {"S", "V"}
    assert len(s["sweep"]["tau"]) == 101
    assert (tmp_path / "runs" / "oneshot_p2_sweep.png").exists()
    assert o["crops_csv"].endswith("oneshot_p2_crops.csv")


def test_p2_without_oov_crops_still_reports_the_known_side(tmp_path):
    cfg = _cfg(tmp_path, "p2", eval={"image_subset": "test"})
    _encoder(cfg, _FULL)
    _crops(tmp_path)
    for cls in ("Symbol_6", "Symbol_7"):
        for p in (tmp_path / "data" / "derived" / "crops_fgc" / "val" / cls).iterdir():
            p.unlink()
    EvaluationPipeline(cfg).run()
    o = json.loads((tmp_path / "runs" / "metrics.json").read_text())["oneshot"]
    assert o["open_set"]["auroc"] is None and o["open_set"]["oscr_area"] is None
    assert o["groups"]["S"]["macro_top1"] == 1.0


def test_p2_on_all_images_warns(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        _run(tmp_path, "p2")
    assert any("image_subset=all" in r.message for r in caplog.records)


def test_vote_runs_and_its_top1_is_the_matchers_choice(tmp_path):
    _, out = _run(
        tmp_path,
        "p2",
        eval={"image_subset": "test"},
        oneshot={
            "aggregate": "vote",
            "query_views": {"scales": [1.0, 1.25], "orientations": [0, 90]},
        },
    )
    assert (
        out["oneshot"]["aggregate"] == "vote" and len(out["oneshot"]["query_views"]["scales"]) == 2
    )
    rows = _csv(tmp_path, "p2")
    assert all(r["top1"] in {"gate", "ball", "globe", "butterfly"} for r in rows)


def test_rank_names_agrees_with_match_views_for_both_aggregates():
    g = torch.Generator().manual_seed(0)
    directory = PrototypeDirectory(
        names=list("abcde"),
        embeddings=torch.nn.functional.normalize(torch.randn(5, 16, generator=g)),
    )
    for _ in range(20):
        emb = torch.nn.functional.normalize(torch.randn(8, 16, generator=g))
        for agg in ("vote", "mean", "max"):
            ranked = rank_names(directory, emb, agg, 3)
            match = directory.match_views(emb, None, agg)
            assert ranked[0][0] == match.name and ranked[0][1] == pytest.approx(
                match.cosine, abs=1e-5
            )
            assert len({n for n, _ in ranked}) == 3
    with pytest.raises(ValueError, match="aggregate"):
        rank_names(directory, emb, "median")


def test_rank_names_lists_classes_once_when_a_class_has_two_forms():
    e = torch.eye(4)
    directory = PrototypeDirectory(names=["a", "b", "b@B", "c"], embeddings=e.clone())
    emb = torch.nn.functional.normalize(e[[2, 2, 1]] + 0.01, dim=1)
    for agg in ("vote", "mean", "max"):
        ranked = rank_names(directory, emb, agg, 3)
        assert ranked[0][0] == "b" and [n for n, _ in ranked] == list(
            dict.fromkeys(n for n, _ in ranked)
        )


# ------------------------------------------------------------------ #
# p1
# ------------------------------------------------------------------ #


def test_p1_scores_h_in_and_h_out_from_train_and_val(tmp_path):
    _, out = _run(tmp_path, "p1", fine=_LO, excluded=["Symbol_3", "Symbol_7"])
    o = out["oneshot"]
    assert o["protocol"] == "p1" and o["splits"] == ["train", "val"]
    rows = _csv(tmp_path, "p1")
    assert {r["class"] for r in rows} == {"Symbol_3", "Symbol_7"}
    assert {r["split"] for r in rows} == {"train", "val"}
    assert len(rows) == 2 * (4 + 3) and o["n_crops"] == 14
    assert {r["group"] for r in rows} == {"H_chon", "OOV_heldout"}
    assert {r["expected"] for r in rows if r["class"] == "Symbol_3"} == {"globe"}
    pc = o["per_class"]
    assert pc["Symbol_3"]["n"] == 7 and pc["Symbol_3"]["top1"] == 1.0
    assert pc["Symbol_7"]["top1"] is None and pc["Symbol_7"]["most_predicted"] in {
        "gate", "ball", "globe", "butterfly"
    }  # fmt: skip
    assert o["h_in"]["macro_top1"] == 1.0 and o["h_in"]["macro_top3"] == 1.0
    assert o["h_out"]["n_classes"] == 1
    assert o["open_set"]["n_known"] == 7 and o["open_set"]["n_unknown"] == 7
    assert o["open_set"]["other_at_tau"]["tau"] == 0.5
    assert 0.0 <= o["open_set"]["margin_auroc"] <= 1.0
    assert (tmp_path / "runs" / "oneshot_p1_sweep.png").exists()


def test_p1_raises_before_scoring_when_a_tested_class_is_in_the_checkpoint(tmp_path):
    cfg = _cfg(tmp_path, "p1")
    _encoder(cfg, _FULL)  # R1b-like: trained on h_in and h_out
    _crops(tmp_path)
    with pytest.raises(ValueError, match="trained on"):
        EvaluationPipeline(cfg).run()
    assert not (tmp_path / "runs" / "oneshot_p1_crops.csv").exists()
    with pytest.raises(ValueError, match="Symbol_3"):
        check_no_leak(["Symbol_3", "Symbol_7"], _FULL)
    check_no_leak(["Symbol_3", "Symbol_7"], _LO)  # fine


def test_p1_cannot_use_an_image_subset(tmp_path):
    cfg = _cfg(tmp_path, "p1", eval={"image_subset": "test"})
    _encoder(cfg, _LO, ["Symbol_3", "Symbol_7"])
    _crops(tmp_path)
    with pytest.raises(ValueError, match="image_subset"):
        EvaluationPipeline(cfg).run()


# ------------------------------------------------------------------ #
# tau
# ------------------------------------------------------------------ #


def test_tau_reads_s_classes_of_val_sel_and_never_the_own_template(tmp_path):
    _, out = _run(tmp_path, "tau", eval={"image_subset": "val_sel"})
    o = out["oneshot"]
    assert o["protocol"] == "tau" and o["classes"] == ["Symbol_1", "Symbol_2", "Symbol_3"]
    assert o["rule_a"]["n_crops"] == 3 * 2 and o["rule_a"]["n_classes"] == 3
    assert o["rule_a"]["crops_per_class"] == {"Symbol_1": 2, "Symbol_2": 2, "Symbol_3": 2}
    rows = _csv(tmp_path, "tau")
    assert {r["image_id"] for r in rows} == set(_VAL_SEL)
    for r in rows:
        assert float(r["best_cos"]) > 0.999  # the crop IS its template
        assert r["top1"] == r["expected"]
        # with its own template left out the crop cannot score ~1 any more
        assert float(r["pseudo_cos"]) < 0.999 and float(r["pseudo_cos_no_twins"]) < 0.999
    known = [float(r["best_cos"]) for r in rows]
    assert min(known) - 1e-5 <= o["rule_a"]["tau"] <= max(known) + 1e-5  # CSV rounds to 6 places
    assert o["rule_b"]["tau"] < min(known) - 1e-6 and 0.0 < o["rule_b"]["youden_j"] <= 1.0
    assert o["rule_b_no_twins"]["twin_cos"] == 0.95
    assert isinstance(o["rule_b_no_twins"]["twin_pairs"], list)
    assert "NOT written to config" in o["note"]
    m = o["margin_rule_a"]
    margin = [float(r["best_cos"]) - float(r["second_cos"]) for r in rows]
    assert min(margin) - 1e-5 <= m["delta"] <= max(margin) + 1e-5
    assert m["tpr"] == 0.95 and m["n_crops"] == 6
    assert m["known_accepted"] >= 0.95 - 1e-9
    assert m["known_accepted_with_rule_a_tau"] <= min(m["known_accepted"], 1.0)


def test_tau_twin_pairs_are_reported_and_dropped(tmp_path):
    """Two near-identical templates: the pair is listed, and the crop's pseudo-unknown no
    longer matches its twin."""
    cfg = _cfg(tmp_path, "tau", eval={"image_subset": "val_sel"})
    _encoder(cfg, _FULL)
    _crops(tmp_path)
    directory = PrototypeDirectory.load(cfg.oneshot.prototypes)
    emb = directory.embeddings.clone()
    emb[1] = torch.nn.functional.normalize(emb[0] + 0.01 * torch.randn(_DIM), dim=0)  # ball ~ gate
    PrototypeDirectory(directory.names, emb, directory.coarse, directory.meta).save(
        cfg.oneshot.prototypes
    )
    o = OneShotEvaluation(cfg, tmp_path / "out").run()
    pairs = {(p["class"], p["twin"]) for p in o["rule_b_no_twins"]["twin_pairs"]}
    assert ("Symbol_1", "ball") in pairs and ("Symbol_2", "gate") in pairs
    assert all(p["cos"] > 0.95 for p in o["rule_b_no_twins"]["twin_pairs"])
    by_class = {}
    for r in _csv_rows(tmp_path / "out" / "oneshot_tau_crops.csv"):
        by_class.setdefault(r["class"], r)
    # Symbol_1 without gate alone still hits its twin ball; without both it cannot
    assert float(by_class["Symbol_1"]["pseudo_cos"]) > float(
        by_class["Symbol_1"]["pseudo_cos_no_twins"]
    )


def _csv_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("subset", ["all", "test"])
def test_tau_refuses_anything_but_val_sel(tmp_path, subset):
    cfg = _cfg(tmp_path, "tau", eval={"image_subset": subset})
    _encoder(cfg, _FULL)
    _crops(tmp_path)
    with pytest.raises(ValueError, match="val_sel"):
        EvaluationPipeline(cfg).run()


def test_tau_on_the_leave_out_encoder_uses_only_its_four_style_s_classes(tmp_path):
    _, out = _run(tmp_path, "tau", fine=_LO, excluded=["Symbol_3", "Symbol_7"],
                  eval={"image_subset": "val_sel"})  # fmt: skip
    assert out["oneshot"]["classes"] == ["Symbol_1", "Symbol_2"]
    assert {r["class"] for r in _csv(tmp_path, "tau")} == {"Symbol_1", "Symbol_2"}


def test_tau_class_guard_refuses_held_out_classes():
    holdout = Holdout(h_in=["Symbol_3"], h_out=["Symbol_7"], h_extra=[])
    with pytest.raises(ValueError, match="held out"):
        validate_tau_classes(["Symbol_1", "Symbol_3"], _LO, holdout)
    with pytest.raises(ValueError, match="Symbol_7"):
        validate_tau_classes(["Symbol_7"], _LO, holdout)
    validate_tau_classes(["Symbol_1", "Symbol_3"], _FULL, holdout)  # seen by the full encoder


# ------------------------------------------------------------------ #
# ImageNet-only baseline (B4a): trained on no class
# ------------------------------------------------------------------ #


def _reference(tmp_path, fine, excluded=()):
    """A checkpoint read only for its class lists (eval.oneshot.groups_ckpt)."""
    path = tmp_path / "ref.pt"
    torch.save({"fine_classes": list(fine), "excluded_classes": list(excluded)}, path)
    return str(path)


def test_p1_on_an_encoder_trained_on_no_class_passes_the_leak_guard(tmp_path, caplog):
    """The scripts/make_imagenet_encoder.py layout: an empty head, every class declared
    excluded. Every tested class is unseen, so P1 runs; nothing warns about undeclared ones."""
    with caplog.at_level("WARNING"):
        _, out = _run(tmp_path, "p1", fine=[], excluded=_FULL)
    assert not any("not declared excluded" in r.message for r in caplog.records)
    o = out["oneshot"]
    assert o["encoder_classes"] == 0 and o["groups_checkpoint"] is None
    assert {r["group"] for r in _csv(tmp_path, "p1")} == {"H_chon", "OOV_heldout"}
    assert o["per_class"]["Symbol_3"]["top1"] == 1.0  # crops are template copies
    check_no_leak(["Symbol_3", "Symbol_7"], [])


def test_tau_on_an_encoder_trained_on_no_class_needs_groups_ckpt(tmp_path):
    cfg = _cfg(tmp_path, "tau", eval={"image_subset": "val_sel"})
    _encoder(cfg, [], _FULL)
    _crops(tmp_path)
    with pytest.raises(ValueError, match="no group-S class"):
        OneShotEvaluation(cfg, tmp_path / "out").run()


def test_tau_with_groups_ckpt_reads_the_leave_out_encoders_s_classes(tmp_path):
    """Same classes as the leave-out encoder's own tau run (test above: Symbol_1, Symbol_2)."""
    ref = _reference(tmp_path, _LO, ["Symbol_3", "Symbol_7"])
    _, out = _run(tmp_path, "tau", fine=[], excluded=_FULL,
                  eval={"image_subset": "val_sel", "oneshot": {"groups_ckpt": ref}})  # fmt: skip
    o = out["oneshot"]
    assert o["classes"] == ["Symbol_1", "Symbol_2"] and o["groups_checkpoint"] == ref
    assert o["encoder_classes"] == 0
    assert {r["class"] for r in _csv(tmp_path, "tau")} == {"Symbol_1", "Symbol_2"}


def test_tau_with_groups_ckpt_never_reads_a_held_out_class(tmp_path):
    """A full-vocabulary reference would put the P1 test class Symbol_3 into group S."""
    cfg = _cfg(tmp_path, "tau", eval={"image_subset": "val_sel"})
    cfg.eval.oneshot.groups_ckpt = _reference(tmp_path, _FULL)
    _encoder(cfg, [], _FULL)
    _crops(tmp_path)
    with pytest.raises(ValueError, match=r"\['Symbol_3'\], held out"):
        OneShotEvaluation(cfg, tmp_path / "out").run()


def test_groups_ckpt_does_not_weaken_the_p1_leak_guard(tmp_path):
    cfg = _cfg(tmp_path, "p1")
    cfg.eval.oneshot.groups_ckpt = _reference(tmp_path, _LO, ["Symbol_3", "Symbol_7"])
    _encoder(cfg, _FULL)  # the encoder actually loaded trained on the tested classes
    _crops(tmp_path)
    with pytest.raises(ValueError, match="trained on"):
        OneShotEvaluation(cfg, tmp_path / "out").run()


# ------------------------------------------------------------------ #
# validation and the closed-set path
# ------------------------------------------------------------------ #


def test_validate_names_the_missing_pieces(tmp_path):
    cfg = _cfg(tmp_path, "p2", oneshot={"encoder_ckpt": None})
    with pytest.raises(ValueError, match="encoder_ckpt"):
        EvaluationPipeline(cfg).validate()
    cfg = _cfg(tmp_path, "p2")
    _encoder(cfg, _FULL)
    for bad in ("p3", None):
        with pytest.raises(ValueError, match="protocol"):
            EvaluationPipeline(
                OmegaConf.merge(cfg, {"eval": {"oneshot": {"protocol": bad}}})
            ).validate()
    with pytest.raises(ValueError, match="holdout"):
        EvaluationPipeline(OmegaConf.merge(cfg, {"eval": {"image_subset": "holdout"}})).validate()
    assert EvaluationPipeline(cfg).validate() is True


def test_detection_modes_refuse_an_image_subset(tmp_path):
    cfg = _cfg(tmp_path, "p2", eval={"mode": "det_only", "image_subset": "test"})
    with pytest.raises(ValueError, match="crop-level"):
        EvaluationPipeline(cfg).validate()


def test_fgc_only_applies_the_image_subset_and_records_it(tmp_path):
    cfg = _cfg(
        tmp_path, "p2",
        eval={"mode": "fgc_only", "split": "val", "image_subset": "test", "topk": [1, 3]},
        fgc={"global_encoder": True, "ckpt_dir": str(tmp_path / "ckpt"),
             "route": {"other_min_cosine": 0.35}},
    )  # fmt: skip
    (tmp_path / "ckpt").mkdir()
    torch.manual_seed(0)
    model = build_model(cfg, GLOBAL_GROUP, len(_FULL))
    torch.save({"model": model.state_dict(), "fine_classes": _FULL, "excluded_classes": []},
               tmp_path / "ckpt" / f"{GLOBAL_GROUP}.pt")  # fmt: skip
    _crops(tmp_path)
    EvaluationPipeline(cfg).run()
    out = json.loads((tmp_path / "runs" / "metrics.json").read_text())
    assert out["image_subset"] == "test" and out["fgc"]["n_samples"] == 8 * 2

    with pytest.raises(ValueError, match="val"):
        EvaluationPipeline(OmegaConf.merge(cfg, {"eval": {"split": "train"}})).validate()
    cfg_all = OmegaConf.merge(cfg, {"eval": {"image_subset": "all"}})
    EvaluationPipeline(cfg_all).run()
    out = json.loads((tmp_path / "runs" / "metrics.json").read_text())
    assert out["image_subset"] == "all" and out["fgc"]["n_samples"] == 8 * 4
    assert np.isfinite(out["fgc"]["top1"])


# ------------------------------------------------------------------ #
# p4 (OPEN100-ISO)
# ------------------------------------------------------------------ #

_VOCAB_P4 = _VOCAB + "A5,pump,pump,equipment,no,\n"  # no DigitizePID class: H_tu_nhien
_P4_TEMPLATE_SEED = {**_TEMPLATE_SEED, "pump": 5}
_P4_VOCAB = list(_P4_TEMPLATE_SEED)  # gate, ball, globe, butterfly, pump
_P4_SEED = {**_P4_TEMPLATE_SEED, "other_iso": 61, "other_isa": 71, "other_generic": 81}
_P4_CATS = list(_P4_SEED)


def _p4_glyph(cat, sheet, degraded):
    """The glyph drawn for `cat`: its own template, except on degraded sheets (every vocab
    symbol drawn as the NEXT template, so none is named right) and for `gate` on the clean
    sheet 1 (drawn as `ball`)."""
    if cat in _P4_VOCAB and degraded:
        cat = _P4_VOCAB[(_P4_VOCAB.index(cat) + 1) % len(_P4_VOCAB)]
    elif cat == "gate" and sheet == 1:
        cat = "ball"
    return _glyph(_P4_SEED[cat])


def _p4_world(tmp_path, categories=None, image_keys=("sheet", "variant", "degraded"), ignore=False):
    """2 sheets x 2 variants x (clean, degraded); one instance per category per image. The box
    is the glyph shrunk by 2.5 px per side, so box + CROP_MARGIN_FRAC is the 30 px glyph.
    `ignore`: plus an IGNORE_CATEGORY category and one iscrowd region per image (B6d)."""
    data = tmp_path / "data" / "canonical"
    (data / "images").mkdir(parents=True, exist_ok=True)
    (tmp_path / "vocab.csv").write_text(_VOCAB_P4)
    images, anns = [], []
    for sheet in (0, 1):
        for variant in (0, 1):
            for degraded in (False, True):
                stem = f"iso{sheet:02d}_v{variant:03d}" + ("_deg" if degraded else "")
                canvas = Image.new("RGB", (10 + 40 * len(_P4_CATS), 50), "white")
                image_id = len(images) + 1
                for i, cat in enumerate(_P4_CATS):
                    canvas.paste(_p4_glyph(cat, sheet, degraded), (10 + 40 * i, 10))
                    box = [12.5 + 40 * i, 12.5, 25, 25]
                    anns.append({"id": len(anns) + 1, "image_id": image_id,
                                 "category_id": i + 1, "bbox": box})  # fmt: skip
                if ignore:
                    anns.append({"id": len(anns) + 1, "image_id": image_id, "iscrowd": 1,
                                 "category_id": len(_P4_CATS) + 1, "bbox": [0, 0, 5, 5],
                                 "ignore_kind": "flag"})  # fmt: skip
                canvas.save(data / "images" / f"{stem}.png")
                full = {"sheet": sheet, "variant": variant, "degraded": degraded}
                images.append({"id": image_id, "file_name": f"synthetic/{stem}.png",
                               **{k: full[k] for k in image_keys}})  # fmt: skip
    cats = [{"id": i + 1, "name": n} for i, n in enumerate(categories or _P4_CATS)]
    if ignore:
        cats.append({"id": len(cats) + 1, "name": IGNORE_CATEGORY, "fgc": False})
    coco = {"images": images, "annotations": anns, "categories": cats}
    (data / "annotations.coco.json").write_text(json.dumps(coco))


def _run_p4(tmp_path, fine=_FULL, excluded=(), ignore=False, **over):
    cfg = _cfg(tmp_path, "p4", **over)
    _encoder(cfg, fine, excluded, _P4_TEMPLATE_SEED)
    _p4_world(tmp_path, ignore=ignore)
    EvaluationPipeline(cfg).run()
    return cfg, json.loads((tmp_path / "runs" / "metrics.json").read_text())["oneshot"]


def _iso_groups(tmp_path, fine, excluded=()):
    (tmp_path / "symbol_map.csv").write_text(_SYMBOL_MAP)
    (tmp_path / "vocab.csv").write_text(_VOCAB_P4)
    vocab = read_vocab(tmp_path / "vocab.csv")
    classes = assign_groups(read_symbol_map(tmp_path / "symbol_map.csv"), vocab, fine, excluded)
    return assign_iso_groups(classes, [r["fine_name"] for r in vocab.values()])


def test_iso_groups_follow_the_checkpoint_classes(tmp_path):
    full = {"gate": "S", "ball": "S", "globe": "S", "butterfly": "V", "pump": "H_tu_nhien"}
    assert _iso_groups(tmp_path, _FULL) == full
    # leave-out: the held-out ISO class's template becomes H_chon
    assert _iso_groups(tmp_path, _LO, ["Symbol_3", "Symbol_7"]) == {**full, "globe": "H_chon"}
    # trained on no class: every mapped ISO template is H_chon, a VARIANT-only one V_held
    assert _iso_groups(tmp_path, [], _FULL) == {
        "gate": "H_chon", "ball": "H_chon", "globe": "H_chon", "butterfly": "V_held",
        "pump": "H_tu_nhien",
    }  # fmt: skip


def test_iso_group_s_wins_over_a_trained_variant_and_v_over_a_held_iso_class(tmp_path):
    # ball: ISO Symbol_2 held out, its VARIANT Symbol_4 trained -> V, not H_chon
    lo = [c for c in _FULL if c != "Symbol_2"]
    assert _iso_groups(tmp_path, lo, ["Symbol_2"])["ball"] == "V"


def test_p4_groups_clean_and_degraded_are_never_pooled(tmp_path):
    cfg, o = _run_p4(tmp_path)
    assert o["protocol"] == "p4" and o["primary"] == "degraded"
    assert o["other_min_cosine"] == 0.5 and o["max_per_class"] is None and o["seed"] == 0
    assert o["template_groups"] == {
        "gate": "S", "ball": "S", "globe": "S", "butterfly": "V", "pump": "H_tu_nhien"
    }  # fmt: skip
    assert o["group_classes"]["S"] == ["ball", "gate", "globe"]
    assert o["n_annotations"] == 64 and o["n_crops"] == 64

    rows = _csv(tmp_path, "p4")
    assert set(rows[0]) >= {"split", "path", "image_id", "class", "group", "expected", "top1",
                            "cos1", "best_cos", "second_cos", "sheet", "variant", "degraded",
                            "ann_id"}  # fmt: skip
    assert len({r["ann_id"] for r in rows}) == 64
    assert {(r["split"], r["degraded"]) for r in rows} == {("clean", "0"), ("degraded", "1")}
    assert {r["group"] for r in rows if r["class"].startswith("other")} == {
        "other_iso", "other_isa", "other_generic"
    }  # fmt: skip
    assert all(r["expected"] == "" for r in rows if r["class"].startswith("other"))

    clean, deg = o["clean"], o["degraded"]
    assert clean["n_crops"] == deg["n_crops"] == 32 and clean["n_images"] == 4
    # degraded sheets draw every vocab symbol as another template: all wrong; clean: only
    # gate on sheet 1. A pooled number would sit in between.
    assert {g: deg["groups"][g]["macro_top1"] for g in ("S", "V", "H_tu_nhien")} == {
        "S": 0.0, "V": 0.0, "H_tu_nhien": 0.0
    }  # fmt: skip
    assert clean["per_class"]["gate"]["top1"] == 0.5
    assert clean["groups"]["S"]["macro_top1"] == pytest.approx((0.5 + 1 + 1) / 3)
    assert clean["groups"]["V"]["macro_top1"] == 1.0
    assert clean["groups"]["H_tu_nhien"] == {
        "n_classes": 1, "n_crops": 4, "macro_top1": 1.0, "macro_top3": 1.0, "top1": 1.0,
        "top3": 1.0,
    }  # fmt: skip
    assert clean["groups"]["H_chon"]["n_crops"] == 0 and clean["groups"]["H_chon"]["top1"] is None
    assert clean["n_available"]["gate"] == 4
    assert (tmp_path / "runs" / "oneshot_p4_degraded_sweep.png").exists()
    assert (tmp_path / "runs" / "oneshot_p4_clean_sweep.png").exists()


def test_p4_reports_per_sheet_then_mean_and_std_over_sheets(tmp_path):
    _, o = _run_p4(tmp_path)
    clean = o["clean"]
    assert set(clean["per_sheet"]) == {"iso00", "iso01"}
    s0, s1 = (clean["per_sheet"][k]["groups"]["S"]["macro_top1"] for k in ("iso00", "iso01"))
    assert s0 == 1.0 and s1 == pytest.approx(2 / 3)
    ms = clean["sheet_mean_std"]["groups"]["S"]["macro_top1"]
    assert ms["n"] == 2 and ms["mean"] == pytest.approx(5 / 6)
    assert ms["std"] == pytest.approx(float(np.std([1.0, 2 / 3], ddof=1)))
    assert clean["per_sheet"]["iso00"]["n_crops"] == 16
    assert clean["sheet_mean_std"]["auroc"]["n"] == 2
    assert o["degraded"]["sheet_mean_std"]["groups"]["S"]["macro_top1"]["mean"] == 0.0


def test_p4_other_iso_is_the_positive_and_other_isa_is_reported_apart(tmp_path):
    _, o = _run_p4(tmp_path)
    for view in ("degraded", "clean"):
        b = o[view]
        assert b["open_set"]["n_known"] == 5 * 4 and b["open_set"]["n_unknown"] == 4
        assert b["open_set"]["other_at_tau"]["tau"] == 0.5
        assert 0.0 <= b["open_set"]["auroc"] <= 1.0 and "margin_auroc" in b["open_set"]
        assert b["other"]["other_iso"]["auroc"] == pytest.approx(b["open_set"]["auroc"])
        for name in ("other_isa", "other_generic"):
            assert b["other"][name]["n"] == 4
            assert 0.0 <= b["other"][name]["reject_rate_at_tau"] <= 1.0
            assert b["per_class"][name]["top1"] is None
        assert set(b["open_set_by_group"]) == {"S", "V", "H_tu_nhien"}


def test_p4_on_the_leave_out_and_imagenet_style_encoders(tmp_path):
    _, o = _run_p4(tmp_path, fine=_LO, excluded=["Symbol_3", "Symbol_7"])
    assert o["template_groups"]["globe"] == "H_chon"
    assert o["clean"]["groups"]["H_chon"]["n_classes"] == 1
    assert o["clean"]["groups"]["S"]["n_classes"] == 2

    _, o = _run_p4(tmp_path, fine=[], excluded=_FULL)
    assert o["encoder_classes"] == 0
    assert o["group_classes"] == {"S": [], "V": [], "H_chon": ["ball", "gate", "globe"],
                                  "V_held": ["butterfly"], "H_tu_nhien": ["pump"]}  # fmt: skip
    # with groups_ckpt the ImageNet encoder's rows carry R1a's groups
    ref = _reference(tmp_path, _LO, ["Symbol_3", "Symbol_7"])
    _, o = _run_p4(tmp_path, fine=[], excluded=_FULL, eval={"oneshot": {"groups_ckpt": ref}})
    assert o["groups_checkpoint"] == ref and o["template_groups"]["globe"] == "H_chon"
    assert o["template_groups"]["gate"] == "S"


def test_p4_max_per_class_samples_per_class_and_view_deterministically(tmp_path):
    _, o = _run_p4(tmp_path, eval={"oneshot": {"max_per_class": 1, "seed": 3}})
    assert o["max_per_class"] == 1 and o["seed"] == 3
    rows = _csv(tmp_path, "p4")
    assert len(rows) == len(_P4_CATS) * 2 and o["clean"]["n_available"]["gate"] == 4
    first = sorted(r["ann_id"] for r in rows)
    _run_p4(tmp_path, eval={"oneshot": {"max_per_class": 1, "seed": 3}})
    assert sorted(r["ann_id"] for r in _csv(tmp_path, "p4")) == first


def test_sample_per_class_is_seeded_and_independent_of_the_other_keys():
    ids = {("a", False): list(range(100)), ("a", True): list(range(100, 200)),
           ("b", False): [7, 3]}  # fmt: skip
    one = sample_per_class(ids, 5, 0)
    assert one == sample_per_class(dict(reversed(ids.items())), 5, 0)
    assert one[("b", False)] == [3, 7] and all(len(v) == 5 for k, v in one.items() if k[0] == "a")
    alone = {("a", False): ids[("a", False)]}
    assert one[("a", False)] == sample_per_class(alone, 5, 0)[("a", False)]
    assert one != sample_per_class(ids, 5, 1)
    assert sample_per_class(ids, None, 0)[("a", True)] == list(range(100, 200))


def test_p4_refuses_an_image_subset(tmp_path):
    cfg = _cfg(tmp_path, "p4", eval={"image_subset": "test"})
    _encoder(cfg, _FULL, (), _P4_TEMPLATE_SEED)
    _p4_world(tmp_path)
    with pytest.raises(ValueError, match="image_subset"):
        EvaluationPipeline(cfg).run()


def test_p4_refuses_a_coco_that_is_not_open100_iso(tmp_path):
    cfg = _cfg(tmp_path, "p4")
    _encoder(cfg, _FULL, (), _P4_TEMPLATE_SEED)
    _p4_world(tmp_path, categories=["Symbol_1", *_P4_CATS[1:]])  # DigitizePID-style name
    with pytest.raises(ValueError, match=r"\['Symbol_1'\]"):
        OneShotEvaluation(cfg, tmp_path / "out").run()
    _p4_world(tmp_path, image_keys=("sheet", "variant"))  # no `degraded`
    with pytest.raises(ValueError, match="degraded"):
        OneShotEvaluation(cfg, tmp_path / "out").run()
    assert not (tmp_path / "out" / "oneshot_p4_crops.csv").exists()


def test_p4_skips_ignore_regions(tmp_path):
    """B6d ignore regions (iscrowd, IGNORE_CATEGORY): never a crop, never known, never Other."""
    _, o = _run_p4(tmp_path, ignore=True)
    assert o["n_annotations"] == 64 and o["n_ignored"] == 8 and o["n_crops"] == 64
    rows = _csv(tmp_path, "p4")
    assert IGNORE_CATEGORY not in {r["class"] for r in rows}
    assert IGNORE_CATEGORY not in o["clean"]["n_available"]
    deg = o["degraded"]["open_set"]
    assert deg["n_known"] == 20 and deg["n_unknown"] == 4


# ------------------------------------------------------------------ #
# center (CL2N)
# ------------------------------------------------------------------ #

_CENTER_CLASSES = sorted(set(_FULL) - {"Symbol_3", "Symbol_7"})  # minus h_in / h_out


def _center_file(tmp_path, fine=_FULL, excluded=(), **over):
    cfg, out = _run(tmp_path, "center", fine=fine, excluded=excluded, **over)
    return cfg, out["oneshot"]


def _manual_center(cfg, classes, crops_per_class=None):
    from pipeline.oneshot_infer import load_oneshot_classifier

    clf = load_oneshot_classifier(cfg, build_eval_transform(cfg), "cpu")
    root = cfg.data_root + "/derived/crops_fgc/train"
    embs = []
    for cls in classes:
        files = sorted((Path(root) / cls).glob("*.png"))
        for f in files[:crops_per_class]:
            embs.append(clf.embed_crop(np.array(Image.open(f).convert("RGB"))))
    return torch.cat(embs).mean(0)


def test_center_is_the_mean_view_embedding_of_seen_train_crops_outside_the_holdout(tmp_path):
    from utils.oneshot import load_center

    cfg, o = _center_file(tmp_path)
    assert o["classes"] == _CENTER_CLASSES
    assert o["held_out_dropped"] == ["Symbol_3", "Symbol_7"]
    assert o["split"] == "train" and o["n_crops"] == 3 * len(_CENTER_CLASSES)
    assert o["center"] is None  # the run itself is not centered
    center, meta = load_center(o["center_file"])
    assert meta["encoder_sha256"] == checkpoint_sha256(cfg.oneshot.encoder_ckpt)
    assert torch.allclose(center, _manual_center(cfg, _CENTER_CLASSES), atol=1e-6)


def test_center_samples_max_per_class(tmp_path):
    _, o = _center_file(tmp_path, eval={"oneshot": {"max_per_class": 1}})
    assert o["n_crops"] == len(_CENTER_CLASSES)
    assert set(o["per_class"].values()) == {1}


def test_center_of_an_encoder_trained_on_no_class_needs_groups_ckpt(tmp_path):
    cfg = _cfg(tmp_path, "center")
    _encoder(cfg, [], _FULL)
    _crops(tmp_path)
    with pytest.raises(ValueError, match="groups_ckpt"):
        OneShotEvaluation(cfg, tmp_path / "out").run()
    ref = _reference(tmp_path, _LO, ["Symbol_3", "Symbol_7"])
    cfg.eval.oneshot.groups_ckpt = ref
    o = OneShotEvaluation(cfg, tmp_path / "out").run()
    assert o["classes"] == sorted(_LO) and o["held_out_dropped"] == []


def test_center_refuses_an_image_subset_and_a_centered_classifier(tmp_path):
    cfg, o = _center_file(tmp_path)
    cfg.eval.image_subset = "val_sel"
    with pytest.raises(ValueError, match="crops_fgc/train only"):
        OneShotEvaluation(cfg, tmp_path / "out").run()
    cfg.eval.image_subset = "all"
    cfg.oneshot.center = o["center_file"]
    with pytest.raises(ValueError, match="drop oneshot.center"):
        OneShotEvaluation(cfg, tmp_path / "out").run()


def test_a_centered_run_scores_cl2n_cosines(tmp_path):
    import torch.nn.functional as F

    from utils.oneshot import load_center

    cfg, o = _center_file(tmp_path, fine=_LO, excluded=["Symbol_3", "Symbol_7"])
    center, _ = load_center(o["center_file"])
    cfg.eval.oneshot.protocol = "p1"
    out = OneShotEvaluation(cfg, tmp_path / "raw").run()
    cfg.oneshot.center = o["center_file"]
    cen = OneShotEvaluation(cfg, tmp_path / "cen").run()
    assert out["center"] is None and cen["center"] == o["center_file"]

    protos = PrototypeDirectory.load(cfg.oneshot.prototypes)
    crop = np.array(_glyph(_OOV_SEED["Symbol_7"]))
    from pipeline.oneshot_infer import load_oneshot_classifier

    raw_clf = load_oneshot_classifier(OmegaConf.merge(cfg, {"oneshot": {"center": None}}),
                                      build_eval_transform(cfg), "cpu")  # fmt: skip
    q = F.normalize(F.normalize(raw_clf.embed_crop(crop) - center, dim=-1).mean(0), dim=0)
    p = F.normalize(protos.embeddings - center, dim=-1)
    expected = float((p @ q).max())
    with (tmp_path / "cen" / "oneshot_p1_crops.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if r["class"] == "Symbol_7"]
    assert rows and all(abs(float(r["best_cos"]) - expected) < 1e-5 for r in rows)
    with (tmp_path / "raw" / "oneshot_p1_crops.csv").open() as f:
        raw = [r for r in csv.DictReader(f) if r["class"] == "Symbol_7"]
    assert abs(float(raw[0]["best_cos"]) - expected) > 1e-4  # the space really changed


def test_a_center_built_with_another_encoder_is_refused(tmp_path):
    from pipeline.oneshot_infer import load_oneshot_classifier

    cfg, o = _center_file(tmp_path)
    _encoder(cfg, _LO, ["Symbol_3", "Symbol_7"])  # a different checkpoint at the same path
    cfg.oneshot.center = o["center_file"]
    with pytest.raises(ValueError, match="sha256 differs"):
        load_oneshot_classifier(cfg, build_eval_transform(cfg), "cpu")


def test_a_missing_center_file_is_named_before_loading(tmp_path):
    from pipeline.oneshot_infer import require_oneshot_inputs

    cfg = _cfg(tmp_path, "p1")
    _encoder(cfg, _LO)
    cfg.oneshot.center = str(tmp_path / "nope.pt")
    with pytest.raises(FileNotFoundError, match="oneshot.center"):
        require_oneshot_inputs(cfg)


# --------------------------------------------------------------------------- #
# oneshot.template_align (PLAN_PHASE_C §2E.3)
# --------------------------------------------------------------------------- #


def _aligned(tmp_path, align, **osh):
    cfg, o = _center_file(tmp_path, fine=_LO, excluded=["Symbol_3", "Symbol_7"])
    cfg.oneshot.center = o["center_file"]
    cfg = OmegaConf.merge(cfg, {"oneshot": {"template_align": align, **osh}})
    return cfg


def _pairs_file(tmp_path, cfg, names, views=None, nudge=0.0):
    import torch

    from utils.linear_align import save_pairs

    protos = PrototypeDirectory.load(cfg.oneshot.prototypes)
    rows = protos.embeddings[[protos.names.index(n) for n in names]] + nudge
    g = torch.Generator().manual_seed(0)
    views = views or [torch.randn(3, 2, protos.dim, generator=g) for _ in names]
    meta = {"input_norm": cfg.oneshot.get("input_norm", None)}
    return save_pairs(tmp_path / "pairs.pt", names, rows, views, meta), views


def test_domain_center_moves_only_the_prototypes(tmp_path):
    import torch.nn.functional as F

    from pipeline.oneshot_infer import load_oneshot_classifier

    cfg = _aligned(tmp_path, "domain_center")
    clf = load_oneshot_classifier(cfg, build_eval_transform(cfg), "cpu")
    raw = PrototypeDirectory.load(cfg.oneshot.prototypes).embeddings
    assert torch_allclose(clf.directory.embeddings, F.normalize(raw - raw.mean(0), dim=-1))
    assert clf.template_align == {"kind": "domain_center"} and clf.center is not None
    cfg.eval.oneshot.protocol = "p1"
    out = OneShotEvaluation(cfg, tmp_path / "f0a").run()
    assert out["template_align"] == {"kind": "domain_center"}


def test_pair_shift_adds_the_mean_pair_residual_computed_with_the_run_center(tmp_path):
    import torch
    import torch.nn.functional as F

    from pipeline.oneshot_infer import load_oneshot_classifier
    from utils.oneshot import load_center

    cfg = _aligned(tmp_path, "pair_shift")
    names = PrototypeDirectory.load(cfg.oneshot.prototypes).names[:2]
    path, views = _pairs_file(tmp_path, cfg, names)
    cfg.oneshot.template_pairs = str(path)
    clf = load_oneshot_classifier(cfg, build_eval_transform(cfg), "cpu")
    center, _ = load_center(cfg.oneshot.center)
    raw = PrototypeDirectory.load(cfg.oneshot.prototypes)
    x = F.normalize(raw.embeddings - center, dim=-1)
    y = []
    for v in views:
        q = F.normalize(F.normalize(v - center, dim=-1).mean(1), dim=-1)
        y.append(F.normalize(q.mean(0), dim=-1))
    b = (torch.stack(y) - x[:2]).mean(0)
    assert torch_allclose(clf.directory.embeddings, F.normalize(x + b, dim=-1))
    assert clf.template_align["pairs"] == names and clf.template_align["n_crops"] == [3, 3]
    assert abs(clf.template_align["shift_norm"] - float(b.norm())) < 1e-5


def test_pair_shift_refuses_pairs_of_other_templates(tmp_path):
    from pipeline.oneshot_infer import load_oneshot_classifier

    cfg = _aligned(tmp_path, "pair_shift")
    names = PrototypeDirectory.load(cfg.oneshot.prototypes).names[:2]
    path, _ = _pairs_file(tmp_path, cfg, names, nudge=0.01)
    cfg.oneshot.template_pairs = str(path)
    with pytest.raises(ValueError, match="differ from the prototypes"):
        load_oneshot_classifier(cfg, build_eval_transform(cfg), "cpu")


def test_template_align_needs_a_center_a_known_kind_and_its_pairs(tmp_path):
    from pipeline.oneshot_infer import load_oneshot_classifier, require_oneshot_inputs

    cfg = _aligned(tmp_path, "domain_center")
    no_center = OmegaConf.merge(cfg, {"oneshot": {"center": None}})
    with pytest.raises(ValueError, match="needs oneshot.center"):
        load_oneshot_classifier(no_center, build_eval_transform(cfg), "cpu")
    bad = OmegaConf.merge(cfg, {"oneshot": {"template_align": "rotate"}})
    with pytest.raises(ValueError, match="template_align"):
        require_oneshot_inputs(bad)
    shift = OmegaConf.merge(cfg, {"oneshot": {"template_align": "pair_shift"}})
    with pytest.raises(ValueError, match="template_pairs"):
        require_oneshot_inputs(shift)


def torch_allclose(a, b, atol=1e-6):
    import torch

    return torch.allclose(a.float(), b.float(), atol=atol)
