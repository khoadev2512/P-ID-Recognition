"""eval.mode=oneshot end to end on a tiny world (PLAN B6b-1): protocols p1 / p2 / tau.

The world: 8 DigitizePID-like classes whose crops are EXACT copies of the ISO template glyph
of their class (so a one-shot match is a cosine of ~1 and the expected name is knowable) or
random glyphs (OOV), a real untrained ArcFace encoder saved the way pid-fgc-train does, and
prototypes built from it with the provenance pid-iso-prototypes records.
"""

from __future__ import annotations

import csv
import json

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
    validate_tau_classes,
)
from utils.classmap import GLOBAL_GROUP  # noqa: E402
from utils.dataset import build_eval_transform  # noqa: E402
from utils.model import build_model, checkpoint_sha256, load_arcface_encoder  # noqa: E402
from utils.oneshot import Holdout, PrototypeDirectory, build_prototypes  # noqa: E402

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


def _encoder(cfg, fine, excluded=()):
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
    templates = {n: _glyph(s) for n, s in _TEMPLATE_SEED.items()}
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
        for agg in ("vote", "mean"):
            ranked = rank_names(directory, emb, agg, 3)
            match = directory.match_views(emb, None, agg)
            assert ranked[0][0] == match.name and ranked[0][1] == pytest.approx(
                match.cosine, abs=1e-5
            )
            assert len({n for n, _ in ranked}) == 3
    with pytest.raises(ValueError, match="aggregate"):
        rank_names(directory, emb, "max")


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
