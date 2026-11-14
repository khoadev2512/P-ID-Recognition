"""Open-set metrics, class groups and the image-subset filter of the one-shot evaluation
(PLAN B6b-1). Pure numpy except the image-subset tests (utils.dataset needs torchvision).

The end-to-end runs of eval.mode=oneshot are in tests/test_oneshot_eval.py.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from utils.oneshot_metrics import (
    OTHER_NAMES,
    assign_groups,
    assign_iso_groups,
    auroc,
    class_stats,
    gated_other_prf,
    margin_auroc,
    margin_rule_a,
    margins,
    mean_std,
    open_set_summary,
    oscr,
    other_auroc,
    other_prf,
    read_symbol_map,
    read_vocab,
    tau_rule_a,
    tau_rule_b,
    tau_sweep,
)

_ROOT = Path(__file__).resolve().parents[1]
_RNG = np.random.default_rng(0)


# ------------------------------------------------------------------ #
# AUROC
# ------------------------------------------------------------------ #


def test_auroc_is_one_on_perfect_separation_and_zero_on_the_reverse():
    assert auroc([3.0, 4.0, 5.0], [0.0, 1.0, 2.0]) == 1.0
    assert auroc([0.0, 1.0], [2.0, 3.0]) == 0.0


def test_auroc_is_half_on_identical_distributions_and_ties_count_half():
    x = _RNG.normal(size=200)
    assert auroc(x, x.copy()) == pytest.approx(0.5, abs=0.01)
    assert auroc([1.0, 1.0], [1.0, 1.0, 1.0]) == 0.5
    assert auroc([2.0, 1.0], [1.0]) == pytest.approx(0.75)  # (1 win + 1 tie) / 2


def test_auroc_matches_the_pairwise_definition():
    pos, neg = _RNG.integers(0, 6, 40), _RNG.integers(0, 6, 55)
    pairs = (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
    assert auroc(pos, neg) == pytest.approx(pairs)


def test_other_auroc_scores_unknown_as_low_cosine():
    assert other_auroc(known_cos=[0.9, 0.8, 0.7], unknown_cos=[0.3, 0.2]) == 1.0
    assert other_auroc(known_cos=[0.3, 0.2], unknown_cos=[0.9, 0.8]) == 0.0


# ------------------------------------------------------------------ #
# OSCR
# ------------------------------------------------------------------ #


def test_oscr_area_is_one_when_separated_and_named_perfectly():
    fpr, ccr, area = oscr([True] * 3, [0.9, 0.8, 0.7], [0.3, 0.2])
    assert area == pytest.approx(1.0)
    assert (fpr[0], ccr[0]) == (0.0, 0.0) and (fpr[-1], ccr[-1]) == (1.0, 1.0)


def test_oscr_ignores_unnamed_known_crops_but_auroc_does_not():
    """Separation is perfect (AUROC 1) but half the known crops are named wrong: the CCR
    never exceeds 0.5, so the OSCR area is 0.5."""
    known_cos, unknown_cos = [0.9, 0.8, 0.7, 0.6], [0.3, 0.2]
    _, ccr, area = oscr([True, False, True, False], known_cos, unknown_cos)
    assert other_auroc(known_cos, unknown_cos) == 1.0
    assert ccr.max() == pytest.approx(0.5) and area == pytest.approx(0.5)


def test_oscr_curve_is_monotone_and_handles_ties_between_known_and_unknown():
    fpr, ccr, area = oscr([True, True, False], [0.5, 0.5, 0.4], [0.5, 0.1])
    assert np.all(np.diff(fpr) >= 0) and np.all(np.diff(ccr) >= 0)
    assert 0.0 <= area <= 1.0


def test_oscr_length_mismatch_is_an_error():
    with pytest.raises(ValueError, match="known_correct"):
        oscr([True], [0.5, 0.6], [0.1])


# ------------------------------------------------------------------ #
# tau sweep and the two rules
# ------------------------------------------------------------------ #


def test_sweep_is_monotone_in_tau():
    known = _RNG.uniform(0.3, 1.0, 300)
    correct = known > 0.5
    unknown = _RNG.uniform(0.0, 0.8, 200)
    s = tau_sweep(correct, known, unknown, np.linspace(0, 1, 51))
    assert np.all(np.diff(s["known_acc"]) <= 1e-12)
    assert np.all(np.diff(s["known_reject_rate"]) >= -1e-12)
    assert np.all(np.diff(s["unknown_reject_rate"]) >= -1e-12)
    assert s["known_reject_rate"][0] == 0.0 and s["known_reject_rate"][-1] == 1.0
    assert s["known_acc"][0] == pytest.approx(correct.mean())
    assert s["known_acc_accepted"][-1] is None  # nothing passes tau=1.0


def test_rule_a_accepts_95_percent_of_the_known_crops():
    known = _RNG.uniform(0.2, 1.0, 1000)
    tau = tau_rule_a(known, 0.95)
    assert (known >= tau).mean() == pytest.approx(0.95, abs=0.002)
    assert (known >= tau + 1e-9).mean() < 0.95  # the largest such tau
    assert tau_rule_a([0.5], 0.95) == 0.5
    assert tau_rule_a([0.1, 0.9], 1.0) == 0.1


def test_rule_b_lands_between_known_and_simulated_unknown():
    tau, j = tau_rule_b([0.9, 0.8, 0.85], [0.3, 0.2, 0.4])
    assert 0.4 < tau <= 0.8 and j == pytest.approx(1.0)


def test_rule_b_is_pushed_up_by_a_pseudo_unknown_that_looks_known():
    """The sensitivity PLAN B8 names: a near-twin prototype makes the simulated unknown score
    like the known crops, so tau rises; without it the optimum drops."""
    known = [0.9, 0.85, 0.8, 0.95]
    with_twin = tau_rule_b(known, [0.93, 0.92, 0.91, 0.2])[0]
    without = tau_rule_b(known, [0.3, 0.2, 0.25, 0.1])[0]
    assert with_twin > without


def test_other_prf_at_tau():
    r = other_prf(known_cos=[0.9, 0.4], unknown_cos=[0.3, 0.2, 0.6], tau=0.5)
    assert (r["tp"], r["fp"], r["fn"]) == (2, 1, 1)
    assert r["precision"] == pytest.approx(2 / 3) and r["recall"] == pytest.approx(2 / 3)
    assert r["f1"] == pytest.approx(2 / 3)


def test_margin_is_top1_minus_top2_and_infinite_without_a_second_prototype():
    m = margins([0.9, 0.7, 0.6], [0.5, 0.69, float("nan")])
    assert m[0] == pytest.approx(0.4) and m[1] == pytest.approx(0.01) and m[2] == np.inf


def test_margin_rule_a_is_rule_a_on_the_margin_and_off_when_that_margin_is_infinite():
    known = _RNG.uniform(0.0, 0.3, 1000)
    delta = margin_rule_a(known, 0.95)
    assert delta == tau_rule_a(known, 0.95)
    assert (known >= delta).mean() == pytest.approx(0.95, abs=0.002)
    assert margin_rule_a([np.inf, np.inf, 0.2], 0.5) == 0.0  # never "reject everything finite"


def test_margin_auroc_scores_unknown_as_small_margin():
    assert margin_auroc([0.3, 0.2], [0.01, 0.02]) == 1.0
    assert margin_auroc([0.01], [0.3]) == 0.0


def test_gated_prf_counts_any_gate_and_other_prf_is_the_cosine_gate():
    """A near unknown (high cosine, tiny margin) passes the cosine gate and is caught by the
    OR-gate; the known crop with a wide margin stays accepted."""
    known_cos, known_m = np.array([0.9, 0.8]), np.array([0.2, 0.15])
    unk_cos, unk_m = np.array([0.95, 0.3]), np.array([0.005, 0.1])
    tau, delta = 0.5, 0.05
    cos_only = other_prf(known_cos, unk_cos, tau)
    both = gated_other_prf((known_cos < tau) | (known_m < delta), (unk_cos < tau) | (unk_m < delta))
    assert cos_only["recall"] == 0.5 and both["recall"] == 1.0 and both["fp"] == 0
    assert {k: v for k, v in cos_only.items() if k != "tau"} == gated_other_prf(
        known_cos < tau, unk_cos < tau
    )


# ------------------------------------------------------------------ #
# empty unknown set
# ------------------------------------------------------------------ #


def test_an_empty_unknown_set_gives_none_with_a_warning_and_no_crash(caplog):
    with caplog.at_level(logging.WARNING):
        assert other_auroc([0.9, 0.8], []) is None
        assert auroc([], [1.0]) is None
        assert oscr([True, True], [0.9, 0.8], []) is None
        assert other_prf([0.9], [], 0.5) is None
        assert tau_rule_b([0.9, 0.8], []) is None
        summary = open_set_summary([True, False], [0.9, 0.4], [], 0.5, [0.0, 0.5, 1.0])
    assert any("unknown" in r.message for r in caplog.records)
    assert summary["auroc"] is None and summary["oscr_area"] is None
    assert summary["other_at_tau"] is None and summary["n_unknown"] == 0
    assert summary["sweep"]["unknown_reject_rate"] == [None, None, None]
    assert summary["sweep"]["known_acc"][0] == 0.5


def test_a_known_set_is_required_by_the_tau_rules():
    with pytest.raises(ValueError):
        tau_rule_a([])
    with pytest.raises(ValueError):
        tau_rule_b([], [0.1])


def test_class_stats_top1_top3_and_most_predicted():
    names = [["a", "b", "c"], ["b", "a", "c"], ["c", "d", "a"], ["a", "x", "y"]]
    s = class_stats(names, "a", [0.9, 0.8, 0.7, 0.6])
    assert s["n"] == 4 and s["top1"] == 0.5 and s["top3"] == 1.0
    assert s["most_predicted"] == "a" and s["mean_best_cos"] == pytest.approx(0.75)
    u = class_stats(names, None, [0.9, 0.8, 0.7, 0.6])
    assert u["top1"] is None and u["top3"] is None


# ------------------------------------------------------------------ #
# class groups, against the real symbol_map + vocab + config
# ------------------------------------------------------------------ #


@pytest.fixture(scope="module")
def real():
    from utils.oneshot import holdout_classes

    symbol_map = read_symbol_map(_ROOT / "docs" / "symbol_mapping" / "symbol_map.csv")
    vocab = read_vocab(_ROOT / "data" / "raw" / "iso_symbols" / "vocab.csv")
    holdout = holdout_classes(OmegaConf.load(_ROOT / "configs" / "config.yaml"))
    return symbol_map, vocab, holdout


def _by_group(groups):
    out: dict[str, set[str]] = {}
    for info in groups.values():
        out.setdefault(info.group, set()).add(info.name)
    return out


def test_r1b_groups_match_the_plan_table(real):
    symbol_map, vocab, holdout = real
    full = [f"Symbol_{i}" for i in range(1, 33)]
    assert set(full) == set(symbol_map)
    groups = assign_groups(symbol_map, vocab, full, [], holdout)
    by = _by_group(groups)
    assert len(by["S"]) == 10 and len(by["V"]) == 6
    assert {groups[n].template for n in by["S"]} == {
        "valve_gate", "valve_globe", "valve_general", "valve_ball", "valve_needle",
        "open_disc", "interchangeable_disc", "interchangeable_disc_open",
        "flanged_connection", "flow_direction",
    }  # fmt: skip
    assert {groups[n].template for n in by["V"]} == {
        "valve_butterfly", "valve_control", "check_valve_general", "reducer", "hx_coil",
        "piping_heated_cooled_insulated",
    }  # fmt: skip
    # drawing conventions of S symbols are neither S nor V
    assert by["V_of_S"] == {"Symbol_2", "Symbol_12", "Symbol_13"}
    assert {groups[n].template for n in by["V_of_S"]} <= {"valve_ball", "valve_general"}
    assert len(by["OOV_seen"]) == 11 and len(by["UNSURE"]) == 2
    assert not by.get("H_chon") and not by.get("h_extra") and not by.get("OOV_heldout")
    assert all(groups[n].template is None for n in by["OOV_seen"] | by["UNSURE"])


def test_r1a_groups_chia_a(real):
    symbol_map, vocab, holdout = real
    full = [f"Symbol_{i}" for i in range(1, 33)]
    r1a = [c for c in full if c not in holdout.excluded]
    assert len(r1a) == 19
    groups = assign_groups(symbol_map, vocab, r1a, holdout.excluded, holdout)
    by = _by_group(groups)
    assert by["S"] == {"Symbol_1", "Symbol_17", "Symbol_21", "Symbol_24"}
    assert by["H_chon"] == set(holdout.h_in)
    assert by["OOV_heldout"] == set(holdout.h_out)
    assert by["h_extra"] == set(holdout.h_extra)
    assert len(by["V"]) == 6 and not by.get("V_of_S")
    assert len(by["OOV_seen"]) == 7 and len(by["UNSURE"]) == 2
    assert sum(len(v) for v in by.values()) == 32
    # an h_in class's expected template is still its ISO fine_name
    assert groups["Symbol_3"].template == "valve_globe"


def test_a_class_missing_without_being_declared_excluded_warns(real, caplog):
    symbol_map, vocab, _ = real
    with caplog.at_level(logging.WARNING):
        groups = assign_groups(symbol_map, vocab, [f"Symbol_{i}" for i in range(2, 33)])
    assert groups["Symbol_1"].group == "H_chon"
    assert any("Symbol_1" in r.message for r in caplog.records)


def test_an_unknown_status_is_an_error(real):
    _, vocab, _ = real
    with pytest.raises(ValueError, match="status"):
        assign_groups({"Symbol_1": {"status": "TODO"}}, vocab, ["Symbol_1"])


def _iso_by_group(real, fine, excluded, names=None):
    symbol_map, vocab, holdout = real
    templates = [r["fine_name"] for r in vocab.values()]
    iso = assign_iso_groups(assign_groups(symbol_map, vocab, fine, excluded, holdout), templates)
    out: dict[str, set[str]] = {}
    for name, g in iso.items():
        if names is None or name in names:
            out.setdefault(g, set()).add(name)
    return out


def _open100_names():
    """Vocab categories with instances on OPEN100-ISO (skip when the export is not local)."""
    path = _ROOT / "data" / "iso_open100" / "canonical" / "annotations.coco.json"
    if not path.exists():
        pytest.skip("OPEN100-ISO export not present")
    coco = json.loads(path.read_text())
    used = {a["category_id"] for a in coco["annotations"]}
    names = {c["name"] for c in coco["categories"] if c["id"] in used}
    return names - set(OTHER_NAMES)


def test_iso_groups_match_the_plan_table(real):
    """PLAN "Lớp ISO thuộc nhóm nào": R1b 10 S / 6 V / 26 H tự nhiên over the 42 templates; on
    OPEN100-ISO 9 S (no open_disc) / 4 V / 17 H tự nhiên. R1a: its h_in templates are H_chon.
    ImageNet-only (trained on no class): no S / V."""
    symbol_map, vocab, holdout = real
    full = [f"Symbol_{i}" for i in range(1, 33)]
    by = _iso_by_group(real, full, [])
    assert {g: len(v) for g, v in by.items()} == {"S": 10, "V": 6, "H_tu_nhien": 26}
    r1a = [c for c in full if c not in holdout.excluded]
    by = _iso_by_group(real, r1a, holdout.excluded)
    groups = assign_groups(symbol_map, vocab, full, [], holdout)
    assert by["H_chon"] == {groups[c].template for c in holdout.h_in}
    assert by["S"] == {"valve_gate", "open_disc", "flanged_connection", "flow_direction"}
    assert len(by["V"]) == 6 and len(by["H_tu_nhien"]) == 26
    by = _iso_by_group(real, [], full)
    assert {g: len(v) for g, v in by.items()} == {"H_chon": 10, "V_held": 6, "H_tu_nhien": 26}

    names = _open100_names()
    assert len(names) == 30
    by = _iso_by_group(real, full, [], names)
    assert {g: len(v) for g, v in by.items()} == {"S": 9, "V": 4, "H_tu_nhien": 17}
    assert "open_disc" not in names


def test_mean_std_is_over_the_units_that_have_a_value():
    assert mean_std([1.0, None, 0.5]) == {
        "mean": 0.75, "std": pytest.approx(float(np.std([1.0, 0.5], ddof=1))), "n": 2
    }  # fmt: skip
    assert mean_std([0.3]) == {"mean": 0.3, "std": None, "n": 1}
    assert mean_std([None]) == {"mean": None, "std": None, "n": 0}


# ------------------------------------------------------------------ #
# image-subset filter
# ------------------------------------------------------------------ #

_SPLITS = {"val_sel": ["10", "11"], "test": ["12", "13"]}


def _crops(tmp_path):
    from PIL import Image

    root = tmp_path / "crops" / "val"
    for cls in ("Symbol_1", "Symbol_2"):
        (root / cls).mkdir(parents=True)
        for img_id in ("10", "11", "12", "13", "120"):
            for n in (0, 1):
                Image.new("RGB", (8, 8)).save(root / cls / f"{img_id}_{n}.png")
    return root


def _subset_cfg(tmp_path, subset, extra=None):
    f = tmp_path / "splits.json"
    f.write_text(json.dumps(_SPLITS))
    return OmegaConf.create(
        {"eval": {"image_subset": subset, "splits_file": str(f), **(extra or {})}}
    )


def test_image_subset_keeps_only_the_listed_images(tmp_path):
    pytest.importorskip("torchvision")
    from utils.dataset import crop_image_id, image_subset_ids, scan_crops

    root = _crops(tmp_path)
    ids = image_subset_ids(_subset_cfg(tmp_path, "test"), "val")
    assert ids == {"12", "13"}
    samples, _ = scan_crops(root, ["Symbol_1", "Symbol_2"], ids)
    assert len(samples) == 2 * 2 * 2  # 2 classes x 2 images x 2 crops; image 120 is not 12
    assert {crop_image_id(p.name) for p, _ in samples} == {"12", "13"}
    assert len(scan_crops(root, ["Symbol_1", "Symbol_2"], None)[0]) == 2 * 5 * 2


def test_image_subset_all_is_no_filter_whatever_the_split(tmp_path):
    pytest.importorskip("torchvision")
    from utils.dataset import image_subset_ids

    assert image_subset_ids(_subset_cfg(tmp_path, "all"), "val") is None
    assert image_subset_ids(_subset_cfg(tmp_path, "all"), "train") is None
    assert image_subset_ids(OmegaConf.create({"eval": {}}), "test") is None


def test_image_subset_refuses_a_split_other_than_val_and_unknown_names(tmp_path):
    pytest.importorskip("torchvision")
    from utils.dataset import image_subset_ids

    for split in ("train", "test"):
        with pytest.raises(ValueError, match="val"):
            image_subset_ids(_subset_cfg(tmp_path, "val_sel"), split)
    with pytest.raises(ValueError, match="holdout"):
        image_subset_ids(_subset_cfg(tmp_path, "holdout"), "val")


def test_the_real_splits_file_partitions_the_val_images():
    pytest.importorskip("torchvision")
    from utils.dataset import image_subset_ids

    cfg = lambda s: OmegaConf.create({"eval": {"image_subset": s}})  # noqa: E731
    sel, test = image_subset_ids(cfg("val_sel"), "val"), image_subset_ids(cfg("test"), "val")
    assert len(sel) == len(test) == 50 and not sel & test
