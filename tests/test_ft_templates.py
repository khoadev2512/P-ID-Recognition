"""scripts/ft_templates.py (PLAN_PHASE_C §2L): clusters, folds, WiSE-FT, LOCO assembly, the rule,
reproducible training samples."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import ft_templates as ft  # noqa: E402

from utils.oneshot import ViewSpec  # noqa: E402
from utils.template_variants import PipeAxis  # noqa: E402


def test_components_are_the_connected_pieces_above_the_threshold():
    names = ["a", "b", "c", "d", "e"]
    sim = torch.eye(5)
    sim[0, 1] = sim[1, 0] = 0.9  # a-b
    sim[1, 2] = sim[2, 1] = 0.86  # b-c -> a, b, c chained
    sim[3, 4] = sim[4, 3] = 0.84  # below 0.85: no edge
    assert ft.components(names, sim) == [["a", "b", "c"]]


def test_folds_cover_disjoint_clusters_and_k1_is_always_trained():
    k2, k345 = set(ft.held_out("k2")), set(ft.held_out("k345"))
    assert not k2 & k345
    assert k2 == set(ft.CLUSTERS["K2"]) and len(k345) == 8
    assert not set(ft.CLUSTERS["K1"]) & (k2 | k345)
    items = [("valve_butterfly", 1), ("valve_butterfly", 2), ("valve_ball", 3), ("blind_disc", 4)]
    assert ft.train_items(items, "k2") == [("valve_ball", 3), ("blind_disc", 4)]
    assert len(ft.train_items(items, None)) == 4


def test_wise_interpolates_float_tensors_only():
    a = {"w": torch.zeros(2), "n": torch.tensor([1])}
    b = {"w": torch.ones(2), "n": torch.tensor([7])}
    m = ft.wise(a, b, 0.5)
    assert torch.allclose(m["w"], torch.full((2,), 0.5)) and m["n"].item() == 7
    assert torch.equal(ft.wise(a, b, 1.0)["w"], b["w"])
    with pytest.raises(ValueError):
        ft.wise(a, {"w": b["w"]}, 0.5)


def test_assemble_takes_each_class_from_the_fold_that_held_it_out():
    f1 = pd.DataFrame({"ann_id": [1, 2], "class": ["x", "y"], "top1": ["x", "x"]})
    f2 = pd.DataFrame({"ann_id": [1, 2], "class": ["x", "y"], "top1": ["y", "y"]})
    out = ft.assemble({"k2": f1, "k345": f2}, {"k2": ["x"], "k345": ["y"]})
    assert out.set_index("class").top1.to_dict() == {"x": "x", "y": "y"}


def _ok():
    return {
        "d_macro": 0.06,
        "ci": (0.01, 0.1),
        "d_per_class": {"a": 0.2, "b": -0.05},
        "p1": {"k2": {"d_top1": 0.0, "d_auroc": 0.0}, "k345": {"d_top1": -0.005, "d_auroc": 0}},
        "viso": {"k2": {"a": 0.0}, "k345": {"valve_three_way": 0.0}},
    }


def test_rule_passes_only_when_every_check_holds():
    assert ft.decide(**_ok())["pass"]
    for key, bad in (
        ("d_macro", 0.04),
        ("ci", (-0.01, 0.1)),
        ("d_per_class", {"a": 0.3, "b": -0.11}),
        ("p1", {"k2": {"d_top1": -0.02, "d_auroc": 0.0}}),
        ("p1", {"k2": {"d_top1": 0.0, "d_auroc": -0.02}}),
        ("viso", {"k345": {"valve_three_way": -1.0}}),  # a template thrown out: guard (a)
    ):
        args = _ok() | {key: bad}
        assert not ft.decide(**args)["pass"], key


def test_sample_is_reproducible_and_one_view():
    img = Image.new("RGB", (60, 30), "white")
    ImageDraw.Draw(img).rectangle((10, 8, 50, 22), outline="black", width=2)
    item = ("x", img, [PipeAxis("L", 0.5, 0.0), PipeAxis("R", 0.5, 1.0)])
    spec = ViewSpec((1.0, 1.25), (0, 90, 180, 270))
    a = ft.sample(item, 3, "k2", spec, (255, 255, 255), lambda im: im)
    b = ft.sample(item, 3, "k2", spec, (255, 255, 255), lambda im: im)
    c = ft.sample(item, 3, "k345", spec, (255, 255, 255), lambda im: im)
    assert np.array_equal(np.asarray(a), np.asarray(b))
    assert isinstance(a, Image.Image) and not np.array_equal(np.asarray(a), np.asarray(c))


def test_deploy_meta_holds_nothing_out_and_records_the_recipe(tmp_path):
    m = ft.ft_meta(ft.DEPLOY, 0.5, ["a", "b"])
    assert m["fold"] == "all" and m["held_out"] == [] and m["alpha"] == 0.5
    assert m["recipe"]["ft_steps"] == ft.FT_STEPS and m["recipe"]["p_clean"] == ft.P_CLEAN
    assert ft.ft_meta("k2", 0.5, [])["held_out"] == ft.held_out("k2")
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc")
    assert ft.sha256(f).startswith("ba7816bf")  # sha256("abc")
