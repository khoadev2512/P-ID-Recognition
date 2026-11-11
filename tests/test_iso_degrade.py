"""Unit tests for utils/iso_degrade.py (V7): limits that follow the stroke width, an affine that is
the same for pixels and coordinates, ink boxes that follow the ink, a deterministic degradation."""

from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from utils.iso_degrade import (
    MIN_PEAK,
    DegradeParams,
    affine_for,
    apply_affine,
    clamp_params,
    corner_box,
    degrade_image,
    extreme_params,
    ink_box_after,
    limits,
    moved_geometry,
    peak_contrast,
    sample_params,
)
from utils.iso_mapping import load_config
from utils.iso_templates import load_templates
from utils.iso_variants import STAGES, derive_seed, plan_variant, render_variant
from utils.page_svg import rasterize
from utils.pid2graph import load_graphml, pipe_runs

ROOT = Path(__file__).resolve().parent.parent
_OPEN100 = ROOT / "data" / "raw" / "pid2graph" / "open100"
_TEMPLATES = ROOT / "data" / "raw" / "template_iso_svg"
_VARIANTS = ROOT / "data" / "iso_open100" / "variants"
needs_open100 = pytest.mark.skipif(
    not (_OPEN100.exists() and _TEMPLATES.exists()), reason="OPEN100 data missing"
)

OFF = DegradeParams(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 100, 1)  # everything off (JPEG at 100)
CFG = {
    "degrade": {
        "rotate_deg": [-1.0, 1.0], "shear_deg": [-0.5, 0.5], "morph": [-1.0, 1.0],
        "blur": [0.0, 1.0], "pixelate": [0.0, 1.0], "lighting": [0.0, 0.2],
        "noise_sigma": [0.0, 10.0], "jpeg_quality": [55, 90],
    }
}  # fmt: skip


def _page(w=480, h=320):
    return np.full((h, w), 255, np.uint8)


def test_limits_follow_the_stroke_width():
    assert limits(4.0) == {"morph_px": 2.0, "blur_sigma": 2.0, "pixelate_block": 2.0}
    assert limits(2.0)["pixelate_block"] == 1.0
    big = DegradeParams(0, 0, -9.0, 9.0, 9.0, 0, 0, 90, 1)
    c = clamp_params(big, 3.0)
    assert c.morph_px >= -1.5 and c.blur_sigma <= 1.5 and c.pixelate_block <= 1.5
    assert peak_contrast(3.0, c) >= MIN_PEAK  # and a line of that width is still dark enough


def test_sampled_and_extreme_params_never_exceed_the_limits():
    rng = random.Random(0)
    for _ in range(300):
        w = rng.uniform(2.0, 4.0)
        for p in (
            sample_params(CFG, w, rng),
            extreme_params(CFG, w, -1),
            extreme_params(CFG, w, 1),
        ):
            assert abs(p.morph_px) <= w / 2 + 1e-9
            assert 0 <= p.blur_sigma <= w / 2 + 1e-9
            assert 1.0 <= p.pixelate_block <= max(1.0, w / 2) + 1e-9
            assert abs(p.rotate_deg) <= 1.0 and abs(p.shear_deg) <= 1.0
    e = extreme_params(CFG, 4.0, morph=-1)  # at w = 4 nothing needs lowering
    assert (e.morph_px, e.rotate_deg, e.jpeg_quality) == (-2.0, 1.0, 55)
    assert 0.5 < e.blur_sigma <= 2.0  # lowered so that a line of width 4 keeps its darkness
    only = extreme_params(CFG, 3.0, only="blur")
    assert only.blur_sigma == 1.5 and only.rotate_deg == 0 and only.noise_sigma == 0
    assert only.jpeg_quality == 90


def test_sample_params_is_a_function_of_the_seed():
    a = sample_params(CFG, 3.0, random.Random(5))
    assert a == sample_params(CFG, 3.0, random.Random(5))
    assert a != sample_params(CFG, 3.0, random.Random(6))


def test_affine_identity_and_fixed_centre():
    m = affine_for(400, 200, 0.0, 0.0)
    assert np.allclose(apply_affine(m, [(10, 20)]), [(10, 20)])
    m = affine_for(400, 200, 1.0, 0.5)
    assert np.allclose(apply_affine(m, [(200, 100)]), [(200, 100)])
    # clockwise on screen: a point right of the centre moves down
    assert apply_affine(affine_for(400, 200, 1.0, 0.0), [(300, 100)])[0][1] > 100


@pytest.mark.parametrize("rot, shear", [(1.0, 0.0), (-0.8, 0.0), (0.0, 0.5), (0.7, -0.4)])
def test_the_image_and_the_coordinates_move_the_same_way(rot, shear):
    img = _page()
    img[100:106, 300:306] = 0  # a dark square whose centre is the page point (303, 103)
    p = DegradeParams(rot, shear, 0.0, 0.0, 1.0, 0.0, 0.0, 100, 1)
    out, m, _ = degrade_image(img, p)
    ys, xs = np.nonzero(255 - out.astype(float) > 0)
    weights = (255 - out[ys, xs]).astype(float)
    cx = float((xs + 0.5) @ weights / weights.sum())
    cy = float((ys + 0.5) @ weights / weights.sum())
    assert np.allclose(apply_affine(m, [(303, 103)])[0], [cx, cy], atol=0.35)


def test_ink_box_after_hugs_the_moved_ink_and_beats_the_corner_box():
    img = _page(900, 700)
    cv2.ellipse(img, (450, 350), (80, 300), 0, 0, 360, 0, 2)  # a tall vessel outline
    m = affine_for(900, 700, 1.0, 0.0)
    box = (369.5, 49.5, 530.5, 650.5)
    tight = ink_box_after(m, img, box, pad=0.0)
    loose = corner_box(m, box)
    out, _, _ = degrade_image(img, DegradeParams(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 100, 1))
    ys, xs = np.nonzero(out < 128)  # the moved ink, measured on the warped image
    assert tight[0] == pytest.approx(xs.min(), abs=1.5) and tight[2] == pytest.approx(
        xs.max() + 1, abs=1.5
    )
    assert tight[1] == pytest.approx(ys.min(), abs=1.5) and tight[3] == pytest.approx(
        ys.max() + 1, abs=1.5
    )
    assert tight[2] - tight[0] < loose[2] - loose[0] - 4  # corners add ~ 600 * sin(1 deg) of width
    assert ink_box_after(m, img, (0, 0, 20, 20)) is None  # no ink there


def test_degrade_is_deterministic_and_noise_follows_its_seed():
    img = _page()
    img[100:104, 50:400] = 0
    p = DegradeParams(0.5, 0.2, -1.0, 0.8, 1.5, 0.1, 6.0, 70, 11)
    a, _, ja = degrade_image(img, p)
    b, _, jb = degrade_image(img, p)
    assert np.array_equal(a, b) and ja == jb
    c, _, _ = degrade_image(img, DegradeParams(**{**p.__dict__, "noise_seed": 12}))
    assert not np.array_equal(a, c)
    assert a.shape == img.shape and a.dtype == np.uint8


def _width(img):
    """Mean ink coverage across a horizontal line: its width in px."""
    col = img[:, 240].astype(float)
    return float((255 - col).sum() / 255)


def test_morph_changes_the_stroke_width_by_the_asked_amount():
    img = _page()
    img[150:154, :] = 0  # a 4 px line
    base = _width(degrade_image(img, OFF)[0])
    assert base == pytest.approx(4.0, abs=0.3)
    thin = _width(degrade_image(img, DegradeParams(**{**OFF.__dict__, "morph_px": -2.0}))[0])
    fat = _width(degrade_image(img, DegradeParams(**{**OFF.__dict__, "morph_px": 2.0}))[0])
    assert thin == pytest.approx(2.0, abs=0.4) and fat == pytest.approx(6.0, abs=0.4)
    half = _width(degrade_image(img, DegradeParams(**{**OFF.__dict__, "morph_px": -1.0}))[0])
    assert thin < half < base


def test_a_thin_line_keeps_its_darkness_at_the_strongest_setting():
    for w in (2.0, 3.0, 4.0):
        p = extreme_params(CFG, w, morph=-1)
        assert peak_contrast(w, p) >= MIN_PEAK  # the measure used by the clamp ...
        img = _page()
        img[150 : 150 + int(w), :] = 0
        clean = degrade_image(img, replace(p, noise_sigma=0.0, lighting=0.0, jpeg_quality=100))[0]
        peak = (255 - clean[130:175, 60:420].min(axis=0)) / 255
        assert peak.min() >= MIN_PEAK - 0.1  # ... agrees with the pixels of a real image
    loose = DegradeParams(0, 0, -1.0, 2.0, 1.0, 0, 0, 90, 1)
    assert peak_contrast(2.0, loose) < MIN_PEAK  # an unclamped setting would lose the line


# --- on a real sheet ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def drawn():
    cfg = load_config()
    templates = load_templates(cfg["template_dir"], cfg["taxonomy_csv"], cfg["scope"])
    sheet = load_graphml(_OPEN100 / "3.graphml", Image.open(_OPEN100 / "3.png").size)
    runs = pipe_runs(sheet)
    draft = plan_variant(sheet, runs, templates, cfg, 3, 0)
    return cfg, draft, render_variant(draft, sheet, runs, cfg)


@needs_open100
def test_degrade_stage_leaves_the_other_stages_alone(drawn):
    cfg, draft, _ = drawn
    base = cfg["variants"]["base_seed"]
    assert draft.seeds["degrade"] == derive_seed(base, 3, 0, "degrade")
    assert len({draft.seeds[s] for s in draft.seeds}) == len(STAGES)
    stored = _VARIANTS / "sheet3" / "v000.json"
    if stored.exists():  # the V6 picture of the same variant is still what its seeds give
        meta = json.loads(stored.read_text())
        assert meta["signature"] == draft.signature
        assert all(meta["seeds"][k] == draft.seeds[k] for k in ("assign", "layout", "style"))


@needs_open100
def test_moved_geometry_follows_the_affine(drawn):
    cfg, draft, rendered = drawn
    symbols = rasterize(rendered.page.layer("symbols"))
    h, w = symbols.shape
    ident = moved_geometry(affine_for(w, h, 0.0, 0.0), rendered, symbols, OFF)
    for a in ident["annotations"]:
        assert np.allclose(a["bbox"], a["bbox_clean"], atol=1.5), a["id"]
    p = DegradeParams(1.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 90, 1)
    m = affine_for(w, h, 1.0, 0.5)
    geo = moved_geometry(m, rendered, symbols, p)
    assert set(geo["nodes"]) == set(rendered.sheet.nodes)
    assert len(geo["ports"]) == len(rendered.page.ports)
    bubbles = [a for a in geo["annotations"] if a["kind"] == "bubble"]
    assert bubbles
    for a in bubbles:  # a bubble's node is its circle, moved
        c = geo["nodes"][a["node"]]["center"]
        x0, y0, x1, y1 = a["bbox_clean"]
        assert np.allclose(c, apply_affine(m, [((x0 + x1) / 2, (y0 + y1) / 2)])[0], atol=0.02)
    for a in geo["annotations"]:  # the ink box is inside the (looser) corner box
        b, c = a["bbox"], a["bbox_corners"]
        assert b[0] >= c[0] - 1 and b[1] >= c[1] - 1 and b[2] <= c[2] + 1 and b[3] <= c[3] + 1


@needs_open100
def test_degrade_one_writes_a_consistent_document(tmp_path):
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import make_degraded

    from utils.iso_variants import generate_sheet

    cfg = load_config()
    cfg["variants"].update(count=1, max_attempts=2)
    templates = load_templates(cfg["template_dir"], cfg["taxonomy_csv"], cfg["scope"])
    sheet = load_graphml(_OPEN100 / "3.graphml", Image.open(_OPEN100 / "3.png").size)
    runs = pipe_runs(sheet)
    generate_sheet(3, sheet, runs, templates, cfg, tmp_path)
    meta = json.loads((tmp_path / "sheet3" / "v000.json").read_text())
    doc, jpeg, secs = make_degraded.degrade_one(
        meta, tmp_path / "sheet3" / "v000.png", sheet, runs, templates, cfg
    )
    doc2, jpeg2, _ = make_degraded.degrade_one(
        meta, tmp_path / "sheet3" / "v000.png", sheet, runs, templates, cfg
    )
    assert jpeg == jpeg2 and doc == doc2  # a function of the variant's seeds alone
    decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert decoded.shape == (meta["height"], meta["width"])
    p = doc["params"]
    assert doc["seed"] == derive_seed(cfg["variants"]["base_seed"], 3, 0, "degrade")
    assert np.allclose(
        doc["affine"], affine_for(meta["width"], meta["height"], p["rotate_deg"], p["shear_deg"]),
        atol=5e-3,  # the angles in `params` are rounded to 4 decimals, the affine is not
    )  # fmt: skip
    assert abs(p["morph_px"]) <= meta["stroke_px"] / 2 and p["blur_sigma"] <= meta["stroke_px"] / 2
    assert len(doc["annotations"]) == len(meta["annotations"])
    assert doc["boxes_outside_page"] == 0 and set(doc) >= {"nodes", "ports", "stems"}
    assert secs["degrade"] > 0


def _survival():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import check_degrade_survival

    return check_degrade_survival


def _bow(bar: bool, bar_ink: int = 0) -> np.ndarray:
    """A frame and a dot (the shared part, like a valve's two triangles) with a 3 px vertical bar
    through the dot (valve_butterfly-like) or without (valve_globe-like)."""
    img = np.full((60, 60), 255, np.uint8)
    cv2.rectangle(img, (4, 8), (55, 51), 0, 3)
    cv2.circle(img, (30, 30), 6, 0, -1)
    if bar:
        img[14:47, 29:32] = np.minimum(img[14:47, 29:32], bar_ink)
    return img


def test_the_detail_core_is_the_middle_of_what_only_one_drawing_has():
    sv = _survival()
    with_bar, without = _bow(True), _bow(False)
    core = sv.detail_core(with_bar, without)
    assert core.any() and set(np.nonzero(core)[1]) == {30}  # the bar's middle column only
    assert not sv.detail_core(without, with_bar).any()  # nothing only the dot drawing has
    assert sv.detail_darkness(with_bar, core) == pytest.approx(1.0)
    assert sv.detail_darkness(without, core) == pytest.approx(0.0)
    assert sv.halfway(with_bar, without, core) == pytest.approx(0.5)
    # a bar faded to 70 % ink is still nearer the barred drawing; one at 30 % is not
    assert sv.detail_darkness(_bow(True, bar_ink=77), core) > 0.5
    assert sv.detail_darkness(_bow(True, bar_ink=179), core) < 0.5


def test_whole_symbol_ncc_barely_sees_the_bar_the_detail_check_does():
    sv = _survival()
    with_bar, without = _bow(True), _bow(False)
    refs = {"bar": with_bar, "dot": without}
    core = sv.detail_core(with_bar, without)
    faded = _bow(True, bar_ink=230)  # the bar all but gone
    # whole-symbol NCC: the shared frame keeps the score up, it moves by a few hundredths
    assert sv.best_match(with_bar, refs)["bar"] - sv.best_match(faded, refs)["bar"] < 0.1
    # the detail: from full ink to under a tenth
    assert sv.detail_darkness(with_bar, core) - sv.detail_darkness(faded, core) > 0.85
