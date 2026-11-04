"""Unit tests for utils/iso_templates.py — the ISO template loader.

A hand-written template directory (SVG + ports.csv + templates.csv + taxonomy) pins the loader's
rules: defs / ids / editor attributes are stripped, strokes follow `w / w_base * W` (also under
a scaled group), and the 8 orientations move ports, sides and `matrix` consistently. The tests on
the real `data/raw/template_iso_svg/` are skipped when that (gitignored) directory is absent.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pytest

from utils.iso_templates import (
    ROTATIONS,
    Template,
    _apply,
    _own_stroke_width,
    _parse_body,
    _walk,
    load_templates,
)

_ROOT = Path(__file__).resolve().parent.parent
_REAL_DIR = _ROOT / "data" / "raw" / "template_iso_svg"
_TAXONOMY = _ROOT / "docs" / "symbol_mapping" / "iso_taxonomy.csv"

_SVG = """<?xml version='1.0' encoding='utf-8'?>
<svg xmlns="http://www.w3.org/2000/svg"
  xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
  width="12mm" height="6mm" viewBox="50 20 12 6">
  <defs id="defs1"><marker id="m1"><path d="M 0,0 1,1"/></marker></defs>
  <sodipodi:namedview id="base" inkscape:zoom="1"/>
  <metadata id="md"><x/></metadata>
  <g inkscape:groupmode="layer" id="layer1">
    <g id="g1" inkscape:label="REG#:T1_symbol" transform="translate(0,10)">
      <path id="p1" inkscape:connector-curvature="0"
        style="fill:none;stroke:#000000;stroke-width:0.25" d="M 51,16 H 61"/>
      <path id="p2" style="fill:none;stroke:#000000;stroke-width:0.25" d="M 51,12 H 61"/>
      <path id="p3" style="fill:none;stroke:#000000;stroke-width:1" d="M 56,12 V 16"/>
      <path id="p4" style="fill:#000000;stroke:#000000;stroke-width:0" d="M 52,13 h 1 v 1 z"/>
      <g transform="translate(55,14) scale(0.25)">
        <path id="p5" style="stroke:#000000;stroke-width:0.5" d="M 0,0 4,0"/>
      </g>
    </g>
  </g>
</svg>
"""


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def tpl_dir(tmp_path: Path) -> tuple[Path, Path]:
    """One 12 x 6 mm template `T1` (ports L and R) + `T2`, an extra-scope single-port one."""
    out = tmp_path / "svg"
    out.mkdir()
    (out / "T1.svg").write_text(_SVG, encoding="utf-8")
    (out / "T2.svg").write_text(_SVG, encoding="utf-8")
    port = {"file": "T1", "reg_number": "T1", "variant": "1", "fine_name": "valve_t"}
    _write_csv(
        out / "ports.csv",
        [
            {
                **port,
                "port_id": "0",
                "side": "R",
                "x_mm": "11.5",
                "y_mm": "2",
                "u": "0.9583",
                "v": "0.3333",
                "source": "stub",
            },
            {
                **port,
                "port_id": "1",
                "side": "L",
                "x_mm": "0.5",
                "y_mm": "2",
                "u": "0.0417",
                "v": "0.3333",
                "source": "stub",
            },
            {
                "file": "T2",
                "reg_number": "T2",
                "variant": "1",
                "fine_name": "funnel_t",
                "port_id": "0",
                "side": "B",
                "x_mm": "6",
                "y_mm": "5.5",
                "u": "0.5",
                "v": "0.9167",
                "source": "override",
            },
        ],
    )
    common = {"in_vocab": "yes", "width_mm": "12", "height_mm": "6", "source": "x.svg", "ok": "yes"}
    _write_csv(
        out / "templates.csv",
        [
            {
                "file": "T1",
                "reg_number": "T1",
                "variant": "1",
                "group": "G1",
                "fine_name": "valve_t",
                "scope": "html",
                "expected": "2_straight",
                "detected": "2_straight",
                "sides": "LR",
                **common,
            },
            {
                "file": "T2",
                "reg_number": "T2",
                "variant": "1",
                "group": "G5",
                "fine_name": "funnel_t",
                "scope": "extra",
                "expected": "1",
                "detected": "1",
                "sides": "B",
                **common,
            },
        ],
    )
    tax = tmp_path / "taxonomy.csv"
    _write_csv(
        tax,
        [
            {
                "reg_number": "T1",
                "variant": "1",
                "port_config": "2_straight",
                "directional": "yes",
                "placement": "pipe",
            },
            {
                "reg_number": "T2",
                "variant": "1",
                "port_config": "1",
                "directional": "no",
                "placement": "end",
            },
        ],
    )
    return out, tax


def test_scope_filters_templates(tpl_dir):
    out, tax = tpl_dir
    assert list(load_templates(out, tax, "html")) == ["T1"]
    assert list(load_templates(out, tax, "all")) == ["T1", "T2"]
    with pytest.raises(ValueError):
        load_templates(out, tax, "everything")


def test_fields_join_taxonomy_and_ports(tpl_dir):
    out, tax = tpl_dir
    t = load_templates(out, tax, "all")["T1"]
    assert (t.group, t.fine_name, t.in_vocab) == ("G1", "valve_t", True)
    assert (t.port_config, t.directional, t.placement) == ("2_straight", True, "pipe")
    assert (t.w_mm, t.h_mm) == (12.0, 6.0)
    assert t.sides == ("L", "R")
    right = next(p for p in t.ports if p.side == "R")
    assert (right.x_mm, right.y_mm) == (11.5, 2.0)
    assert right.u == pytest.approx(11.5 / 12)
    assert right.v == pytest.approx(2 / 6)
    # body (sheet) coordinates -> local: the viewBox origin (50, 20) goes to (0, 0)
    assert _apply(t.matrix, 50, 20) == (0.0, 0.0)


def test_rejects_inconsistent_inputs(tpl_dir):
    out, tax = tpl_dir
    text = (out / "templates.csv").read_text()
    (out / "templates.csv").write_text(text.replace("LR", "LL"))
    with pytest.raises(ValueError, match="sides"):
        load_templates(out, tax)
    (out / "templates.csv").write_text(text.replace(",yes\n", ",no\n"))
    with pytest.raises(ValueError, match="ok="):
        load_templates(out, tax)
    (out / "templates.csv").write_text(text)
    (out / "T1.svg").write_text(_SVG.replace('width="12mm"', 'width="13mm"'), encoding="utf-8")
    with pytest.raises(ValueError, match="mm"):
        load_templates(out, tax)


def test_body_strips_defs_ids_and_editor_attributes(tpl_dir):
    out, tax = tpl_dir
    body = load_templates(out, tax)["T1"].body()
    for junk in ("defs", "marker", "metadata", "namedview", "inkscape", "sodipodi", "xmlns", "id="):
        assert junk not in body
    assert body.count("<path") == 5
    assert "H 61" in body


def test_body_rescales_strokes_to_the_common_width(tpl_dir):
    out, tax = tpl_dir
    t = load_templates(out, tax)["T1"]
    # w_base = 0.25 (3 of the 5 elements): 0.25 -> W, 1 -> 4 W, 0 stays 0
    widths = [float(w) for w in re.findall(r"stroke-width:([\d.e-]+)", t.body(stroke=0.1))]
    assert widths[:4] == pytest.approx([0.1, 0.1, 0.4, 0.0])
    # p5 sits under scale(0.25): 0.5 / 0.25 * 0.1 = 0.2 drawn, so 0.2 / 0.25 = 0.8 declared
    assert widths[4] == pytest.approx(0.8)
    # without `stroke` the file's own widths survive
    assert re.findall(r"stroke-width:([\d.e-]+)", t.body()) == ["0.25", "0.25", "1", "0", "0.5"]


def test_canonical_orientation_is_identity(tpl_dir):
    out, tax = tpl_dir
    t = load_templates(out, tax)["T1"]
    same = t.oriented(0, False)
    assert (same.ports, same.matrix, same.w_mm, same.h_mm) == (t.ports, t.matrix, 12.0, 6.0)


@pytest.mark.parametrize(
    ("rot", "flip", "sides"),
    [
        (0, True, ("L", "R")),
        (90, False, ("B", "T")),
        (180, False, ("L", "R")),
        (270, False, ("B", "T")),
        (90, True, ("B", "T")),
    ],
)
def test_oriented_sides_and_size(tpl_dir, rot, flip, sides):
    out, tax = tpl_dir
    t = load_templates(out, tax)["T1"].oriented(rot, flip)
    assert t.sides == sides
    assert (t.w_mm, t.h_mm) == ((6.0, 12.0) if rot in (90, 270) else (12.0, 6.0))
    assert (t.rot, t.flip) == (rot, flip)


def test_turn_moves_sides_clockwise_and_flip_swaps_left_right(tpl_dir):
    out, tax = tpl_dir
    t = load_templates(out, tax, "all")["T2"]  # one port, at the bottom
    assert [t.oriented(r).ports[0].side for r in ROTATIONS] == ["B", "L", "T", "R"]
    # a mirror leaves a bottom port alone, so the turns are the same
    assert [t.oriented(r, True).ports[0].side for r in ROTATIONS] == ["B", "L", "T", "R"]
    valve = load_templates(out, tax)["T1"]  # port 0 on the right, port 1 on the left
    assert [p.side for p in valve.oriented(0, True).ports] == ["L", "R"]
    assert [p.side for p in valve.oriented(90, True).ports] == ["T", "B"]  # mirror, then turn


def test_oriented_is_absolute_not_stacked(tpl_dir):
    out, tax = tpl_dir
    t = load_templates(out, tax)["T1"]
    assert t.oriented(90).oriented(90).ports == t.oriented(90).ports
    assert t.oriented(90).oriented(0, False).ports == t.ports
    assert t.oriented(270, True).base is t


def test_all_orientations_are_eight(tpl_dir):
    out, tax = tpl_dir
    t = load_templates(out, tax)["T1"]
    poses = {(o.rot, o.flip) for o in t.all_orientations()}
    assert len(poses) == 8


def _check_matrix_consistency(t: Template) -> None:
    """For every orientation: matrix(body corners of the viewBox) fill the oriented box, and
    each port is where the canonical port goes under the same mapping."""
    base = t if t.base is None else t.base
    vx, vy = -base.matrix[4], -base.matrix[5]
    for o in t.all_orientations():
        corners = [
            _apply(o.matrix, vx + dx, vy + dy) for dx in (0, base.w_mm) for dy in (0, base.h_mm)
        ]
        assert min(c[0] for c in corners) == pytest.approx(0, abs=1e-9)
        assert min(c[1] for c in corners) == pytest.approx(0, abs=1e-9)
        assert max(c[0] for c in corners) == pytest.approx(o.w_mm)
        assert max(c[1] for c in corners) == pytest.approx(o.h_mm)
        for p, q in zip(base.ports, o.ports, strict=True):
            x, y = _apply(o.matrix, vx + p.x_mm, vy + p.y_mm)
            assert (x, y) == pytest.approx((q.x_mm, q.y_mm))
            assert (q.u, q.v) == pytest.approx((q.x_mm / o.w_mm, q.y_mm / o.h_mm))


def test_matrix_agrees_with_ports_in_all_orientations(tpl_dir):
    out, tax = tpl_dir
    for t in load_templates(out, tax, "all").values():
        _check_matrix_consistency(t)


# --- real templates (skipped when data/raw/template_iso_svg is absent) -------------------------

real = pytest.mark.skipif(not (_REAL_DIR / "templates.csv").exists(), reason="no template_iso_svg")


@pytest.fixture(scope="module")
def real_templates() -> dict[str, Template]:
    return load_templates(_REAL_DIR, _TAXONOMY, "all")


@real
def test_real_templates_load(real_templates):
    assert len(real_templates) == 63
    # 45 of the 5-group proposal + 511 flanged_connection (2026-11-01) + 241 flow_direction
    # (2026-11-02, an overlay: no rule draws it, arrows carry it)
    assert sum(t.scope == "html" for t in real_templates.values()) == 47
    assert len(load_templates(_REAL_DIR, _TAXONOMY, "html")) == 47
    for t in real_templates.values():
        assert t.ports, t.file
        assert t.body(stroke=0.5).startswith("<g")


@real
def test_real_strokes_all_resolve_to_the_target(real_templates):
    """Every stroked element declares its width (none falls back to the SVG default of 1), and
    the template's base stroke comes out exactly at W."""
    for name, t in real_templates.items():
        canonical = _parse_body(str(t.svg_path))
        for top in canonical.elements:
            for e in _walk(top):
                if e.tag not in ("text", "tspan") and "stroke:#" in e.get("style", ""):
                    assert _own_stroke_width(e) is not None, name
        widths = [float(w) for w in re.findall(r"stroke-width:([\d.e-]+)", t.body(stroke=0.1))]
        if canonical.w_base is None:  # filled shapes only (241, the arrowhead): nothing to scale
            assert all(w == 0 for w in widths), name
            continue
        assert any(w == pytest.approx(0.1) for w in widths), name


@real
def test_real_matrix_agrees_with_ports(real_templates):
    for t in real_templates.values():
        _check_matrix_consistency(t)


def _render(o: Template, px_per_mm: int = 20):
    """Ink pixel coordinates (in mm, local frame of `o`) of the template drawn under its matrix."""
    pymupdf = pytest.importorskip("pymupdf")
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{o.w_mm * px_per_mm}" '
        f'height="{o.h_mm * px_per_mm}" viewBox="0 0 {o.w_mm} {o.h_mm}">'
        f'<rect width="{o.w_mm}" height="{o.h_mm}" fill="white"/>'
        f'<g transform="{o.transform}">{o.body(stroke=0.25)}</g></svg>'
    )
    pix = pymupdf.open(stream=svg.encode(), filetype="svg")[0].get_pixmap(dpi=72, alpha=False)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[..., 0]
    ys, xs = np.nonzero(img < 128)
    scale = pix.width / o.w_mm
    return xs / scale, ys / scale, 1 / scale


_RENDERED = ["2101", "2301", "2102", "X8087", "2511", "2040"]


@real
@pytest.mark.parametrize("name", _RENDERED)
def test_real_render_stays_inside_the_box(real_templates, name):
    """Drawn under each orientation's matrix, the ink fills a box no bigger than the one the
    orientation reports (so `matrix` and `w_mm` / `h_mm` agree)."""
    for o in real_templates[name].all_orientations():
        xs, ys, px = _render(o)
        assert xs.size, (name, o.rot, o.flip)
        assert xs.min() >= -0.05 and xs.max() + px <= o.w_mm + 0.05, (name, o.rot, o.flip)
        assert ys.min() >= -0.05 and ys.max() + px <= o.h_mm + 0.05, (name, o.rot, o.flip)


@real
@pytest.mark.parametrize("name", _RENDERED)
def test_real_ports_touch_the_body(real_templates, name):
    for o in real_templates[name].all_orientations():
        xs, ys, _ = _render(o)
        for p in o.ports:
            assert np.hypot(xs - p.x_mm, ys - p.y_mm).min() <= 0.3, (name, p.side)
