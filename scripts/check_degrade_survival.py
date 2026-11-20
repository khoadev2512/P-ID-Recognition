"""V7 detail-survival check: does a symbol, degraded as hard as V7 allows, stay itself?

Each drawable template of the groups that matter (G1, G2 valves; G5 end devices; the agitators;
`flanged_connection` against the rest of G3) is drawn alone at the target size and stroke width,
degraded with `utils.iso_degrade` at the *maximum* parameters, and compared by normalised
cross-correlation (max over +-4 px shifts) with the clean drawing of every template of its group.
It passes when the best match is a template of its own class (`fine_name`); `strict` also asks
for the very same file (two drawings of one class are allowed to be taken for each other).

Whole-symbol NCC cannot test a template whose clean drawing is already within
`MIN_CLEAN_MARGIN` of another class (`testable` 0): the shared part outweighs the detail that tells
them apart (valve_globe vs valve_butterfly, check_valve_globe vs check_valve_swing differ by one
bar through the middle dot, ~5% of the ink; fading the bar from full to none moves NCC by ~0.02).
Such a template is checked on the difference region instead (`detail_*`): the core of what only
it draws must stay darker than halfway between its clean darkness and the other's there, and the
core of what only the other draws must stay lighter than halfway (no detail of the other class
grows out of blur or dilation). The region and its rule come from the two templates, so nothing
is estimated.

Scenarios: every parameter at its strongest end (eroding, and dilating), and each parameter alone
at its strongest end, to say which one breaks a template. Writes a CSV (one row per template,
scenario, size k px/mm and stroke w) and prints the failures. Read-only on data/raw/.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from itertools import product
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.iso_degrade import _RANGE_KEYS, degrade_image, extreme_params  # noqa: E402
from utils.iso_mapping import load_config  # noqa: E402
from utils.iso_templates import load_templates  # noqa: E402
from utils.page_svg import _compose, oriented_ink_box, rasterize  # noqa: E402

PAD_PX = 14  # blank border round the largest drawing of a group
MIN_CLEAN_MARGIN = 0.03  # a clean drawing closer than this to another class cannot be tested
SHIFT = 4  # px the match may slide (a rotation moves a drawing by about a pixel)
DETAIL_MIN_PX = 4  # px: a smaller piece of a difference region is anti-aliasing, not a detail
GROUPS = {  # name -> (templates to test, templates to compare them with)
    "G1": (lambda t: t.group == "G1", lambda t: t.group == "G1"),
    "G2": (lambda t: t.group == "G2", lambda t: t.group == "G2"),
    "G5": (lambda t: t.group == "G5", lambda t: t.group == "G5"),
    "agitator": (lambda t: t.placement == "attach", lambda t: t.placement == "attach"),
    "flange": (lambda t: t.fine_name == "flanged_connection", lambda t: t.group == "G3"),
}


def draw(tpl, k: float, w: float, size: tuple[int, int]) -> np.ndarray:
    """`tpl` alone, ink centred on a white canvas of `size` px, `k` px per mm, stroke `w` px."""
    x0, y0, x1, y1 = oriented_ink_box(tpl)
    px, py = size[0] / 2 - k * (x0 + x1) / 2, size[1] / 2 - k * (y0 + y1) / 2
    m = _compose((k, 0.0, 0.0, k, px, py), tpl.matrix)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size[0]}" height="{size[1]}" '
        f'viewBox="0 0 {size[0]} {size[1]}"><rect width="100%" height="100%" fill="white"/>'
        f'<g transform="matrix({" ".join(f"{v:.6g}" for v in m)})">{tpl.body(stroke=w / k)}</g>'
        "</svg>"
    )
    return rasterize(svg)


def best_match(degraded: np.ndarray, refs: dict[str, np.ndarray]) -> dict[str, float]:
    """NCC of `degraded` with each clean reference, best over small shifts."""
    out = {}
    for name, ref in refs.items():
        core = ref[SHIFT:-SHIFT, SHIFT:-SHIFT]
        out[name] = float(cv2.matchTemplate(degraded, core, cv2.TM_CCOEFF_NORMED).max())
    return out


def _dark(img: np.ndarray) -> np.ndarray:
    return 1.0 - img.astype(np.float64) / 255


def detail_core(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The middle line of what only `a` draws (two clean drawings on one canvas): pixels inked
    in `a` where `b` is blank paper, kept per connected piece of at least `DETAIL_MIN_PX` pixels
    at their deepest (distance to the piece's edge within half a pixel of its largest)."""
    region = ((_dark(a) > 0.5) & (_dark(b) < 0.02)).astype(np.uint8)
    n, labels = cv2.connectedComponents(region)
    dist = cv2.distanceTransform(region, cv2.DIST_L2, 3)
    core = np.zeros(region.shape, bool)
    for i in range(1, n):
        piece = labels == i
        if piece.sum() >= DETAIL_MIN_PX:
            core |= piece & (dist >= dist[piece].max() - 0.5)
    return core


def detail_darkness(img: np.ndarray, core: np.ndarray) -> float:
    """How dark `img` stays on `core`: the median over its pixels of the darkest pixel within
    1 px (a rotation moves a thin line by about a pixel); NaN for an empty core."""
    if not core.any():
        return float("nan")
    near = cv2.dilate(_dark(img), np.ones((3, 3), np.uint8))
    return float(np.median(near[core]))


def halfway(a: np.ndarray, b: np.ndarray, core: np.ndarray) -> float:
    """The darkness halfway between clean drawings `a` and `b` on `core`: a degraded drawing
    darker than this there is nearer `a`'s, lighter is nearer `b`'s."""
    return (detail_darkness(a, core) + detail_darkness(b, core)) / 2


def scenarios(cfg: dict, w: float) -> dict[str, object]:
    out = {
        "clean": None,
        "all_erode": extreme_params(cfg, w, morph=-1),
        "all_dilate": extreme_params(cfg, w, morph=+1),
    }
    for key in (*_RANGE_KEYS, "jpeg_quality"):
        if key == "morph":
            out["only_erode"] = extreme_params(cfg, w, morph=-1, only="morph")
            out["only_dilate"] = extreme_params(cfg, w, morph=+1, only="morph")
        else:
            out[f"only_{key}"] = extreme_params(cfg, w, only=key)
    return out


def evaluate(cfg, templates, ks, ws, only=None):
    """Result rows for every group x (k, w) x scenario (`only`: a set of scenario names)."""
    rows = []
    for gname, (is_test, is_ref) in GROUPS.items():
        tests = [t for t in templates if is_test(t)]
        refs_t = [t for t in templates if is_ref(t)]
        fine_of = {t.file: t.fine_name for t in refs_t}
        boxes = [oriented_ink_box(t) for t in refs_t]
        for k, w in product(ks, ws):
            size = (
                int(max(b[2] - b[0] for b in boxes) * k) + 2 * PAD_PX,
                int(max(b[3] - b[1] for b in boxes) * k) + 2 * PAD_PX,
            )
            refs = {t.file: draw(t, k, w, size) for t in refs_t}
            clean_margin: dict[str, float] = {}
            # per untestable template: (other template, own core, other's core, both thresholds)
            details: dict[str, list[tuple[str, np.ndarray, np.ndarray, float, float]]] = {}
            for scen, params in scenarios(cfg, w).items():
                if only is not None and scen not in only:
                    continue
                for n, t in enumerate(tests):
                    img = refs[t.file]
                    if t.file not in clean_margin:
                        base = best_match(img, refs)
                        clean_margin[t.file] = base[t.file] - max(
                            (v for f, v in base.items() if fine_of[f] != t.fine_name), default=0
                        )
                        details[t.file] = []
                        for f, v in base.items():
                            if fine_of[f] == t.fine_name or base[t.file] - v >= MIN_CLEAN_MARGIN:
                                continue
                            own, other = detail_core(img, refs[f]), detail_core(refs[f], img)
                            mid_own = halfway(img, refs[f], own)
                            mid_other = halfway(img, refs[f], other)
                            details[t.file].append((f, own, other, mid_own, mid_other))
                    if params is not None:
                        img = degrade_image(img, replace(params, noise_seed=1000 * n + 7))[0]
                    score = best_match(img, refs)
                    best = max(score, key=score.get)
                    keeps, grows, ok = [], [], True
                    for _, own, other, mid_own, mid_other in details[t.file]:
                        if own.any():
                            keeps.append(detail_darkness(img, own))
                            ok &= keeps[-1] > mid_own
                        if other.any():
                            grows.append(detail_darkness(img, other))
                            ok &= grows[-1] < mid_other
                    same = {f for f in score if fine_of[f] == t.fine_name}
                    others = [v for f, v in score.items() if f not in same]
                    rows.append(
                        {
                            "group": gname, "scenario": scen, "k": k, "w": w, "template": t.file,
                            "fine_name": t.fine_name, "best": best, "best_fine": fine_of[best],
                            "ncc_self": round(score[t.file], 3),
                            "ncc_best_other_class": round(max(others, default=float("nan")), 3),
                            "pass": int(best in same), "strict": int(best == t.file),
                            "clean_margin": round(clean_margin[t.file], 3),
                            "testable": int(clean_margin[t.file] >= MIN_CLEAN_MARGIN),
                            "margin": round(score[t.file] - max(others, default=0), 3),
                            "detail_vs": "+".join(f for f, *_ in details[t.file]),
                            "detail_own": round(min(keeps), 3) if keeps else "",
                            "detail_other": round(max(grows), 3) if grows else "",
                            "detail_pass": int(ok) if details[t.file] else "",
                        }
                    )  # fmt: skip
    return rows


# the parameter each `only_*` scenario isolates (the config key whose strong end it uses)
ONLY = {
    "only_erode": "morph", "only_dilate": "morph", "only_rotate_deg": "rotate_deg",
    "only_shear_deg": "shear_deg", "only_blur": "blur", "only_pixelate": "pixelate",
    "only_lighting": "lighting", "only_noise_sigma": "noise_sigma",
    "only_jpeg_quality": "jpeg_quality",
}  # fmt: skip


def with_cap(d: dict, scen: str, f: float) -> dict:
    """A copy of the `degrade` block with the strong end that `scen` uses moved to the fraction
    `f` of the way from the mild end (0) to where it is now (1)."""
    key, out = ONLY[scen], {k: list(v) for k, v in d.items()}
    lo, hi = out[key]
    if key == "morph":
        out[key] = [lo * f, hi] if scen == "only_erode" else [lo, hi * f]
    elif key in ("rotate_deg", "shear_deg"):
        out[key] = [lo * f, hi * f]
    elif key == "jpeg_quality":
        out[key] = [hi - f * (hi - lo), hi]
    else:
        out[key] = [lo, lo + f * (hi - lo)]
    return out


def passes(rows: list[dict]) -> bool:
    """Every testable template is still taken for its own class, and every other one keeps its
    distinguishing detail without growing the other class's."""
    return all((r["pass"] or not r["testable"]) and r["detail_pass"] != 0 for r in rows)


def tune(cfg, templates, ks, ws, step=0.1):
    """Largest fraction of each parameter's range (in steps of `step`) at which the templates
    still pass when that parameter alone is at its strong end, then dilation, blur and
    pixelation together: the `degrade` block is rewritten in place.
    Returns {scenario: fraction kept}."""
    kept = {}
    for scen in ONLY:
        f = 1.0
        while f > 0:
            trial = {**cfg, "degrade": with_cap(cfg["degrade"], scen, f)}
            if passes(evaluate(trial, templates, ks, ws, {scen})):
                break
            f = round(f - step, 2)
        kept[scen] = f
        cfg["degrade"] = with_cap(cfg["degrade"], scen, f)
    # together: only the parameters that blur or thicken the ink can add up to a confusion
    scale = 1.0
    while scale > 0 and not passes(evaluate(cfg, templates, ks, ws, {"all_erode", "all_dilate"})):
        scale = round(scale - step, 2)
        for scen in ("only_dilate", "only_blur", "only_pixelate"):
            cfg["degrade"] = with_cap(cfg["degrade"], scen, 1 - step)
    kept["together_extra_scale_of_dilate_blur_pixelate"] = scale
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("docs/experiments/iso_render/v7/survival.csv"))
    ap.add_argument("--k", type=float, nargs="*", default=[3.0, 3.5, 5.5, 7.5], help="px per mm")
    ap.add_argument("--gate-k", type=float, default=3.5, help="smallest k that must pass")
    ap.add_argument("--w", type=float, nargs="*", default=[2.0, 3.0, 4.0], help="stroke px")
    ap.add_argument("--tune", action="store_true", help="lower the maxima until the gate passes")
    args = ap.parse_args()

    cfg = load_config()
    templates = list(
        load_templates(cfg["template_dir"], cfg["taxonomy_csv"], cfg["scope"]).values()
    )
    if args.tune:
        kept = tune(cfg, templates, [k for k in args.k if k >= args.gate_k], args.w)
        print("fraction of each range that survives alone:", kept)
        print("final degrade block:", cfg["degrade"])
    rows = evaluate(cfg, templates, args.k, args.w)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    nan = float("nan")
    for label, sel_k in (
        (f"gate (k >= {args.gate_k})", lambda k: k >= args.gate_k),
        ("below the gate", lambda k: k < args.gate_k),
    ):
        print(
            f"\n{label}\n{'scenario':<24}{'tests':>6}{'fail':>6}{'strict-fail':>12}"
            f"{'detail':>8}{'detail-fail':>12}{'min own':>9}{'max other':>10}"
        )
        for scen in dict.fromkeys(r["scenario"] for r in rows):
            sel = [r for r in rows if r["scenario"] == scen and sel_k(r["k"])]
            n_fail = sum(1 - r["pass"] for r in sel if r["testable"])
            n_strict = sum(1 - r["strict"] for r in sel)
            det = [r for r in sel if r["detail_pass"] != ""]
            own = [r["detail_own"] for r in det if r["detail_own"] != ""]
            grow = [r["detail_other"] for r in det if r["detail_other"] != ""]
            if sel:
                print(
                    f"{scen:<24}{len(sel):>6}{n_fail:>6}{n_strict:>12}{len(det):>8}"
                    f"{sum(1 - r['detail_pass'] for r in det):>12}"
                    f"{min(own, default=nan):>9.2f}{max(grow, default=nan):>10.2f}"
                )
    print("\ndetail failures (a distinguishing detail faded, or the other class's grew):")
    for r in rows:
        if r["detail_pass"] == 0:
            print(
                f"  {r['scenario']:<16} k={r['k']} w={r['w']} {r['template']} ({r['fine_name']}) "
                f"vs {r['detail_vs']}: own {r['detail_own']} other {r['detail_other']}"
            )
    print("\nfailures of the all-maximum scenarios (class confused):")
    seen = set()
    for r in rows:
        if not r["pass"] and r["testable"] and r["scenario"].startswith("all_"):
            key = (r["scenario"], r["template"], r["best"], r["k"], r["w"])
            if key not in seen:
                seen.add(key)
                where = f"k={r['k']} w={r['w']} {r['group']}"
                print(
                    f"  {r['scenario']:<11} {where} {r['template']} ({r['fine_name']}) -> "
                    f"{r['best']} ({r['best_fine']}) self {r['ncc_self']}"
                )


if __name__ == "__main__":
    main()
