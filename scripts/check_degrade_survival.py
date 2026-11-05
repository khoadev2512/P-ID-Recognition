"""V7 detail-survival check: does a symbol, degraded as hard as V7 allows, stay itself?

Each drawable template of the groups that matter (G1, G2 valves; G5 end devices; the agitators;
`flanged_connection` against the rest of G3) is drawn alone at the target size and stroke width,
degraded with `utils.iso_degrade` at the *maximum* parameters, and compared by normalised
cross-correlation (max over +-4 px shifts) with the clean drawing of every template of its group.
It passes when the best match is a template of its own class (`fine_name`); `strict` also asks
for the very same file (two drawings of one class are allowed to be taken for each other).

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
                    if params is not None:
                        img = degrade_image(img, replace(params, noise_seed=1000 * n + 7))[0]
                    score = best_match(img, refs)
                    best = max(score, key=score.get)
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
    """Every testable template is still taken for its own class."""
    return all(r["pass"] or not r["testable"] for r in rows)


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
    for label, sel_k in (
        (f"gate (k >= {args.gate_k})", lambda k: k >= args.gate_k),
        ("below the gate", lambda k: k < args.gate_k),
    ):
        print(f"\n{label}\n{'scenario':<24}{'tests':>6}{'fail':>6}{'strict-fail':>12}")
        for scen in dict.fromkeys(r["scenario"] for r in rows):
            sel = [r for r in rows if r["scenario"] == scen and sel_k(r["k"])]
            n_fail = sum(1 - r["pass"] for r in sel if r["testable"])
            n_strict = sum(1 - r["strict"] for r in sel)
            if sel:
                print(f"{scen:<24}{len(sel):>6}{n_fail:>6}{n_strict:>12}")
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
