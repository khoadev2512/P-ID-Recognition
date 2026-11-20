"""Extra drawn forms of vocab classes -> data/raw/iso_symbols_forms/<fine_name>@<form>.png
(PLAN_PHASE_C §2K.4 (i)), the input of `++oneshot.extra_forms_dir`.

BK-2020 / docs/symbol_mapping/iso_taxonomy.csv draws one vocab class in two forms: X8075
valve_butterfly, form A (the iso_symbols_clean template) and form B. The extra form is rendered
from its per-symbol SVG (data/raw/template_iso_svg/, stubs already removed) with the SAME
`render` (600 dpi, 6 px pad) that made data/raw/iso_symbols_clean. Consistency check before
writing: the class's own form, rendered the same way, must match its iso_symbols_clean PNG
(same size, ink IoU >= MIN_IOU). Never writes into iso_symbols_clean.

    python scripts/make_iso_forms.py [--out data/raw/iso_symbols_forms]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from extract_iso_symbols_svg import render  # noqa: E402

# output name -> (SVG of the extra form, SVG of the class's own form, its iso_symbols_clean PNG)
FORMS = {"valve_butterfly@B": ("X8075_2", "X8075_1", "X8075")}
DPI, PAD = 600, 6
MIN_IOU = 0.9


def ink_iou(a: Image.Image, b: Image.Image) -> float:
    x = np.asarray(a.convert("L")) < 128
    y = np.asarray(b.convert("L")) < 128
    if x.shape != y.shape:
        return 0.0
    return float((x & y).sum() / max(1, (x | y).sum()))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--svg-dir", type=Path, default=ROOT / "data/raw/template_iso_svg")
    ap.add_argument("--clean-dir", type=Path, default=ROOT / "data/raw/iso_symbols_clean")
    ap.add_argument("--out", type=Path, default=ROOT / "data/raw/iso_symbols_forms")
    args = ap.parse_args(argv)
    if args.out.resolve() == args.clean_dir.resolve():
        raise SystemExit("--out must not be the template library itself")
    args.out.mkdir(parents=True, exist_ok=True)
    for name, (extra, own, reg) in FORMS.items():
        check = render((args.svg_dir / f"{own}.svg").read_bytes(), dpi=DPI, pad=PAD)
        iou = ink_iou(check, Image.open(args.clean_dir / f"{reg}.png"))
        if iou < MIN_IOU:
            raise SystemExit(f"{own} re-rendered vs {reg}.png: ink IoU {iou:.3f} < {MIN_IOU}")
        img = render((args.svg_dir / f"{extra}.svg").read_bytes(), dpi=DPI, pad=PAD)
        img.save(args.out / f"{name}.png")
        print(
            f"{name}.png <- {extra}.svg  ({img.size[0]}x{img.size[1]}; check {own} IoU {iou:.3f})"
        )


if __name__ == "__main__":
    main()
