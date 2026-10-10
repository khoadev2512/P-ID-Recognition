"""Render synthetic P&ID images from the ISO symbol library, with auto COCO annotations.

This replaces DigitizePID (a mixed-standard set) with a 100%-ISO synthetic dataset for
the detection + fine-grained-classification pipeline. Output drops straight into
data/canonical/ so the existing prep -> train pipeline runs unchanged.

Design follows docs/RENDER_RULES.md. Placement is RANDOM (no connectivity rules) — the
same choice SynthPID and PIDreader make, because detection/FGC learn local symbol
appearance, not process topology (that only matters for graph extraction). Each image:

  1. background  : a real empty patch cropped from DigitizePID (lines/notes/scan noise),
                   NOT a blank canvas — the context the detector must learn to ignore.
  2. symbols     : N random ISO symbols, balanced across coarse classes + FGC families,
                   scaled (no rotation/flip — P&ID symbols are orientation-meaningful),
                   pasted white-transparent so only the ink lands on the background.
  3. pipes       : L-shaped lines between some symbols (visual realism; NOT annotated).
  4. tags        : short text labels near symbols (e.g. "V-101"; NOT annotated).
  5. degrade     : noise + blur + light pixelation, to close the clean-vs-scan gap.
  6. annotation  : one COCO bbox per pasted symbol, category = its fine_name.

Run (after extract_iso_symbols + extract_bg_patches):
    uv run python scripts/render_pid.py --n-images 400
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ------------------------------------------------------------------ #
# library + config
# ------------------------------------------------------------------ #
@dataclass
class SymbolTemplate:
    reg: str
    fine_name: str
    coarse: str
    is_fgc: bool
    img: Image.Image  # RGBA, white made transparent


@dataclass
class RenderConfig:
    canvas_min: int = 1024          # patches are 1024; keep it simple
    symbols_min: int = 15
    symbols_max: int = 40
    scale_min: float = 0.7
    scale_max: float = 1.5
    max_place_tries: int = 60       # per symbol, to find a clear non-overlapping spot
    max_bg_ink: float = 0.02        # max background ink under a symbol (avoid text/legend)
    fgc_boost: float = 2.0          # relative sampling weight for FGC-family symbols
    pipe_prob: float = 0.5          # fraction of symbols that get a pipe stub to a peer
    tag_prob: float = 0.6           # fraction of symbols that get a text tag
    # degradation
    noise_sigma: float = 8.0
    blur_max: float = 1.2
    pixelate_min: float = 0.6       # downscale factor floor before upscaling back
    margin: int = 24                # keep symbols off the very edge
    seed: int = 42
    # feature switches (for step-by-step verification)
    do_pipes: bool = True
    do_tags: bool = True
    do_degrade: bool = True


def _white_to_transparent(img: Image.Image, thresh: int = 235) -> Image.Image:
    """RGB symbol (white background) -> RGBA with near-white pixels made transparent,
    so pasting leaves only the ink (black strokes) over the background."""
    rgba = img.convert("RGBA")
    arr = np.array(rgba)
    white = (arr[:, :, 0] > thresh) & (arr[:, :, 1] > thresh) & (arr[:, :, 2] > thresh)
    arr[white, 3] = 0
    return Image.fromarray(arr)


def load_library(symbols_dir: Path, vocab_csv: Path) -> list[SymbolTemplate]:
    """Load the ISO symbol templates named in vocab.csv (skips any missing PNG)."""
    templates = []
    for row in csv.DictReader(vocab_csv.open()):
        png = symbols_dir / f"{row['reg_number']}.png"
        if not png.exists():
            print(f"  warn: missing {png}, skipping {row['reg_number']}")
            continue
        img = _white_to_transparent(Image.open(png))
        templates.append(
            SymbolTemplate(
                reg=row["reg_number"],
                fine_name=row["fine_name"],
                coarse=row["coarse"],
                is_fgc=row["fgc_group"].strip().lower() == "yes",
                img=img,
            )
        )
    return templates


# ------------------------------------------------------------------ #
# placement
# ------------------------------------------------------------------ #
@dataclass
class Placed:
    fine_name: str
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 (pixels)
    anchor: tuple[int, int] = field(default=(0, 0))  # center, for pipes


def _overlaps(box, placed: list[Placed], pad: int = 8) -> bool:
    x0, y0, x1, y1 = box
    for p in placed:
        px0, py0, px1, py1 = p.box
        if x0 - pad < px1 and x1 + pad > px0 and y0 - pad < py1 and y1 + pad > py0:
            return True
    return False


def _sample_template(templates: list[SymbolTemplate], cfg: RenderConfig, rng) -> SymbolTemplate:
    """Weighted pick: boost FGC-family symbols so each fine class gets enough samples."""
    weights = [cfg.fgc_boost if t.is_fgc else 1.0 for t in templates]
    return rng.choices(templates, weights=weights, k=1)[0]


def _region_is_clear(bg_gray: np.ndarray, box, max_ink: float) -> bool:
    """True if the background region under `box` is mostly white (ink ratio <= max_ink).

    Background patches carry the real drawing's note tables / legends / borders. Pasting
    a symbol there would bury it under text. We only place where the background is clear,
    so symbols land on empty drawing space — the same place real P&ID symbols sit.
    """
    x0, y0, x1, y1 = box
    region = bg_gray[y0:y1, x0:x1]
    if region.size == 0:
        return False
    return float((region < 200).mean()) <= max_ink


def place_symbols(
    canvas: Image.Image, templates: list[SymbolTemplate], cfg: RenderConfig, rng
) -> list[Placed]:
    """Paste N random, non-overlapping, scaled symbols onto CLEAR background space."""
    W, H = canvas.size
    bg_gray = np.array(canvas.convert("L"))  # snapshot: judge clearness vs ORIGINAL bg
    n = rng.randint(cfg.symbols_min, cfg.symbols_max)
    placed: list[Placed] = []
    for _ in range(n):
        t = _sample_template(templates, cfg, rng)
        scale = rng.uniform(cfg.scale_min, cfg.scale_max)
        sw = max(8, int(t.img.width * scale))
        sh = max(8, int(t.img.height * scale))
        if sw >= W - 2 * cfg.margin or sh >= H - 2 * cfg.margin:
            continue
        sym = t.img.resize((sw, sh), Image.LANCZOS)
        for _ in range(cfg.max_place_tries):
            x0 = rng.randint(cfg.margin, W - cfg.margin - sw)
            y0 = rng.randint(cfg.margin, H - cfg.margin - sh)
            box = (x0, y0, x0 + sw, y0 + sh)
            if _overlaps(box, placed):
                continue
            # don't paste over the background's own text/legend/border
            if not _region_is_clear(bg_gray, box, cfg.max_bg_ink):
                continue
            canvas.paste(sym, (x0, y0), sym)
            placed.append(Placed(t.fine_name, box, (x0 + sw // 2, y0 + sh // 2)))
            break
    return placed


# ------------------------------------------------------------------ #
# pipes + tags + degradation
# ------------------------------------------------------------------ #
def draw_pipes(canvas: Image.Image, placed: list[Placed], cfg: RenderConfig, rng) -> None:
    """L-shaped lines between some symbol pairs. Visual context only — NOT annotated."""
    draw = ImageDraw.Draw(canvas)
    for p in placed:
        if rng.random() > cfg.pipe_prob or len(placed) < 2:
            continue
        q = rng.choice(placed)
        if q is p:
            continue
        (ax, ay), (bx, by) = p.anchor, q.anchor
        # L-route: horizontal then vertical (Manhattan), 2-3 px black line
        draw.line([(ax, ay), (bx, ay)], fill=(0, 0, 0), width=rng.randint(2, 3))
        draw.line([(bx, ay), (bx, by)], fill=(0, 0, 0), width=rng.randint(2, 3))


_TAG_PREFIX = ["V", "P", "FT", "PT", "TT", "LV", "HX", "E"]


def draw_tags(canvas: Image.Image, placed: list[Placed], cfg: RenderConfig, rng) -> None:
    """Short tags near symbols (e.g. 'V-101'). Realism only — NOT annotated."""
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for p in placed:
        if rng.random() > cfg.tag_prob:
            continue
        tag = f"{rng.choice(_TAG_PREFIX)}-{rng.randint(100, 999)}"
        x0, y0, _, _ = p.box
        draw.text((x0, max(0, y0 - 12)), tag, fill=(40, 40, 40), font=font)


def degrade(canvas: Image.Image, cfg: RenderConfig, rng) -> Image.Image:
    """Noise + blur + light pixelation to mimic a scanned drawing."""
    from PIL import ImageFilter

    img = canvas.convert("RGB")
    # pixelate: downscale then upscale
    if rng.random() < 0.5:
        f = rng.uniform(cfg.pixelate_min, 1.0)
        if f < 0.99:
            w, h = img.size
            img = img.resize((max(1, int(w * f)), max(1, int(h * f))), Image.BILINEAR)
            img = img.resize((w, h), Image.BILINEAR)
    # blur
    b = rng.uniform(0.0, cfg.blur_max)
    if b > 0.1:
        img = img.filter(ImageFilter.GaussianBlur(b))
    # gaussian noise
    arr = np.array(img).astype(np.int16)
    noise = np.random.default_rng(rng.randint(0, 2**31)).normal(0, cfg.noise_sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ------------------------------------------------------------------ #
# COCO assembly
# ------------------------------------------------------------------ #
def build_categories(templates: list[SymbolTemplate]) -> tuple[list[dict], dict[str, int]]:
    """COCO categories: one per fine class, supercategory = its coarse. 1-indexed ids,
    ordered by first appearance so ids are stable across runs with the same vocab.

    Each category also carries `fgc` (bool): whether its coarse is a visually-similar
    family that needs the FGC stage. This comes from vocab.csv's hand-marked fgc_group
    column, NOT from "coarse has >1 fine" — with a detailed ISO vocab many coarse classes
    have several fine children that are NOT visually similar (pump centrifugal vs gear vs
    diaphragm look clearly different), so the >1-fine heuristic over-triggers. build_classes
    reads this flag when present; see its fgc_groups logic.
    """
    fine_to_coarse: dict[str, str] = {}
    coarse_is_fgc: dict[str, bool] = {}
    for t in templates:
        fine_to_coarse.setdefault(t.fine_name, t.coarse)
        # a coarse is an FGC family if ANY of its symbols is marked fgc (they agree in vocab)
        coarse_is_fgc[t.coarse] = coarse_is_fgc.get(t.coarse, False) or t.is_fgc
    cats, name_to_id = [], {}
    for i, (fine, coarse) in enumerate(fine_to_coarse.items(), start=1):
        cats.append({"id": i, "name": fine, "supercategory": coarse, "fgc": coarse_is_fgc[coarse]})
        name_to_id[fine] = i
    return cats, name_to_id


def render_dataset(
    bg_dir: Path,
    templates: list[SymbolTemplate],
    out_dir: Path,
    n_images: int,
    cfg: RenderConfig,
    val_ratio: float = 0.2,
) -> None:
    rng = random.Random(cfg.seed)
    bg_files = sorted(bg_dir.glob("*.png"))
    if not bg_files:
        raise FileNotFoundError(f"No background patches in {bg_dir} (run extract_bg_patches first)")

    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    cats, name_to_id = build_categories(templates)

    coco_images, coco_anns = [], []
    split_map: dict[str, str] = {}
    ann_id = 0
    n_val = int(n_images * val_ratio)

    for i in range(n_images):
        bg = Image.open(rng.choice(bg_files)).convert("RGB").copy()
        placed = place_symbols(bg, templates, cfg, rng)
        if cfg.do_pipes:
            draw_pipes(bg, placed, cfg, rng)
        if cfg.do_tags:
            draw_tags(bg, placed, cfg, rng)
        final = degrade(bg, cfg, rng) if cfg.do_degrade else bg.convert("RGB")

        image_id = f"iso_{i:05d}"
        final.save(images_dir / f"{image_id}.jpg", quality=92)
        split_map[image_id] = "val" if i < n_val else "train"

        coco_images.append(
            {"id": i + 1, "file_name": f"synthetic/{image_id}.jpg",
             "width": final.width, "height": final.height}
        )
        for p in placed:
            x0, y0, x1, y1 = p.box
            ann_id += 1
            coco_anns.append(
                {"id": ann_id, "image_id": i + 1, "category_id": name_to_id[p.fine_name],
                 "bbox": [x0, y0, x1 - x0, y1 - y0], "area": (x1 - x0) * (y1 - y0), "iscrowd": 0}
            )

    coco = {"images": coco_images, "annotations": coco_anns, "categories": cats}
    (out_dir / "annotations.coco.json").write_text(json.dumps(coco))
    (out_dir / "digitizepid_splits.json").write_text(json.dumps(split_map, indent=2))
    n_coarse = len({c["supercategory"] for c in cats})
    print(f"Rendered {n_images} images, {ann_id} annotations -> {out_dir}")
    print(f"  categories: {len(cats)} fine classes, {n_coarse} coarse")
    print(f"  split: {n_images - n_val} train / {n_val} val")


def main() -> None:
    p = argparse.ArgumentParser(description="Render synthetic ISO P&ID images + COCO")
    p.add_argument("--symbols-dir", default="data/raw/iso_symbols_clean")
    p.add_argument("--vocab", default="data/raw/iso_symbols/vocab.csv")
    p.add_argument("--bg-dir", default="data/raw/bg_patches")
    p.add_argument("--out", default="data/canonical")
    p.add_argument("--n-images", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-pipes", action="store_true")
    p.add_argument("--no-tags", action="store_true")
    p.add_argument("--no-degrade", action="store_true")
    args = p.parse_args()

    cfg = RenderConfig(
        seed=args.seed,
        do_pipes=not args.no_pipes,
        do_tags=not args.no_tags,
        do_degrade=not args.no_degrade,
    )
    templates = load_library(Path(args.symbols_dir), Path(args.vocab))
    print(f"Loaded {len(templates)} symbol templates "
          f"({sum(t.is_fgc for t in templates)} in FGC families)")
    render_dataset(Path(args.bg_dir), templates, Path(args.out), args.n_images, cfg)


if __name__ == "__main__":
    main()
