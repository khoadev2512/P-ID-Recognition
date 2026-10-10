"""Extract individual ISO 10628-2 symbol images from the standard's symbol-sheet PDF.

The PDF (ISO_10628-2_2012_Symbols.pdf, 7 pages, ~304 symbols) lays symbols out in a
regular grid: each symbol's graphic sits to the LEFT of a two-line text label
`REG#:<id>` / `DESC:<description>`. We locate every REG# text span, then crop a fixed
window to its left (where the graphic lives) from a high-DPI render of the page.

Output:
    data/raw/iso_symbols/<reg>.png        one cropped symbol per REG#
    data/raw/iso_symbols/symbols.csv      reg_number, desc, page, group (if detectable)

This is the symbol LIBRARY for the synthetic P&ID renderer (scripts/render_pid.py,
coming next) — the ISO-standard replacement for the mixed-standard DigitizePID set.

Semi-automatic: the crop window is a heuristic (symbols vary in size), so eyeball the
output and widen --sym-w / --sym-h or fix individual crops by hand if a few clip.

Run:
    uv run python scripts/extract_iso_symbols.py
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pymupdf

# Grid geometry read off the PDF (page 0): REG# text columns sit at x≈261/828/1395/1962,
# columns are ~567px apart, and the symbol graphic occupies the band to the left of its
# REG#. These are defaults in PDF points; --dpi scales the actual render.
_REG_RE = re.compile(r"REG#\s*:?\s*(\S+)")
_DESC_RE = re.compile(r"DESC\s*:?\s*(.+)")

# Hand-tuned crop boxes (page_index, x0, y0, x1, y1 in PDF points) for the handful of
# symbols the automatic vector-bbox misses — small valve/piping glyphs on page 6 that
# sit flush against group-box rules, so the auto filter either clips them or grabs the
# rule. Verified by eye. Any REG# here overrides the automatic crop.
_OVERRIDE_BOXES: dict[str, tuple[int, float, float, float, float]] = {
    "2101": (5, 72, 40, 227, 82),     # valve (general)
    "X8074": (5, 72, 777, 227, 819),  # valve, gate
    "2102": (5, 43, 118, 198, 160),   # valve, angle
    "2103": (5, 43, 206, 227, 248),   # valve, three-way
    "X8075": (5, 72, 855, 227, 897),  # valve, butterfly
    "405": (5, 1773, 30, 1998, 72),   # pipeline
    "511": (5, 1250, 845, 1370, 895),  # flanged connection
    "X2124": (5, 639, 35, 794, 115),  # safety valve, spring loaded
    "301": (0, 155, 60, 228, 180),    # tank, vessel (tight: body + short stubs)
}


# Every sheet repeats a LEGEND box in its bottom-right corner that uses REG#:301
# (Tank, vessel) as a worked example. That REG# text sits at a fixed spot (~x>1850,
# y>1350 in PDF points) on all 7 pages, so without excluding it we'd extract the legend
# example 7 times and overwrite the real symbol. Skip any REG# inside this region.
_LEGEND_X_MIN = 1850.0
_LEGEND_Y_MIN = 1350.0


def _in_legend(x: float, y: float) -> bool:
    return x > _LEGEND_X_MIN and y > _LEGEND_Y_MIN


def _page_label_items(page) -> list[dict]:
    """Return [{reg, desc, x, y}] — one per symbol on the page, from its REG#/DESC text.

    REG# and DESC are separate lines at the same x; we pair a DESC to the nearest REG#
    just below-ish it (same column, DESC sits ~18px under REG#). REG#s inside the
    bottom-right legend box are skipped (see _in_legend) so the legend's worked example
    doesn't masquerade as a real symbol.
    """
    regs: list[tuple[float, float, str]] = []
    descs: list[tuple[float, float, str]] = []
    for b in page.get_text("dict")["blocks"]:
        for line in b.get("lines", []):
            txt = " ".join(s["text"] for s in line.get("spans", [])).strip()
            x0, y0 = line["bbox"][0], line["bbox"][1]
            if _in_legend(x0, y0):
                continue
            m = _REG_RE.search(txt)
            if m:
                regs.append((x0, y0, m.group(1)))
                continue
            m = _DESC_RE.search(txt)
            if m:
                descs.append((x0, y0, m.group(1).strip()))

    items = []
    for x, y, reg in regs:
        # nearest DESC in the same column (|dx|<40) just below the REG# (0<dy<60)
        best = None
        for dx_, dy_, desc in descs:
            if abs(dx_ - x) < 40 and 0 < (dy_ - y) < 60:
                if best is None or (dy_ - y) < best[0]:
                    best = (dy_ - y, desc)
        items.append({"reg": reg, "desc": best[1] if best else "", "x": x, "y": y})
    return items


def _safe_name(reg: str) -> str:
    """Filesystem-safe symbol file stem from a REG# (e.g. 'X8074' or 'C0001')."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", reg)


def _symbol_bbox_from_vectors(page, rx: float, ry: float, pad: float = 6.0):
    """Tight bbox of a symbol's graphic, from the vector strokes in its grid cell.

    A cell's graphic sits just LEFT of its REG# at (rx, ry). We gather drawing rects
    that: (a) end left of the REG# (x1 < rx), (b) start within ~230pt of it (same cell,
    not the neighbour), (c) sit on the REG#/DESC row (vertical center near ry), and
    (d) aren't table rules (too wide/flat) or stray text ticks (too tiny). Returns a
    padded pymupdf.Rect, or None if nothing plausible is found (caller falls back to a
    fixed window).
    """
    cand = []
    for d in page.get_drawings():
        r = d["rect"]
        cy = (r.y0 + r.y1) / 2
        # Tight row band (±48/+70 of the REG#) so a symbol in the row above/below the
        # one we want doesn't get swept in.
        if not (r.x1 < rx - 2 and r.x0 > rx - 230 and (ry - 48) < cy < (ry + 70)):
            continue
        # Drop grid rules: long vertical cell dividers (tall, hairline) and long
        # horizontal row/group separators (wide, hairline). Both bleed in otherwise.
        is_vertical_rule = r.height > 120 and r.width < 4
        is_horizontal_rule = r.width > 130 and r.height < 4
        if is_vertical_rule or is_horizontal_rule:
            continue
        if 2 < r.height < 150 and r.width < 200:
            cand.append(r)
    if not cand:
        return None
    # Keep the cluster nearest the REG# (rightmost strokes): symbol graphic hugs the
    # right of its cell. Anchor on the rightmost stroke, keep strokes within one symbol
    # width of it — drops a stray left-edge rule the filters missed.
    cand.sort(key=lambda r: r.x1)
    right = cand[-1].x1
    core = [r for r in cand if r.x0 > right - 170]
    if not core:
        core = cand
    x0 = min(r.x0 for r in core) - pad
    y0 = min(r.y0 for r in core) - pad
    x1 = max(r.x1 for r in core) + pad
    y1 = max(r.y1 for r in core) + pad
    return pymupdf.Rect(max(0, x0), max(0, y0), x1, y1)


def extract(
    pdf_path: Path,
    out_dir: Path,
    dpi: int = 300,
    sym_w: float = 150.0,
    sym_h: float = 95.0,
    x_gap: float = 8.0,
    only: set[str] | None = None,
) -> list[dict]:
    """Crop every symbol graphic to out_dir/<reg>.png, return the metadata rows.

    Prefers a tight vector-derived bbox (accurate); falls back to a fixed window
    (sym_w x sym_h, left of REG#) when no strokes are found. `only` restricts to a set
    of REG#s (e.g. the chosen vocab) — handy for cleanly re-cropping just those.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(pdf_path)
    mat = pymupdf.Matrix(dpi / 72.0, dpi / 72.0)  # PDF is 72 dpi; scale up to target dpi

    rows: list[dict] = []
    for page_index, page in enumerate(doc):
        items = _page_label_items(page)
        for it in items:
            if only is not None and it["reg"] not in only:
                continue
            rx, ry = it["x"], it["y"]
            override = _OVERRIDE_BOXES.get(it["reg"])
            if override is not None and override[0] == page_index:
                clip = pymupdf.Rect(*override[1:])
            else:
                clip = _symbol_bbox_from_vectors(page, rx, ry)
            if clip is None or clip.is_empty:
                # fallback: fixed window to the LEFT of the REG#, centered on the row.
                x1 = rx - x_gap
                clip = pymupdf.Rect(
                    max(0, x1 - sym_w), max(0, ry - sym_h * 0.55), x1, ry + sym_h * 0.45
                )
            if clip.is_empty:
                continue
            sub = page.get_pixmap(matrix=mat, clip=clip)
            name = _safe_name(it["reg"])
            sub.save(str(out_dir / f"{name}.png"))
            rows.append(
                {
                    "reg_number": it["reg"],
                    "desc": it["desc"],
                    "page": page_index + 1,
                    "file": f"{name}.png",
                }
            )

    # write the library manifest
    csv_path = out_dir / "symbols.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["reg_number", "desc", "page", "file"])
        w.writeheader()
        w.writerows(rows)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Extract ISO 10628-2 symbols from the symbol-sheet PDF")
    p.add_argument("--pdf", default="ISO_10628-2_2012_Symbols.pdf")
    p.add_argument("--out", default="data/raw/iso_symbols")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--sym-w", type=float, default=150.0, help="crop width (PDF points)")
    p.add_argument("--sym-h", type=float, default=95.0, help="crop height (PDF points)")
    p.add_argument("--vocab", default=None,
                   help="CSV with a reg_number column; crop ONLY those symbols (cleanly)")
    args = p.parse_args()

    only = None
    if args.vocab:
        only = {r["reg_number"] for r in csv.DictReader(open(args.vocab))}

    rows = extract(
        Path(args.pdf), Path(args.out), dpi=args.dpi, sym_w=args.sym_w, sym_h=args.sym_h, only=only
    )
    print(f"Extracted {len(rows)} symbols -> {args.out}")
    print(f"Manifest: {args.out}/symbols.csv")
    print("\nFirst few:")
    for r in rows[:6]:
        print(f"  {r['reg_number']:8} p{r['page']}  {r['desc'][:45]}")


if __name__ == "__main__":
    main()
