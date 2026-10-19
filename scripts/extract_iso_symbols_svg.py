"""Render the one-shot ISO 10628-2 template library from the BK-2020-04 vector sheets.

Source: Steven Baltakatei Sandoval, BK-2020-04-PID-1 (CC BY-SA 4.0),
https://gitlab.com/baltakatei/BK-2020-04, files DWG/BK-2020-04-PID-1-SHT{1..7}.svg. The
PDFs that scripts/extract_iso_symbols.py crops from are exports of the same drawings, so
this changes HOW the templates are cut, not what they depict.

Why vector instead of cropping the PDF render: every symbol in the SVG is its own group
labelled `REG#:<id>_symbol`, so isolating one is exact. The PDF crop is a heuristic
window, which produced a broken X8071 (a tiny symbol plus a stray rule) and aliased
strokes on others (docs/experiments/iso_templates_bk2020/).

Connection stubs: the sheets draw each symbol with short pipe stubs, each a separate
path that is ONE straight segment exactly 5 drawing units long. They are removed by
default so every template is the bare symbol (the old library was inconsistent: some
had stubs, most not). A symbol made of nothing but such segments keeps them — there the
lines are the symbol. --keep-stubs renders them all (an ablation: DigitizePID crops
show pipes attached to the symbol).

Known renderer gap: MuPDF ignores SVG markers, so a symbol whose arrowhead is a marker
(241, flow direction) loses it. Such REG#s are listed in --copy-from-old and copied from
the previous library unchanged.

Output: <out>/<reg>.png (white background, RGB) for every vocab row + <out>/symbols.csv.

Run:
    git clone --depth 1 https://gitlab.com/baltakatei/BK-2020-04.git /tmp/bk2020
    uv run python scripts/extract_iso_symbols_svg.py --svg-dir /tmp/bk2020/DWG
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image

_INK_LABEL = "{http://www.inkscape.org/namespaces/inkscape}label"
_SHEET_GLOB = "BK-2020-04-PID-1-SHT*.svg"

# Elements that never draw anything but may be referenced by what does (gradients,
# markers, CSS) — kept when everything else is pruned.
_STRUCTURAL = ("defs", "style", "namedview", "metadata")

# Connection stub length in drawing units (mm) on these sheets.
_STUB_LENGTH = 5.0
_STUB_TOL = 0.05

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_PAIR = rf"({_NUM})\s*,?\s*({_NUM})"
# "m x,y h dx" / "M x,y H x2" / "m x,y v dy" / "M x,y V y2" / "m x,y l dx,dy" / "M x,y L x2,y2"
_ONE_SEGMENT = re.compile(rf"^\s*([mM])\s*{_PAIR}\s*(?:([hHvV])\s*({_NUM})|([lL])\s*{_PAIR})\s*$")


def symbol_label(reg: str) -> str:
    return f"REG#:{reg}_symbol"


def segment_endpoints(d: str | None) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Start and end point of a path that is exactly one straight segment, else None."""
    if not d:
        return None
    m = _ONE_SEGMENT.match(d)
    if m is None:
        return None
    # The FIRST moveto of a path is absolute even when written "m" (SVG spec), so the
    # start point is (x, y) either way; only the second command's case matters.
    _, x, y, hv, hv_val, line, lx, ly = m.groups()
    x, y = float(x), float(y)
    if hv is not None:
        v = float(hv_val)
        end = {"h": (x + v, y), "v": (x, y + v), "H": (v, y), "V": (x, v)}[hv]
    else:
        dx, dy = float(lx), float(ly)
        end = (x + dx, y + dy) if line == "l" else (dx, dy)
    return (x, y), end


def segment_length(d: str | None) -> float | None:
    """Length of a path that is exactly one straight segment, else None."""
    ends = segment_endpoints(d)
    if ends is None:
        return None
    (x0, y0), (x1, y1) = ends
    return math.hypot(x1 - x0, y1 - y0)


_DASHARRAY = re.compile(r"stroke-dasharray\s*:\s*([^;]+)")


def expand_dashes(target: ET.Element) -> None:
    """Rewrite dashed one-segment paths as explicit dash sub-paths.

    MuPDF ignores stroke-dasharray, so a dashed line (the column internals of X8100)
    would render solid — a different symbol. Only straight single segments are
    expanded; that covers every dashed line in the vocab symbols.
    """
    for elem in target.iter():
        style = elem.get("style") or ""
        m = _DASHARRAY.search(style)
        if m is None or m.group(1).strip() == "none":
            continue
        ends = segment_endpoints(elem.get("d"))
        pattern = [float(v) for v in re.split(r"[\s,]+", m.group(1).strip()) if v]
        if ends is None or not pattern or sum(pattern) <= 0:
            continue
        if len(pattern) % 2:
            pattern *= 2  # SVG: an odd list is repeated to make it even
        (x0, y0), (x1, y1) = ends
        length = math.hypot(x1 - x0, y1 - y0)
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        parts, pos, i = [], 0.0, 0
        while pos < length:
            seg = min(pattern[i % len(pattern)], length - pos)
            if i % 2 == 0:  # even entries draw, odd ones are gaps
                a, b = pos, pos + seg
                parts.append(
                    f"M {x0 + ux * a:.5f},{y0 + uy * a:.5f} L {x0 + ux * b:.5f},{y0 + uy * b:.5f}"
                )
            pos += seg
            i += 1
        elem.set("d", " ".join(parts))
        elem.set("style", _DASHARRAY.sub("stroke-dasharray:none", style))


def is_stub(elem: ET.Element) -> bool:
    if not elem.tag.endswith("path"):
        return False
    length = segment_length(elem.get("d"))
    return length is not None and abs(length - _STUB_LENGTH) <= _STUB_TOL


def isolate(root: ET.Element, reg: str, keep_stubs: bool = False) -> bool:
    """Prune `root` in place down to the REG#:<reg>_symbol group (and its ancestors).

    Returns False when the sheet has no such group. Pruning (not display:none) because
    MuPDF's SVG renderer does not honour display:none in a style attribute.
    """
    parent = {c: p for p in root.iter() for c in p}
    target = next((e for e in root.iter() if e.get(_INK_LABEL) == symbol_label(reg)), None)
    if target is None:
        return False
    keep = set(target.iter())
    node = target
    while node in parent:
        node = parent[node]
        keep.add(node)
    for p in list(root.iter()):
        if p not in keep:
            continue
        for c in list(p):
            if c not in keep and not any(c.tag.endswith(t) for t in _STRUCTURAL):
                p.remove(c)

    # Stubs are found on the original single-segment paths, so dashes are expanded after.
    if not keep_stubs:
        child_parent = {c: p for p in target.iter() for c in p}
        for s in find_stubs(root, target):
            child_parent[s].remove(s)
    expand_dashes(target)
    return True


def _ink_box(root: ET.Element, dpi: int) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1) of the rendered ink on the uncropped page, or None if blank."""
    doc = pymupdf.open(stream=ET.tostring(root), filetype="svg")
    pix = doc[0].get_pixmap(dpi=dpi, alpha=False)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)[..., :3]
    ys, xs = np.nonzero(img.min(axis=2) < 200)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _without(root: ET.Element, target: ET.Element, drop: list[ET.Element]) -> ET.Element:
    """Deep copy of `root` with the `drop` leaves of `target` removed (matched by position)."""
    leaves = [e for e in target.iter() if e is not target and len(e) == 0]
    drop_idx = {i for i, e in enumerate(leaves) if any(e is d for d in drop)}
    path_to_target = []
    node, parent = target, {c: p for p in root.iter() for c in p}
    while node in parent:
        path_to_target.append(list(parent[node]).index(node))
        node = parent[node]
    clone = copy.deepcopy(root)
    t = clone
    for i in reversed(path_to_target):
        t = t[i]
    clone_leaves = [e for e in t.iter() if e is not t and len(e) == 0]
    clone_parent = {c: p for p in t.iter() for c in p}
    for i in sorted(drop_idx):
        clone_parent[clone_leaves[i]].remove(clone_leaves[i])
    return clone


def find_stubs(root: ET.Element, target: ET.Element, dpi: int = 150) -> list[ET.Element]:
    """Connection stubs of an isolated symbol: 5-unit straight segments OUTSIDE its body.

    Length alone is not enough — a gate valve's stem bar (X8074) or the internals of
    2033 are 5-unit segments inside the symbol. So a candidate counts as a stub only if
    less than half of its rendered ink falls inside the bounding box of everything that
    is not a candidate (the body). A symbol made only of candidates (511, a flange drawn
    as bars) has no body and keeps everything: there the lines are the symbol.
    """
    leaves = [e for e in target.iter() if e is not target and len(e) == 0]
    candidates = [e for e in leaves if is_stub(e)]
    if not candidates or len(candidates) == len(leaves):
        return []
    body = _ink_box(_without(root, target, candidates), dpi)
    if body is None:
        return []
    bx0, by0, bx1, by1 = body
    stubs = []
    for cand in candidates:
        alone = _ink_box(_without(root, target, [e for e in leaves if e is not cand]), dpi)
        if alone is None:
            continue
        x0, y0, x1, y1 = alone
        area = (x1 - x0 + 1) * (y1 - y0 + 1)
        ix = max(0, min(x1, bx1) - max(x0, bx0) + 1)
        iy = max(0, min(y1, by1) - max(y0, by0) + 1)
        if ix * iy < 0.5 * area:
            stubs.append(cand)
    return stubs


def render(svg_bytes: bytes, dpi: int, pad: int) -> Image.Image | None:
    """Rasterize and crop to the ink bounding box (+pad) on white. None if blank."""
    doc = pymupdf.open(stream=svg_bytes, filetype="svg")
    pix = doc[0].get_pixmap(dpi=dpi, alpha=False)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)[..., :3]
    ys, xs = np.nonzero(img.min(axis=2) < 200)
    if xs.size == 0:
        return None
    y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad + 1)
    return Image.fromarray(np.ascontiguousarray(img[y0:y1, x0:x1]), mode="RGB")


def _row(reg: str, vocab_row: dict[str, str], source: str, png: Path) -> dict[str, str]:
    return {
        "reg_number": reg,
        "desc": vocab_row.get("desc", ""),
        "source": source,
        "file": png.name,
    }


def find_sheet(sheets: list[Path], reg: str) -> Path | None:
    needle = f'"{symbol_label(reg)}"'
    return next((s for s in sheets if needle in s.read_text(encoding="utf-8")), None)


def extract(
    svg_dir: Path,
    vocab_csv: Path,
    out_dir: Path,
    dpi: int = 600,
    pad: int = 6,
    keep_stubs: bool = False,
    copy_from_old: dict[str, Path] | None = None,
) -> list[dict[str, str]]:
    sheets = sorted(svg_dir.glob(_SHEET_GLOB))
    if not sheets:
        raise FileNotFoundError(f"no {_SHEET_GLOB} under {svg_dir}")
    with vocab_csv.open(newline="", encoding="utf-8") as f:
        vocab = list(csv.DictReader(f))
    copy_from_old = copy_from_old or {}
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    missing: list[str] = []
    for row in vocab:
        reg = row["reg_number"].strip()
        out_png = out_dir / f"{reg}.png"
        if reg in copy_from_old:
            shutil.copyfile(copy_from_old[reg], out_png)
            rows.append(_row(reg, row, "old_pdf_crop", out_png))
            continue
        sheet = find_sheet(sheets, reg)
        if sheet is None:
            missing.append(reg)
            continue
        tree = ET.parse(sheet)
        isolate(tree.getroot(), reg, keep_stubs=keep_stubs)
        img = render(ET.tostring(tree.getroot()), dpi=dpi, pad=pad)
        if img is None:
            missing.append(reg)
            continue
        img.save(out_png)
        rows.append(_row(reg, row, sheet.name, out_png))
    if missing:
        raise RuntimeError(f"no renderable symbol group for REG# {missing}")

    with (out_dir / "symbols.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["reg_number", "desc", "source", "file"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--svg-dir", type=Path, required=True, help="BK-2020-04 DWG/ directory")
    ap.add_argument("--vocab", type=Path, default=Path("data/raw/iso_symbols/vocab.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/raw/iso_symbols_clean"))
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--pad", type=int, default=6, help="white margin around the ink, px")
    ap.add_argument("--keep-stubs", action="store_true", help="keep 5-unit connection stubs")
    ap.add_argument(
        "--copy-from-old",
        nargs="*",
        default=["241"],
        help="REG#s the SVG render gets wrong (markers); copied from --old-dir unchanged",
    )
    ap.add_argument("--old-dir", type=Path, default=Path("data/raw/iso_symbols_clean"))
    args = ap.parse_args()

    # Copy the old PNGs aside first: --old-dir and --out are the same directory by default.
    with tempfile.TemporaryDirectory() as stash:
        old = {}
        for reg in args.copy_from_old:
            src = args.old_dir / f"{reg}.png"
            if not src.exists():
                raise FileNotFoundError(f"--copy-from-old {reg}: {src} does not exist")
            old[reg] = Path(shutil.copy(src, Path(stash) / src.name))
        rows = extract(args.svg_dir, args.vocab, args.out, args.dpi, args.pad, args.keep_stubs, old)
    print(f"wrote {len(rows)} templates -> {args.out}")


if __name__ == "__main__":
    main()
