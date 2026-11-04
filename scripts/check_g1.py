"""G1 check for PLAN_ISO_RENDER_PID2GRAPH: is the V5 page drawing good enough to go on?

Draws one OPEN100 sheet (V2 -> V3 -> V4 -> V5, one seed) and measures the three automatic G1
criteria; the fourth (a person looks at the picture) is the PNG this writes.

  1. every pipe-to-symbol port has ink on both the pipe layer and the symbol layer within 1 px
  2. stroke thickness of the symbol layer and of the pipe layer differ by at most 1 px
  3. no symbol pair overlaps that did not overlap in the PID2Graph boxes (ISO symbols involved)
  5. every graph edge between two drawn symbols has ink joining them, or the symbols touch
  6. a signal line to a tag does not cut any symbol's ink between its ends (a valve's stem, which
     starts in the middle of the valve, is left out)
  7. no two tags' circles overlap by more than a stroke

Writes, under --out: sheet<i>.svg, sheet<i>.png, g1_report.txt. Read-only on data/raw/.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.iso_layout import transform_layout  # noqa: E402
from utils.iso_mapping import assign_symbols, load_config  # noqa: E402
from utils.iso_templates import load_templates  # noqa: E402
from utils.page_svg import Page, rasterize, render_page, touching_pairs  # noqa: E402
from utils.pid2graph import load_graphml, pipe_runs  # noqa: E402

PORT_TOL_PX = 1  # criterion 1: ink within this many px of the port
THICKNESS_TOL_PX = 1.0  # criterion 2
STEM_END_PX = 6.0  # criterion 6: the part of a signal line at either end that may sit on ink


def port_ink(page: Page, pipes: np.ndarray, symbols: np.ndarray, tol: int = PORT_TOL_PX):
    """Ports where the pipe and the symbol do not meet; returns (ports checked, misses).

    A port passes when the pipe layer has ink within `tol` px of it and the symbol layer has ink
    within `tol` px of that pipe ink near the port: pipe and symbol touch there. (The pipe runs
    half a stroke into the symbol, so a port a fraction of a mm off the drawing still joins.)

    A signal line that enters a flange pair's gap (`p["gap"]`: pipe axis, half the gap) starts in
    empty space between the two marks, as in OPEN100: it passes when the pipe layer has ink
    there and the symbol layer has ink on both sides of it, along the pipe axis, within the gap.
    """
    kernel = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    touch = cv2.dilate(symbols.astype(np.uint8), kernel).astype(bool)
    misses: list[dict] = []
    for p in page.ports:
        x, y = int(round(p["x"])), int(round(p["y"]))
        near = (slice(max(y - tol, 0), y + tol + 1), slice(max(x - tol, 0), x + tol + 1))
        zone = (slice(max(y - tol - 2, 0), y + tol + 3), slice(max(x - tol - 2, 0), x + tol + 3))
        lacks = []
        if not pipes[near].any():
            lacks.append("pipe")
        if "gap" in p:
            reach = int(np.ceil(p["gap"]["half"])) + tol + 1
            if p["gap"]["axis"] == "H":
                sides = (
                    symbols[near[0], max(x - reach, 0) : x],
                    symbols[near[0], x + 1 : x + reach + 1],
                )
            else:
                sides = (
                    symbols[max(y - reach, 0) : y, near[1]],
                    symbols[y + 1 : y + reach + 1, near[1]],
                )
            if not all(side.any() for side in sides):
                lacks.append("gap")
        elif not (pipes[zone] & touch[zone]).any():
            lacks.append("touch")
        if lacks:
            misses.append({**p, "lacks": lacks})
    return len(page.ports), misses


def edges_joined(page: Page, runs, pipes: np.ndarray, symbols: np.ndarray, margin: int = 40):
    """Pairs of drawn symbols linked in the graph that the picture leaves apart.

    For every pair joined by a run, look at the window around the two symbols with every other
    symbol's drawing rubbed out (pipes stay): the pair passes when their ink boxes touch, or
    when a piece of ink in the window reaches both. A node drawn as several boxes (a disc and
    the flanges it sits between) counts as their union. Returns (pairs checked, [(a, b, gap)]).
    """
    boxes = {a["id"]: a["bbox"] for a in page.annotations}
    node = {a["id"]: a.get("node", a["id"]) for a in page.annotations}
    drawn: dict[str, list[float]] = {}
    for aid, b in boxes.items():
        u = drawn.setdefault(node[aid], list(b))
        u[:] = [min(u[0], b[0]), min(u[1], b[1]), max(u[2], b[2]), max(u[3], b[3])]
    pairs = {
        frozenset((r.a, r.b))
        for r in runs
        if not r.internal and r.a != r.b and r.a in drawn and r.b in drawn
    }
    misses = []
    for pair in sorted(tuple(sorted(p)) for p in pairs):
        a, b = pair
        ba, bb = drawn[a], drawn[b]
        gap = max(max(ba[0], bb[0]) - min(ba[2], bb[2]), max(ba[1], bb[1]) - min(ba[3], bb[3]), 0)
        if gap <= 1.5:
            continue
        x0, y0 = int(min(ba[0], bb[0])) - margin, int(min(ba[1], bb[1])) - margin
        x1, y1 = int(max(ba[2], bb[2])) + margin, int(max(ba[3], bb[3])) + margin
        x0, y0 = max(x0, 0), max(y0, 0)
        win_pipes = pipes[y0:y1, x0:x1]
        win_syms = symbols[y0:y1, x0:x1].copy()

        def rect(box, grow=2, x0=x0, y0=y0):
            return (
                slice(max(int(box[1]) - grow - y0, 0), max(int(box[3]) + grow + 1 - y0, 0)),
                slice(max(int(box[0]) - grow - x0, 0), max(int(box[2]) + grow + 1 - x0, 0)),
            )

        keep = np.zeros_like(win_syms, dtype=bool)
        keep[rect(ba)] = True
        keep[rect(bb)] = True
        for other, box in boxes.items():
            apart_from_window = box[2] < x0 or box[0] > x1 or box[3] < y0 or box[1] > y1
            if node[other] not in pair and not apart_from_window:
                rub = np.zeros_like(win_syms, dtype=bool)
                rub[rect(box, 0)] = True
                win_syms[rub & ~keep] = False  # other symbols are not a way across, pipes are
        win = (win_pipes | win_syms).astype(np.uint8)
        _, labels = cv2.connectedComponents(win, connectivity=8)
        la, lb = (
            np.unique(labels[rect(ba)] * (win[rect(ba)] > 0)),
            np.unique(labels[rect(bb)] * (win[rect(bb)] > 0)),
        )
        if not (set(la) & set(lb) - {0}):
            misses.append((a, b, round(float(gap), 1)))
    return len(pairs), misses


def stems_clear(page: Page, symbols: np.ndarray) -> tuple[int, list[dict]]:
    """Signal lines (`page.stems`, a valve's own left out) that run over symbol ink away from
    their ends: the line starts on its symbol and ends on the tag's circle, and in between it
    must be over blank paper. Returns (lines checked, [{id, bubble, x, y}])."""
    h, w = symbols.shape
    checked, cuts = 0, []
    for st in page.stems:
        if st["from_centre"]:
            continue
        checked += 1
        pts = st["points"]
        for a, b in zip(pts, pts[1:], strict=False):
            n = max(int(math.dist(a, b) * 2), 1)
            for k in range(n + 1):
                x = a[0] + (b[0] - a[0]) * k / n
                y = a[1] + (b[1] - a[1]) * k / n
                if (
                    math.dist((x, y), pts[0]) <= STEM_END_PX
                    or math.dist((x, y), pts[-1]) <= STEM_END_PX
                ):
                    continue
                r, c = int(round(y)), int(round(x))
                if 0 <= r < h and 0 <= c < w and symbols[r, c]:
                    cuts.append({"id": st["id"], "bubble": st["bubble"], "x": x, "y": y})
                    break
            else:
                continue
            break
    return checked, cuts


def bubble_overlaps(page: Page, w_px: float) -> tuple[int, list[tuple[str, str, float]]]:
    """Pairs of tags (bubbles) whose circles overlap by more than one stroke: the box of a tag is
    its circle's square grown by half a stroke. Returns (tags, [(a, b, overlap px)])."""
    circles = {}
    for a in page.annotations:
        if a["kind"] == "bubble":
            x0, y0, x1, y1 = a["bbox"]
            circles[a["id"]] = ((x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0 - w_px) / 2)
    out = []
    for (a, ca), (b, cb) in combinations(circles.items(), 2):
        over = ca[2] + cb[2] - math.hypot(ca[0] - cb[0], ca[1] - cb[1])
        if over > w_px:
            out.append((a, b, round(over, 1)))
    return len(circles), out


def stroke_thickness(ink: np.ndarray, max_px: float = 12.0) -> float:
    """Median line width in px: twice the distance transform on the ridge, less the centre pixel.

    Ridge = local maxima of the transform; blobs thicker than `max_px` (filled arrows, dots) are
    left out so they do not count as line width.
    """
    dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    ridge = (dist >= cv2.dilate(dist, np.ones((3, 3), np.uint8))) & (dist > 0.5)
    vals = dist[ridge]
    vals = vals[2 * vals <= max_px]
    return float(2 * np.median(vals) - 1) if vals.size else float("nan")


def _overlap(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def new_overlaps(page: Page, source, tol: float) -> tuple[list[tuple[str, str, bool]], int]:
    """Pairs of drawn symbols whose ink boxes overlap but whose source boxes were neither
    overlapping nor within `tol` px (the rule V5 itself uses, `page_svg.touching_pairs`).

    Returns ([(a, b, iso_involved)], touching pairs in the source)."""
    boxes = {a["id"]: a["bbox"] for a in page.annotations}
    kinds = {a["id"]: a["kind"] for a in page.annotations}
    node = {a["id"]: a.get("node", a["id"]) for a in page.annotations}  # flanges: their disc's
    touching = touching_pairs(source, sorted(set(node.values())), tol)
    out = []
    for a, b in combinations(boxes, 2):
        if node[a] == node[b]:
            continue  # a disc and the flanges it sits between
        if _overlap(boxes[a], boxes[b]) and frozenset((node[a], node[b])) not in touching:
            out.append((a, b, "iso" in (kinds[a], kinds[b])))
    return out, len(touching)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=Path("data/raw/pid2graph/open100"))
    ap.add_argument("--out", type=Path, default=Path("docs/experiments/iso_render/v5_g1"))
    ap.add_argument("--sheet", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    image = Image.open(args.src / f"{args.sheet}.png")
    sheet = load_graphml(args.src / f"{args.sheet}.graphml", image.size)
    runs = pipe_runs(sheet)
    templates = load_templates(cfg["template_dir"], cfg["taxonomy_csv"], cfg["scope"])
    rng = random.Random(args.seed)
    assignments = assign_symbols(sheet, runs, templates, rng, cfg)  # V3 on source px
    sheet4, runs4 = transform_layout(sheet, runs, rng, cfg)  # V4
    page = render_page(sheet4, runs4, assignments, cfg, sheet)  # V5

    img = rasterize(page.svg)
    Image.fromarray(img).save(args.out / f"sheet{args.sheet}.png")
    (args.out / f"sheet{args.sheet}.svg").write_text(page.svg)
    pipes = rasterize(page.layer("pipes")) < 128
    symbols = rasterize(page.layer("symbols")) < 128

    lines = [f"G1 check: sheet {args.sheet}, seed {args.seed}, {page.width}x{page.height} px"]
    kinds: dict[str, int] = {}
    for a in page.annotations:
        key = a["fine_name"] or a["label"] or a["kind"]
        kinds[f"{a['kind']}:{key}"] = kinds.get(f"{a['kind']}:{key}", 0) + 1
    lines.append("drawn: " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
    lines.append("log: " + ", ".join(f"{k}={v}" for k, v in page.log.items()))

    n, misses = port_ink(page, pipes, symbols)
    ok1 = not misses
    lines.append(
        f"[1] pipe and symbol meet within {PORT_TOL_PX} px: {n - len(misses)}/{n} ports"
        f"  -> {'PASS' if ok1 else 'FAIL'}"
    )
    for m in misses[:20]:
        lines.append(
            f"      {m['id']} ({m['kind']}) at ({m['x']:.1f}, {m['y']:.1f}) lacks {m['lacks']}"
        )

    t_sym, t_pipe = stroke_thickness(symbols), stroke_thickness(pipes)
    ok2 = abs(t_sym - t_pipe) <= THICKNESS_TOL_PX
    lines.append(
        f"[2] stroke thickness: symbols {t_sym:.2f} px, pipes {t_pipe:.2f} px "
        f"(|diff| {abs(t_sym - t_pipe):.2f} <= {THICKNESS_TOL_PX})  -> {'PASS' if ok2 else 'FAIL'}"
    )

    new, original = new_overlaps(page, sheet, float(cfg["layout"]["column_tol"]))
    iso_new = [p for p in new if p[2]]
    ok3 = not iso_new
    lines.append(
        f"[3] new overlapping pairs vs the {original} source pairs within column_tol: "
        f"{len(iso_new)} with an ISO symbol, {len(new) - len(iso_new)} between non-ISO "
        f"shapes (ignored)  -> "
        f"{'PASS' if ok3 else 'FAIL'}"
    )
    for a, b, _ in iso_new[:20]:
        lines.append(f"      {a} x {b}")
    n_edges, apart = edges_joined(page, runs, pipes, symbols)
    ok5 = not apart
    lines.append(
        f"[5] graph edges between drawn symbols joined or touching: "
        f"{n_edges - len(apart)}/{n_edges}  -> {'PASS' if ok5 else 'FAIL'}"
    )
    for a, b, gap in apart[:20]:
        lines.append(f"      {a} - {b}: {gap} px apart")
    n_stems, cuts = stems_clear(page, symbols)
    ok6 = not cuts
    lines.append(
        f"[6] signal lines clear of symbol ink: {n_stems - len(cuts)}/{n_stems}"
        f"  -> {'PASS' if ok6 else 'FAIL'}"
    )
    for c in cuts[:20]:
        lines.append(f"      {c['id']} -> {c['bubble']} over ink at ({c['x']:.1f}, {c['y']:.1f})")
    n_tags, tag_over = bubble_overlaps(page, float(cfg["page"]["stroke_px"]))
    ok7 = not tag_over
    lines.append(
        f"[7] tag circles overlapping by more than a stroke: {len(tag_over)} pairs of {n_tags} tags"
        f"  -> {'PASS' if ok7 else 'FAIL'}"
    )
    for a, b, over in tag_over[:20]:
        lines.append(f"      {a} x {b}: {over} px")
    lines.append("[4] look at the picture: " + str(args.out / f"sheet{args.sheet}.png"))
    all_ok = ok1 and ok2 and ok3 and ok5 and ok6 and ok7
    lines.append(f"automatic criteria: {'ALL PASS' if all_ok else 'NOT ALL PASS'}")

    report = "\n".join(lines)
    (args.out / "g1_report.txt").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
