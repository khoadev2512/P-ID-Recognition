"""PLAN_PHASE_C §2M: count the drawing faults of OPEN100-ISO pages, before / after the renderer fix.

Every kept variant (<root>/variants/sheet<i>/v<NNN>.json) is drawn again with the current renderer
(utils.iso_variants.plan_variant + render_variant, the code make_variants.py runs) and measured:

  jog          a short step (< JOG_FRAC of the page width): inside one pipe polyline, two parallel
               stretches joined by a short perpendicular one; or at a node with exactly two pipe
               runs arriving along one axis from opposite sides (an arrow, an inline valve), the
               two ends offset across the axis (after V4): a near-straight pipe drawn with a kink
  in_vessel    a pipe with a stretch inside a vessel / tank / column body (box shrunk by IN_MARGIN
               on every side), or an arrowhead (flow arrow or leader) inside such a box: a line
               drawn into the equipment
  arrows_flip  two flow arrows on one straight pipe run (between them) pointing opposite ways
  dead_end     a pipe run ending at a connector / crossing no other run reaches: a hanging line

    python scripts/check_render_artifacts.py --root data/iso_open100 [--sheets 1 5] [--out f.json]

Read-only on --root. Prints one row per sheet and the totals; --out writes them as JSON.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.iso_mapping import load_config  # noqa: E402
from utils.iso_templates import load_templates  # noqa: E402
from utils.iso_variants import plan_variant, render_variant  # noqa: E402
from utils.pid2graph import load_graphml, pipe_runs  # noqa: E402

JOG_FRAC = 0.008  # step shorter than this share of the page width
IN_MARGIN = 0.15  # vessel box shrunk by this share on each side (keeps out a dished end's corners)
VESSEL_NAMES = ("tank", "column", "vessel", "silo", "drum")
KINDS = ("jog", "in_vessel", "arrows_flip", "dead_end")
POLY = re.compile(r'<polyline points="([^"]+)"')

Point = tuple[float, float]


# --------------------------------------------------------------------------- #
# pure measures (tests/test_check_render_artifacts.py)
# --------------------------------------------------------------------------- #


def polylines(svg: str) -> list[list[Point]]:
    out = []
    for m in POLY.finditer(svg):
        nums = [float(v) for v in re.split(r"[ ,]+", m.group(1).strip())]
        out.append(list(zip(nums[0::2], nums[1::2], strict=True)))
    return out


def _axis(a: Point, b: Point, tol: float = 1e-3) -> str | None:
    if abs(a[1] - b[1]) <= tol and abs(a[0] - b[0]) > tol:
        return "H"
    if abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) > tol:
        return "V"
    return None


def _len(a: Point, b: Point) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def jogs(pl: list[Point], max_step: float) -> list[Point]:
    """Steps inside a polyline: segment i+1 short and perpendicular, segments i and i+2
    parallel and running the same way. Returns where each step is."""
    out = []
    for i in range(len(pl) - 3):
        a, b, c, d = pl[i : i + 4]
        s1, s2, s3 = _axis(a, b), _axis(b, c), _axis(c, d)
        if s1 is None or s1 != s3 or s2 is None or s2 == s1:
            continue
        if _len(b, c) >= max_step:
            continue
        k = 0 if s1 == "H" else 1
        if (b[k] - a[k]) * (d[k] - c[k]) > 0:
            out.append(b)
    return out


def _shrink(box, frac):
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) * frac, (y1 - y0) * frac
    return x0 + dx, y0 + dy, x1 - dx, y1 - dy


def crosses_box(pl: list[Point], box, samples: int = 20) -> Point | None:
    """The first sampled point of `pl` strictly inside `box`, else None."""
    x0, y0, x1, y1 = box
    for a, b in zip(pl, pl[1:], strict=False):
        for t in range(samples + 1):
            x = a[0] + (b[0] - a[0]) * t / samples
            y = a[1] + (b[1] - a[1]) * t / samples
            if x0 < x < x1 and y0 < y < y1:
                return (x, y)
    return None


def _seg_dist(p: Point, a: Point, b: Point) -> float:
    ax, ay = b[0] - a[0], b[1] - a[1]
    n2 = ax * ax + ay * ay
    t = 0.0 if n2 == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * ax + (p[1] - a[1]) * ay) / n2))
    return _len(p, (a[0] + t * ax, a[1] + t * ay))


def deep_head(head, vessel, frac: float = 0.05) -> Point | None:
    """An arrowhead box inside a vessel box, deeper than `frac` of the vessel's size from the
    walls it points across (a horizontal head: the side walls; a vertical one: top / bottom).
    A head just past the outline is an arrow entering through the wall (a dished end's curve
    lies inside its box), not one."""
    x, y = (head[0] + head[2]) / 2, (head[1] + head[3]) / 2
    x0, y0, x1, y1 = vessel
    if not (x0 < x < x1 and y0 < y < y1):
        return None
    if head[2] - head[0] >= head[3] - head[1]:  # horizontal arrow
        deep = min(x - x0, x1 - x) > frac * (x1 - x0)
    else:
        deep = min(y - y0, y1 - y) > frac * (y1 - y0)
    return (x, y) if deep else None


def _run_ends(runs) -> dict[str, list[tuple[Point, Point]]]:
    ends: dict[str, list[tuple[Point, Point]]] = {}
    for r in runs:
        if r.internal or r.nozzle or len(r.points) < 2:
            continue
        ends.setdefault(r.a, []).append((r.points[0], r.points[1]))
        ends.setdefault(r.b, []).append((r.points[-1], r.points[-2]))
    return ends


def node_jogs(runs, max_step: float, min_step: float = 0.5) -> list[Point]:
    """At a node (an arrow, an inline valve, a tee...), two run ends arriving along one axis
    from opposite sides but offset across it by min_step..max_step: the pipe steps there."""
    out = []
    for arrivals in _run_ends(runs).values():
        for ax in ("H", "V"):
            k = 0 if ax == "H" else 1  # along the axis
            side = {+1: [], -1: []}
            for p, p_in in arrivals:
                if _axis(p_in, p, tol=0.5) == ax:
                    side[1 if p[k] > p_in[k] else -1].append(p)
            # exactly one pipe in from each side: a pipe through the node (a vessel's nozzles
            # on opposite walls, or a valve's signal line, are other pipes)
            if len(side[1]) == 1 and len(side[-1]) == 1:
                p, q = side[1][0], side[-1][0]
                if min_step < abs(p[1 - k] - q[1 - k]) < max_step:
                    out.append(p)
    return out


def dead_ends(
    sheet, runs, labels=(), near: float = 0.0, drawn=None, boxes=(), touch: float = 0.0
) -> list[Point]:
    """Ends of runs at a connector / crossing that no other run reaches and no label box (`labels`,
    grown by `near`) is next to: a line left hanging. A leader line ending at its tag is not one
    (PID2Graph labels a tag's leader line as pipe; V5c writes the tag at its free end).
    `drawn` (pipe polylines of the page): only ends where a drawn pipe still ends count, and
    that touch no other drawn pipe (within `touch`) and no symbol box (`boxes`): the graph may
    leave a pipe unjoined that runs into another one on the page."""
    live = [r for r in runs if not r.internal and not r.nozzle]
    seen = Counter(n for r in live for n in (r.a, r.b))

    def hanging(nid: str) -> bool:
        node = sheet.nodes.get(nid)
        return node is not None and node.label in ("connector", "crossing") and seen[nid] == 1

    def labelled(q: Point) -> bool:
        return any(
            b[0] - near <= q[0] <= b[2] + near and b[1] - near <= q[1] <= b[3] + near
            for b in labels
        )

    out = []
    for r in live:
        for nid, q in ((r.a, r.points[0]), (r.b, r.points[-1])):
            if not hanging(nid) or labelled(q):
                continue
            if drawn is not None:
                mine = [pl for pl in drawn if any(_len(q, e) <= 1.0 for e in (pl[0], pl[-1]))]
                if not mine:
                    continue  # not drawn
                if any(
                    _seg_dist(q, a, b) <= touch
                    for pl in drawn
                    if pl not in mine
                    for a, b in zip(pl, pl[1:], strict=False)
                ):
                    continue
            if any(
                b[0] - touch <= q[0] <= b[2] + touch and b[1] - touch <= q[1] <= b[3] + touch
                for b in boxes
            ):
                continue
            out.append(q)
    return out


def flipped_arrows(runs, assignments) -> list[Point]:
    """Straight runs between two arrows lying along the run, the arrows pointing opposite ways
    (where: the run's middle)."""
    out = []
    for r in runs:
        if r.internal or r.nozzle or r.elbows or r.a == r.b:
            continue
        a, b = assignments.get(r.a), assignments.get(r.b)
        if a is None or b is None or a.kind != "arrow" or b.kind != "arrow":
            continue
        axis = _axis(r.points[0], r.points[-1], tol=2.0)
        if axis is None or a.axis != axis or b.axis != axis:
            continue
        if bool(a.forward) != bool(b.forward):
            p, q = r.points[0], r.points[-1]
            out.append(((p[0] + q[0]) / 2, (p[1] + q[1]) / 2))
    return out


# --------------------------------------------------------------------------- #


def locate(rendered, assignments) -> dict[str, list[Point]]:
    """Where each kind of fault is on one drawn page."""
    page = rendered.page
    pls = [pl for pl in polylines(page.pipes) if len(pl) >= 2]
    vessels = [
        a["bbox"]
        for a in page.annotations
        if any(k in str(a.get("fine_name", "")) for k in VESSEL_NAMES)
    ]
    step = JOG_FRAC * page.width
    stroke = float(rendered.cfg["page"]["stroke_px"])
    labels = [t["box"] for t in page.texts]
    inside = [crosses_box(pl, _shrink(v, IN_MARGIN)) for pl in pls for v in vessels]
    heads = [a["bbox"] for a in page.annotations if a.get("kind") in ("arrow", "leader")]
    inside += [q for h in heads for v in vessels if (q := deep_head(h, v)) is not None]
    return {
        "jog": [q for pl in pls for q in jogs(pl, step)]
        + node_jogs(rendered.runs, step, min_step=stroke),  # thinner than a stroke: not seen
        "in_vessel": [q for q in inside if q is not None],
        "arrows_flip": flipped_arrows(rendered.runs, assignments),
        "dead_end": dead_ends(
            rendered.sheet,
            rendered.runs,
            labels,
            near=15 * stroke,
            drawn=pls,
            boxes=[a["bbox"] for a in page.annotations],
            touch=2 * stroke,
        ),  # fmt: skip
    }


def mark(image: Path, where: dict[str, list[Point]], out: Path, offset: Point = (0.0, 0.0)) -> None:
    """The page with a labelled circle on every fault (for checking the measures by eye);
    `offset` moves the drawing's points onto a framed page."""
    from PIL import ImageDraw, ImageFont

    colour = {"jog": "red", "in_vessel": "blue", "arrows_flip": "orange", "dead_end": "magenta"}
    im = Image.open(image).convert("RGB")
    d = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=40)
    for k, pts in where.items():
        for x, y in pts:
            x, y = x + offset[0], y + offset[1]
            d.ellipse((x - 45, y - 45, x + 45, y + 45), outline=colour[k], width=8)
            d.text((x + 50, y - 50), k, fill=colour[k], font=font)
    im.save(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=None, help="default: export.data_root")
    ap.add_argument("--src", type=Path, default=ROOT / "data/raw/pid2graph/open100")
    ap.add_argument("--sheets", type=int, nargs="*", default=None)
    ap.add_argument("--max-variants", type=int, default=None)
    ap.add_argument("--index", type=int, default=None, help="only this variant index")
    ap.add_argument(
        "--set", action="append", default=[], metavar="BLOCK.KEY=VALUE",
        help="override one render_iso setting, e.g. layout.straighten_pipes=true (repeatable)",
    )  # fmt: skip
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--mark", type=Path, default=None, help="PNG of the first variant, faults marked"
    )
    args = ap.parse_args(argv)
    cfg = load_config()
    for item in args.set:
        key, value = item.split("=", 1)
        if "." in key:
            block, name = key.split(".", 1)
            cfg[block] = {**cfg[block], name: json.loads(value)}
        else:
            cfg[key] = json.loads(value)
    args.root = args.root or ROOT / cfg["export"]["data_root"]
    # the measures read the drawing; a sheet frame (§2M item 7) round it changes nothing in it
    cfg["frame"] = {**cfg.get("frame", {}), "enabled": False}
    templates = load_templates(cfg["template_dir"], cfg["taxonomy_csv"], cfg["scope"])
    sheets = args.sheets or sorted(
        int(p.name.removeprefix("sheet")) for p in (args.root / "variants").glob("sheet*")
    )
    table, total = {}, Counter()
    for s in sheets:
        sheet = load_graphml(args.src / f"{s}.graphml", Image.open(args.src / f"{s}.png").size)
        runs = pipe_runs(sheet)
        files = sorted((args.root / "variants" / f"sheet{s}").glob("v*.json"))
        if args.index is not None:
            files = [f for f in files if f.stem == f"v{args.index:03d}"]
        counts, n = Counter(), 0
        for jf in files[: args.max_variants]:
            meta = json.loads(jf.read_text())
            index = int(meta["index"])
            draft = plan_variant(sheet, runs, templates, cfg, s, index)
            where = locate(render_variant(draft, sheet, runs, cfg), draft.assignments)
            counts.update({k: len(v) for k, v in where.items()})
            if args.mark and n == 0:
                frame = meta.get("frame")
                offset = tuple(frame["content"][:2]) if frame else (0.0, 0.0)
                mark(jf.with_suffix(".png"), where, args.mark, offset)
            n += 1
        table[s] = {"variants": n, **{k: counts[k] for k in KINDS}}
        total.update(counts)
        print(f"sheet {s:2d} ({n} variants): " + "  ".join(f"{k} {counts[k]}" for k in KINDS))
    print("TOTAL: " + "  ".join(f"{k} {total[k]}" for k in KINDS))
    if args.out:
        args.out.write_text(json.dumps({"sheets": table, "total": dict(total)}, indent=1))


if __name__ == "__main__":
    main()
