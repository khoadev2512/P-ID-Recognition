"""Many renderings of one OPEN100 sheet, near-duplicates removed (V6 of PLAN_ISO_RENDER_PID2GRAPH).

A variant is the V3 -> V4 -> V5 pipeline run with its own random choices:

  assign   V3 symbol per node (and flanges, agitators, clearance)
  layout   V4 axis mapping, `layout.gap_scale`
  style    stroke width `w` and one factor on every symbol's size
  degrade  the whole-image degradation of V7 (utils.iso_degrade; drawn after the page, so it
           cannot change what the three stages above produce)
  text     the labels of V5c (utils.page_text: tag and line numbers in blank space; they never move
           what the stages above drew)

Each stage has its own `random.Random` seeded from (base seed, source id, variant index, stage
name; `derive_seed`) by a hash, so a draw added to one stage never moves another's, and variant
`i` of a sheet is the same whatever else was generated before it. `w` and the size factor reach
V5 without touching it: `w` through a copy of `cfg` with `page.stroke_px` replaced, the size
through a copy of the source sheet whose symbol boxes (`sized_nodes`) are scaled about their
centres (V5 fits every symbol to its source box). Which pairs already touched (and so may overlap)
is read from the unscaled sheet, by V5 (`touch_source`) and by G1 criterion 3 alike: an enlarged
box must not make two symbols that were apart in the data count as touching.

`generate_sheet` runs variants of one sheet in index order and drops, cheapest test first:
  signature   the assignment is identical to a kept variant's (nothing was drawn yet)
  duplicate   fewer than `variants.min_node_diff` of the nodes got a different choice than in
              some kept variant (`node_choices`, utils.diversity.ContentDeduper; nothing drawn yet)
  error       V5 raised
  quality     a `variants.quality.max_log` counter of `Page.log` is over its limit
  g1          an automatic criterion in `variants.g1_drop` failed (`check` is supplied by the
              caller: scripts/check_g1.py keeps the criteria); `g1_flag` failures are kept and
              recorded
until `variants.count` are kept or `variants.max_attempts` were tried. The picture's ahash / dhash
and its Hamming distance to the closest kept picture are recorded, not filtered on: the coarse
hash sees the layout, which all variants of a sheet share, and not the symbols. Kept variants are
written as `<out>/sheet<i>/v<NNN>.png` + `.svg` + `.json` (seeds, parameters, signature, both
nearest distances, hashes, log, annotations).
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from PIL import Image

from utils.diversity import ContentDeduper, Deduper, ahash, dhash
from utils.iso_layout import transform_layout
from utils.iso_mapping import Assignment, assign_symbols
from utils.iso_templates import Template
from utils.page_frame import Frame, add_frame, frame_settings, shift_runs, shift_sheet
from utils.page_svg import Page, rasterize, render_page
from utils.pid2graph import PipeRun, Sheet

STAGES = ("assign", "layout", "style", "degrade", "text")
HASH_HEX_DIGITS = 4  # bits per hex digit
# G1 reference scale: `page.stroke_px` is 3 at `layout.scale` 3, and `variants.stroke` is the
# range of w in units of scale / 3
_STROKE_UNIT = 3.0


def derive_seed(base: int, source: int | str, index: int, stage: str) -> int:
    """A 64-bit seed for one stage of one variant; stable across runs and platforms (a hash of
    the four values, not Python's per-process `hash`)."""
    digest = hashlib.blake2b(f"{base}|{source}|{index}|{stage}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


@dataclass(frozen=True)
class VariantSettings:
    """The `variants` block of render_iso.yaml, checked."""

    count: int  # variants to keep per sheet
    max_attempts: int  # candidates tried per sheet at most
    base_seed: int
    stroke: tuple[float, float]  # w range in px (the config's range x scale / 3)
    size: tuple[float, float]  # factor on every symbol's size
    hash_size: int  # ahash / dhash grid side; hashes have hash_size² bits (recorded only)
    min_node_diff: float  # duplicate when fewer than this share of nodes differ from a kept one
    max_log: dict[str, int]  # Page.log counters that may not exceed these
    g1_drop: tuple[int, ...]  # G1 criteria whose failure drops a variant
    g1_flag: tuple[int, ...]  # G1 criteria whose failure is recorded, the variant kept


def variant_settings(cfg: Mapping[str, Any]) -> VariantSettings:
    """Read and check `cfg["variants"]` (`cfg` is the `render_iso` mapping)."""
    v = cfg["variants"]
    unit = float(cfg["layout"]["scale"]) / _STROKE_UNIT
    stroke = (float(v["stroke"][0]) * unit, float(v["stroke"][1]) * unit)
    size = (float(v["size"][0]), float(v["size"][1]))
    s = VariantSettings(
        count=int(v["count"]),
        max_attempts=int(v["max_attempts"]),
        base_seed=int(v["base_seed"]),
        stroke=stroke,
        size=size,
        hash_size=int(v["hash_size"]),
        min_node_diff=float(v["min_node_diff"]),
        max_log={k: int(n) for k, n in (v.get("quality") or {}).get("max_log", {}).items()},
        g1_drop=tuple(int(c) for c in (v.get("g1") or {}).get("drop", [])),
        g1_flag=tuple(int(c) for c in (v.get("g1") or {}).get("flag", [])),
    )
    if s.count < 1 or s.max_attempts < s.count:
        raise ValueError(
            f"variants: 1 <= count <= max_attempts needed: {s.count}, {s.max_attempts}"
        )
    if not 0 < stroke[0] <= stroke[1] or not 0 < size[0] <= size[1]:
        raise ValueError(f"variants: stroke / size ranges must be ascending and positive: {v}")
    if s.hash_size < 2 or not 0 <= s.min_node_diff <= 1:
        raise ValueError("variants: hash_size >= 2 and 0 <= min_node_diff <= 1 required")
    if set(s.g1_drop) & set(s.g1_flag):
        raise ValueError("variants.g1: a criterion cannot be both dropped on and flagged")
    return s


def assignment_signature(assignments: Mapping[str, Assignment]) -> str:
    """A short digest of what V3 chose for every node: kind, label, template and orientation,
    flanges, agitator and its clearance, arrow direction. Equal digests = the same choices."""

    def tpl(t: Template | None) -> str:
        return "-" if t is None else f"{t.file}/{t.rot}/{int(t.flip)}"

    parts = [
        "|".join(
            (
                nid,
                a.kind,
                str(a.label),
                tpl(a.template),
                str(a.axis),
                str(a.forward),
                tpl(a.flanges),
                tpl(a.agitator),
                f"{a.clearance:.2f}",
            )
        )
        for nid, a in sorted(assignments.items())
    ]
    return hashlib.blake2b("\n".join(parts).encode(), digest_size=8).hexdigest()


def node_choices(assignments: Mapping[str, Assignment]) -> dict[str, str]:
    """What was chosen for each node, as one string per node id: kind, label, template and its
    orientation, arrow axis and direction, flange pair, agitator. Two variants of one sheet differ
    at a node when these strings differ. The agitator's clearance (a shaft length) is left out:
    it changes how a symbol is drawn, not which, like the stroke width and the size factor."""

    def tpl(t: Template | None) -> str:
        return "-" if t is None else f"{t.file}/{t.rot}/{int(t.flip)}"

    return {
        nid: "|".join(
            (
                a.kind,
                str(a.label),
                tpl(a.template),
                str(a.axis),
                str(a.forward),
                tpl(a.flanges),
                "-" if a.agitator is None else a.agitator.file,
            )
        )
        for nid, a in assignments.items()
    }


def scale_symbols(sheet: Sheet, factor: float, ids: set[str] | frozenset[str]) -> Sheet:
    """`sheet` with the boxes of the nodes `ids` scaled by `factor` about their centres: V5 fits
    every placed symbol to its source box, so this is what sizes them."""
    if factor == 1.0:
        return sheet
    nodes = dict(sheet.nodes)
    for nid in ids:
        n = nodes[nid]
        cx, cy = n.center
        hw, hh = n.width * factor / 2, n.height * factor / 2
        nodes[nid] = replace(n, xmin=cx - hw, xmax=cx + hw, ymin=cy - hh, ymax=cy + hh)
    return replace(sheet, nodes=nodes)


def sized_nodes(assignments: Mapping[str, Assignment]) -> set[str]:
    """Nodes whose drawing V5 sizes from the source box: placed ISO symbols and flow arrows. A
    vessel (`multi`) is drawn on its V4 box and a plain box, bubble or flag on theirs, so a factor
    there would only make the source disagree with the picture (which pairs touch)."""
    return {
        nid
        for nid, a in assignments.items()
        if a.kind == "arrow"
        or (a.kind == "iso" and a.template is not None and a.template.port_config != "multi")
    }


@dataclass
class Draft:
    """The choices of one variant before anything is drawn."""

    source: int
    index: int
    seeds: dict[str, int]
    stroke_px: float
    size_factor: float
    assignments: dict[str, Assignment]
    signature: str
    choices: dict[str, str]  # node_choices(assignments)


@dataclass
class Rendered:
    """A drawn variant: `source` is the sheet before V4 and before the size factor, the one the
    G1 checks must be given (criterion 3 asks which pairs touched in the data)."""

    page: Page
    runs: list[PipeRun]
    source: Sheet
    cfg: dict[str, Any]
    sheet: Sheet | None = None  # the sheet after V4 (node boxes in page px)
    frame: Frame | None = None  # the sheet frame round the drawing (config `frame`), if any


def plan_variant(
    sheet: Sheet,
    runs: list[PipeRun],
    templates: Mapping[str, Template],
    cfg: Mapping[str, Any],
    source: int,
    index: int,
) -> Draft:
    """Draw the stroke width, the size factor and the V3 assignment of variant `index`."""
    st = variant_settings(cfg)
    seeds = {s: derive_seed(st.base_seed, source, index, s) for s in STAGES}
    style = random.Random(seeds["style"])
    stroke = round(style.uniform(*st.stroke), 2)
    size = round(style.uniform(*st.size), 3)
    assignments = assign_symbols(sheet, runs, templates, random.Random(seeds["assign"]), cfg)
    return Draft(
        source, index, seeds, stroke, size, assignments, assignment_signature(assignments),
        node_choices(assignments),
    )  # fmt: skip


def render_variant(
    draft: Draft, sheet: Sheet, runs: list[PipeRun], cfg: Mapping[str, Any]
) -> Rendered:
    """V4 layout (its own seed) and V5 page of a draft, with its `w` and size factor."""
    cfg_v = {**cfg, "page": {**cfg["page"], "stroke_px": draft.stroke_px}}
    sheet4, runs4 = transform_layout(sheet, runs, random.Random(draft.seeds["layout"]), cfg)
    # symbols are fitted to the scaled boxes; which pairs may overlap is read from the data's
    sized = scale_symbols(sheet, draft.size_factor, sized_nodes(draft.assignments))
    page = render_page(
        sheet4, runs4, draft.assignments, cfg_v, sized, touch_source=sheet,
        text_rng=random.Random(draft.seeds["text"]),
    )  # fmt: skip
    frame = frame_settings(cfg)
    if frame is not None:  # PLAN_PHASE_C §2M item 7: drawn last, the drawing moves into it
        seed = derive_seed(variant_settings(cfg).base_seed, draft.source, draft.index, "frame")
        page, fr = add_frame(page, draft.stroke_px, frame, random.Random(seed))
        dx, dy = fr.layout.offset
        sheet4 = shift_sheet(sheet4, dx, dy, page.width, page.height)
        runs4 = shift_runs(runs4, dx, dy)
        return Rendered(page, runs4, sheet, cfg_v, sheet4, fr)
    return Rendered(page, runs4, sheet, cfg_v, sheet4)


def quality_failures(log: Mapping[str, int], max_log: Mapping[str, int]) -> list[str]:
    """Counters of `Page.log` over their limit, as 'name=value' strings."""
    return [f"{k}={log.get(k, 0)}" for k, limit in sorted(max_log.items()) if log.get(k, 0) > limit]


def _hex(value: int, bits: int) -> str:
    return f"{value:0{bits // HASH_HEX_DIGITS}x}"


CheckFn = Callable[[Rendered], dict[int, bool]]


def generate_sheet(
    source: int,
    sheet: Sheet,
    runs: list[PipeRun],
    templates: Mapping[str, Template],
    cfg: Mapping[str, Any],
    out_dir: Path | None,
    check: CheckFn | None = None,
) -> list[dict[str, Any]]:
    """Try variants 0, 1, ... of one sheet until `count` are kept (see the module doc).

    Returns one record per attempt: index, status ("kept" / "dropped"), reason (None, or
    "signature" / "duplicate" / "error" / "quality" / "g1"), detail, seeds, stroke_px,
    size_factor, signature, nearest_content (share of nodes differing from the closest kept
    variant; None while nothing is kept), nearest_hash (Hamming distance of the picture to the
    closest kept one, when drawn and something is kept), seconds per phase, and for a kept one
    also its files, annotations' class counts and G1 flags. With `out_dir=None` nothing is
    written. `check(rendered)` returns {criterion: passed}.
    """
    st = variant_settings(cfg)
    content = ContentDeduper(st.min_node_diff)
    pictures = Deduper(0)  # records hash distances only
    bits = st.hash_size**2
    signatures: set[str] = set()
    records: list[dict[str, Any]] = []
    kept = 0
    sheet_dir = None if out_dir is None else Path(out_dir) / f"sheet{source}"
    if sheet_dir is not None:
        sheet_dir.mkdir(parents=True, exist_ok=True)

    for index in range(st.max_attempts):
        if kept >= st.count:
            break
        t0 = time.perf_counter()
        draft = plan_variant(sheet, runs, templates, cfg, source, index)
        rec: dict[str, Any] = {
            "sheet": source,
            "index": index,
            "status": "dropped",
            "reason": None,
            "detail": "",
            "seeds": draft.seeds,
            "stroke_px": draft.stroke_px,
            "size_factor": draft.size_factor,
            "signature": draft.signature,
            "nearest_content": None,
            "nearest_hash": None,
            "seconds": {"assign": round(time.perf_counter() - t0, 3)},
        }
        records.append(rec)
        if draft.signature in signatures:
            rec["reason"] = "signature"
            continue
        duplicate, nearest = content.is_duplicate(draft.choices)
        rec["nearest_content"] = None if nearest is None else round(nearest, 4)
        if duplicate:
            rec["reason"] = "duplicate"
            continue

        t1 = time.perf_counter()
        try:
            rendered = render_variant(draft, sheet, runs, cfg)
            img = rasterize(rendered.page.svg)
        except Exception as exc:  # a layout V5 cannot draw is a dropped variant, not a crash
            rec["reason"], rec["detail"] = "error", f"{type(exc).__name__}: {exc}"
            continue
        rec["seconds"]["render"] = round(time.perf_counter() - t1, 3)
        page = rendered.page
        rec["log"] = dict(page.log)
        fails = quality_failures(page.log, st.max_log)
        if fails:
            rec["reason"], rec["detail"] = "quality", ",".join(fails)
            continue

        t2 = time.perf_counter()
        a, d = ahash(img, st.hash_size), dhash(img, st.hash_size)
        rec["nearest_hash"] = pictures.distance(a, d)
        rec["seconds"]["hash"] = round(time.perf_counter() - t2, 3)

        flags: list[int] = []
        if check is not None:
            t3 = time.perf_counter()
            passed = check(rendered)
            rec["seconds"]["g1"] = round(time.perf_counter() - t3, 3)
            rec["g1"] = {str(c): ok for c, ok in sorted(passed.items())}
            failed = [c for c, ok in passed.items() if not ok]
            dropping = [c for c in failed if c in st.g1_drop]
            flags = [c for c in failed if c in st.g1_flag]
            if dropping:
                rec["reason"], rec["detail"] = "g1", ",".join(map(str, dropping))
                continue

        content.add(draft.choices)
        pictures.add(a, d)
        signatures.add(draft.signature)
        kept += 1
        rec["status"], rec["g1_flags"] = "kept", flags
        classes: dict[str, int] = {}
        for ann in page.annotations:
            classes[ann["label"] or ann["kind"]] = classes.get(ann["label"] or ann["kind"], 0) + 1
        rec["classes"] = classes
        if sheet_dir is not None:
            t4 = time.perf_counter()
            stem = f"v{index:03d}"
            Image.fromarray(img).save(sheet_dir / f"{stem}.png")
            (sheet_dir / f"{stem}.svg").write_text(page.svg)
            meta = {
                **{k: rec[k] for k in ("sheet", "index", "seeds", "stroke_px", "size_factor")},
                "signature": draft.signature,
                "width": page.width,
                "height": page.height,
                "ahash": _hex(a, bits),
                "dhash": _hex(d, bits),
                "nearest_content": rec["nearest_content"],
                "nearest_hash": rec["nearest_hash"],
                "log": rec["log"],
                "g1": rec.get("g1"),
                "g1_flags": flags,
                "annotations": page.annotations,
                "texts": page.texts,  # V5c labels: not COCO objects
            }
            if rendered.frame is not None:  # where the drawing is; the rest is frame (not COCO)
                lay = rendered.frame.layout
                meta["frame"] = {
                    "content": list(lay.content), "border": list(lay.border),
                    "column": list(lay.column), "title_block": list(rendered.frame.title_block),
                    "revision_table": list(rendered.frame.revision_table),
                }  # fmt: skip
            (sheet_dir / f"{stem}.json").write_text(json.dumps(meta, indent=1))
            rec["files"] = [f"sheet{source}/{stem}.{ext}" for ext in ("png", "svg", "json")]
            rec["seconds"]["write"] = round(time.perf_counter() - t4, 3)
        rec["seconds"]["total"] = round(time.perf_counter() - t0, 3)
    for rec in records:
        rec["seconds"].setdefault("total", round(sum(rec["seconds"].values()), 3))
    return records
