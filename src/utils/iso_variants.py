"""Many renderings of one OPEN100 sheet, near-duplicates removed (V6 of PLAN_ISO_RENDER_PID2GRAPH).

A variant is the V3 -> V4 -> V5 pipeline run with its own random choices:

  assign   V3 symbol per node (and flanges, agitators, clearance)
  layout   V4 axis mapping, `layout.gap_scale`
  style    stroke width `w` and one factor on every symbol's size

Each stage has its own `random.Random` seeded from (base seed, source id, variant index, stage
name; `derive_seed`) by a hash, so a draw added to one stage never moves another's, and variant
`i` of a sheet is the same whatever else was generated before it. `w` and the size factor reach
V5 without touching it: `w` through a copy of `cfg` with `page.stroke_px` replaced, the size
through a copy of the source sheet whose symbol boxes (`sized_nodes`) are scaled about their
centres (V5 fits every symbol to its source box).

`generate_sheet` runs variants of one sheet in index order and drops, cheapest test first:
  signature   the assignment is identical to a kept variant's (nothing was drawn yet)
  error       V5 raised
  quality     a `variants.quality.max_log` counter of `Page.log` is over its limit
  duplicate   the picture is within `variants.min_distance` bits of a kept one (utils.diversity)
  g1          an automatic criterion in `variants.g1_drop` failed (`check` is supplied by the
              caller: scripts/check_g1.py keeps the criteria); `g1_flag` failures are kept and
              recorded
until `variants.count` are kept or `variants.max_attempts` were tried. Kept variants are written as
`<out>/sheet<i>/v<NNN>.png` + `.svg` + `.json` (seeds, parameters, signature, hashes, log,
annotations).
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

from utils.diversity import Deduper, ahash, dhash
from utils.iso_layout import transform_layout
from utils.iso_mapping import Assignment, assign_symbols
from utils.iso_templates import Template
from utils.page_svg import Page, rasterize, render_page
from utils.pid2graph import PipeRun, Sheet

STAGES = ("assign", "layout", "style")
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
    hash_size: int  # ahash / dhash grid side; hashes have hash_size² bits
    min_distance: int  # duplicate when both hashes differ by fewer bits than this
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
        min_distance=int(v["min_distance"]),
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
    if s.hash_size < 2 or s.min_distance < 0:
        raise ValueError("variants: hash_size >= 2 and min_distance >= 0 required")
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


@dataclass
class Rendered:
    """A drawn variant: `source` is the size-scaled sheet V5 and the G1 checks must be given."""

    page: Page
    runs: list[PipeRun]
    source: Sheet
    cfg: dict[str, Any]


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
    return Draft(source, index, seeds, stroke, size, assignments, assignment_signature(assignments))


def render_variant(
    draft: Draft, sheet: Sheet, runs: list[PipeRun], cfg: Mapping[str, Any]
) -> Rendered:
    """V4 layout (its own seed) and V5 page of a draft, with its `w` and size factor."""
    cfg_v = {**cfg, "page": {**cfg["page"], "stroke_px": draft.stroke_px}}
    sheet4, runs4 = transform_layout(sheet, runs, random.Random(draft.seeds["layout"]), cfg)
    source = scale_symbols(sheet, draft.size_factor, sized_nodes(draft.assignments))
    page = render_page(sheet4, runs4, draft.assignments, cfg_v, source)
    return Rendered(page, runs4, source, cfg_v)


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
    "signature" / "error" / "quality" / "duplicate" / "g1"), detail, seeds, stroke_px,
    size_factor, signature, nearest (Hamming distance to the closest kept image, when hashed),
    seconds per phase, and for a kept one also its files, annotations' class counts and G1
    flags. With `out_dir=None` nothing is written. `check(rendered)` returns {criterion: passed}.
    """
    st = variant_settings(cfg)
    dedup = Deduper(st.min_distance)
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
            "nearest": None,
            "seconds": {"assign": round(time.perf_counter() - t0, 3)},
        }
        records.append(rec)
        if draft.signature in signatures:
            rec["reason"] = "signature"
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
        duplicate, rec["nearest"] = dedup.is_duplicate(a, d)
        rec["seconds"]["hash"] = round(time.perf_counter() - t2, 3)
        if duplicate:
            rec["reason"] = "duplicate"
            continue

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

        dedup.add(a, d)
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
                "nearest": rec["nearest"],
                "log": rec["log"],
                "g1": rec.get("g1"),
                "g1_flags": flags,
                "annotations": page.annotations,
            }
            (sheet_dir / f"{stem}.json").write_text(json.dumps(meta, indent=1))
            rec["files"] = [f"sheet{source}/{stem}.{ext}" for ext in ("png", "svg", "json")]
            rec["seconds"]["write"] = round(time.perf_counter() - t4, 3)
        rec["seconds"]["total"] = round(time.perf_counter() - t0, 3)
    for rec in records:
        rec["seconds"].setdefault("total", round(sum(rec["seconds"].values()), 3))
    return records
