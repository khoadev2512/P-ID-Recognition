"""PID2Graph node -> ISO symbol (V3 of PLAN_ISO_RENDER_PID2GRAPH).

For every symbol node of a sheet, `assign_symbols` decides what to draw: an ISO 10628-2 template
(already oriented), an arrow overlay, a flag, an ISA bubble, a plain box, or nothing (a symbol
nested inside a bigger one). The rules are the flat `rules` list of `configs/render_iso.yaml`;
the symbols inside a group come from the taxonomy through `iso_templates.load_templates`, and
the match is by port sides, never by name.

Port rules (plan §V3, "3 nguyên tắc"). A template fits a node when, after some mirror / turn:
  - its port sides equal the node's port sides, or
  - the node has fewer ports, every one of them lies on a side the template has, and the
    template's `placement` lets a port stay open (`pipe_or_end`, `end`: a vent at the end of
    a pipe); a node with more ports than the template never fits (a pipe would have nowhere to
    end).
Two templates are special: `multi` (a vessel; any node with >= 1 port, drawn as is, its ports
are placed on the outline by V5) and `2` (pumps and compressors, whose two ports are not
collinear in use: the orientations that agree with the node on the most sides win and V5 bends
the pipe with an L).

One class (`fine_name`) is drawn uniformly from the candidates, then one of its templates, then
one orientation uniformly from the valid ones (the 8 of `all_orientations()`, duplicates kept).
Weighting classes by how often they were already drawn is deferred (plan).
"""

from __future__ import annotations

import logging
import os
import random
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from utils.iso_templates import Template
from utils.pid2graph import (
    Node,
    PipeRun,
    Port,
    PortConfig,
    Sheet,
    port_config,
    symbol_ports,
)

logger = logging.getLogger(__name__)

NOT_SYMBOLS = ("connector", "crossing", "background")
OPEN_PLACEMENTS = ("pipe_or_end", "end")  # may leave a template port unconnected
KINDS = ("iso", "arrow", "leader", "flag", "bubble", "generic")
# COCO label of the non-ISO kinds; an ISO template is `fine_name`, or OTHER_ISO outside vocab.
KIND_LABELS = {
    "arrow": "flow_direction",
    "leader": "leader_arrow",
    "flag": None,
    "bubble": "other_isa",
    "generic": "other_generic",
}
OTHER_ISO = "other_iso"
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "render_iso.yaml"


@dataclass(frozen=True)
class Assignment:
    """What to draw for one node.

    kind      iso | arrow | leader | flag | bubble | generic | nested
    label     COCO category: fine_name / other_iso (iso), flow_direction (arrow), leader_arrow
              (leader: an arrow on a line from a tag to the pipe, not a flow direction; kept apart
              so it can be merged into one `arrow` class like PID2Graph's, or left out),
              other_isa (bubble), other_generic (generic); None for flag and nested
    template  the oriented Template (iso; an arrow carries 241, turned to point its way)
    axis      "H" / "V": the pipe direction an arrow or flag lies along
    forward   which way an arrow / flag points along `axis` (True = right / down); OPEN100 has
              no flow direction, so it is drawn at random
    flanges   the flange pair a disc or plate is held between (the `flanges` block of the
              config): the `flanges.symbol` template turned to the disc's pipe axis, drawn and
              labelled as a symbol of its own; None = the disc alone, which is just as correct
    agitator  an agitator template (an `attach` symbol) mounted in this vessel, drawn and labelled
              as a symbol of its own; PID2Graph has no such node, so it exists only in the picture
              and the COCO file (V5b)
    clearance the gap between the agitator's blade and the vessel's bottom, in vessel diameters
              (ISO 10628-2 X8006 draws 0.5): the shaft is drawn as long as that needs
    """

    kind: str
    label: str | None
    template: Template | None = None
    axis: str | None = None
    forward: bool | None = None
    flanges: Template | None = None
    agitator: Template | None = None
    clearance: float = 0.5


CONFIG_ENV = "PID_RENDER_CONFIG"  # path of the render config to use instead of the default


def config_path(path: str | Path | None = None) -> Path:
    """`path`, else `$PID_RENDER_CONFIG`, else configs/render_iso.yaml."""
    return Path(path or os.environ.get(CONFIG_ENV) or DEFAULT_CONFIG)


def load_config_node(path: str | Path | None = None) -> Any:
    """The whole config file (an OmegaConf node, not resolved). A file with `extends: <file>`
    (relative to its own folder) is that file with this one merged over it: an overlay names only
    what it changes (configs/render_iso_v2.yaml, PLAN_PHASE_C §2M)."""
    from omegaconf import OmegaConf

    p = config_path(path)
    node = OmegaConf.load(p)
    base = node.pop("extends", None)
    if base is None:
        return node
    return OmegaConf.merge(load_config_node(p.parent / base), node)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """The `render_iso` mapping of a render config (see `config_path`, `load_config_node`)."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.to_container(load_config_node(path), resolve=True)
    return cfg["render_iso"]  # type: ignore[index,return-value]


def size_class(node: Node, config: PortConfig, small_max: float, large_min: float) -> str:
    """small / medium / large of a node, in source-image px (before any scaling).

    A `2_straight` node (a fitting on a pipe) is measured across the pipe: a thin, long box
    is still small. Every other port layout uses the longer side.
    """
    if config.kind == "2_straight":
        side = node.height if config.axis == "H" else node.width
    else:
        side = max(node.width, node.height)
    if side <= small_max:
        return "small"
    return "large" if side >= large_min else "medium"


def _fits(template_sides: Counter[str], node_sides: Counter[str], placement: str) -> bool:
    if template_sides == node_sides:
        return True
    leftover = node_sides - template_sides  # node sides the template cannot take
    return (
        placement in OPEN_PLACEMENTS
        and sum(node_sides.values()) >= 1
        and not leftover
        and sum(node_sides.values()) < sum(template_sides.values())
    )


def orientations_for(
    template: Template,
    node_sides: list[str],
    aspect: float | None = None,
    allow_empty: bool = False,
) -> list[Template]:
    """The orientations of `template` that may stand for a node with these port sides.

    `aspect` is the node's height / width in source px; a `multi` vessel wider than tall
    (< 1) is turned a quarter so that its dished ends stay at the ends of the long axis.
    `allow_empty`: a `multi` vessel may have no port at all (a tank whose only pipes are tag
    lines)."""
    node = Counter(node_sides)
    n = sum(node.values())
    kind = template.port_config
    if kind == "attach":
        return []
    if kind == "multi":
        if n < 1 and not allow_empty:
            return []
        lying = aspect is not None and aspect < 1.0 and template.fine_name != "column"
        return [template.oriented(90 if lying else 0, False)]
    if kind == "2":
        if n != 2:
            return []
        if len(node) == 1:
            # both ports on one side (a pump piped from the top): the ports go on the two sides
            # at right angles to it, so each pipe turns once instead of circling the body
            (side,) = node
            across = ("L", "R") if side in "TB" else ("B", "T")
            return [o for o in template.all_orientations() if o.sides == across]
        scored = [(sum((Counter(o.sides) & node).values()), o) for o in template.all_orientations()]
        best = max(s for s, _ in scored)
        return [o for s, o in scored if s == best]
    return [
        o for o in template.all_orientations() if _fits(Counter(o.sides), node, template.placement)
    ]


def _pick(
    templates: Mapping[str, Template],
    groups: list[str],
    configs: list[str] | None,
    node_sides: list[str],
    rng: random.Random,
    classes: list[str] | None = None,
    aspect: float | None = None,
    allow_empty: bool = False,
    not_classes: list[str] | None = None,
) -> Template | None:
    by_class: dict[str, list[list[Template]]] = {}  # fine_name -> [valid orientations / template]
    for t in templates.values():
        if t.group not in groups or (configs is not None and t.port_config not in configs):
            continue
        if classes is not None and t.fine_name not in classes:
            continue
        if not_classes is not None and t.fine_name in not_classes:
            continue
        valid = orientations_for(t, node_sides, aspect, allow_empty)
        if valid:
            by_class.setdefault(t.fine_name, []).append(valid)
    if not by_class:
        return None
    # the class is the unit Stage 2 is scored on, so it is drawn first: X8075's two drawings
    # share a class and must not make it twice as likely as the others
    return rng.choice(rng.choice(by_class[rng.choice(list(by_class))]))


def _flanges_for(
    disc: Template, flange_tpls: list[Template], rng: random.Random
) -> Template | None:
    """A flange template turned so that its ports lie on the same sides as the disc's (the
    pipe axis); None when no flange template is in scope."""
    fits = [o for t in flange_tpls for o in t.all_orientations() if set(o.sides) == set(disc.sides)]
    return rng.choice(fits) if fits else None


def _agitator_for(
    node: Node,
    picked: Template,
    templates: Mapping[str, Template],
    cfg: Mapping[str, Any],
    rng: random.Random,
) -> tuple[Template, float] | None:
    """An agitator for a vessel and its blade's clearance (drawn in `cfg["clearance"]`), or
    None. Only in the `host` class (2062), on a node of one of `labels` (a `tank`: a large
    `general` drawn as 2062 is no vessel in OPEN100), standing (height / width within
    `aspect`); the class is drawn uniformly, then its template. A column or a lying vessel (a
    heater) never gets one."""
    if not cfg or picked.fine_name != cfg.get("host"):
        return None
    if node.label not in cfg.get("labels", [node.label]):
        return None
    lo, hi = cfg["aspect"]
    if not (lo <= node.height / node.width < hi):
        return None
    if rng.random() >= float(cfg["p"]):
        return None
    by_class: dict[str, list[Template]] = {}
    for t in templates.values():
        if t.placement == "attach" and t.fine_name in cfg["classes"]:
            by_class.setdefault(t.fine_name, []).append(t)
    if not by_class:
        return None
    tpl = rng.choice(by_class[rng.choice(list(by_class))]).oriented(0, False)
    lo, hi = cfg.get("clearance", [0.5, 0.5])
    return tpl, rng.uniform(float(lo), float(hi))


_TIP_SIDE = {("H", True): "R", ("H", False): "L", ("V", True): "B", ("V", False): "T"}
_OPPOSITE = {"L": "R", "R": "L", "T": "B", "B": "T"}


def _tip_side(node: Node, ports: list[Port], runs: list[PipeRun]) -> str | None:
    """The side an arrow points to, read from the drawing; None to draw it at random.

    On a straight pipe: the side whose run is at most one arrow long (a bend, a junction or a
    symbol right at its head). In OPEN100, 264 of 362 such arrows sit against something on one
    side, and 257 of the 262 whose head could be told from the pixels point into it. An arrow
    in the middle of a pipe has no side. With one port (a flow arrow against a symbol or at the
    end of a pipe; leaders are another kind): into the symbol when its run is that short, else
    away from the pipe (5 of the 7 in OPEN100 follow these two)."""

    def length(run: PipeRun) -> float:
        pts = run.points
        return sum(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in zip(pts, pts[1:], strict=False))

    size = max(node.width, node.height)
    kind = port_config(ports).kind
    if kind == "1":
        (p,) = ports
        return p.side if length(runs[p.run]) <= size else _OPPOSITE[p.side]
    if kind != "2_straight":
        return None
    near = min(ports, key=lambda p: length(runs[p.run]))
    return near.side if length(runs[near.run]) <= size else None


def _arrow_template(overlays: list[Template], axis: str, forward: bool) -> Template | None:
    """The flow arrow template (an `overlay`, 241) turned to point along `axis`, `forward` =
    right / down. Its tip is the base drawing's R port (BK-2020 draws it pointing right)."""
    for t in overlays:
        tip = next((q.port_id for q in t.ports if q.side == "R"), None)
        if tip is None:
            continue
        for rot in (0, 90, 180, 270):
            o = t.oriented(rot)
            if next(q.side for q in o.ports if q.port_id == tip) == _TIP_SIDE[(axis, forward)]:
                return o
    return None


def _straight_axis(run: PipeRun) -> str | None:
    (x0, y0), (x1, y1) = run.points[0], run.points[-1]
    if run.elbows:
        return None
    if abs(y0 - y1) <= 1e-6 and abs(x0 - x1) > 1e-6:
        return "H"
    if abs(x0 - x1) <= 1e-6 and abs(y0 - y1) > 1e-6:
        return "V"
    return None


def arrow_chains(runs: list[PipeRun], assignments: Mapping[str, Assignment]) -> list[list[str]]:
    """Flow arrows strung on one straight pipe: groups of arrows joined by straight runs (between
    two arrows) along the arrows' own axis; each group in order of first appearance in `runs`."""
    parent: dict[str, str] = {}

    def find(n: str) -> str:
        while parent.setdefault(n, n) != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    order: list[str] = []
    for run in runs:
        if run.internal or run.nozzle or run.a == run.b:
            continue
        a, b = assignments.get(run.a), assignments.get(run.b)
        if a is None or b is None or a.kind != "arrow" or b.kind != "arrow":
            continue
        axis = _straight_axis(run)
        if axis is None or a.axis != axis or b.axis != axis:
            continue
        for n in (run.a, run.b):
            if n not in parent:
                order.append(n)
        parent[find(run.a)] = find(run.b)
    groups: dict[str, list[str]] = {}
    for n in order:
        groups.setdefault(find(n), []).append(n)
    return list(groups.values())


def unify_arrow_chains(
    runs: list[PipeRun],
    assignments: dict[str, Assignment],
    tipped: set[str],
    arrow_tpls: list[Template],
) -> dict[str, Assignment]:
    """Every arrow of a chain (`arrow_chains`) points the same way (PLAN_PHASE_C §2M, user rule
    2026-11-23): the way of its arrows read from the drawing (`tipped`, the majority; the first on
    a tie), else the way drawn for the chain's first arrow. No random draw, so nothing else of
    the variant changes."""
    out = dict(assignments)
    for chain in arrow_chains(runs, assignments):
        votes = [bool(assignments[n].forward) for n in chain if n in tipped]
        if votes:
            forward = votes.count(True) > votes.count(False) or (
                votes.count(True) == votes.count(False) and votes[0]
            )
        else:
            forward = bool(assignments[chain[0]].forward)
        for n in chain:
            a = assignments[n]
            if bool(a.forward) != forward:
                tpl = _arrow_template(arrow_tpls, a.axis or "H", forward)
                out[n] = replace(a, forward=forward, template=tpl)
    return out


def _is_nested(sheet: Sheet, node_id: str, runs_of: dict[str, list[PipeRun]]) -> bool:
    """Centre inside another symbol's bbox and linked to the rest only by `internal` runs."""
    cx, cy = sheet.nodes[node_id].center
    inside = any(
        m.xmin < cx < m.xmax and m.ymin < cy < m.ymax
        for mid, m in sheet.nodes.items()
        if mid != node_id and m.label not in NOT_SYMBOLS
    )
    return inside and all(r.internal for r in runs_of.get(node_id, []))


def _check_rules(rules: list[Mapping[str, Any]]) -> None:
    for rule in rules:
        kind = rule.get("kind", "iso")
        if kind not in KINDS:
            raise ValueError(f"rule {dict(rule)}: kind must be one of {KINDS}")
        if kind == "iso" and not rule.get("groups"):
            raise ValueError(f"rule {dict(rule)}: an iso rule needs `groups`")


def assign_symbols(
    sheet: Sheet,
    runs: list[PipeRun],
    templates: Mapping[str, Template],
    rng: random.Random,
    cfg: Mapping[str, Any],
) -> dict[str, Assignment]:
    """One Assignment per symbol node (connector / crossing / background are skipped).

    `cfg` is the `render_iso` mapping (see `load_config`); `templates` come from
    `iso_templates.load_templates`, whose `scope` decides which are drawable. Nodes go in
    sheet order and every random draw comes from `rng`, so a seed reproduces the result.
    The first rule of `cfg["rules"]` whose filters (label, size, ports, next_to, not_next_to,
    tagged, min_aspect / max_aspect = height / width in [min, max)) match and that yields a symbol
    wins; a `replace` rule first rolls `p_replace`. No rule
    -> a plain box. A symbol of a `cfg["flanges"]["classes"]` class gets flanges with chance `p`.
    `cfg["vocab_only"]` (PLAN_PHASE_C §2N): templates outside the Stage 2 vocabulary are never
    drawn; a rule that only had those falls through like any rule with no template left.
    """
    if cfg.get("vocab_only", False):  # PLAN_PHASE_C §2N rule 2: Stage 1's library = the vocab
        templates = {k: t for k, t in templates.items() if t.in_vocab}
    rules = list(cfg["rules"])
    _check_rules(rules)
    small_max, large_min = cfg["size_class"]["small_max"], cfg["size_class"]["large_min"]
    p_replace = float(cfg["p_replace"])
    flange_cfg = cfg.get("flanges") or {}
    flange_classes = set(flange_cfg.get("classes", []))
    p_flanges = float(flange_cfg.get("p", 0.0))
    flange_tpls = [t for t in templates.values() if t.fine_name == flange_cfg.get("symbol")]
    agitator_cfg = cfg.get("agitators") or {}
    arrow_tpls = [
        t
        for t in templates.values()
        if t.placement == "overlay" and t.fine_name == KIND_LABELS["arrow"]
    ]

    runs_of: dict[str, list[PipeRun]] = {}
    for run in runs:
        runs_of.setdefault(run.a, []).append(run)
        if run.b != run.a:
            runs_of.setdefault(run.b, []).append(run)

    out: dict[str, Assignment] = {}
    tipped: set[str] = set()  # arrows whose way was read from the drawing
    fallbacks: Counter[tuple[str, str]] = Counter()
    for nid, node in sheet.nodes.items():
        if node.label in NOT_SYMBOLS:
            continue
        if _is_nested(sheet, nid, runs_of):
            out[nid] = Assignment("nested", None)
            continue
        ports = symbol_ports(sheet, nid, runs)
        pcfg = port_config(ports)
        sides = [p.side for p in ports]
        # a tag (an `instrumentation` node) on a run of this node; `symbol_ports` skips bubbles
        tagged = any(
            sheet.nodes[r.b if r.a == nid else r.a].label == "instrumentation"
            for r in runs_of.get(nid, [])
            if r.a != r.b
        )
        size = size_class(node, pcfg, small_max, large_min)
        result: Assignment | None = None
        for rule in rules:
            if rule.get("label") not in (None, node.label):
                continue
            if rule.get("size") not in (None, size):
                continue
            if "ports" in rule and pcfg.kind not in rule["ports"]:
                continue
            neighbors = {sheet.nodes[p.neighbor].label for p in ports}
            if rule.get("next_to") and not neighbors & set(rule["next_to"]):
                continue
            if rule.get("not_next_to") and neighbors & set(rule["not_next_to"]):
                continue
            if rule.get("tagged") not in (None, tagged):
                continue
            aspect = node.height / node.width if node.width > 0 else 1.0
            if aspect < float(rule.get("min_aspect", 0.0)):
                continue
            if aspect >= float(rule.get("max_aspect", float("inf"))):
                continue
            if rule.get("replace") and rng.random() >= p_replace:
                continue
            kind = rule.get("kind", "iso")
            if kind == "iso":
                picked = _pick(
                    templates,
                    rule["groups"],
                    rule.get("configs"),
                    sides,
                    rng,
                    rule.get("classes"),
                    aspect,
                    bool(rule.get("allow_empty", False)),
                    rule.get("not_classes"),
                )
                if picked is None:
                    continue
                label = picked.fine_name if picked.in_vocab else OTHER_ISO
                flanges = None
                if picked.fine_name in flange_classes and rng.random() < p_flanges:
                    flanges = _flanges_for(picked, flange_tpls, rng)
                agitator = (
                    _agitator_for(node, picked, templates, agitator_cfg, rng)
                    if picked.port_config == "multi"
                    else None
                )
                result = Assignment(
                    "iso",
                    label,
                    template=picked,
                    flanges=flanges,
                    agitator=agitator[0] if agitator else None,
                    clearance=agitator[1] if agitator else 0.5,
                )
            elif kind in ("arrow", "flag"):
                if sides:
                    axis = "H" if rng.choice(sides) in "LR" else "V"
                else:  # no port to read the pipe direction from: follow the box
                    axis = "H" if node.width >= node.height else "V"
                forward = rng.random() < 0.5
                if kind == "arrow" and (tip := _tip_side(node, ports, runs)) is not None:
                    axis, forward = ("H" if tip in "LR" else "V"), tip in "RB"
                    tipped.add(nid)
                tpl = _arrow_template(arrow_tpls, axis, forward) if kind == "arrow" else None
                result = Assignment(
                    kind, KIND_LABELS[kind], template=tpl, axis=axis, forward=forward
                )
            elif kind == "leader":  # the arrowhead is the flow arrow's, drawn at any angle by V5
                tpl = arrow_tpls[0].oriented(0) if arrow_tpls else None
                result = Assignment(kind, KIND_LABELS[kind], template=tpl)
            else:
                result = Assignment(kind, KIND_LABELS[kind])
            break
        if result is None:
            result = Assignment("generic", KIND_LABELS["generic"])
            fallbacks[(node.label, pcfg.kind)] += 1
        out[nid] = result

    if cfg.get("arrow_chains", False):  # PLAN_PHASE_C §2M (OPEN100-ISO v2)
        out = unify_arrow_chains(runs, out, tipped, arrow_tpls)
    if fallbacks:
        detail = ", ".join(f"{lb}/{cf}: {n}" for (lb, cf), n in sorted(fallbacks.items()))
        logger.info(
            "%s: %d nodes without a rule -> other_generic (%s)",
            sheet.name,
            sum(fallbacks.values()),
            detail,
        )
    return out
