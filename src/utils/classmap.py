"""Loader and accessor for canonical/classes.yaml — the fine<->coarse class map.

classes.yaml is described in the report as the single most important file: it defines
the fine vocabulary C, the coarse super-categories, the fine->coarse mapping, and which
super-categories form "visually similar groups" that get routed to the FGC stage
(valve family, instrument-bubble family, fitting family; §2.2.2).

Expected schema (see configs/classes.example.yaml):

    coarse:            # super-categories (Stage 1 label set)
      - valve
      - instrument_bubble
      - pump
      - fitting
    fine:              # fine class -> parent coarse class
      gate_valve: valve
      globe_valve: valve
      ...
    fgc_groups:        # coarse categories that require the FGC stage
      - valve
      - instrument_bubble
      - fitting
    rare_fine:         # optional: fine classes flagged rare (long-tail; §2.2.2)
      - needle_valve
      - pinch_valve
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ClassMap:
    coarse_classes: list[str]
    fine_to_coarse: dict[str, str]
    fgc_groups: list[str]
    rare_fine: list[str]
    _fine_classes: list[str] = field(init=False, repr=False)
    _coarse_index: dict[str, int] = field(init=False, repr=False)
    _fine_index: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # dict insertion order is stable in Python 3.7+, so this order matches
        # the fine: block order in classes.yaml -> reproducible YOLO fine ids.
        self._fine_classes = list(self.fine_to_coarse.keys())
        self._coarse_index = {name: i for i, name in enumerate(self.coarse_classes)}
        self._fine_index = {name: i for i, name in enumerate(self._fine_classes)}

    @property
    def fine_classes(self) -> list[str]:
        return self._fine_classes

    def coarse_id(self, name: str) -> int:
        return self._coarse_index[name]

    def fine_id(self, name: str) -> int:
        return self._fine_index[name]

    def needs_fgc(self, coarse_class: str) -> bool:
        return coarse_class in self.fgc_groups

    def fine_of_group(self, coarse_class: str) -> list[str]:
        return [fine for fine, coarse in self.fine_to_coarse.items() if coarse == coarse_class]


def load_classmap(path: str | Path) -> ClassMap:
    """Parse classes.yaml into a ClassMap, validating it matches the documented schema."""
    path = Path(path)
    with path.open("r") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")

    missing = {"coarse", "fine", "fgc_groups"} - raw.keys()
    if missing:
        raise ValueError(f"{path}: missing required key(s) {sorted(missing)}")

    coarse_classes = list(raw["coarse"])
    fine_to_coarse = dict(raw["fine"])
    fgc_groups = list(raw["fgc_groups"])
    rare_fine = list(raw.get("rare_fine", []))

    unknown_groups = set(fgc_groups) - set(coarse_classes)
    if unknown_groups:
        raise ValueError(
            f"{path}: fgc_groups references unknown coarse classes {sorted(unknown_groups)}"
        )

    unknown_parents = set(fine_to_coarse.values()) - set(coarse_classes)
    if unknown_parents:
        raise ValueError(
            f"{path}: fine classes reference unknown coarse parents {sorted(unknown_parents)}"
        )

    unknown_rare = set(rare_fine) - set(fine_to_coarse.keys())
    if unknown_rare:
        raise ValueError(
            f"{path}: rare_fine references unknown fine classes {sorted(unknown_rare)}"
        )

    return ClassMap(
        coarse_classes=coarse_classes,
        fine_to_coarse=fine_to_coarse,
        fgc_groups=fgc_groups,
        rare_fine=rare_fine,
    )
