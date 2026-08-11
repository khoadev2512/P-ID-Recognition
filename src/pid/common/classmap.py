"""Loader and accessor for canonical/classes.yaml — the fine<->coarse class map.

classes.yaml is described in the report as the single most important file: it defines
the fine vocabulary C, the coarse super-categories, the fine->coarse mapping, and which
super-categories form "visually similar groups" that get routed to the FGC stage
(valve family, instrument-bubble family, fitting family; §2.2.2).

Expected schema (see configs/data/classes.example.yaml):

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

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ClassMap:
    coarse_classes: list[str]
    fine_to_coarse: dict[str, str]
    fgc_groups: list[str]
    rare_fine: list[str]

    @property
    def fine_classes(self) -> list[str]:
        raise NotImplementedError  # TODO: stable-ordered keys of fine_to_coarse

    def coarse_id(self, name: str) -> int:
        raise NotImplementedError  # TODO: index into coarse_classes (YOLO coarse id)

    def fine_id(self, name: str) -> int:
        raise NotImplementedError  # TODO: index into fine_classes (YOLO fine id)

    def needs_fgc(self, coarse_class: str) -> bool:
        raise NotImplementedError  # TODO: coarse_class in fgc_groups

    def fine_of_group(self, coarse_class: str) -> list[str]:
        raise NotImplementedError  # TODO: fine classes whose parent == coarse_class


def load_classmap(path: str | Path) -> ClassMap:
    """Parse classes.yaml into a ClassMap. TODO: implement + validate schema."""
    raise NotImplementedError
