"""Core data structures shared across stages.

These mirror the detection tuple D = {(b_i, c_i, s_i)} formalised in report §2.2.1,
extended with the coarse/fine label split that the two-stage pipeline needs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BBox:
    """Axis-aligned box in pixel space, xywh with (x, y) as the center.

    Matches the report's b_i = (x_i, y_i, w_i, h_i) convention (§2.2.1).
    """

    x: float
    y: float
    w: float
    h: float

    def to_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x - self.w / 2, self.y - self.h / 2, self.x + self.w / 2, self.y + self.h / 2)

    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float) -> BBox:
        return cls(x=(x1 + x2) / 2, y=(y1 + y2) / 2, w=x2 - x1, h=y2 - y1)


@dataclass
class Detection:
    """One detected symbol.

    coarse_class comes from Stage 1 (detector); fine_class is filled by Stage 2 (FGC)
    only for symbols routed through the fine-grained head, else it equals coarse_class.
    """

    bbox: BBox
    coarse_class: str
    score: float
    fine_class: str | None = None
    fine_score: float | None = None


@dataclass
class SymbolInstance:
    """A ground-truth annotation carrying BOTH label levels.

    Populated from canonical/annotations.coco.json, which stores fine + coarse
    (report data layout). This is the object the whole data layer converts into
    YOLO txt / crops / manifest rows.
    """

    image_id: str
    bbox: BBox
    fine_class: str
    coarse_class: str
