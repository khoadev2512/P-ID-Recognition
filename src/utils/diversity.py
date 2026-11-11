"""Near-duplicate filtering (V6 of PLAN_ISO_RENDER_PID2GRAPH): by content, with picture hashes
kept as a record.

Two renderings of one source sheet share their topology and, up to V4's axis mapping, their
layout, so neither a graph hash (WL) nor a coarse picture hash tells them apart: a 16 x 16 hash of
a 7000 px page sees ~450 px cells, i.e. the layout, not which symbol was drawn where. What does
tell them apart is the choice made for each node. `ContentDeduper` holds, per kept variant, one
key per node (`{node id: choice}`, the same node ids for every variant of a sheet) and calls a
candidate a duplicate when the share of nodes whose choice differs from a kept variant's
(`content_distance`) is below `min_node_diff`. SynthPID compares label histograms instead
(its variants change the graph, so their nodes do not correspond); with corresponding nodes the
per-node share also sees two symbols swapping places, which a histogram does not.

`ahash` records which cells of a coarse grid are darker than the grid's mean, `dhash` which
cells are darker than their right neighbour. Both are bit strings held in an int; `hamming`
counts the bits where two differ. `Deduper` measures the distance of a picture to the kept ones
(the larger of the two Hamming distances); V6 records it and no longer drops on it.

Pages are mostly white with thin ink, so the image is reduced with a box filter (each cell is
the mean of its area, which keeps thin lines as grey) and then inverted: a cell's value is its
ink coverage.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from PIL import Image


def _ink_grid(img: np.ndarray | Image.Image, width: int, height: int) -> np.ndarray:
    """Mean ink coverage (0 = paper, 255 = black) of a gray image on a `height` x `width` grid."""
    pil = img if isinstance(img, Image.Image) else Image.fromarray(np.asarray(img, np.uint8))
    small = pil.convert("L").resize((width, height), Image.Resampling.BOX)
    return 255.0 - np.asarray(small, np.float64)


def _pack(bits: np.ndarray) -> int:
    value = 0
    for b in bits.ravel():
        value = (value << 1) | int(b)
    return value


def ahash(img: np.ndarray | Image.Image, size: int = 16) -> int:
    """Average hash: `size`² bits, 1 where a cell's ink is above the mean ink of all cells."""
    grid = _ink_grid(img, size, size)
    return _pack(grid > grid.mean())


def dhash(img: np.ndarray | Image.Image, size: int = 16) -> int:
    """Difference hash: `size`² bits, 1 where a cell has more ink than its right neighbour
    (the grid is `size` + 1 cells wide)."""
    grid = _ink_grid(img, size + 1, size)
    return _pack(grid[:, :-1] > grid[:, 1:])


def hamming(a: int, b: int) -> int:
    """Number of bits in which two hashes differ."""
    return (a ^ b).bit_count()


@dataclass
class Deduper:
    """Keeps images that are not near-duplicates of an earlier kept one.

    A candidate (its two hashes) is a duplicate of a kept image when both Hamming distances are
    below `min_distance`. `nearest` is the smallest `max(da, dd)` against the kept images (None
    when nothing is kept yet). `is_duplicate` only asks; `add` remembers an image (call it once
    the candidate has passed every other filter); `offer` does both.
    """

    min_distance: int
    kept: list[tuple[int, int]] = field(default_factory=list)

    def distance(self, a: int, d: int) -> int | None:
        """Smallest, over the kept images, of the larger of the two Hamming distances."""
        return min((max(hamming(a, ka), hamming(d, kd)) for ka, kd in self.kept), default=None)

    def is_duplicate(self, a: int, d: int) -> tuple[bool, int | None]:
        """(whether the hashes are within `min_distance` of a kept image, that distance)."""
        nearest = self.distance(a, d)
        return nearest is not None and nearest < self.min_distance, nearest

    def add(self, a: int, d: int) -> None:
        self.kept.append((a, d))

    def offer(self, a: int, d: int) -> tuple[bool, int | None]:
        """Keep the candidate unless it is a duplicate; returns (accepted, nearest)."""
        duplicate, nearest = self.is_duplicate(a, d)
        if not duplicate:
            self.add(a, d)
        return not duplicate, nearest


def content_distance(a: Mapping[str, str], b: Mapping[str, str]) -> float:
    """Share of nodes (the union of both key sets) whose choice differs; 0.0 for two empty maps.
    A node present in only one map counts as differing."""
    nodes = a.keys() | b.keys()
    if not nodes:
        return 0.0
    return sum(a.get(n) != b.get(n) for n in nodes) / len(nodes)


@dataclass
class ContentDeduper:
    """Keeps variants whose per-node choices are not near-copies of a kept one.

    A candidate is a duplicate when `content_distance` to some kept variant is below
    `min_node_diff` (0.05 = fewer than 5% of nodes differ). `distance` is the smallest distance
    to the kept variants (None when nothing is kept yet); `add` remembers a variant (call it once
    the candidate has passed every other filter).
    """

    min_node_diff: float
    kept: list[dict[str, str]] = field(default_factory=list)

    def distance(self, choices: Mapping[str, str]) -> float | None:
        return min((content_distance(choices, k) for k in self.kept), default=None)

    def is_duplicate(self, choices: Mapping[str, str]) -> tuple[bool, float | None]:
        """(whether `choices` is within `min_node_diff` of a kept variant, that distance)."""
        nearest = self.distance(choices)
        return nearest is not None and nearest < self.min_node_diff, nearest

    def add(self, choices: Mapping[str, str]) -> None:
        self.kept.append(dict(choices))
