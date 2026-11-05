"""Perceptual hashes and near-duplicate filtering (V6 of PLAN_ISO_RENDER_PID2GRAPH).

Two renderings of one source sheet share their topology, so a graph hash (WL) cannot tell them
apart; their pictures can. `ahash` records which cells of a coarse grid are darker than the
grid's mean, `dhash` which cells are darker than their right neighbour. Both are bit strings
held in an int; `hamming` counts the bits where two differ. `Deduper` drops an image that is within
`min_distance` bits of a kept one in both hashes (the two must agree, so one chance coincidence
of the coarse average hash does not discard a picture).

Pages are mostly white with thin ink, so the image is reduced with a box filter (each cell is
the mean of its area, which keeps thin lines as grey) and then inverted: a cell's value is its
ink coverage.
"""

from __future__ import annotations

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
