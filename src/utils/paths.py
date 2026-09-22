"""Resolve the data/ layout (raw -> canonical -> derived) from config.

The three-tier layout is fixed by the project (see data/ tree):
    raw/        tier 0 — immutable source (synthetic/, real/)
    canonical/  tier 1 — source of truth (images/, annotations.coco.json,
                         classes.yaml, manifest.csv)
    derived/    tier 2 — regenerable (yolo_fine/, yolo_coarse/, tiles/, crops_fgc/)

This module centralises those paths so no stage hardcodes them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataPaths:
    root: Path  # the data/ directory

    @property
    def canonical(self) -> Path:
        return self.root / "canonical"

    @property
    def coco_json(self) -> Path:
        return self.canonical / "annotations.coco.json"

    @property
    def classes_yaml(self) -> Path:
        return self.canonical / "classes.yaml"

    @property
    def manifest_csv(self) -> Path:
        return self.canonical / "manifest.csv"

    @property
    def derived(self) -> Path:
        return self.root / "derived"

    @property
    def yolo_coarse(self) -> Path:
        return self.derived / "yolo_coarse"

    @property
    def yolo_fine(self) -> Path:
        return self.derived / "yolo_fine"

    @property
    def tiles(self) -> Path:
        return self.derived / "tiles"

    @property
    def crops_fgc(self) -> Path:
        return self.derived / "crops_fgc"

    _IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    def find_image(self, image_id: str) -> Path:
        """Resolve `canonical/images/<image_id>.*` — manifest.csv stores ids without
        an extension, so every stage that needs the actual file goes through this.
        """
        for ext in self._IMAGE_EXTENSIONS:
            candidate = self.canonical / "images" / f"{image_id}{ext}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"No image file for id={image_id!r} under {self.canonical / 'images'}"
        )
