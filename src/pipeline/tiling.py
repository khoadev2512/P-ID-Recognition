"""Overlap-preserving tiling of full-resolution diagrams (SAHI strategy, ref [40]).

P&ID sheets exceed 10,000 x 7,000 px (§2.2.1); naive tiling truncates symbols at
patch boundaries and drops recall (Gap 4). This step slices each canonical image
into fixed-size patches with a configurable overlap margin so every symbol is fully
visible in at least one patch, and rewrites the YOLO labels into patch-local
coordinates. WBF re-merges patch detections at inference time (see pipeline.detect_infer).

Config: cfg.data.tiling.{size, overlap, min_visibility}. Output: derived/tiles/.

Ported from PID_Symbol_Detection's `utils/patch_generator.py` (PatchGenerator),
which delegates the crop + bbox reprojection/clip/visibility-filter entirely to
Albumentations (A.Crop + A.BboxParams(format="yolo", min_visibility=...)). Fixes
that reference implementation's edge-coverage bug: its
`range(0, height - patch_h + 1, step)` silently drops a trailing strip whenever
`height` isn't exactly `patch_h + k*step`. Here the last row/col start is always
pulled back flush against the image edge (see `_axis_starts`), so every pixel is
covered by at least one tile even when an image dimension is smaller than the
patch size.
"""

from __future__ import annotations

import albumentations as A
import cv2
import hydra
import pandas as pd
import yaml
from omegaconf import DictConfig

from pipeline.base import BasePipeline
from utils import bbox_utils
from utils.cli import CONFIG_DIR
from utils.types import BBox


class TilingPipeline(BasePipeline):
    """Slice canonical images (+ their full-image YOLO labels) into overlapping tiles."""

    def validate(self) -> bool:
        if not self.paths.manifest_csv.exists():
            return False
        labels_dir = self._labels_src_root / "labels"
        return labels_dir.exists() and any(labels_dir.glob("*.txt"))

    @property
    def _labels_src_root(self):
        label_level = str(self.cfg.data.label_level)
        if label_level == "coarse":
            return self.paths.yolo_coarse
        if label_level == "fine":
            return self.paths.yolo_fine
        raise ValueError(f"cfg.data.label_level must be 'coarse' or 'fine', got {label_level!r}")

    def run(self) -> None:
        if not self.validate():
            raise FileNotFoundError(
                f"{self.paths.manifest_csv} missing, or {self._labels_src_root / 'labels'} "
                "missing/empty; run pid-prep-manifest and pid-prep-yolo first."
            )

        label_level = str(self.cfg.data.label_level)
        size = int(self.cfg.data.tiling.size)
        overlap_frac = float(self.cfg.data.tiling.overlap)
        min_visibility = float(self.cfg.data.tiling.min_visibility)

        overlap_px = round(overlap_frac * size)
        step = size - overlap_px
        if step <= 0:
            raise ValueError(
                f"cfg.data.tiling.overlap={overlap_frac} implies overlap_px={overlap_px} >= "
                f"size={size}; step would be non-positive."
            )

        manifest = pd.read_csv(self.paths.manifest_csv, dtype={"id": str})
        labels_dir = self._labels_src_root / "labels"

        splits_with_tiles: set[str] = set()

        for _, row in manifest.iterrows():
            image_id = row["id"]
            split = row["split"]

            image_path = self.paths.find_image(image_id)
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                raise ValueError(f"Failed to read image: {image_path}")
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]

            label_path = labels_dir / f"{image_id}.txt"
            src_rows = bbox_utils.read_yolo_txt(label_path) if label_path.exists() else []
            class_labels = [class_id for class_id, _ in src_rows]
            bboxes = [[box.x, box.y, box.w, box.h] for _, box in src_rows]

            row_starts = self._axis_starts(height, size, step)
            col_starts = self._axis_starts(width, size, step)

            for r, y in enumerate(row_starts):
                for c, x in enumerate(col_starts):
                    x_max = min(x + size, width)
                    y_max = min(y + size, height)

                    crop = A.Compose(
                        [A.Crop(x_min=x, y_min=y, x_max=x_max, y_max=y_max)],
                        bbox_params=A.BboxParams(
                            format="yolo",
                            min_visibility=min_visibility,
                            label_fields=["class_labels"],
                        ),
                    )
                    transformed = crop(image=image, bboxes=bboxes, class_labels=class_labels)
                    cropped_img = transformed["image"]
                    cropped_bboxes = transformed["bboxes"]
                    cropped_labels = transformed["class_labels"]

                    tile_name = f"{image_id}_{r}_{c}"

                    img_out = self.paths.tiles / "images" / split / f"{tile_name}.jpg"
                    img_out.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(img_out), cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))

                    out_rows = [
                        (int(class_id), BBox(x=bx, y=by, w=bw, h=bh))
                        for class_id, (bx, by, bw, bh) in zip(
                            cropped_labels, cropped_bboxes, strict=True
                        )
                    ]
                    label_out = self.paths.tiles / "labels" / split / f"{tile_name}.txt"
                    bbox_utils.write_yolo_txt(label_out, out_rows)

                    splits_with_tiles.add(split)

        self._write_data_yaml(splits_with_tiles, label_level)

    @staticmethod
    def _axis_starts(dim: int, patch: int, step: int) -> list[int]:
        """Sliding-window start positions along one axis, edge-coverage-safe.

        `range(0, dim - patch, step)` (the reference repo's approach, off-by-one
        aside) can leave a trailing strip of the image uncovered whenever `dim` isn't
        exactly `patch + k*step`. Appending `max(dim - patch, 0)` as an explicit final
        start pulls the last tile back flush against the far edge, guaranteeing full
        coverage; the dedup avoids emitting the same tile twice when it already lines
        up with the last computed start.
        """
        starts = list(range(0, max(dim - patch, 1), step))
        last = max(dim - patch, 0)
        if not starts or starts[-1] != last:
            starts.append(last)
        return starts

    def _write_data_yaml(self, splits_with_tiles: set[str], label_level: str) -> None:
        if label_level == "coarse":
            names = self.classmap.coarse_classes
        else:
            names = self.classmap.fine_classes

        data: dict = {"nc": len(names), "names": list(names)}
        for split in ("train", "val", "test"):
            if split in splits_with_tiles:
                data[split] = str((self.paths.tiles / "images" / split).resolve())

        data_yaml = self.paths.tiles / "data.yaml"
        data_yaml.parent.mkdir(parents=True, exist_ok=True)
        with data_yaml.open("w") as f:
            yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    TilingPipeline(cfg).run()


if __name__ == "__main__":
    main()
