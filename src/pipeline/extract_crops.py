"""Extract per-symbol crops for FGC training (report §4.2.3).

Training the fine-grained stage on cropped symbol regions — not full diagrams — is
the mechanism that makes rare-class balancing practical (§2.2.2 long tail). This step
cuts one crop per ground-truth symbol from canonical images and writes them into a
per-fine-class folder tree that torchvision ImageFolder can read directly:

    derived/crops_fgc/{train,val,test}/<fine_class>/<image_id>_<inst>.png

Only symbols in fgc_groups (valve / instrument_bubble / fitting) are extracted by
default; controlled by cfg.fgc.groups (classmap.needs_fgc is the source of truth,
should match).
"""

from __future__ import annotations

import json

import cv2
import hydra
import pandas as pd
from omegaconf import DictConfig

from pipeline.base import BasePipeline
from utils import bbox_utils
from utils.cli import CONFIG_DIR

# Margin added around each ground-truth box before cropping, as a fraction of the
# box's own size per side. Symbol detectors/crops benefit from a little context
# around the tight box (stems, adjoining lines used to disambiguate fine classes);
# 10% is a reasonable default that doesn't drag in neighbouring symbols on typical
# P&ID symbol spacing.
CROP_MARGIN_FRAC = 0.10


class ExtractCropsPipeline(BasePipeline):
    """Crop one PNG per fgc-group ground-truth symbol, bucketed by fine class + split."""

    def validate(self) -> bool:
        return self.paths.manifest_csv.exists() and self.paths.classes_yaml.exists()

    def run(self) -> None:
        if not self.validate():
            raise FileNotFoundError(
                f"{self.paths.manifest_csv} and/or {self.paths.classes_yaml} missing; "
                "run pid-prep-classes and pid-prep-manifest first."
            )

        with self.paths.coco_json.open("r") as f:
            coco = json.load(f)

        manifest = pd.read_csv(self.paths.manifest_csv, dtype={"id": str})
        split_by_id = dict(zip(manifest["id"], manifest["split"], strict=True))

        category_id_to_category = {c["id"]: c for c in coco.get("categories", [])}

        images_by_id = {img["id"]: img for img in coco.get("images", [])}

        anns_by_image: dict[int, list[dict]] = {}
        for ann in coco.get("annotations", []):
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

        for coco_image_id, img in images_by_id.items():
            image_id = self._stem(img["file_name"])
            split = split_by_id.get(image_id)
            if split is None:
                # Image in COCO but absent from manifest.csv (stale manifest) — skip
                # rather than guessing a split for it.
                continue

            fgc_anns = [
                ann
                for ann in anns_by_image.get(coco_image_id, [])
                if self.classmap.needs_fgc(
                    category_id_to_category[ann["category_id"]]["supercategory"]
                )
            ]
            if not fgc_anns:
                continue

            image_path = self.paths.find_image(image_id)
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Failed to read image: {image_path}")
            height, width = image.shape[:2]

            for ann_index, ann in enumerate(fgc_anns):
                category = category_id_to_category[ann["category_id"]]
                fine_class = category["name"]

                x, y, w, h = ann["bbox"]
                xyxy = (x, y, x + w, y + h)
                xyxy = self._add_margin(xyxy, w, h)
                xyxy_int = bbox_utils.clip_xyxy(xyxy, width, height)
                crop = bbox_utils.crop_xyxy(image, xyxy_int)

                out_dir = self.paths.crops_fgc / split / fine_class
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{image_id}_{ann_index}.png"
                cv2.imwrite(str(out_path), crop)

    @staticmethod
    def _add_margin(
        xyxy: tuple[float, float, float, float], box_w: float, box_h: float
    ) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = xyxy
        mx = box_w * CROP_MARGIN_FRAC
        my = box_h * CROP_MARGIN_FRAC
        return (x1 - mx, y1 - my, x2 + mx, y2 + my)

    @staticmethod
    def _stem(file_name: str) -> str:
        name = file_name.split("/")[-1]
        return name.rsplit(".", 1)[0]


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    ExtractCropsPipeline(cfg).run()


if __name__ == "__main__":
    main()
