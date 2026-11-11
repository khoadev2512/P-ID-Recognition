"""Extract per-symbol crops for FGC training (report §4.2.3).

Training the fine-grained stage on cropped symbol regions — not full diagrams — is
the mechanism that makes rare-class balancing practical (§2.2.2 long tail). This step
cuts one crop per ground-truth symbol from canonical images and writes them into a
per-fine-class folder tree that torchvision ImageFolder can read directly:

    derived/crops_fgc/{train,val,test}/<fine_class>/<image_id>_<inst>.png

Only symbols in fgc_groups (valve / instrument_bubble / fitting) are extracted by
default; controlled by cfg.fgc.groups (classmap.needs_fgc is the source of truth,
should match).

cfg.fgc.global_encoder=true extracts EVERY symbol instead, including the coarse
families with a single fine child that the per-group path has no head for. That is
required by the one-shot ISO path: the global embedding encoder trains over the whole
fine vocabulary, so a class with no crops here would simply be missing from the
embedding space it is supposed to be separable in.

Each PNG is the symbol's stored CONTEXT (box grown by CROP_CONTEXT_FRAC per side), not
the view it is classified on; `<fine_class>/_boxes.csv` records the box and the crop's
origin so the dataset cuts the view (box + CROP_MARGIN_FRAC, the same as at inference)
or a jittered one at train time. See utils.crop_box.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import hydra
import pandas as pd
from omegaconf import DictConfig

from pipeline.base import BasePipeline
from utils.cli import CONFIG_DIR
from utils.crop_box import StoredCrop, context_region, write_boxes


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

        stored: dict[Path, list[tuple[str, StoredCrop]]] = {}  # per class dir: its _boxes.csv
        for coco_image_id, img in images_by_id.items():
            image_id = self._stem(img["file_name"])
            split = split_by_id.get(image_id)
            if split is None:
                # Image in COCO but absent from manifest.csv (stale manifest) — skip
                # rather than guessing a split for it.
                continue

            fgc_anns = self._wanted_anns(
                anns_by_image.get(coco_image_id, []), category_id_to_category
            )
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
                box = (float(x), float(y), float(x + w), float(y + h))
                x1, y1, x2, y2 = context_region(box, width, height)
                if x2 <= x1 or y2 <= y1:
                    continue  # degenerate box (outside the image): nothing to crop
                crop = image[y1:y2, x1:x2]

                out_dir = self.paths.crops_fgc / split / fine_class
                out_dir.mkdir(parents=True, exist_ok=True)
                name = f"{image_id}_{ann_index}.png"
                cv2.imwrite(str(out_dir / name), crop)
                stored.setdefault(out_dir, []).append((name, StoredCrop(box, (x1, y1))))

        for out_dir, rows in stored.items():
            write_boxes(out_dir, rows)

    def _wanted_anns(
        self, anns: list[dict], category_id_to_category: dict[int, dict]
    ) -> list[dict]:
        """Annotations to crop: every symbol under cfg.fgc.global_encoder, else only
        those whose coarse family is routed to a fine head (classmap.needs_fgc)."""
        if bool(self.cfg.fgc.get("global_encoder", False)):
            return list(anns)
        return [
            ann
            for ann in anns
            if self.classmap.needs_fgc(
                category_id_to_category[ann["category_id"]]["supercategory"]
            )
        ]

    @staticmethod
    def _stem(file_name: str) -> str:
        name = file_name.split("/")[-1]
        return name.rsplit(".", 1)[0]


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    ExtractCropsPipeline(cfg).run()


if __name__ == "__main__":
    main()
