"""Convert canonical COCO annotations into YOLO txt label sets.

Emits BOTH label sets from the same source, selected by cfg.data.label_level:
    coarse -> derived/yolo_coarse/  (Stage 1 detector training; report §4.2.2)
    fine   -> derived/yolo_fine/    (full-fine-label ablation; §4.2.2)

Class ids come from utils.classmap (coarse_id / fine_id) so ids stay stable across
runs. Boxes are normalised to YOLO center-xywh in [0, 1].

Output is FULL-canonical-image-scale labels (one .txt per canonical image, mirroring
canonical/images/), NOT tiled — pipeline.tiling consumes this directory as its source.
"""

from __future__ import annotations

import json

import hydra
import yaml
from omegaconf import DictConfig

from pipeline.base import BasePipeline
from utils import bbox_utils
from utils.cli import CONFIG_DIR


class CocoToYoloPipeline(BasePipeline):
    """Emit full-image YOLO label txts (+ an Ultralytics data.yaml) from COCO."""

    def validate(self) -> bool:
        return self.paths.coco_json.exists() and self.paths.classes_yaml.exists()

    @property
    def _out_root(self):
        label_level = str(self.cfg.data.label_level)
        if label_level == "coarse":
            return self.paths.yolo_coarse
        if label_level == "fine":
            return self.paths.yolo_fine
        raise ValueError(f"cfg.data.label_level must be 'coarse' or 'fine', got {label_level!r}")

    def run(self) -> None:
        if not self.validate():
            raise FileNotFoundError(
                f"{self.paths.coco_json} and/or {self.paths.classes_yaml} missing; "
                "run pid-prep-classes first."
            )

        with self.paths.coco_json.open("r") as f:
            coco = json.load(f)

        label_level = str(self.cfg.data.label_level)
        out_root = self._out_root
        labels_dir = out_root / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)

        category_id_to_category = {c["id"]: c for c in coco.get("categories", [])}

        anns_by_image: dict[int, list[dict]] = {}
        for ann in coco.get("annotations", []):
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

        for img in coco.get("images", []):
            image_id = self._stem(img["file_name"])
            width, height = img["width"], img["height"]

            rows: list[tuple[int, object]] = []
            for ann in anns_by_image.get(img["id"], []):
                category = category_id_to_category[ann["category_id"]]
                x, y, w, h = ann["bbox"]
                xyxy = (x, y, x + w, y + h)
                box = bbox_utils.xyxy_to_yolo(xyxy, width, height)

                if label_level == "coarse":
                    class_id = self.classmap.coarse_id(category["supercategory"])
                else:
                    class_id = self.classmap.fine_id(category["name"])
                rows.append((class_id, box))

            bbox_utils.write_yolo_txt(labels_dir / f"{image_id}.txt", rows)

        self._write_data_yaml(out_root, label_level)

    def _write_data_yaml(self, out_root, label_level: str) -> None:
        if label_level == "coarse":
            names = self.classmap.coarse_classes
        else:
            names = self.classmap.fine_classes
        data = {"nc": len(names), "names": list(names)}
        with (out_root / "data.yaml").open("w") as f:
            yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)

    @staticmethod
    def _stem(file_name: str) -> str:
        name = file_name.split("/")[-1]
        return name.rsplit(".", 1)[0]


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    CocoToYoloPipeline(cfg).run()


if __name__ == "__main__":
    main()
