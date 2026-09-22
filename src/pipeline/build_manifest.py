"""Build canonical/manifest.csv — one row per image.

Columns (per data/ layout spec):
    id, source, split, n_symbols, classes_present

source is synthetic|real (tier-0 origin); split is train|val|test. The real-world
held-out set (Phase 3) is marked split=test and source=real, and is never used for
training (report §4.2.4). This file is the authoritative train/val/test partition
that every other stage reads.

classes_present is serialised as a semicolon-joined string of sorted, unique fine
class names present in that image, e.g. "gate_valve;globe_valve" (empty string if
the image has no annotations) — simple and readable in a CSV cell without needing a
nested-list encoding.
"""

from __future__ import annotations

import json
import random

import hydra
import pandas as pd
from omegaconf import DictConfig

from pipeline.base import BasePipeline
from utils.cli import CONFIG_DIR


class BuildManifestPipeline(BasePipeline):
    """Materialise canonical/manifest.csv from COCO images/annotations + classes.yaml."""

    def validate(self) -> bool:
        return self.paths.coco_json.exists() and self.paths.classes_yaml.exists()

    def run(self) -> None:
        if not self.validate():
            raise FileNotFoundError(
                f"{self.paths.coco_json} and/or {self.paths.classes_yaml} missing; "
                "run pid-prep-classes first."
            )

        with self.paths.coco_json.open("r") as f:
            coco = json.load(f)

        images = coco.get("images", [])
        annotations = coco.get("annotations", [])
        if not images:
            raise ValueError(f"{self.paths.coco_json}: no `images` found")

        # COCO annotations key classes by category_id, so we still need id->name from
        # the COCO categories block itself; self.classmap (built by build_classes from
        # this same COCO file) is the authoritative fine-name vocabulary, so validate
        # against it rather than trusting COCO category names blindly.
        category_id_to_fine = {c["id"]: c["name"] for c in coco.get("categories", [])}
        unknown_fine = set(category_id_to_fine.values()) - set(self.classmap.fine_classes)
        if unknown_fine:
            raise ValueError(
                f"{self.paths.coco_json}: category names {sorted(unknown_fine)} are not "
                f"present in {self.paths.classes_yaml}; run pid-prep-classes again?"
            )

        anns_by_image: dict[int, list[dict]] = {}
        for ann in annotations:
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

        rows: list[dict] = []
        for img in images:
            image_id = self._stem(img["file_name"])
            source = self._source_of(img["file_name"])
            img_anns = anns_by_image.get(img["id"], [])

            fine_names = sorted(
                {
                    category_id_to_fine[ann["category_id"]]
                    for ann in img_anns
                    if ann["category_id"] in category_id_to_fine
                }
            )

            rows.append(
                {
                    "id": image_id,
                    "source": source,
                    "n_symbols": len(img_anns),
                    "classes_present": ";".join(fine_names),
                }
            )

        splits = self._assign_splits(rows)
        for row in rows:
            row["split"] = splits[row["id"]]

        df = pd.DataFrame(rows, columns=["id", "source", "split", "n_symbols", "classes_present"])
        df = df.sort_values("id").reset_index(drop=True)

        self.paths.manifest_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.paths.manifest_csv, index=False)

    @staticmethod
    def _stem(file_name: str) -> str:
        """`"synthetic/0001.jpg"` -> `"0001"` — manifest ids have no extension."""
        name = file_name.split("/")[-1]
        return name.rsplit(".", 1)[0]

    @staticmethod
    def _source_of(file_name: str) -> str:
        """Infer tier-0 origin from the COCO `file_name` prefix (see module docstring:
        canonical/ was assembled from raw/synthetic/ and raw/real/, and file_name is
        assumed to carry that prefix, e.g. "synthetic/0001.jpg"). Raise loudly if this
        assumption doesn't hold rather than silently misclassifying an image's source.
        """
        prefix = file_name.split("/")[0]
        if prefix not in ("synthetic", "real"):
            raise ValueError(
                f"Cannot infer source from file_name={file_name!r}: expected a "
                "'synthetic/...' or 'real/...' prefix (see raw/ tree)."
            )
        return prefix

    def _assign_splits(self, rows: list[dict]) -> dict[str, str]:
        """real -> always test (held-out); synthetic -> train/val per cfg.data.split."""
        splits: dict[str, str] = {}
        synthetic_ids: list[str] = []
        for row in rows:
            if row["source"] == "real":
                splits[row["id"]] = "test"
            else:
                synthetic_ids.append(row["id"])

        ratios = self.cfg.data.split.ratios
        train_ratio = float(ratios["train"])
        seed = int(self.cfg.data.split.seed)

        ordered = sorted(synthetic_ids)
        rng = random.Random(seed)
        rng.shuffle(ordered)

        n_train = round(train_ratio * len(ordered))
        for image_id in ordered[:n_train]:
            splits[image_id] = "train"
        for image_id in ordered[n_train:]:
            splits[image_id] = "val"

        return splits


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    BuildManifestPipeline(cfg).run()


if __name__ == "__main__":
    main()
