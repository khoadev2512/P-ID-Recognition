"""Derive / validate canonical/classes.yaml from the COCO categories.

COCO categories in annotations.coco.json carry both fine names and (by convention)
a supercategory field for the coarse level. This step materialises classes.yaml —
the fine<->coarse map, fgc_groups, and rare_fine list — so every downstream stage
reads one authoritative file (see utils.classmap).

Hydra entrypoint. Config: configs/config.yaml (data: section).
"""

from __future__ import annotations

import json
from collections import Counter

import hydra
import yaml
from omegaconf import DictConfig

from pipeline.base import BasePipeline
from utils.cli import CONFIG_DIR


class BuildClassesPipeline(BasePipeline):
    """Materialise canonical/classes.yaml from COCO `categories`."""

    def validate(self) -> bool:
        coco_json = self.paths.coco_json
        return coco_json.exists() and coco_json.stat().st_size > 0

    def run(self) -> None:
        if not self.validate():
            raise FileNotFoundError(
                f"{self.paths.coco_json} does not exist or is empty; "
                "nothing to derive classes.yaml from."
            )

        with self.paths.coco_json.open("r") as f:
            coco = json.load(f)

        categories = coco.get("categories", [])
        if not categories:
            raise ValueError(f"{self.paths.coco_json}: no `categories` found")
        annotations = coco.get("annotations", [])

        # fine -> coarse, preserving COCO category-id order (not alphabetical) so
        # fine ids stay meaningful / stable across re-derivations of the same COCO file.
        categories_sorted = sorted(categories, key=lambda c: c["id"])
        fine_to_coarse = {c["name"]: c["supercategory"] for c in categories_sorted}

        # coarse classes: sorted unique supercategory values, alphabetical for
        # reproducibility (unlike fine ids, coarse ids don't need to track COCO order).
        coarse_classes = sorted({c["supercategory"] for c in categories})

        # fgc_groups: a coarse class needs the FGC stage iff it's a VISUALLY SIMILAR
        # family (detector alone can't tell its children apart). Two ways to decide:
        #
        #   1. If the COCO categories carry an explicit `fgc` flag (the ISO renderer
        #      writes one, from vocab.csv's hand-marked fgc_group column) — trust it.
        #      With a detailed vocab, "coarse has >1 fine" over-triggers: pump has
        #      centrifugal/gear/diaphragm children that look clearly different and need
        #      no FGC. The explicit flag captures real visual similarity.
        #   2. Otherwise fall back to the >1-fine heuristic (DigitizePID path, where a
        #      coarse with multiple fine children is in fact a look-alike family).
        explicit_fgc = {
            c["supercategory"] for c in categories if c.get("fgc") is True
        }
        if any("fgc" in c for c in categories):
            fgc_groups = [c for c in coarse_classes if c in explicit_fgc]
        else:
            fine_count_per_coarse: Counter[str] = Counter(fine_to_coarse.values())
            fgc_groups = [c for c in coarse_classes if fine_count_per_coarse[c] > 1]

        # rare_fine: flag the bottom decile (<=10th percentile) of fine classes by
        # training-instance frequency across ALL annotations (build_manifest hasn't
        # split train/val/test yet at this stage, so this is a coarse pre-split
        # long-tail signal, refined later if needed). This is a heuristic, not an
        # exact quantile computation — guard the degenerate case of very few classes
        # where a strict percentile cut would flag everything or nothing.
        category_id_to_fine = {c["id"]: c["name"] for c in categories}
        fine_instance_counts: Counter[str] = Counter()
        for ann in annotations:
            fine_name = category_id_to_fine.get(ann["category_id"])
            if fine_name is not None:
                fine_instance_counts[fine_name] += 1

        rare_fine = self._flag_rare(fine_to_coarse.keys(), fine_instance_counts)

        classes = {
            "coarse": coarse_classes,
            "fine": fine_to_coarse,
            "fgc_groups": fgc_groups,
            "rare_fine": rare_fine,
        }

        self.paths.classes_yaml.parent.mkdir(parents=True, exist_ok=True)
        with self.paths.classes_yaml.open("w") as f:
            yaml.safe_dump(classes, f, sort_keys=False, default_flow_style=False)

    @staticmethod
    def _flag_rare(fine_names, fine_instance_counts: Counter[str]) -> list[str]:
        """Bottom-decile (<=10th percentile count) fine classes by instance frequency.

        Classes absent from `fine_instance_counts` (zero training instances) count as
        frequency 0 and are the rarest by definition.
        """
        fine_names = list(fine_names)
        if len(fine_names) <= 1:
            # A single (or zero) fine class can't meaningfully be "rare" relative to
            # itself — heuristic degenerates, so don't flag anything.
            return []

        counts = {name: fine_instance_counts.get(name, 0) for name in fine_names}
        sorted_counts = sorted(counts.values())
        # 10th percentile via nearest-rank on the sorted count list.
        idx = max(0, int(0.10 * (len(sorted_counts) - 1)))
        threshold = sorted_counts[idx]
        rare = [name for name in fine_names if counts[name] <= threshold]

        # Degenerate case: threshold cut flags every class (e.g. all counts equal) or
        # would flag none — a heuristic that produces a no-op signal isn't useful but
        # also isn't an error; only avoid the "flags literally everything" extreme
        # since that carries no long-tail information.
        if len(rare) >= len(fine_names):
            return []
        return sorted(rare)


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    BuildClassesPipeline(cfg).run()


if __name__ == "__main__":
    main()
