"""Export the ISO re-renderings of PID2Graph OPEN100 as a data root the pipeline reads (V8).

Input: `<root>/variants/` (V6: clean `v<NNN>.png` + `.json`) and `<root>/degraded/` (V7: `.jpg` +
`.json`), made by scripts/make_variants.py and scripts/make_degraded.py. Output: the
`canonical/` + `graphml/` + `splits.json` layout described in `utils.iso_export`, and
`canonical/classes.yaml` / `manifest.csv` made by the pipeline's own `BuildClassesPipeline` /
`BuildManifestPipeline`. Nothing is written outside `<root>` and nothing in it is read from
`data/canonical/`.

    PID_DATA_ROOT=data/iso_open100 pid-render-iso render_iso.export.images=degraded

Config: `render_iso.export` of configs/render_iso.yaml. Overrides are Hydra's (`a.b=c`), composed
with the compose API so that no `outputs/` or `runs/` directory is created.
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from PIL import Image

from pipeline.base import BasePipeline
from pipeline.build_classes import BuildClassesPipeline
from pipeline.build_manifest import SPLIT_MAP_FILENAME, BuildManifestPipeline
from utils import iso_export as ex
from utils.cli import CONFIG_DIR
from utils.iso_templates import load_templates
from utils.iso_variants import plan_variant, render_variant
from utils.pid2graph import load_graphml, pipe_runs

IMAGES = ("clean", "degraded", "both")
BBOX_TOL = 0.02  # px: a redrawn clean variant must give the boxes stored in its json


def _export_sheet(job: tuple[int, dict, dict, str]) -> tuple[list[dict], dict[str, int]]:
    """Worker: every variant of one sheet. Returns (image records, counters)."""
    sheet_id, render_cfg, opt, root_s = job
    root = Path(root_s)
    src = Path(opt["src_dir"])
    templates = load_templates(
        render_cfg["template_dir"], render_cfg["taxonomy_csv"], render_cfg["scope"]
    )
    sheet = load_graphml(src / f"{sheet_id}.graphml", Image.open(src / f"{sheet_id}.png").size)
    runs = pipe_runs(sheet)
    src_nodes, src_edges = ex.read_source_graphml(src / f"{sheet_id}.graphml")
    vocab = {v["fine"] for v in ex.load_vocab(opt["vocab_csv"])}
    known = vocab | set(opt["other_labels"])
    drop = set(opt["drop_labels"])
    counters: Counter[str] = Counter()
    records = []
    for jf in sorted((root / "variants" / f"sheet{sheet_id}").glob("v*.json")):
        meta = json.loads(jf.read_text())
        index = int(meta["index"])
        for degraded in (False, True):
            if (
                degraded
                and opt["images"] == "clean"
                or not degraded
                and opt["images"] == "degraded"
            ):
                continue
            if degraded:
                dj = root / "degraded" / f"sheet{sheet_id}" / jf.name
                if not dj.exists():
                    counters["degraded_missing"] += 1
                    continue
                doc = json.loads(dj.read_text())
                anns = doc["annotations"]
                fallback = {nid: n["box"] for nid, n in doc["nodes"].items()}
                image = dj.with_suffix(".jpg")
                graph = {"image": "", "affine": json.dumps(doc["affine"])}
            else:
                draft = plan_variant(sheet, runs, templates, render_cfg, sheet_id, index)
                rendered = render_variant(draft, sheet, runs, render_cfg)
                anns = meta["annotations"]
                by_id = {a["id"]: a for a in rendered.page.annotations}
                for a in anns:  # the json must be what this config draws
                    b = by_id.get(a["id"])
                    if (
                        b is None
                        or max(abs(p - q) for p, q in zip(a["bbox"], b["bbox"], strict=True))
                        > BBOX_TOL
                    ):
                        raise RuntimeError(f"{jf}: not what the current config draws ({a['id']})")
                fallback = {
                    nid: (n.xmin, n.ymin, n.xmax, n.ymax) for nid, n in rendered.sheet.nodes.items()
                }
                image = jf.with_suffix(".png")
                graph = {"image": "", "affine": "identity"}
            stem = ex.stem_of(sheet_id, index, degraded)
            graph["image"] = f"{stem}{image.suffix}"
            cats = {a["id"]: ex.coco_category(a, known, drop) for a in anns}
            geometry = ex.node_geometry(anns, fallback)
            nodes = ex.graph_nodes(src_nodes, geometry, anns, cats)
            ex.write_graphml(root / "graphml" / f"{stem}.graphml", nodes, src_edges, graph)
            counters[
                ex.link_file(
                    image,
                    root / "canonical" / "images" / image.name.replace(image.stem, stem),
                    opt["link"],
                )
            ] += 1
            records.append(
                {
                    "stem": stem, "ext": image.suffix, "sheet": sheet_id, "index": index,
                    "degraded": degraded, "width": meta["width"], "height": meta["height"],
                    "anns": [
                        {"category": cats[a["id"]], "box": a["bbox"], "id": a["id"],
                         "node": a.get("node", a["id"]), "added": bool(a.get("added"))}
                        for a in anns if cats[a["id"]] is not None
                    ],
                    "dropped": sum(cats[a["id"]] is None for a in anns),
                }
            )  # fmt: skip
    return records, dict(counters)


class RenderIsoPipeline(BasePipeline):
    """Write the `canonical/`, `graphml/` and `splits.json` of the ISO data root."""

    @property
    def opt(self) -> dict[str, Any]:
        return dict(OmegaConf.to_container(self.cfg.render_iso.export, resolve=True))  # type: ignore[arg-type]

    def validate(self) -> bool:
        root = self.paths.root.resolve()
        if root == Path("data").resolve() or (root / "raw").exists():
            raise ValueError(f"{root} looks like the Stage 1/2 data root; refusing to write there")
        if self.opt["images"] not in IMAGES:
            raise ValueError(f"export.images must be one of {IMAGES}, got {self.opt['images']!r}")
        return (root / "variants").is_dir()

    def run(self) -> None:
        if not self.validate():
            raise FileNotFoundError(f"{self.paths.root / 'variants'} missing: run make_variants.py")
        root, opt = self.paths.root, self.opt
        if opt["images"] != "clean" and not (root / "degraded").is_dir():
            raise FileNotFoundError(f"{root / 'degraded'} missing: run make_degraded.py")
        render_cfg = dict(OmegaConf.to_container(self.cfg.render_iso, resolve=True))  # type: ignore[arg-type]
        sheets = sorted(
            int(p.name.removeprefix("sheet"))
            for p in (root / "variants").glob("sheet*")
            if p.is_dir()
        )
        split = ex.assign_splits(
            sheets,
            opt["split"]["mode"],
            float(opt["split"]["test_fraction"]),
            int(opt["split"]["seed"]),
        )
        for stale in (root / "graphml", self.paths.canonical / "images"):
            shutil.rmtree(stale, ignore_errors=True)
        for stale in (self.paths.coco_json, self.paths.classes_yaml, self.paths.manifest_csv):
            stale.unlink(missing_ok=True)

        jobs = [(s, render_cfg, opt, str(root)) for s in sheets]
        records: list[dict] = []
        counters: Counter[str] = Counter()
        with ProcessPoolExecutor(int(opt["workers"])) as pool:
            for recs, cnt in pool.map(_export_sheet, jobs):
                records += recs
                counters.update(cnt)
        records.sort(key=lambda r: (r["sheet"], r["index"], r["degraded"]))

        vocab = ex.load_vocab(opt["vocab_csv"])
        cats = ex.build_categories(vocab, opt["other_labels"], opt["other_coarse"])
        cat_id = {c["name"]: c["id"] for c in cats}
        images, annotations = [], []
        for i, r in enumerate(records, start=1):
            images.append(
                {"id": i, "file_name": f"synthetic/{r['stem']}{r['ext']}", "width": r["width"],
                 "height": r["height"], "sheet": r["sheet"], "variant": r["index"],
                 "degraded": r["degraded"]}
            )  # fmt: skip
            for a in r["anns"]:
                box = ex.coco_box(a["box"])
                ann = {
                    "id": len(annotations) + 1, "image_id": i, "category_id": cat_id[a["category"]],
                    "bbox": box, "area": round(box[2] * box[3], 2), "iscrowd": 0,
                    "node": a["node"], "ann_id": a["id"],
                }  # fmt: skip
                if a["added"]:
                    ann["added"] = True
                annotations.append(ann)
        ex.dump_json(
            self.paths.coco_json, {"images": images, "annotations": annotations, "categories": cats}
        )

        per_image = {r["stem"]: split[r["sheet"]] for r in records}
        ex.dump_json(root / "splits.json", {
            "mode": opt["split"]["mode"], "seed": opt["split"]["seed"],
            "sheets": {str(s): v for s, v in split.items()}, "images": per_image,
        })  # fmt: skip
        ex.dump_json(self.paths.canonical / SPLIT_MAP_FILENAME, per_image)

        stage_cfg = OmegaConf.create(
            {"data_root": str(root), "run_dir": str(root / "runs"),
             "data": {"split": {"ratios": {"train": 0.8, "val": 0.2}, "seed": 0}}}
        )  # fmt: skip
        BuildClassesPipeline(stage_cfg).run()
        BuildManifestPipeline(stage_cfg).run()
        summary = {
            "images": len(images), "annotations": len(annotations),
            "dropped_not_in_coco": sum(r["dropped"] for r in records), **counters,
        }  # fmt: skip
        ex.dump_json(root / "export_summary.json", summary)
        print(json.dumps(summary))


def main(argv: list[str] | None = None) -> None:
    """`pid-render-iso [hydra overrides]`; the data root is `$PID_DATA_ROOT` (default
    `data/iso_open100`) unless `render_iso.export.data_root=` says otherwise."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="render_iso", overrides=list(sys.argv[1:] if argv is None else argv)
        )
    top = OmegaConf.create({"data_root": cfg.render_iso.export.data_root, "run_dir": "runs"})
    top.render_iso = cfg.render_iso
    RenderIsoPipeline(top).run()


if __name__ == "__main__":
    main()
