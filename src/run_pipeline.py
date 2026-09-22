"""Single CLI entrypoint over every pipeline stage (mirrors PID_Symbol_Detection's
src/run_pipeline.py). Config is still Hydra/OmegaConf underneath -- via the Compose
API rather than the per-stage @hydra.main decorator used by the individual pid-*
console scripts (see pyproject.toml) -- so every configs/config.yaml field remains
overridable on the command line exactly the same way.

Usage:
    python src/run_pipeline.py prep [--step classes|manifest|yolo|tiles|crops|all]
    python src/run_pipeline.py detect --train | --infer
    python src/run_pipeline.py fgc --train | --infer
    python src/run_pipeline.py evaluate

Hydra overrides work the same as the individual pid-* commands, just appended:
    python src/run_pipeline.py detect --train detector.imgsz=1280 detector.epochs=50
    python src/run_pipeline.py evaluate eval.mode=det_only
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from pipeline.build_classes import BuildClassesPipeline
from pipeline.build_manifest import BuildManifestPipeline
from pipeline.coco_to_yolo import CocoToYoloPipeline
from pipeline.detect_infer import DetectInferPipeline
from pipeline.detect_train import DetectTrainPipeline
from pipeline.evaluation import EvaluationPipeline
from pipeline.extract_crops import ExtractCropsPipeline
from pipeline.fgc_infer import FGCInferPipeline
from pipeline.fgc_train import FGCTrainPipeline
from pipeline.tiling import TilingPipeline
from utils.cli import CONFIG_DIR

logger = logging.getLogger(__name__)

# Order matters when --step all: classes -> manifest -> yolo -> tiles -> crops,
# matching the data/ layout's dependency chain (each step reads the previous one's
# output -- see each pipeline class's validate()).
DATA_PREP_STEPS = {
    "classes": BuildClassesPipeline,
    "manifest": BuildManifestPipeline,
    "yolo": CocoToYoloPipeline,
    "tiles": TilingPipeline,
    "crops": ExtractCropsPipeline,
}


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="P&ID Symbol Detection pipeline")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prep", help="Data prep steps (classes/manifest/yolo/tiles/crops)")
    prep.add_argument("--step", choices=[*DATA_PREP_STEPS, "all"], default="all")

    detect = sub.add_parser("detect", help="Stage 1 detector (YOLOv8 + SAHI/WBF)")
    detect_mode = detect.add_mutually_exclusive_group(required=True)
    detect_mode.add_argument("--train", action="store_true")
    detect_mode.add_argument("--infer", action="store_true")

    fgc = sub.add_parser("fgc", help="Stage 2 fine-grained classifier (ResNet-34)")
    fgc_mode = fgc.add_mutually_exclusive_group(required=True)
    fgc_mode.add_argument("--train", action="store_true")
    fgc_mode.add_argument("--infer", action="store_true")

    sub.add_parser("evaluate", help="Run evaluation (detection + FGC metrics)")

    # Anything left over that doesn't match a defined flag is treated as a Hydra
    # override (key=value) -- same convention the individual pid-* scripts use.
    args, overrides = parser.parse_known_args()
    return args, overrides


def _compose_cfg(overrides: list[str]) -> DictConfig:
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config", overrides=overrides)

    # @hydra.main normally resolves hydra.run.dir (configs/config.yaml) to a fresh
    # timestamped directory per invocation. The Compose API used here doesn't set
    # HydraConfig up, so BasePipeline.output_dir can't read it and falls back to
    # cfg.run_dir directly (see pipeline/base.py) -- replicate the timestamping by
    # hand so each run still gets its own checkpoint/metrics directory.
    cfg.run_dir = str(Path(str(cfg.run_dir)) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    return cfg


def main() -> None:
    args, overrides = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    cfg = _compose_cfg(overrides)

    if args.command == "prep":
        steps = DATA_PREP_STEPS.values() if args.step == "all" else [DATA_PREP_STEPS[args.step]]
        for pipeline_cls in steps:
            logger.info("Running %s", pipeline_cls.__name__)
            pipeline_cls(cfg).run()

    elif args.command == "detect":
        (DetectTrainPipeline(cfg) if args.train else DetectInferPipeline(cfg)).run()

    elif args.command == "fgc":
        (FGCTrainPipeline(cfg) if args.train else FGCInferPipeline(cfg)).run()

    elif args.command == "evaluate":
        EvaluationPipeline(cfg).run()

    else:  # pragma: no cover -- argparse enforces `required=True` on the subparsers
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
