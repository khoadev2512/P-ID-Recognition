# P&ID Symbol Detection and Fine-Grained Classification

Base source for a **detection-then-fine-grained-classification** pipeline for P&ID
symbol recognition (thesis: *Symbol Detection and Fine-Grained Classification in P&ID*).

This repository is currently a **scaffold**: structure, configs, and typed interfaces
are in place; stage logic is marked `TODO` / `NotImplementedError`.

## Pipeline

| Stage | Module | Role |
|-------|--------|------|
| 1. Detect | `pid.detect` | YOLOv8 localization + **coarse** (super-category) classification, over SAHI tiles, merged with WBF |
| 2. FGC | `pid.fgc` | ResNet-34 **within-group fine** classification on symbol crops (valve / instrument-bubble / fitting families) |
| Eval | `pid.eval` | detection metrics **+** FGC-specific metrics (within-group confusion, per-family accuracy, top-k, rare-class macro-F1) |

The two stages are deliberately **separable** so the FGC contribution is measurable as
an ablation (report §4.2.1).

## Data layout (three tiers)

```
data/raw/         tier 0 — immutable source (synthetic/, real/)
data/canonical/   tier 1 — source of truth: images/, annotations.coco.json,
                           classes.yaml (fine<->coarse map), manifest.csv
data/derived/     tier 2 — regenerable: yolo_coarse/, yolo_fine/, tiles/, crops_fgc/
```

`classes.yaml` is the most important file — see `configs/data/classes.example.yaml`.

## Setup (uv)

```bash
uv sync                      # create .venv and install deps
# GPU torch: install the CUDA wheel per https://pytorch.org, then `uv sync`
uv run pytest                # scaffold smoke tests
```

## Running (Hydra, config-driven)

Configs compose from groups in `configs/` (`data`, `detector`, `fgc`, `eval`, `phase`).

```bash
# Phase 1 — data prep from public synthetic data
uv run pid-prep-classes  phase=phase1
uv run pid-prep-manifest phase=phase1
uv run pid-prep-yolo     phase=phase1 data.label_level=coarse
uv run pid-prep-tiles    phase=phase1
uv run pid-prep-crops    phase=phase1

# Detector (Stage 1)
uv run pid-detect-train  phase=phase1 detector.label_level=coarse
uv run pid-detect-infer  phase=phase1

# FGC (Stage 2)
uv run pid-fgc-train     phase=phase2
uv run pid-fgc-infer     phase=phase2

# Evaluation (both metric families; ablation via eval.mode)
uv run pid-eval          eval.mode=det_only
uv run pid-eval          eval.mode=det_plus_fgc
uv run pid-eval          phase=phase3           # synthetic-to-real gap
```

Override any field on the CLI, e.g. `detector.imgsz=1280 fgc.loss=ce`.

## Phases (dataset strategy, report §4.2.4)

- **phase1** — public synthetic baseline (Weeks 1-3)
- **phase2** — programmatic synthesis with coarse+fine labels + degradation aug
- **phase3** — real-world held-out eval (never trained on)

## Layout

```
src/pid/
  common/   types, classes.yaml loader, path resolution
  data/     COCO -> classes.yaml / YOLO / tiles / crops / manifest
  detect/   YOLOv8 train + SAHI/WBF inference
  fgc/      ResNet-34 dataset / model / train / infer
  eval/     detection metrics, FGC metrics, driver
configs/    Hydra config groups + phase presets
scripts/    (reserved for standalone utilities)
tests/      scaffold smoke tests
```
