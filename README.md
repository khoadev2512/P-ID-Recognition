# P&ID Symbol Detection and Fine-Grained Classification

Base source for a **detection-then-fine-grained-classification** pipeline for P&ID
symbol recognition (thesis: *Symbol Detection and Fine-Grained Classification in P&ID*).

Stage logic (data prep, detector train/infer, FGC train/infer, evaluation) is
implemented end-to-end, organized as one `BasePipeline` subclass per stage
(`src/pipeline/base.py`) behind Hydra entrypoints — layout mirrors
`PID_Symbol_Detection`'s `src/pipeline` + `src/utils` split. It has not yet been run
against real data: `data/canonical/` is still empty placeholders, so training/inference
have only been exercised via unit tests and synthetic fixtures, not a live run.

## Pipeline

| Stage | Module | Role |
|-------|--------|------|
| 1. Detect | `pipeline.detect_train` / `pipeline.detect_infer` | YOLOv8 localization + **coarse** (super-category) classification, over SAHI tiles, merged with WBF |
| 2. FGC | `pipeline.fgc_train` / `pipeline.fgc_infer` | ResNet-34 **within-group fine** classification on symbol crops (valve / instrument-bubble / fitting families). Optional **ArcFace** head (`fgc.metric_learning=true`) adds an out-of-vocabulary **"Other"** reject gate |
| 2b. One-shot ISO | `pipeline.iso_prototypes` | embeds the **ISO 10628-2 symbol library** with the frozen Stage-2 encoder into a **prototype directory**, so ISO symbols are recognized by cosine distance to one template each — no training on ISO data |
| Eval | `pipeline.evaluation` | detection metrics **+** FGC-specific metrics (within-group confusion, per-family accuracy, top-k, rare-class macro-F1) |

The two stages are deliberately **separable** so the FGC contribution is measurable as
an ablation (report §4.2.1).

### FGC head: linear vs ArcFace (out-of-vocabulary "Other")

`fgc.metric_learning=false` (default) trains a plain linear head with CE/focal loss.
`fgc.metric_learning=true` swaps in an **ArcFace** angular-margin head (§4.2.3): the
backbone learns an L2-normalized embedding with one class-center per fine class, giving
(a) tighter within-group separation on the long-tail families, and (b) an
**out-of-vocabulary reject gate** — at inference a crop whose cosine to its nearest
class center is below `fgc.route.other_min_cosine` is labelled **"Other"**
(`fine_id = -2`) instead of being forced into the nearest known class. The fine-label
decision itself stays softmax-argmax (unchanged output format); the class centers are
used only as the reject signal. `other_min_cosine` must be **calibrated on a val split**
(the eval report's `fgc.groups[*].other_gate.false_reject_rate` is the signal for this);
the default `0.35` is a placeholder, not a tuned value.

```bash
uv run pid-fgc-train  fgc.metric_learning=true
uv run pid-fgc-infer  fgc.metric_learning=true fgc.ckpt_dir=runs/<train-run>/fgc_checkpoints
```

### Embedding scope: one global encoder vs. per-group heads

`fgc.global_encoder=false` (default) trains **one head per `fgc_group`** — the
closed-set, within-family configuration. `fgc.global_encoder=true` trains **a single
head over the entire fine vocabulary** (checkpoint `__global__.pt`), which is what the
one-shot ISO path requires: per-group heads learn **separate** embedding spaces and
cosines from different spaces are not comparable, so a single ISO prototype directory
and a vocabulary-wide "Other" gate both need every class in **one** space (as in
OSSR-PID). It also covers the ISO families (pump, vessel, compressor, motor, piping,
fitting) that no per-group head corresponds to.

The flag reaches three stages, and all three must agree:

| Stage | `global_encoder=true` behaviour |
|-------|--------------------------------|
| `pid-prep-crops` | crops **every** symbol, not only `fgc_group` ones — a class with no crops would be missing from the embedding space it is supposed to be separable in |
| `pid-fgc-train` | trains one head sized to the whole fine vocabulary |
| `pid-fgc-infer` | **refuses** — it loads one checkpoint per group and routes by coarse class; the one-shot prototype path is its counterpart |

```bash
uv run pid-prep-crops  fgc.global_encoder=true
uv run pid-fgc-train   fgc.global_encoder=true fgc.metric_learning=true
```

### One-shot ISO 10628-2 recognition (prototype directory)

The ArcFace embedding is also what makes the **ISO vocabulary** reachable without any
ISO training data. `pid-iso-prototypes` pushes each ISO symbol template (one PNG per
`reg_number`, from `data/raw/iso_symbols_clean/`) through the **frozen** Stage-2 encoder
and stores the resulting unit vectors as `{fine_name -> embedding}` — a *prototype
directory*. A detector crop is then labelled by whichever prototype its own embedding is
closest to in cosine, and rejected as **"Other"** when the best cosine falls below
`oneshot.other_min_cosine`.

Two consequences worth stating plainly:

- **Adding a symbol is adding a row.** A new ISO symbol = one line in
  `oneshot.vocab_csv` + one PNG + re-running this stage. No retraining, which is the
  concrete difference from OSSR-PID (arXiv 2109.03849), whose conclusion still requires
  training when a class is added.
- **Query views vote.** A crop is embedded at several *box-fit* scales × 4 orientations
  (`oneshot.query_views`) and the views vote on the label (OSSR-PID technique #3), so a
  symbol drawn rotated still matches its upright template. Only views whose own best
  cosine clears the gate get a vote — otherwise the three uninformative orientations of
  an asymmetric symbol would out-vote the one correct orientation.

`oneshot.other_min_cosine` is a **separate** number from `fgc.route.other_min_cosine`
(trained class centers vs. single-template prototypes are different cosine
distributions) and is likewise a placeholder until calibrated. The builder logs the
mean/max **inter-prototype** cosine, which the threshold has to sit above to mean
anything.

```bash
# Whole ISO vocabulary (39 symbols), one global encoder:
uv run pid-iso-prototypes \
  oneshot.encoder_ckpt=runs/<train-run>/fgc_checkpoints/__global__.pt

# Per-group: pid-fgc-train writes one checkpoint per fgc_group, so pair each encoder
# with the ISO coarse families it was actually trained to separate.
uv run pid-iso-prototypes \
  oneshot.encoder_ckpt=runs/<train-run>/fgc_checkpoints/valve.pt \
  'oneshot.coarse_filter=[valve,check_valve]' \
  oneshot.prototypes=data/derived/iso_prototypes_valve.pt
```

## Data layout (three tiers)

```
data/raw/         tier 0 — immutable source (synthetic/, real/)
data/canonical/   tier 1 — source of truth: images/, annotations.coco.json,
                           classes.yaml (fine<->coarse map), manifest.csv
data/derived/     tier 2 — regenerable: yolo_coarse/, yolo_fine/, tiles/, crops_fgc/
```

`classes.yaml` is the most important file — see `configs/classes.example.yaml`.

## Setup (uv)

```bash
uv sync                      # create .venv and install deps
# GPU torch: install the CUDA wheel per https://pytorch.org, then `uv sync`
uv run pytest                # scaffold smoke tests
```

## Running (Hydra, config-driven)

One flat config: `configs/config.yaml` (`data`, `detector`, `fgc`, `eval` sections, no
config-group composition). Override any field on the CLI, e.g. `detector.imgsz=1280
fgc.loss=ce`.

Two equivalent ways to run a stage — pick whichever you prefer, both read/override the
same `configs/config.yaml`:

**A. `run_pipeline.py`** — one entrypoint, mirrors `PID_Symbol_Detection`'s
`src/run_pipeline.py`:

```bash
uv run python src/run_pipeline.py prep --step all
uv run python src/run_pipeline.py detect --train  detector.label_level=coarse
uv run python src/run_pipeline.py detect --infer  detector.ckpt=runs/<train-run>/weights/best.pt
uv run python src/run_pipeline.py fgc    --train
uv run python src/run_pipeline.py fgc    --infer  fgc.ckpt_dir=runs/<train-run>/fgc_checkpoints
uv run python src/run_pipeline.py evaluate eval.mode=det_only
```

**B. Individual `pid-*` console scripts** — one per stage, installed by `uv sync`:

```bash
# Data prep
uv run pid-prep-classes
uv run pid-prep-manifest
uv run pid-prep-yolo     data.label_level=coarse
uv run pid-prep-tiles
uv run pid-prep-crops

# Detector (Stage 1)
uv run pid-detect-train  detector.label_level=coarse
uv run pid-detect-infer  detector.ckpt=runs/<train-run>/weights/best.pt

# FGC (Stage 2)
uv run pid-fgc-train
uv run pid-fgc-infer     fgc.ckpt_dir=runs/<train-run>/fgc_checkpoints

# One-shot ISO prototype directory (Stage 2b; needs an ArcFace checkpoint)
uv run pid-iso-prototypes oneshot.encoder_ckpt=runs/<train-run>/fgc_checkpoints/__global__.pt

# Evaluation (both metric families; ablation via eval.mode)
uv run pid-eval          eval.mode=det_only
uv run pid-eval          eval.mode=det_plus_fgc
uv run pid-eval          eval.split=test         # real-world held-out (see Phases below)
```

## Phases (dataset strategy, report §4.2.4)

These are stages of the data-collection/training methodology, not separate config
files — each is just the default config with a couple of fields set by hand as shown:

- **phase1** — public synthetic baseline (Weeks 1-3). Defaults already match it
  (`data.label_level=coarse`, `detector.label_level=coarse`).
- **phase2** — programmatic synthesis with coarse+fine labels + degradation aug.
  Defaults already match it (`fgc.balanced_sampler=true`); swap in the phase-2 dataset
  under `data/raw/synthetic/` and re-run the data-prep steps. The degradation aug is off
  by default; it degrades train images online like the V7 test pages (no rotation, limits
  from the measured stroke width; `src/utils/scan_degrade.py`):
  `pid-detect-train +detector.aug.degrade.p=0.5` and `pid-fgc-train +fgc.aug.degrade.p=0.5`
  (ranges such as `'+fgc.aug.degrade.noise_sigma=[0,6]'` override the V7 defaults). On
  resume, pass the same override again.
- **phase3** — real-world held-out eval (never trained on). Run
  `pid-eval eval.split=test eval.mode=det_plus_fgc` and compare against the same
  command run on the synthetic test split — real-source images are always assigned
  `split=test` by `pid-prep-manifest`, so this never mixes into training regardless.

## Layout

```
src/
  run_pipeline.py   single CLI entrypoint (argparse subcommands -> pipeline classes)
  pipeline/   one BasePipeline subclass per Hydra entrypoint:
                base.py                                   shared BasePipeline
                build_classes / build_manifest /
                coco_to_yolo / tiling / extract_crops      data prep
                detect_train / detect_infer                Stage 1 (YOLOv8 + SAHI/WBF)
                fgc_train / fgc_infer                      Stage 2 (ResNet-34 FGC)
                iso_prototypes                             Stage 2b (ISO one-shot prototypes)
                evaluation                                 detection + FGC metrics driver
  utils/      shared helpers, no Hydra entrypoints:
                types, classmap, paths, cli, bbox_utils,
                dataset, model, oneshot,
                detection_metrics, fgc_metrics
configs/      config.yaml (single flat config) + classes.example.yaml
models/       base_yolo_models/yolov8n.pt (vendored COCO-pretrained init);
              trained checkpoints land under runs/<timestamp>/ instead (see below)
scripts/      (reserved for standalone utilities)
tests/        unit + smoke tests
```

Just two sibling packages, `pipeline/` and `utils/`, no top-level package wrapper —
mirrors `PID_Symbol_Detection`'s `src/pipeline` + `src/utils` layout (rather than a
`src/<package_name>/` src-layout). One practical consequence: `pipeline`/`utils` are
generic top-level import names, so avoid installing this project's env alongside
another package that also publishes a top-level `pipeline`/`utils` module.
