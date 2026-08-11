"""P&ID symbol detection and fine-grained classification.

Two-stage pipeline (see report ch.4.2):

    Stage 1  detect/  — YOLOv8 localization + coarse (super-category) classification,
                        run over SAHI tiles and merged with WBF.
    Stage 2  fgc/     — ResNet-34 within-group fine classification on cropped symbols.

Shared code lives in common/; data preparation (canonical COCO -> derived artefacts)
lives in data/; the evaluation protocol (detection + FGC metrics) lives in eval/.
"""

__version__ = "0.1.0"
