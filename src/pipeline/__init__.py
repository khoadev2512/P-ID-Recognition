"""Pipeline stages for the P&ID detection-then-FGC system (report ch.4.2).

    base.py                        BasePipeline — shared Hydra-cfg / paths / classmap
    build_classes / build_manifest /
    coco_to_yolo / tiling / extract_crops     data prep (canonical COCO -> derived artefacts)
    detect_train / detect_infer     Stage 1 — YOLOv8 + SAHI/WBF
    fgc_train / fgc_infer           Stage 2 — ResNet-34 fine-grained classification
    evaluation                      detection + FGC metrics driver

Mirrors PID_Symbol_Detection's `src/pipeline/` (one stage class per file, sharing a
BasePipeline); helper logic that isn't itself a pipeline stage lives in `utils/`.
"""
