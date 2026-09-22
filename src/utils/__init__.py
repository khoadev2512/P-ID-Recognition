"""Shared helpers used across pipeline stages — no Hydra entrypoints live here.

    types.py              BBox / Detection / SymbolInstance dataclasses
    classmap.py           classes.yaml (fine<->coarse map) loader + accessors
    paths.py              data/ layout (raw -> canonical -> derived) resolution
    cli.py                CONFIG_DIR resolution for @hydra.main entrypoints
    bbox_utils.py         pixel/normalized box conversions + clipping
    dataset.py, model.py  FGC crop dataset / balanced sampler / ResNet-34 model / loss
    detection_metrics.py  IoU, AP, mAP (report §2.2.3)
    fgc_metrics.py        within-group confusion/accuracy, top-k, rare macro-F1

Mirrors PID_Symbol_Detection's `src/utils/`.
"""
