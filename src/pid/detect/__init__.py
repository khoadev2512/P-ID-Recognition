"""Stage 1: YOLOv8 detector — localization + coarse (super-category) classification.

Trained on coarse labels by default so within-group discrimination is left to the FGC
stage (report §4.2.2). Inference tiles the input (SAHI) and merges patch detections
with WBF into a diagram-level symbol inventory.
"""
