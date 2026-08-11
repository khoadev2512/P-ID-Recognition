"""Stage 2: fine-grained classification (ResNet-34) on cropped symbol regions.

Applied selectively to detections whose coarse class is a visually similar group
(valve / instrument_bubble / fitting). Trained on crops (derived/crops_fgc) with
aggressive per-class balancing for the long tail (report §4.2.3, §2.2.2).
"""
