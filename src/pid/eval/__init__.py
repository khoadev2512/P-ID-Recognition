"""Evaluation protocol: detection metrics + FGC-specific metrics (report §2.2.3).

Reporting BOTH families together — not aggregate mAP alone — is a stated contribution
(addresses Gap 5). detection.py covers the standard family; fgc_metrics.py covers the
within-group family that isolates the dominant residual error mode.
"""
