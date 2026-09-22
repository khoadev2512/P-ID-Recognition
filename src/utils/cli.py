"""Shared Hydra entrypoint helper.

Locating configs/ by a relative path from an *installed* package is fragile
(it resolves against site-packages, not the project). We resolve it via the
PID_CONFIG_DIR env var if set, else by walking up from this file to the repo
root that contains configs/. Entrypoints pass CONFIG_DIR to @hydra.main.
"""

from __future__ import annotations

import os
from pathlib import Path


def _find_config_dir() -> str:
    env = os.environ.get("PID_CONFIG_DIR")
    if env:
        return str(Path(env).resolve())
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "configs"
        if (candidate / "config.yaml").exists():
            return str(candidate)
    # Fallback: assume repo root is two levels above src/utils/
    return str(here.parents[2] / "configs")


CONFIG_DIR = _find_config_dir()
