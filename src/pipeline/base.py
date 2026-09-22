"""Shared pipeline-stage base class.

Mirrors PID_Symbol_Detection's `BasePipeline` shape (config accessors + `run()`), but
takes an already-composed Hydra `DictConfig` instead of loading its own YAML file:
composition (defaults, phase overlay, CLI overrides) happens once, in the
`@hydra.main`-decorated `main()` of each entrypoint module, not per pipeline instance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from omegaconf import DictConfig

from utils.classmap import ClassMap, load_classmap
from utils.paths import DataPaths


class BasePipeline(ABC):
    """Base class for all pipeline stages."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.paths = DataPaths(root=Path(str(cfg.data_root)))
        self._classmap: ClassMap | None = None

    @property
    def classmap(self) -> ClassMap:
        """Lazily-loaded fine<->coarse class map (canonical/classes.yaml)."""
        if self._classmap is None:
            self._classmap = load_classmap(self.paths.classes_yaml)
        return self._classmap

    @property
    def output_dir(self) -> Path:
        """Resolved Hydra run directory for this invocation (checkpoints/metrics land here).

        Hydra>=1.2 no longer chdir's into the run dir by default, so this reads the
        actual resolved output dir from HydraConfig rather than assuming cwd. Falls
        back to `cfg.run_dir` directly when not running under `@hydra.main` (e.g. tests
        constructing a pipeline with a plain DictConfig).
        """
        try:
            from hydra.core.hydra_config import HydraConfig

            return Path(HydraConfig.get().runtime.output_dir)
        except Exception:
            return Path(str(self.cfg.run_dir))

    @abstractmethod
    def run(self) -> None:
        """Run the complete pipeline stage."""

    def validate(self) -> bool:
        """Validate pipeline stage inputs and configuration.

        Default no-op; stages with real preconditions (required files/dirs, a trained
        checkpoint) override this and raise a clear error instead of failing deep
        inside the run.
        """
        return True
