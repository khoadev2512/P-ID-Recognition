"""Dispatch tests for src/run_pipeline.py — the single CLI the Colab notebooks call.

The notebooks only reach a stage through this entrypoint, so a stage missing here is a
stage Colab cannot run (the one-shot `prototypes` step was exactly that gap). The
pipelines themselves are replaced by a recorder: what is pinned is the routing and that
Hydra overrides reach the stage's config.
"""

from __future__ import annotations

import sys

import pytest

pytest.importorskip("torch")
pytest.importorskip("ultralytics")

import run_pipeline  # noqa: E402


class _Recorder:
    calls: list = []

    def __init__(self, cfg):
        self.cfg = cfg

    def run(self):
        _Recorder.calls.append(self.cfg)


@pytest.fixture
def recorder(monkeypatch):
    _Recorder.calls = []
    monkeypatch.setattr(run_pipeline, "IsoPrototypePipeline", _Recorder)
    return _Recorder


def _main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", *argv])
    run_pipeline.main()


def test_prototypes_subcommand_runs_the_iso_prototype_stage(monkeypatch, tmp_path, recorder):
    _main(
        monkeypatch,
        "prototypes",
        f"run_dir={tmp_path}",
        "oneshot.encoder_ckpt=/x/__global_lo__.pt",
        "oneshot.prototypes=/x/iso_prototypes_lo.pt",
    )
    (cfg,) = recorder.calls
    assert cfg.oneshot.encoder_ckpt == "/x/__global_lo__.pt"
    assert cfg.oneshot.prototypes == "/x/iso_prototypes_lo.pt"


def test_list_overrides_reach_the_config(monkeypatch, tmp_path, recorder):
    """The notebook passes the held-out list as one quoted Hydra list override."""
    _main(
        monkeypatch,
        "prototypes",
        f"run_dir={tmp_path}",
        "fgc.exclude_classes=[Symbol_3,Symbol_10]",
    )
    (cfg,) = recorder.calls
    assert list(cfg.fgc.exclude_classes) == ["Symbol_3", "Symbol_10"]
