"""Experiment tracking. MLflow by default (local file store — no account needed);
a null tracker keeps everything working (and testable) with tracking disabled.

Every metric is ALSO written to metrics.csv inside the run directory, so results
survive even with tracking turned off entirely.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from vision_pipeline.config import TrackingConfig

log = logging.getLogger(__name__)


class NullTracker:
    def log_params(self, params: dict[str, Any]) -> None: ...

    def log_metrics(self, metrics: dict[str, float], step: int) -> None: ...

    def log_artifact(self, path: str | Path) -> None: ...

    def finish(self) -> None: ...


class MlflowTracker:
    def __init__(self, cfg: TrackingConfig, run_name: str):
        import mlflow

        mlflow.set_tracking_uri(cfg.tracking_uri)
        mlflow.set_experiment(cfg.experiment_name)
        self._mlflow = mlflow
        self._run = mlflow.start_run(run_name=run_name)

    def log_params(self, params: dict[str, Any]) -> None:
        try:
            self._mlflow.log_params({k: str(v) for k, v in params.items()})
        except Exception as exc:  # never let tracking break training
            log.warning("MLflow log_params failed: %s", exc)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        try:
            self._mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)
        except Exception as exc:
            log.warning("MLflow log_metrics failed: %s", exc)

    def log_artifact(self, path: str | Path) -> None:
        try:
            self._mlflow.log_artifact(str(path))
        except Exception as exc:
            log.warning("MLflow log_artifact failed: %s", exc)

    def finish(self) -> None:
        try:
            self._mlflow.end_run()
        except Exception:  # pragma: no cover
            pass


def make_tracker(cfg: TrackingConfig, run_name: str) -> NullTracker | MlflowTracker:
    if cfg.backend != "mlflow":
        return NullTracker()
    try:
        return MlflowTracker(cfg, run_name)
    except Exception as exc:
        log.warning("MLflow unavailable (%s); falling back to NullTracker", exc)
        return NullTracker()
