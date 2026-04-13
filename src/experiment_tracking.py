from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

try:
    import mlflow
except ModuleNotFoundError:
    mlflow = None


def start_run(experiment_name: str, run_name: str | None = None):
    if mlflow is None:
        return nullcontext()
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run(run_name=run_name)


def start_run_if_enabled(enabled: bool, experiment_name: str, run_name: str | None = None):
    if not enabled or mlflow is None:
        return nullcontext()
    return start_run(experiment_name=experiment_name, run_name=run_name)


def log_training_metadata(params: dict[str, Any], metrics: dict[str, float], artifact_paths: list[Path]) -> None:
    if mlflow is None:
        return
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    for path in artifact_paths:
        if path.exists():
            mlflow.log_artifact(str(path))


def log_metrics_only(params: dict[str, Any], metrics: dict[str, float]) -> None:
    if mlflow is None:
        return
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
