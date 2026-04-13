from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

try:
    from src.data_loader import load_kaggle_csv
    from src.experiment_tracking import log_metrics_only, start_run_if_enabled
except ModuleNotFoundError:
    from data_loader import load_kaggle_csv
    from experiment_tracking import log_metrics_only, start_run_if_enabled


CLASSIFIER_TARGET = "Rain Tomorrow"
REGRESSOR_TARGET = "Precipitation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate walk-forward baseline models.")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--max-splits", type=int, default=0, help="Optional cap for quick smoke runs")
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--mlflow-experiment", type=str, default="stormcast-baselines")
    parser.add_argument("--mlflow-run-name", type=str, default="walkforward-baselines")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking")
    return parser.parse_args()


def _walkforward_splits(n_rows: int, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return list(tscv.split(np.arange(n_rows)))


def _evaluate_classifier_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    rain_ratio = float(y_test.mean()) if len(y_test) else 0.0
    persistence_preds = np.zeros_like(y_test)

    return {
        "classifier_accuracy": float(accuracy_score(y_test, preds)),
        "classifier_f1": float(f1_score(y_test, preds, zero_division=0)),
        "classifier_persistence_accuracy": float(accuracy_score(y_test, persistence_preds)),
        "classifier_test_rain_ratio": rain_ratio,
    }


def _evaluate_regressor_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    # Naive baseline: always predict the train mean.
    mean_baseline_preds = np.full(shape=len(y_test), fill_value=float(np.mean(y_train)))

    return {
        "regressor_mae": float(mean_absolute_error(y_test, preds)),
        "regressor_mse": float(mean_squared_error(y_test, preds)),
        "regressor_r2": float(r2_score(y_test, preds)),
        "regressor_mean_baseline_mae": float(mean_absolute_error(y_test, mean_baseline_preds)),
    }


def run_walkforward(n_splits: int, max_splits: int = 0) -> dict[str, Any]:
    cls_df = load_kaggle_csv(target_col=CLASSIFIER_TARGET)
    reg_df = load_kaggle_csv(target_col=REGRESSOR_TARGET)
    if cls_df is None or reg_df is None:
        raise RuntimeError(
            "Failed to load dataset for baseline evaluation. "
            "Expected data/rain_prediction_dataset.csv to exist "
            "(or configure the loader accordingly)."
        )

    cls_features = cls_df.drop(columns=[CLASSIFIER_TARGET]).to_numpy()
    cls_target = cls_df[CLASSIFIER_TARGET].to_numpy().astype(int)

    reg_features = reg_df.drop(columns=[REGRESSOR_TARGET]).to_numpy()
    reg_target = reg_df[REGRESSOR_TARGET].to_numpy().astype(float)

    cls_splits = _walkforward_splits(len(cls_df), n_splits=n_splits)
    reg_splits = _walkforward_splits(len(reg_df), n_splits=n_splits)

    if max_splits > 0:
        cls_splits = cls_splits[:max_splits]
        reg_splits = reg_splits[:max_splits]

    fold_metrics: list[dict[str, float]] = []

    for i, ((cls_train_idx, cls_test_idx), (reg_train_idx, reg_test_idx)) in enumerate(zip(cls_splits, reg_splits), start=1):
        cls_fold = _evaluate_classifier_fold(
            cls_features[cls_train_idx],
            cls_target[cls_train_idx],
            cls_features[cls_test_idx],
            cls_target[cls_test_idx],
        )
        reg_fold = _evaluate_regressor_fold(
            reg_features[reg_train_idx],
            reg_target[reg_train_idx],
            reg_features[reg_test_idx],
            reg_target[reg_test_idx],
        )

        metrics = {"fold": float(i), **cls_fold, **reg_fold}
        fold_metrics.append(metrics)

    if not fold_metrics:
        raise RuntimeError("No walk-forward folds were produced")

    avg_metrics: dict[str, float] = {}
    metric_keys = [key for key in fold_metrics[0].keys() if key != "fold"]
    for key in metric_keys:
        avg_metrics[key] = float(np.mean([m[key] for m in fold_metrics]))

    return {
        "n_splits": n_splits,
        "evaluated_folds": len(fold_metrics),
        "fold_metrics": fold_metrics,
        "avg_metrics": avg_metrics,
    }


def main() -> None:
    args = parse_args()
    if args.n_splits < 2:
        raise ValueError("--n-splits must be at least 2")

    results = run_walkforward(n_splits=args.n_splits, max_splits=args.max_splits)

    for fold in results["fold_metrics"]:
        print(
            (
                f"Fold {int(fold['fold'])}: "
                f"cls_acc={fold['classifier_accuracy']:.4f}, "
                f"cls_f1={fold['classifier_f1']:.4f}, "
                f"reg_mae={fold['regressor_mae']:.4f}, "
                f"reg_r2={fold['regressor_r2']:.4f}"
            )
        )

    print("\nAverage metrics")
    for key, value in results["avg_metrics"].items():
        print(f"- {key}: {value:.4f}")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    with start_run_if_enabled(
        enabled=not args.no_mlflow,
        experiment_name=args.mlflow_experiment,
        run_name=args.mlflow_run_name,
    ):
        if not args.no_mlflow:
            log_metrics_only(
                params={
                    "run_type": "walkforward_baselines",
                    "n_splits": args.n_splits,
                    "max_splits": args.max_splits,
                },
                metrics=results["avg_metrics"],
            )


if __name__ == "__main__":
    main()
