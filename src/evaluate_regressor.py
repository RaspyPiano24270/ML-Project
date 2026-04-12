import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from src.data_loader import load_kaggle_csv, make_sequences
    from src.experiment_tracking import log_metrics_only, start_run_if_enabled
    from src.pipeline_utils import compute_row_split_index, compute_sequence_split_index, load_artifact
except ModuleNotFoundError:
    from data_loader import load_kaggle_csv, make_sequences
    from experiment_tracking import log_metrics_only, start_run_if_enabled
    from pipeline_utils import compute_row_split_index, compute_sequence_split_index, load_artifact

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = SCRIPT_DIR.parent / "models" / "lstm_regressor.h5"
DEFAULT_ARTIFACT_PATH = SCRIPT_DIR.parent / "models" / "lstm_regressor_artifacts.pkl"

TARGET_COL_NAME = 'Precipitation'
LOOKBACK = 14


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the trained LSTM regressor.")
    parser.add_argument("--lookback", type=int, default=LOOKBACK)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--artifact-path", type=str, default=str(DEFAULT_ARTIFACT_PATH))
    parser.add_argument("--mlflow-experiment", type=str, default="stormcast-regressor-eval")
    parser.add_argument("--mlflow-run-name", type=str, default="regressor-eval")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking")
    return parser.parse_args()


def prepare_test_data(lookback: int, test_size: float, artifact_path: Path) -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    df = load_kaggle_csv(target_col=TARGET_COL_NAME)
    if df is None:
        raise RuntimeError("Dataset could not be loaded. Check your data path and target column.")

    artifacts = load_artifact(artifact_path)
    feature_scaler: MinMaxScaler = artifacts["feature_scaler"]
    target_scaler: MinMaxScaler = artifacts["target_scaler"]
    trained_lookback: int = artifacts["lookback"]
    if lookback != trained_lookback:
        raise ValueError(f"Requested lookback ({lookback}) does not match trained lookback ({trained_lookback})")

    values = df.values
    target_col_idx = df.columns.get_loc(TARGET_COL_NAME)

    row_split_idx = compute_row_split_index(num_rows=len(values), test_size=test_size)
    seq_split_idx = compute_sequence_split_index(row_split_idx=row_split_idx, lookback=lookback, num_rows=len(values))
    scaled_data = feature_scaler.transform(values)

    X, y = make_sequences(scaled_data, lookback, target_col_idx)
    return X[seq_split_idx:], y[seq_split_idx:], target_scaler


def plot_actual_vs_predicted(actuals: np.ndarray, preds: np.ndarray, mae: float) -> None:
    plt.figure(figsize=(10, 8))
    sns.set_style("whitegrid")
    sns.scatterplot(x=actuals, y=preds, alpha=0.5, color='blue', label='Data Points')

    min_val = min(actuals.min(), preds.min())
    max_val = max(actuals.max(), preds.max())
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        color='red',
        linestyle='--',
        linewidth=2,
        label='Perfect Prediction',
    )

    plt.title(f'Regression Performance: Actual vs Predicted Rainfall\nMAE: {mae:.2f} mm', fontsize=14)
    plt.xlabel('Actual Rainfall (mm)', fontsize=12)
    plt.ylabel('Predicted Rainfall (mm)', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    args = parse_args()
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1")

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Regressor model not found at {model_path}")
    artifact_path = Path(args.artifact_path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Regressor artifact file not found at {artifact_path}")

    X_test, y_test, target_scaler = prepare_test_data(args.lookback, args.test_size, artifact_path)

    print("Loading model...")
    model = load_model(model_path)

    print("Predicting future rainfall...")
    predictions_scaled = model.predict(X_test)

    y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1))
    predictions_actual = target_scaler.inverse_transform(predictions_scaled)

    mse = mean_squared_error(y_test_actual, predictions_actual)
    mae = mean_absolute_error(y_test_actual, predictions_actual)
    r2 = r2_score(y_test_actual, predictions_actual)

    print("\n" + "=" * 30)
    print("RESULTS REPORT")
    print("=" * 30)
    print(f"Mean Absolute Error (MAE): {mae:.4f} mm")
    print(f"Mean Squared Error (MSE):  {mse:.4f}")
    print(f"R^2 Score:                 {r2:.4f}")
    print("-" * 30)

    with start_run_if_enabled(
        enabled=not args.no_mlflow,
        experiment_name=args.mlflow_experiment,
        run_name=args.mlflow_run_name,
    ):
        if not args.no_mlflow:
            log_metrics_only(
                params={
                    "run_type": "regressor_evaluation",
                    "test_size": args.test_size,
                    "lookback": args.lookback,
                },
                metrics={
                    "mae": float(mae),
                    "mse": float(mse),
                    "r2": float(r2),
                },
            )

    plot_actual_vs_predicted(y_test_actual.flatten(), predictions_actual.flatten(), mae)


if __name__ == "__main__":
    main()