import argparse
from pathlib import Path

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

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
DEFAULT_MODEL_PATH = SCRIPT_DIR.parent / "models" / "lstm_classifier.h5"
DEFAULT_ARTIFACT_PATH = SCRIPT_DIR.parent / "models" / "lstm_classifier_artifacts.pkl"

LOOKBACK = 14
TARGET_COL_NAME = 'Rain Tomorrow'
THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the trained LSTM classifier.")
    parser.add_argument("--lookback", type=int, default=LOOKBACK)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--artifact-path", type=str, default=str(DEFAULT_ARTIFACT_PATH))
    parser.add_argument("--mlflow-experiment", type=str, default="stormcast-classifier-eval")
    parser.add_argument("--mlflow-run-name", type=str, default="classifier-eval")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking")
    return parser.parse_args()


def prepare_test_data(lookback: int, test_size: float, artifact_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = load_kaggle_csv(target_col=TARGET_COL_NAME)
    if df is None:
        raise RuntimeError("Dataset could not be loaded. Check your data path and target column.")

    artifacts = load_artifact(artifact_path)
    feature_scaler: MinMaxScaler = artifacts["feature_scaler"]
    trained_lookback: int = artifacts["lookback"]
    if lookback != trained_lookback:
        raise ValueError(f"Requested lookback ({lookback}) does not match trained lookback ({trained_lookback})")

    values = df.values
    target_col_idx = df.columns.get_loc(TARGET_COL_NAME)

    row_split_idx = compute_row_split_index(num_rows=len(values), test_size=test_size)
    seq_split_idx = compute_sequence_split_index(row_split_idx=row_split_idx, lookback=lookback, num_rows=len(values))
    scaled_data = feature_scaler.transform(values)

    X, y = make_sequences(scaled_data, lookback, target_col_idx)
    return X[seq_split_idx:], y[seq_split_idx:]


def plot_confusion(y_true: np.ndarray, y_pred_classes: np.ndarray) -> None:
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Predicted Dry', 'Predicted Rain'],
        yticklabels=['Actual Dry', 'Actual Rain'],
    )

    plt.title('Classification Performance: Did we catch the storm?', fontsize=16)
    plt.ylabel('Actual Truth', fontsize=14)
    plt.xlabel('Model Prediction', fontsize=14)
    plt.tight_layout()
    plt.show()


def main() -> None:
    args = parse_args()
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1")
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be between 0 and 1")

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Classifier model not found at {model_path}")
    artifact_path = Path(args.artifact_path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Classifier artifact file not found at {artifact_path}")

    X_test, y_test = prepare_test_data(args.lookback, args.test_size, artifact_path)

    print("Loading model...")
    model = load_model(model_path)

    print("Running predictions...")
    predictions = model.predict(X_test)

    y_pred_classes = (predictions > args.threshold).astype(int)

    print("\n" + "=" * 30)
    print("CLASSIFICATION REPORT")
    print("=" * 30)
    acc = accuracy_score(y_test, y_pred_classes)
    print(f"Accuracy: {acc:.2%}")
    print("-" * 30)
    print(classification_report(y_test, y_pred_classes, target_names=['No Rain', 'Rain']))
    print("=" * 30)

    metrics = {
        "accuracy": float(acc),
        "f1": float(f1_score(y_test, y_pred_classes, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred_classes, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred_classes, zero_division=0)),
    }

    with start_run_if_enabled(
        enabled=not args.no_mlflow,
        experiment_name=args.mlflow_experiment,
        run_name=args.mlflow_run_name,
    ):
        if not args.no_mlflow:
            log_metrics_only(
                params={
                    "run_type": "classifier_evaluation",
                    "threshold": args.threshold,
                    "test_size": args.test_size,
                    "lookback": args.lookback,
                },
                metrics=metrics,
            )

    plot_confusion(y_test, y_pred_classes)


if __name__ == "__main__":
    main()