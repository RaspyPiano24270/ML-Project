import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight

try:
    from src.data_loader import load_kaggle_csv, make_sequences
    from src.experiment_tracking import log_training_metadata, start_run_if_enabled
    from src.pipeline_utils import (
        compute_row_split_index,
        compute_sequence_split_index,
        fit_feature_scaler,
        save_artifact,
    )
except ModuleNotFoundError:
    from data_loader import load_kaggle_csv, make_sequences
    from experiment_tracking import log_training_metadata, start_run_if_enabled
    from pipeline_utils import (
        compute_row_split_index,
        compute_sequence_split_index,
        fit_feature_scaler,
        save_artifact,
    )

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = SCRIPT_DIR.parent / "models" / "lstm_classifier.h5"
DEFAULT_ARTIFACT_PATH = SCRIPT_DIR.parent / "models" / "lstm_classifier_artifacts.pkl"

TARGET_COL_NAME = 'Rain Tomorrow'
LOOKBACK = 14
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
TEST_SIZE = 0.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM classifier for rain prediction.")
    parser.add_argument("--lookback", type=int, default=LOOKBACK)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--artifact-path", type=str, default=str(DEFAULT_ARTIFACT_PATH))
    parser.add_argument("--mlflow-experiment", type=str, default="stormcast-classifier")
    parser.add_argument("--mlflow-run-name", type=str, default="")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking")
    return parser.parse_args()


def prepare_data(
    lookback: int,
    test_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, list[str]]:
    df = load_kaggle_csv(target_col=TARGET_COL_NAME)
    if df is None:
        raise RuntimeError("Dataset could not be loaded. Check your data path and target column.")

    values = df.values
    target_col_idx = df.columns.get_loc(TARGET_COL_NAME)
    feature_columns = df.columns.tolist()

    row_split_idx = compute_row_split_index(num_rows=len(values), test_size=test_size)
    feature_scaler = fit_feature_scaler(values[:row_split_idx])
    scaled_data = feature_scaler.transform(values)

    X, y = make_sequences(scaled_data, lookback, target_col_idx)
    seq_split_idx = compute_sequence_split_index(row_split_idx=row_split_idx, lookback=lookback, num_rows=len(values))

    X_train, X_test = X[:seq_split_idx], X[seq_split_idx:]
    y_train, y_test = y[:seq_split_idx], y[seq_split_idx:]
    return X_train, X_test, y_train, y_test, feature_scaler, feature_columns


def build_model(input_shape: tuple[int, int], learning_rate: float) -> Sequential:
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def compute_class_weights(y_train: np.ndarray) -> dict[int, float]:
    classes = np.unique(y_train)
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train,
    )
    return {int(cls): float(weight) for cls, weight in zip(classes, class_weights)}


def plot_history(history, learning_rate: float) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'Training Progress (LR={learning_rate})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    args = parse_args()
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1")

    X_train, X_test, y_train, y_test, feature_scaler, feature_columns = prepare_data(args.lookback, args.test_size)

    class_weights = compute_class_weights(y_train)
    print(
        "Class Weights:",
        ", ".join([f"class {cls}={weight:.2f}" for cls, weight in class_weights.items()]),
    )

    model = build_model((X_train.shape[1], X_train.shape[2]), args.learning_rate)

    print("Starting classifier training...")
    history = model.fit(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_test, y_test),
        class_weight=class_weights,
        verbose=1,
    )

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    print(f"Classifier model saved to {model_path}")

    artifact_path = Path(args.artifact_path)
    save_artifact(
        artifact_path,
        {
            "feature_scaler": feature_scaler,
            "feature_columns": feature_columns,
            "lookback": args.lookback,
            "target_col": TARGET_COL_NAME,
            "test_size": args.test_size,
        },
    )
    print(f"Classifier preprocessing artifacts saved to {artifact_path}")

    with start_run_if_enabled(
        enabled=not args.no_mlflow,
        experiment_name=args.mlflow_experiment,
        run_name=args.mlflow_run_name or None,
    ):
        if not args.no_mlflow:
            metrics = {
                "final_train_loss": float(history.history["loss"][-1]),
                "final_val_loss": float(history.history["val_loss"][-1]),
                "best_val_loss": float(min(history.history["val_loss"])),
                "final_train_accuracy": float(history.history["accuracy"][-1]),
                "final_val_accuracy": float(history.history["val_accuracy"][-1]),
                "best_val_accuracy": float(max(history.history["val_accuracy"])),
            }
            log_training_metadata(
                params={
                    "model_type": "lstm_classifier",
                    "lookback": args.lookback,
                    "batch_size": args.batch_size,
                    "epochs": args.epochs,
                    "learning_rate": args.learning_rate,
                    "test_size": args.test_size,
                },
                metrics=metrics,
                artifact_paths=[model_path, artifact_path],
            )
            print("MLflow tracking completed for classifier run")

    plot_history(history, args.learning_rate)


if __name__ == "__main__":
    main()