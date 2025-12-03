"""
Train an LSTM regressor to predict rainfall amount ("Precipitation")
using a sliding window over the Kaggle USA rain dataset.

Pipeline:
- src.data_loader.load_regression_data() handles:
    * loading + cleaning CSV
    * filtering to top Location
    * sorting by Date
    * scaling with MinMaxScaler
    * building (lookback, features) sequences
    * splitting into train / val / test

This script focuses ONLY on:
- building the LSTM model
- training & evaluating
- saving the model + metrics/history JSON
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

# Try both import styles so it works whether you run:
#   python src/train_regressor.py
# or:
#   python -m src.train_regressor
try:
    from src.data_loader import load_regression_data
except ImportError:
    from data_loader import load_regression_data


# ---------------------------------------------------------------------
# Paths / Config
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent  # project root


def build_model(input_shape):
    """
    Two-layer LSTM model for regression.

    input_shape: (timesteps, num_features)
    """
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # regression output

    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=["mae"],
    )
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train LSTM regressor on Kaggle USA rain dataset."
    )

    p.add_argument(
        "--file_path",
        default=str(BASE_DIR / "data" / "usa_rain_prediction_dataset.csv"),
        help="Path to the Kaggle CSV file.",
    )
    p.add_argument(
        "--target_col",
        default="Precipitation",
        help="Name of the column to predict (must be numeric).",
    )
    p.add_argument(
        "--lookback",
        type=int,
        default=14,
        help="Number of timesteps (days) to look back for each sample.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs.",
    )
    p.add_argument(
        "--model_path",
        default=str(BASE_DIR / "models" / "lstm_regressor.h5"),
        help="Where to save the trained model.",
    )
    p.add_argument(
        "--history_path",
        default=str(BASE_DIR / "Reports" / "regressor_history.json"),
        help="Where to save training history + test metrics as JSON.",
    )
    p.add_argument(
        "--no_plot",
        action="store_true",
        help="Disable training curves plot.",
    )

    return p.parse_args()


def main():
    args = parse_args()

    file_path = args.file_path
    target_col = args.target_col
    lookback = args.lookback
    batch_size = args.batch_size
    epochs = args.epochs

    model_path = Path(args.model_path)
    history_path = Path(args.history_path)

    # Ensure output dirs exist
    model_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    print(" Loading regression data via data_loader.load_regression_data() ...")
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = load_regression_data(
        filepath=file_path,
        target=target_col,
        lookback=lookback,
    )

    if X_train is None:
        print("Failed to load data. Exiting.")
        return

    print("\n Data shapes:")
    print("X_train:", X_train.shape, " y_train:", y_train.shape)
    print("X_val:  ", X_val.shape, " y_val:", y_val.shape)
    print("X_test: ", X_test.shape, " y_test:", y_test.shape)

    # Make sure targets are 1D
    y_train = np.squeeze(y_train)
    y_val = np.squeeze(y_val)
    y_test = np.squeeze(y_test)

    # Build model
    input_shape = X_train.shape[1:]  # (timesteps, features)
    model = build_model(input_shape)
    model.summary()

    print("\n Starting training...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    # Evaluate on test
    print("\n Evaluating on test set...")
    y_pred = model.predict(X_test).squeeze()
    test_mse = mean_squared_error(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)

    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")

    # Save model
    model.save(model_path)
    print(f"\n Model saved to: {model_path}")

    # Save history + metrics
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    results = {
        "history": hist_dict,
        "test_mse": float(test_mse),
        "test_mae": float(test_mae),
        "file_path": file_path,
        "target_col": target_col,
        "lookback": lookback,
        "batch_size": batch_size,
        "epochs": epochs,
    }

    with history_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f" Training history + metrics saved to: {history_path}")

    # Plot training curves (unless disabled)
    if not args.no_plot:
        plt.figure(figsize=(10, 5))
        plt.plot(history.history["loss"], label="Train Loss (MSE)")
        plt.plot(history.history["val_loss"], label="Val Loss (MSE)")
        plt.title("LSTM Regressor Training Progress")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
