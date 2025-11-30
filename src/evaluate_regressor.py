import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data_loader import load_kaggle_csv

# --- CONFIGURATION ---
FILE_PATH = '../data/usa_rain_prediction_dataset.csv'
MODEL_PATH = '../models/lstm_regression_model.h5' # Make sure to save your regression model as a different name!
LOOKBACK = 14
TARGET_COL_NAME = 'Precipitation' # Changing target to amount

# 1. Load Data
df = load_kaggle_csv(FILE_PATH)
values = df.values
target_col_idx = df.columns.get_loc(TARGET_COL_NAME)

# CRITICAL: We need a separate scaler for the target to "inverse" predictions later
# (We want to see error in "millimeters", not "scaled 0.2")
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(values)

target_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaler.fit(df[[TARGET_COL_NAME]]) # Fit only on the precipitation column

X, y = [], []
for i in range(len(scaled_data) - LOOKBACK):
    X.append(scaled_data[i:i+LOOKBACK, :])
    y.append(scaled_data[i+LOOKBACK, target_col_idx])
X, y = np.array(X), np.array(y)

# Test Split
train_size = int(len(X) * 0.8)
X_test = X[train_size:]
y_test = y[train_size:]

# 2. Predict
model = load_model(MODEL_PATH)
predictions_scaled = model.predict(X_test)

# 3. Inverse Transform (Get back to Millimeters)
# We reshape to (N, 1) to match scaler requirements
y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_actual = target_scaler.inverse_transform(predictions_scaled)

# 4. METRICS IMPLEMENTATION
print("\n--- REGRESSION METRICS ---")

# MSE (Penalizes large errors heavily - good for catching outliers/storms)
mse = mean_squared_error(y_test_actual, y_pred_actual)
print(f"Mean Squared Error (MSE): {mse:.4f}")

# MAE (Average error in millimeters - easier to explain to humans)
mae = mean_absolute_error(y_test_actual, y_pred_actual)
print(f"Mean Absolute Error (MAE): {mae:.4f} mm")

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, label='Actual Rainfall (mm)', color='blue')
plt.plot(y_pred_actual, label='Predicted Rainfall (mm)', color='red', alpha=0.7)
plt.title('Regression: Predicted vs Actual Rainfall Amount')
plt.legend()
plt.show()