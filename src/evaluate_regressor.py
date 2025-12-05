import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_loader import load_kaggle_csv, make_sequences

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, '..', 'models', 'lstm_regressor.h5')

TARGET_COL_NAME = 'Precipitation'
LOOKBACK = 14

if not os.path.exists(MODEL_PATH):
    print("\nCRITICAL ERROR: Regressor model not found!")
    print("Please run 'train_regressor.py' first.")
    exit()

# --- MAIN LOGIC ---
# 1. Load Data
df = load_kaggle_csv(target_col=TARGET_COL_NAME)
if df is None: exit()

values = df.values
target_col_idx = df.columns.get_loc(TARGET_COL_NAME)

# 2. Scale
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(values)

# Specific scaler for target to reverse-engineer the answer later
target_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaler.fit(values[:, target_col_idx].reshape(-1, 1))

# 3. Create Sequences
X, y = make_sequences(scaled_data, LOOKBACK, target_col_idx)

# 4. Split Test Data
train_size = int(len(X) * 0.8)
X_test = X[train_size:]
y_test = y[train_size:]

# 5. Predict
print("Loading model...")
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading .h5 file: {e}")
    exit()

print("Predicting future rainfall...")
predictions_scaled = model.predict(X_test)

# 6. Inverse Transform
y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1))
predictions_actual = target_scaler.inverse_transform(predictions_scaled)

# 7. Metrics
mse = mean_squared_error(y_test_actual, predictions_actual)
mae = mean_absolute_error(y_test_actual, predictions_actual)
r2 = r2_score(y_test_actual, predictions_actual)

print("\n" + "="*30)
print(f"RESULTS REPORT")
print("="*30)
print(f"Mean Absolute Error (MAE): {mae:.4f} mm")
print(f"Mean Squared Error (MSE):  {mse:.4f}")
print(f"R² Score:                  {r2:.4f}")
print("-" * 30)

# 8. Visualization
plt.figure(figsize=(10, 8))
sns.set_style("whitegrid")

actuals = y_test_actual.flatten()
preds = predictions_actual.flatten()

sns.scatterplot(x=actuals, y=preds, alpha=0.5, color='blue', label='Data Points')

# Perfect Prediction Line
min_val = min(actuals.min(), preds.min())
max_val = max(actuals.max(), preds.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Prediction')

plt.title(f'Regression Performance: Actual vs Predicted Rainfall\nMAE: {mae:.2f} mm', fontsize=14)
plt.xlabel('Actual Rainfall (mm)', fontsize=12)
plt.ylabel('Predicted Rainfall (mm)', fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()