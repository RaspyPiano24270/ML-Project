import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data_loader import load_kaggle_csv
import seaborn as sns


# --- CONFIGURATION (BULLETPROOF PATHS) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Build exact paths to the data and model
FILE_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'rain_prediction_dataset.csv')
MODEL_PATH = os.path.join(SCRIPT_DIR, '..', 'models', 'lstm_regressor.h5')

TARGET_COL_NAME = 'Precipitation'
LOOKBACK = 14

print(f"Looking for data at: {os.path.abspath(FILE_PATH)}")
print(f"Looking for model at: {os.path.abspath(MODEL_PATH)}")

# 3. Check if Model Exists BEFORE trying to load it
if not os.path.exists(MODEL_PATH):
    print("\nCRITICAL ERROR: Model file not found!")
    print(f"The file '{os.path.abspath(MODEL_PATH)}' does not exist.")
    print("Please run 'train_regressor.py' again to generate it.")
    exit()

# --- MAIN LOGIC ---
# 1. Load Data
df = load_kaggle_csv(FILE_PATH)
if df is None:
    exit()

values = df.values
target_col_idx = df.columns.get_loc(TARGET_COL_NAME)

# 2. Re-create the Scaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(values)

target_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaler.fit(values[:, target_col_idx].reshape(-1, 1))

# 3. Create Sequences
def create_sequences(data, lookback_steps, target_idx):
    X, y = [], []
    for i in range(len(data) - lookback_steps):
        X.append(data[i:i+lookback_steps, :])
        y.append(data[i+lookback_steps, target_idx])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, LOOKBACK, target_col_idx)

# 4. Split Test Data
train_size = int(len(X) * 0.8)
X_test = X[train_size:]
y_test = y[train_size:]

# 5. Load Model & Predict
print("Loading model...")
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading .h5 file: {e}")
    print("Trying to retrain might fix this if the file is corrupt.")
    exit()

print("Predicting future rainfall...")
predictions_scaled = model.predict(X_test)

# 6. INVERSE TRANSFORM
y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1))
predictions_actual = target_scaler.inverse_transform(predictions_scaled)

# 7. Metrics
mse = mean_squared_error(y_test_actual, predictions_actual)
mae = mean_absolute_error(y_test_actual, predictions_actual)

print("\n" + "="*30)
print(f"RESULTS REPORT")
print("="*30)
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f} mm")
print("-" * 30)


# --- VISUALIZATION: SCATTER PLOT ---
plt.figure(figsize=(10, 8))
sns.set_style("whitegrid")

# 1. Flatten the arrays to make them simple lists
actuals = y_test_actual.flatten()
preds = predictions_actual.flatten()

# 2. Create the Scatter Plot
# Alpha=0.5 makes dots transparent so you can see where they pile up
sns.scatterplot(x=actuals, y=preds, alpha=0.5, color='blue', label='Data Points')

# 3. Add the "Perfect Prediction" Line (The 45-degree diagonal)
# If a dot lands on this red line, the prediction was 100% correct.
min_val = min(actuals.min(), preds.min())
max_val = max(actuals.max(), preds.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Prediction')

# 4. Labels and Title
plt.title(f'Regression Performance: Actual vs Predicted Rainfall\nMAE: {mae:.2f} mm', fontsize=14)
plt.xlabel('Actual Rainfall (mm)', fontsize=12)
plt.ylabel('Predicted Rainfall (mm)', fontsize=12)
plt.legend()

# 5. Show
plt.tight_layout()
plt.show()