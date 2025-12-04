import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from data_loader import load_kaggle_csv

# --- CONFIGURATION (BULLETPROOF PATHS) ---
# 1. Get the folder where THIS script lives
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Build exact paths
# Note: Ensure the CSV name matches exactly what is in your data folder!
FILE_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'rain_prediction_dataset.csv')
MODEL_PATH = os.path.join(SCRIPT_DIR, '..', 'models', 'lstm_classifier.h5') 

LOOKBACK = 14
TARGET_COL_NAME = 'Rain Tomorrow'

# 3. Check if Model Exists
if not os.path.exists(MODEL_PATH):
    print("\nCRITICAL ERROR: Classifier model not found!")
    print(f"I looked for: {MODEL_PATH}")
    print("Did you run 'train_classifier.py' yet? (You might have only run the regressor).")
    exit()

# --- MAIN LOGIC ---
# 1. Load Data
df = load_kaggle_csv(FILE_PATH)
if df is None:
    exit()

values = df.values
target_col_idx = df.columns.get_loc(TARGET_COL_NAME)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(values)

def create_sequences(data, lookback_steps, target_idx):
    X, y = [], []
    for i in range(len(data) - lookback_steps):
        X.append(data[i:i+lookback_steps, :])
        y.append(data[i+lookback_steps, target_idx])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, LOOKBACK, target_col_idx)

# Test Split
train_size = int(len(X) * 0.8)
X_test = X[train_size:]
y_test = y[train_size:]

# 2. Predict
print("Loading model...")
model = load_model(MODEL_PATH)

print("Running predictions...")
predictions = model.predict(X_test)

# Convert Probabilities (0.7) to Classes (1)
y_pred_classes = (predictions > 0.5).astype(int)

# 3. METRICS IMPLEMENTATION
print("\n" + "="*30)
print("CLASSIFICATION REPORT")
print("="*30)
acc = accuracy_score(y_test, y_pred_classes)
print(f"Accuracy: {acc:.2%}")
print("-" * 30)
print(classification_report(y_test, y_pred_classes, target_names=['No Rain', 'Rain']))
print("="*30)

# 4. VISUALIZATION: CONFUSION MATRIX
# This graph shows "False Alarms" vs "Missed Events"
cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2) # Make text bigger for slides

# Create the Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Dry', 'Predicted Rain'],
            yticklabels=['Actual Dry', 'Actual Rain'])

# Labels
plt.title('Classification Performance: Did we catch the storm?', fontsize=16)
plt.ylabel('Actual Truth', fontsize=14)
plt.xlabel('Model Prediction', fontsize=14)

plt.tight_layout()
plt.show()