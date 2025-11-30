import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from data_loader import load_kaggle_csv

# --- CONFIGURATION ---
FILE_PATH = '../data/rain_prediction_dataset.csv'
MODEL_PATH = '../models/lstm_weather_model.h5' # Ensure this is your Classification model
LOOKBACK = 14
TARGET_COL_NAME = 'Rain Tomorrow'

# 1. Load and Prepare Data (Same as training)
df = load_kaggle_csv(FILE_PATH)
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
model = load_model(MODEL_PATH)
predictions = model.predict(X_test)
y_pred_classes = (predictions > 0.5).astype(int)

# 3. METRICS IMPLEMENTATION
print("\n--- CLASSIFICATION METRICS ---")
acc = accuracy_score(y_test, y_pred_classes)
print(f"Accuracy: {acc:.2%}")

# Detailed Report (Precision, Recall, F1)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, target_names=['No Rain', 'Rain']))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Dry', 'Predicted Rain'],
            yticklabels=['Actual Dry', 'Actual Rain'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()