import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from data_loader import load_kaggle_csv, make_sequences

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# We still need the model path since data_loader only handles data
MODEL_PATH = os.path.join(SCRIPT_DIR, '..', 'models', 'lstm_classifier.h5') 

LOOKBACK = 14
TARGET_COL_NAME = 'Rain Tomorrow'

# Check if Model Exists
if not os.path.exists(MODEL_PATH):
    print("\nCRITICAL ERROR: Classifier model not found!")
    print(f"I looked for: {MODEL_PATH}")
    print("Did you run 'train_classifier.py' yet?")
    exit()

# --- MAIN LOGIC ---
# 1. Load Data (Path handled automatically)
df = load_kaggle_csv(target_col=TARGET_COL_NAME)
if df is None: exit()

values = df.values
target_col_idx = df.columns.get_loc(TARGET_COL_NAME)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(values)

# 2. Create Sequences (Using helper)
X, y = make_sequences(scaled_data, LOOKBACK, target_col_idx)

# Test Split
train_size = int(len(X) * 0.8)
X_test = X[train_size:]
y_test = y[train_size:]

# 3. Predict
print("Loading model...")
model = load_model(MODEL_PATH)

print("Running predictions...")
predictions = model.predict(X_test)

# Convert Probabilities to Classes (Threshold 0.5)
y_pred_classes = (predictions > 0.5).astype(int)

# 4. Metrics
print("\n" + "="*30)
print("CLASSIFICATION REPORT")
print("="*30)
acc = accuracy_score(y_test, y_pred_classes)
print(f"Accuracy: {acc:.2%}")
print("-" * 30)
print(classification_report(y_test, y_pred_classes, target_names=['No Rain', 'Rain']))
print("="*30)

# 5. Visualization
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Dry', 'Predicted Rain'],
            yticklabels=['Actual Dry', 'Actual Rain'])

plt.title('Classification Performance: Did we catch the storm?', fontsize=16)
plt.ylabel('Actual Truth', fontsize=14)
plt.xlabel('Model Prediction', fontsize=14)
plt.tight_layout()
plt.show()