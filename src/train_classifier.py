import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam 
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight   
from data_loader import load_kaggle_csv

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'rain_prediction_dataset.csv')
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, '..', 'models', 'lstm_classifier.h5')

TARGET_COL_NAME = 'Rain Tomorrow'
LOOKBACK = 14
BATCH_SIZE = 32
EPOCHS = 30           
LEARNING_RATE = 0.001 

# 1. Load Data
df = load_kaggle_csv(FILE_PATH)
if df is None: exit()

values = df.values
target_col_idx = df.columns.get_loc(TARGET_COL_NAME)

# 2. Scale
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(values)

# 3. Sequences
def create_sequences(data, lookback_steps, target_idx):
    X, y = [], []
    for i in range(len(data) - lookback_steps):
        X.append(data[i:i+lookback_steps, :])
        y.append(data[i+lookback_steps, target_idx])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, LOOKBACK, target_col_idx)

# 4. Split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# --- CLASS WEIGHTS ---
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"Class Weights: No Rain={class_weights[0]:.2f}, Rain={class_weights[1]:.2f}")

# 5. Build Model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.3)) # Increased Dropout slightly
model.add(LSTM(32, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

optimizer = Adam(learning_rate=LEARNING_RATE)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# 6. Train 
print("Starting training with custom optimizer...")
history = model.fit(
    X_train, y_train, 
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE, 
    validation_data=(X_test, y_test),
    class_weight=class_weights_dict, # <--- Apply weights here
    verbose=1
)

model.save(MODEL_SAVE_PATH)
print("Model saved.")

# 7. Plot
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')     # Changed to Loss (more informative than accuracy here)
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title(f'Training Progress (LR={LEARNING_RATE})')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
