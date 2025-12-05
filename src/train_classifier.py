import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam 
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight   
from data_loader import load_kaggle_csv, make_sequences

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, '..', 'models', 'lstm_classifier.h5')

TARGET_COL_NAME = 'Rain Tomorrow'
LOOKBACK = 14
BATCH_SIZE = 32
EPOCHS = 30           
LEARNING_RATE = 0.001 

# 1. Load Data
df = load_kaggle_csv(target_col=TARGET_COL_NAME)
if df is None: exit()

values = df.values
target_col_idx = df.columns.get_loc(TARGET_COL_NAME)

# 2. Scale
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(values)

# 3. Create Sequences
X, y = make_sequences(scaled_data, LOOKBACK, target_col_idx)

# 4. Split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# --- CLASS WEIGHTS ---
# Critical for fixing the "Lazy Model" problem
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
model.add(Dropout(0.3)) 
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
    class_weight=class_weights_dict, 
    verbose=1
)

model.save(MODEL_SAVE_PATH)
print(f"Classifier model saved to {MODEL_SAVE_PATH}")

# 7. Plot
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')     
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title(f'Training Progress (LR={LEARNING_RATE})')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()