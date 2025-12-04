import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping  
from sklearn.preprocessing import MinMaxScaler
from data_loader import load_kaggle_csv

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'rain_prediction_dataset.csv')
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, '..', 'models', 'lstm_regressor.h5')

TARGET_COL_NAME = 'Precipitation'
LOOKBACK = 14
BATCH_SIZE = 32
EPOCHS = 50           
LEARNING_RATE = 0.001

# 1. Load Data
df = load_kaggle_csv(FILE_PATH)
if df is None: exit()

values = df.values
target_col_idx = df.columns.get_loc(TARGET_COL_NAME)

# 2. Scale
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(values)

# 3. Create Sequences
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

print(f"Training Regressor on {len(X_train)} samples...")

# 5. Build Model
model = Sequential()
# Increased neurons slightly (64 -> 128) to capture more complex storm patterns
model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1)) # Linear activation for regression

# Custom Optimizer
optimizer = Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

# --- EARLY STOPPING ---
# restore_best_weights=True ensures we save the BEST version, not the last one.
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    restore_best_weights=True,
    verbose=1
)

# 6. Train
history = model.fit(
    X_train, y_train, 
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE, 
    validation_data=(X_test, y_test),
    callbacks=[early_stop], # <--- Add the callback here
    verbose=1
)

model.save(MODEL_SAVE_PATH)
print(f"Regressor model saved to {MODEL_SAVE_PATH}")

# 7. Plot
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss (MSE)')
plt.plot(history.history['val_loss'], label='Val Loss (MSE)')
plt.title(f'Regressor Training (LR={LEARNING_RATE})')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
