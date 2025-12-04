import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from data_loader import load_kaggle_csv

# --- CONFIGURATION ---
FILE_PATH = '../data/rain_prediction_dataset.csv'
MODEL_SAVE_PATH = '../models/lstm_regressor.h5'
TARGET_COL_NAME = 'Precipitation'  # We are predicting the AMOUNT of rain
LOOKBACK = 14  # Days to look back
BATCH_SIZE = 32
EPOCHS = 20

# 1. Load Data
# This uses your shared data_loader script
df = load_kaggle_csv(FILE_PATH)

if df is None:
    print("Error loading data.")
    exit()

# Extract values
values = df.values

# Identify the column index for 'Precipitation'
# We need this to tell the model which column is the "answer key"
target_col_idx = df.columns.get_loc(TARGET_COL_NAME)

# 2. Scale Data
# LSTMs are sensitive to scale. We squash everything between 0 and 1.
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(values)

# 3. Create Sequences (Sliding Window)
def create_sequences(data, lookback_steps, target_idx):
    X, y = [], []
    for i in range(len(data) - lookback_steps):
        # Input: Past 'lookback' days (all features)
        X.append(data[i:i+lookback_steps, :])
        
        # Target: The specific rainfall amount for the NEXT day
        y.append(data[i+lookback_steps, target_idx])
        
    return np.array(X), np.array(y)

print("Creating sequences...")
X, y = create_sequences(scaled_data, LOOKBACK, target_col_idx)

# 4. Split Train/Test (80/20 split)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples.")

# 5. Build Model (Regression Architecture)
model = Sequential()

# LSTM Layer 1: Return sequences=True to stack another LSTM on top
model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2)) # Prevent overfitting

# LSTM Layer 2
model.add(LSTM(32, return_sequences=False))
model.add(Dropout(0.2))

# Output Layer:
# Units=1 -> Predicting a single number (Rainfall Amount)
# Activation=None (Linear) -> Allows the model to output any range of numbers
model.add(Dense(1)) 

# Compile:
# Loss='mean_squared_error' -> Standard for regression tasks
# Optimizer='adam' -> Smart learning rate
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# 6. Train
print("Starting training...")
history = model.fit(
    X_train, y_train, 
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE, 
    validation_data=(X_test, y_test),
    verbose=1
)

# 7. Save the Model
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

# 8. Visualization (Training Loss)
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss (MSE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
plt.title('Regression Training Progress')
plt.xlabel('Epochs')
plt.ylabel('Loss (Mean Squared Error)')
plt.legend()
plt.show()