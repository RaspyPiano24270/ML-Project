import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'rain_prediction_dataset.csv')

def load_kaggle_csv(filepath=None, target_col=None):
    """
    Loads, filters for the top city, and cleans the weather dataset.
    Args:
        filepath (str): Optional. If None, uses the default path.
        target_col (str): Optional. Checks if this column exists.
    """
    if filepath is None:
        filepath = DEFAULT_DATA_PATH

    print(f"Loading data from: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None

    df = pd.read_csv(filepath)

    # 1. Filter for a Single Location
    if 'Location' in df.columns:
        top_city = df['Location'].value_counts().idxmax()
        print(f"Filtering data for location: {top_city}")
        df = df[df['Location'] == top_city].copy()

    # 2. Fix Date and Sort
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        df.set_index('Date', inplace=True)

    # 3. Handle Missing Values
    df = df.ffill().bfill()

    # 4. Filter Numeric Columns Only
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    
    # Validation Check
    if target_col and target_col not in numeric_df.columns:
        print(f"CRITICAL ERROR: Target column '{target_col}' not found!")
        return None

    print(f"Final dataset shape: {numeric_df.shape}")
    return numeric_df

# --- HELPER: Sequence Creator ---
def make_sequences(data, lookback_steps, target_idx):
    """
    Converts a 2D array into 3D LSTM sequences.
    """
    X, y = [], []
    for i in range(len(data) - lookback_steps):
        X.append(data[i:i+lookback_steps, :])
        y.append(data[i+lookback_steps, target_idx])
    return np.array(X), np.array(y)