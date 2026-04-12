from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = SCRIPT_DIR.parent / "data" / "rain_prediction_dataset.csv"

def load_kaggle_csv(filepath: Optional[str] = None, target_col: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Load and clean the weather dataset used by training and evaluation scripts.

    Steps:
    1. Load CSV from provided path or default path.
    2. Keep only the most common location for a single coherent time series.
    3. Parse/sort dates and set date index when available.
    4. Fill missing values and keep numeric columns only.
    """
    data_path = Path(filepath) if filepath else DEFAULT_DATA_PATH

    print(f"Loading data from: {data_path}")
    
    if not data_path.exists():
        print(f"Error: File not found at {data_path}")
        return None

    df = pd.read_csv(data_path)

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
def make_sequences(data: np.ndarray, lookback_steps: int, target_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert a 2D feature matrix into LSTM sequences and next-step targets."""
    if lookback_steps <= 0:
        raise ValueError("lookback_steps must be greater than 0")
    if data.ndim != 2:
        raise ValueError("data must be a 2D array")
    if not 0 <= target_idx < data.shape[1]:
        raise ValueError("target_idx is out of bounds for data columns")
    if len(data) <= lookback_steps:
        raise ValueError("Not enough rows to build sequences with the selected lookback_steps")

    X, y = [], []
    for i in range(len(data) - lookback_steps):
        X.append(data[i : i + lookback_steps, :])
        y.append(data[i + lookback_steps, target_idx])
    return np.array(X), np.array(y)