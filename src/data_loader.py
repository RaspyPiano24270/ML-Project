import pandas as pd
import numpy as np

def load_kaggle_csv(filepath):
    """
    Loads, filters, and cleans the Kaggle weather dataset.
    Returns a cleaned numeric DataFrame sorted by date.
    """
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    
    # ---- 1. Filter to the most common location ----
    if 'Location' in df.columns:
        top_city = df['Location'].value_counts().idxmax()
        print(f"Filtering for location: {top_city}")
        df = df[df['Location'] == top_city].copy()
    
    # ---- 2. Fix and sort by date ----
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        df.set_index('Date', inplace=True)
    
    # ---- 3. Fill missing values ----
    df = df.ffill().bfill()
    
    # ---- 4. Select numeric columns ----
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    data = df[numeric_cols].copy()
    
    print(f"Final dataset shape (numeric only): {data.shape}")
    return data


# ---------------------------------------------------------------------
# DATA LOADER FOR THE REGRESSION MODEL
# ---------------------------------------------------------------------

def load_regression_data(filepath, target="Precipitation", lookback=14, val_split=0.10, test_split=0.10):
    """
    Loads the Kaggle dataset and prepares it for LSTM regression.

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    With shapes:
        X_* = (samples, lookback, num_features)
        y_* = (samples,)
    """

    df = load_kaggle_csv(filepath)
    if df is None:
        return None, None, None, None, None, None

    # ---- Ensure target column exists ----
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset. Available columns: {df.columns.tolist()}")

    # ---- Remove classifier label if present ----
    if "Rain Tomorrow" in df.columns:
        df = df.drop(columns=["Rain Tomorrow"])
    
    values = df.values.astype(float)
    target_idx = df.columns.get_loc(target)

    # ---- Scale all features with MinMaxScaler ----
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    # ---- Create LSTM sequences ----
    def create_sequences(data, lookback_steps, target_idx):
        X, y = [], []
        for i in range(len(data) - lookback_steps):
            X.append(data[i : i + lookback_steps])
            y.append(data[i + lookback_steps, target_idx])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled, lookback, target_idx)

    # ---- Split train/val/test ----
    total = len(X)
    test_size = int(total * test_split)
    val_size  = int(total * val_split)

    X_train = X[: total - val_size - test_size]
    y_train = y[: total - val_size - test_size]

    X_val   = X[total - val_size - test_size : total - test_size]
    y_val   = y[total - val_size - test_size : total - test_size]

    X_test  = X[total - test_size :]
    y_test  = y[total - test_size :]

    print("Shapes:")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val:  ", X_val.shape,   "y_val:", y_val.shape)
    print("X_test: ", X_test.shape,  "y_test:", y_test.shape)

    return X_train, y_train, X_val, y_val, X_test, y_test
