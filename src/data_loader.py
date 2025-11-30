import pandas as pd
import numpy as np

def load_kaggle_csv(filepath):
    """
    Loads, filters, and cleans the local weather dataset.
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # 1. Filter for a Single Location
    # We automatically pick the city with the most rows of data
    if 'Location' in df.columns:
        top_city = df['Location'].value_counts().idxmax()
        print(f"Filtering data for location: {top_city}")
        df = df[df['Location'] == top_city].copy()
    
    # 2. Fix Date and Sort
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)
    
    # 3. Handle Missing Values
    # Forward fill to handle small gaps in sensor data
    df = df.ffill().bfill()
    
    # 4. Select Features
    # Ensure 'Rain Tomorrow' is included as the target
    # We drop 'Location' and 'Date' (index) from the values used for math
    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Explicitly ensure our target is in the list
    if 'Rain Tomorrow' not in numeric_cols:
         print("Error: 'Rain Tomorrow' column not found or not numeric!")
         return None

    data = df[numeric_cols].copy()
    
    print(f"Final dataset shape: {data.shape}")
    return dataimport pandas as pd

import numpy as np

def load_kaggle_csv(filepath):
    """
    Loads, filters, and cleans the local weather dataset.
    Assumes 'Rain Tomorrow' is already 0/1.
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # 1. Filter for a Single Location
    # We automatically pick the city with the most rows of data
    if 'Location' in df.columns:
        top_city = df['Location'].value_counts().idxmax()
        print(f"Filtering data for location: {top_city}")
        df = df[df['Location'] == top_city].copy()
    
    # 2. Fix Date and Sort
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)
    
    # 3. Handle Missing Values
    # Forward fill to handle small gaps in sensor data
    df = df.ffill().bfill()
    
    # 4. Select Features
    # Ensure 'Rain Tomorrow' is included as the target
    # We drop 'Location' and 'Date' (index) from the values used for math
    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Explicitly ensure our target is in the list
    if 'Rain Tomorrow' not in numeric_cols:
         print("Error: 'Rain Tomorrow' column not found or not numeric!")
         return None

    data = df[numeric_cols].copy()
    
    print(f"Final dataset shape: {data.shape}")
    return data