# src/data_preprocessing.py
# --------------------
# Functions for data loading and preprocessing
# --------------------

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os

# Import configuration relative to src when run as part of the package
try:
    from . import config
except ImportError:
    # Fallback for running the script directly (requires src to be in PYTHONPATH)
    import config


def load_data(file_path):
    """Loads data from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Data loaded successfully from {file_path}. Shape: {df.shape}")
    return df

def preprocess_data(df, date_col, target_col, feature_cols, resample_freq='H'):
    """Basic preprocessing: datetime index, resampling, feature selection."""
    # Make a copy to avoid modifying the original DataFrame
    df_processed = df.copy()

    # Convert to datetime and set index
    try:
        # Attempt common formats first
        df_processed[date_col] = pd.to_datetime(df_processed[date_col], infer_datetime_format=True)
    except ValueError:
         # Fallback to a specific format if inference fails
         print("Warning: Inferring datetime format failed. Trying '%d %m %Y %H:%M'. Adjust if needed.")
         df_processed[date_col] = pd.to_datetime(df_processed[date_col], format='%d %m %Y %H:%M')

    df_processed.set_index(date_col, inplace=True)

    # Handle potential duplicates from setting index (keep first)
    df_processed = df_processed[~df_processed.index.duplicated(keep='first')]

    # Resample data
    if resample_freq:
        df_processed = df_processed.resample(resample_freq).mean()
        print(f"Data resampled to '{resample_freq}' frequency.")

    # Handle missing values (example: forward fill then backward fill)
    initial_nans = df_processed.isnull().sum().sum()
    if initial_nans > 0:
        print(f"Handling {initial_nans} missing values (using ffill then bfill)...")
        df_processed.ffill(inplace=True)
        df_processed.bfill(inplace=True) # Backfill any remaining NaNs at the beginning
    final_nans = df_processed.isnull().sum().sum()
    if final_nans > 0:
        print(f"Warning: {final_nans} missing values still present after ffill/bfill.")
        # Consider other strategies like interpolation or dropping NaNs if appropriate
        # df_processed.interpolate(method='time', inplace=True)
        # df_processed.dropna(inplace=True)


    # Select relevant features + target
    cols_to_keep = feature_cols[:] # Make a copy
    if target_col not in cols_to_keep:
         cols_to_keep.append(target_col) # Ensure target is included if not already a feature

    # Filter out columns not present in the dataframe
    available_cols = [col for col in cols_to_keep if col in df_processed.columns]
    if len(available_cols) < len(cols_to_keep):
        missing_cols = set(cols_to_keep) - set(available_cols)
        print(f"Warning: Requested columns not found in data: {missing_cols}")

    df_processed = df_processed[available_cols]


    print(f"Preprocessing complete. Selected features: {available_cols}. Shape: {df_processed.shape}")
    return df_processed

def split_data_sequential(df, train_split, val_split):
    """Splits data sequentially into train, validation, and test sets."""
    n = len(df)
    if n == 0:
        raise ValueError("Cannot split empty DataFrame.")
    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))

    # Ensure indices are valid
    if not (0 < train_end < n and train_end < val_end <= n):
         raise ValueError(f"Invalid split points: train_end={train_end}, val_end={val_end}, n={n}")

    train_data = df.iloc[:train_end]
    val_data = df.iloc[train_end:val_end]
    test_data = df.iloc[val_end:]

    print(f"Data split sequentially:")
    print(f"  Train: {train_data.shape[0]} samples ({train_data.index.min()} to {train_data.index.max()})")
    print(f"  Validation: {val_data.shape[0]} samples ({val_data.index.min()} to {val_data.index.max()})")
    print(f"  Test: {test_data.shape[0]} samples ({test_data.index.min()} to {test_data.index.max()})")

    return train_data, val_data, test_data

def scale_data(train_data, val_data, test_data):
    """Scales data using MinMaxScaler based on the training set."""
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit scaler only on training data features
    scaler.fit(train_data)

    # Transform train, validation, and test data
    train_scaled = scaler.transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)

    # Convert back to DataFrames
    train_scaled_df = pd.DataFrame(train_scaled, index=train_data.index, columns=train_data.columns)
    val_scaled_df = pd.DataFrame(val_scaled, index=val_data.index, columns=val_data.columns)
    test_scaled_df = pd.DataFrame(test_scaled, index=test_data.index, columns=test_data.columns)

    print("Data scaled using MinMaxScaler (fit on training data).")

    return train_scaled_df, val_scaled_df, test_scaled_df, scaler

def create_sequences(data, sequence_length, target_col_index):
    """Creates input sequences (X) and target values (y) for LSTM."""
    X, y = [], []
    data_np = data.values # Work with numpy array for efficiency
    if len(data_np) <= sequence_length:
        print(f"Warning: Data length ({len(data_np)}) is not greater than sequence length ({sequence_length}). Cannot create sequences.")
        return np.array(X), np.array(y)

    for i in range(len(data_np) - sequence_length):
        # Input sequence: sequence_length steps, all features
        X.append(data_np[i:(i + sequence_length), :])
        # Target value: the value of the target column at the next step
        y.append(data_np[i + sequence_length, target_col_index])
    return np.array(X), np.array(y)

# The __main__ block is for testing the script directly.
# It requires src to be in the Python path to import config.
if __name__ == '__main__':
    print("Testing data_preprocessing module...")
    # Ensure src is discoverable if running directly (e.g., run from project root: python -m src.data_preprocessing)
    try:
        raw_df = load_data(config.RAW_DATA_FILE)
        processed_df = preprocess_data(raw_df, config.DATE_COL, config.TARGET_COL, config.FEATURE_COLS, config.RESAMPLE_FREQ)
        train_df, val_df, test_df = split_data_sequential(processed_df, config.TRAIN_SPLIT, config.VALIDATION_SPLIT)
        train_scaled, val_scaled, test_scaled, scaler = scale_data(train_df, val_df, test_df)
        target_idx = train_scaled.columns.get_loc(config.TARGET_COL)
        X_train, y_train = create_sequences(train_scaled, config.SEQUENCE_LENGTH, target_idx)
        print("\nTest sequences created successfully.")
        print(f"  X_train shape: {X_train.shape}")

    except (ImportError, FileNotFoundError, ValueError, IndexError) as e:
        print(f"\nError during standalone test: {e}")
        print("Ensure you run this from the project root directory using 'python -m src.data_preprocessing'")
        print("And that 'data/wind_turbine_data.csv' exists.")