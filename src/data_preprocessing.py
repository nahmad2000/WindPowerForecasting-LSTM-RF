# filename: project/src/data_preprocessing.py
"""
# src/data_preprocessing.py
# --------------------
# Functions for data loading and preprocessing
# --------------------
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
from typing import Tuple, List, Optional, Union # For type hinting
import logging # Added logging
from tqdm import tqdm # Added tqdm for progress bar

# Get logger for this module
logger = logging.getLogger(__name__)

# Import configuration relative to src when run as part of the package
try:
    from . import config
except ImportError:
    logger.info("Attempting fallback import for config in data_preprocessing.py...")
    try:
       import config
    except ModuleNotFoundError as e:
        logger.error(f"Cannot import 'config'. Ensure 'src' is in PYTHONPATH or run scripts as modules. Details: {e}")
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Loads data from a CSV file."""
    if not os.path.exists(file_path):
        logger.error(f"Data file not found at the specified path: {file_path}")
        raise FileNotFoundError(f"Data file not found at the specified path: {file_path}")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}. Shape: {df.shape}")
        if df.empty:
            logger.warning("Loaded DataFrame is empty.")
        return df
    except Exception as e:
        logger.exception(f"Error loading data from {file_path}: {e}") # Use exception for traceback
        raise

def preprocess_data(df: pd.DataFrame, date_col: str, target_col: str,
                    all_feature_cols: List[str], resample_freq: Optional[str] = 'H'
                    ) -> pd.DataFrame:
    """Basic preprocessing: datetime index, resampling, feature selection, NaN handling."""
    if df.empty:
        logger.warning("Input DataFrame is empty. Skipping preprocessing.")
        return df

    logger.info("Starting data preprocessing...")
    df_processed = df.copy()

    # Convert to datetime and set index
    if date_col not in df_processed.columns:
         logger.error(f"Date column '{date_col}' not found in DataFrame columns: {df_processed.columns}")
         raise KeyError(f"Date column '{date_col}' not found in DataFrame columns: {df_processed.columns}")
    try:
        logger.debug(f"Converting column '{date_col}' to datetime...")
        # Try inferring first, then specific format if needed
        try:
            df_processed[date_col] = pd.to_datetime(df_processed[date_col], infer_datetime_format=True)
        except ValueError:
            logger.warning(f"Inferring datetime format for '{date_col}' failed. Trying '%d %m %Y %H:%M'. Adjust config.py if needed.")
            df_processed[date_col] = pd.to_datetime(df_processed[date_col], format='%d %m %Y %H:%M')

    except Exception as e:
        logger.exception(f"Error converting '{date_col}' to datetime: {e}")
        raise

    df_processed.set_index(date_col, inplace=True)
    logger.debug(f"Index set to '{date_col}'.")

    # Handle potential duplicates from setting index (keep first)
    if df_processed.index.has_duplicates:
        dup_count = df_processed.index.duplicated().sum()
        logger.warning(f"Duplicate indices found. Keeping first occurrence. Count: {dup_count}")
        df_processed = df_processed[~df_processed.index.duplicated(keep='first')]

    # Sort by index
    df_processed.sort_index(inplace=True)
    logger.debug("DataFrame sorted by index.")

    # Resample data
    if resample_freq:
        try:
            logger.info(f"Resampling data to '{resample_freq}' frequency using mean aggregation.")
            df_processed = df_processed.resample(resample_freq).mean()
            logger.debug(f"Resampling complete. New shape: {df_processed.shape}")
        except Exception as e:
            logger.exception(f"Error resampling data to '{resample_freq}': {e}")
            raise

    # Select relevant features + target (use ALL_FEATURE_COLS from config)
    cols_to_keep = list(set(all_feature_cols)) # Ensure target is also included if listed
    available_cols = [col for col in cols_to_keep if col in df_processed.columns]
    if len(available_cols) < len(cols_to_keep):
        missing_cols = set(cols_to_keep) - set(available_cols)
        logger.warning(f"Requested columns not found in data after resampling/initial load: {missing_cols}")
    if not available_cols:
        logger.critical("No specified feature/target columns found in the processed data.")
        raise ValueError("No specified feature/target columns found in the processed data.")

    df_processed = df_processed[available_cols]
    logger.info(f"Initial feature selection complete. Selected columns: {available_cols}.")


    # Handle missing values *after* feature selection and resampling
    initial_nans = df_processed.isnull().sum().sum()
    if initial_nans > 0:
        logger.info(f"Handling {initial_nans} missing value(s) (using ffill then bfill)...")
        df_processed.ffill(inplace=True)
        df_processed.bfill(inplace=True)
    final_nans = df_processed.isnull().sum().sum()
    if final_nans > 0:
        logger.warning(f"{final_nans} missing value(s) still present after ffill/bfill. Consider interpolation or dropping.")
        # Optionally drop rows with remaining NaNs
        # df_processed.dropna(inplace=True)
        # logger.warning(f"Dropped {final_nans} rows with remaining NaN values.")

    logger.info(f"Preprocessing complete. Final shape: {df_processed.shape}")

    # Save processed data (optional)
    try:
        os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
        save_path = config.PROCESSED_DATA_FILE
        df_processed.to_csv(save_path)
        logger.info(f"Preprocessed data saved to {save_path}")
    except Exception as e:
        logger.warning(f"Could not save preprocessed data: {e}")

    return df_processed


def split_data_sequential(df: pd.DataFrame, train_split: float, val_split: float
                         ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits data sequentially into train, validation, and test sets."""
    n = len(df)
    logger.info(f"Splitting {n} data points sequentially (Train: {train_split*100:.0f}%, Val: {val_split*100:.0f}%)...")
    if n == 0:
        logger.error("Cannot split empty DataFrame.")
        raise ValueError("Cannot split empty DataFrame.")

    train_end_idx = int(n * train_split)
    val_end_idx = int(n * (train_split + val_split))

    if not (0 < train_end_idx < n and train_end_idx < val_end_idx <= n):
         msg = f"Invalid split points calculated: train_end={train_end_idx}, val_end={val_end_idx} for data length n={n}. Check split ratios."
         logger.error(msg)
         raise ValueError(msg)

    train_data = df.iloc[:train_end_idx]
    val_data = df.iloc[train_end_idx:val_end_idx]
    test_data = df.iloc[val_end_idx:]

    logger.info("Data split results:")
    if not train_data.empty: logger.info(f"  Train:      {train_data.shape[0]:>6} samples ({train_data.index.min()} to {train_data.index.max()})")
    else: logger.warning("  Train:      0 samples")
    if not val_data.empty: logger.info(f"  Validation: {val_data.shape[0]:>6} samples ({val_data.index.min()} to {val_data.index.max()})")
    else: logger.warning("  Validation: 0 samples")
    if not test_data.empty: logger.info(f"  Test:       {test_data.shape[0]:>6} samples ({test_data.index.min()} to {test_data.index.max()})")
    else: logger.warning("  Test:       0 samples")

    if not train_data.empty and not val_data.empty: assert train_data.index.max() < val_data.index.min()
    if not val_data.empty and not test_data.empty: assert val_data.index.max() < test_data.index.min()

    return train_data, val_data, test_data

def scale_data(train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame
               ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """Scales data using MinMaxScaler based on the training set."""
    logger.info("Scaling data using MinMaxScaler (fitting on training data)...")
    if train_data.empty:
        logger.error("Cannot scale data: Training data is empty.")
        raise ValueError("Cannot scale data: Training data is empty.")

    scaler = MinMaxScaler(feature_range=(0, 1))
    original_cols = train_data.columns
    original_train_idx = train_data.index
    original_val_idx = val_data.index
    original_test_idx = test_data.index

    logger.debug(f"Fitting scaler on {train_data.shape[0]} training samples for columns: {list(original_cols)}")
    scaler.fit(train_data)

    logger.debug("Transforming train, validation, and test sets.")
    train_scaled = scaler.transform(train_data)
    val_scaled = scaler.transform(val_data) if not val_data.empty else np.array([]).reshape(0, train_data.shape[1])
    test_scaled = scaler.transform(test_data) if not test_data.empty else np.array([]).reshape(0, train_data.shape[1])

    train_scaled_df = pd.DataFrame(train_scaled, index=original_train_idx, columns=original_cols)
    val_scaled_df = pd.DataFrame(val_scaled, index=original_val_idx, columns=original_cols)
    test_scaled_df = pd.DataFrame(test_scaled, index=original_test_idx, columns=original_cols)

    logger.info("Data scaling complete.")
    return train_scaled_df, val_scaled_df, test_scaled_df, scaler


def create_sequences_lstm(data: pd.DataFrame, sequence_length: int,
                          target_col_names: Union[str, List[str]],
                          feature_col_names: Optional[List[str]] = None
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates input sequences (X) and target values (y) for LSTM.
    Handles both univariate and multivariate targets.

    Args:
        data (pd.DataFrame): DataFrame containing scaled features and targets.
                             Index should be time-based.
        sequence_length (int): Number of past time steps for input sequence.
        target_col_names (Union[str, List[str]]): Name(s) of the target column(s) to predict.
        feature_col_names (Optional[List[str]]): Names of columns to use as input features.
                                                 If None, uses all columns in data.

    Returns:
        Tuple[np.ndarray, np.ndarray]: X sequences (samples, time_steps, features)
                                       y targets (samples, num_targets)
    """
    logger.info(f"Creating sequences with length {sequence_length}...")
    X, y = [], []
    if data.empty:
        logger.warning("Input data for sequence creation is empty. Returning empty arrays.")
        return np.array(X), np.array(y)

    if isinstance(target_col_names, str):
        target_col_names = [target_col_names]

    if feature_col_names is None:
        feature_col_names = data.columns.tolist()
        logger.debug(f"Using all columns as features: {feature_col_names}")

    # Ensure all target and feature columns exist
    missing_targets = [col for col in target_col_names if col not in data.columns]
    missing_features = [col for col in feature_col_names if col not in data.columns]
    if missing_targets:
        msg = f"Target column(s) not found in data: {missing_targets}. Available: {data.columns}"
        logger.error(msg)
        raise KeyError(msg)
    if missing_features:
        msg = f"Feature column(s) not found in data: {missing_features}. Available: {data.columns}"
        logger.error(msg)
        raise KeyError(msg)

    target_col_indices = [data.columns.get_loc(col) for col in target_col_names]
    feature_col_indices = [data.columns.get_loc(col) for col in feature_col_names]

    data_np = data.values
    if len(data_np) <= sequence_length:
        logger.warning(f"Data length ({len(data_np)}) is not greater than sequence length ({sequence_length}). Cannot create sequences.")
        return np.array(X), np.array(y)

    logger.debug(f"Generating sequences from {len(data_np)} time steps...")
    for i in tqdm(range(len(data_np) - sequence_length), desc="Creating LSTM sequences", unit="seq", leave=False):
        # Input features (X): Select specified feature columns
        X.append(data_np[i:(i + sequence_length), feature_col_indices])
        # Target features (y): Select specified target columns at the step *after* the sequence
        y.append(data_np[i + sequence_length, target_col_indices])

    X_np = np.array(X)
    y_np = np.array(y)

    # Squeeze y if it's univariate target for compatibility with standard Keras loss functions
    if len(target_col_names) == 1:
        y_np = y_np.squeeze(axis=-1)

    logger.info(f"Sequence creation complete. X shape: {X_np.shape}, y shape: {y_np.shape}")
    return X_np, y_np

def prepare_data_for_rf(data: pd.DataFrame,
                         feature_cols: List[str],
                         target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepares data for scikit-learn models like Random Forest.
    Selects features and target, handles potential NaNs from alignment.

    Args:
        data (pd.DataFrame): Scaled DataFrame (e.g., train_scaled_df).
        feature_cols (List[str]): List of column names to use as features.
        target_col (str): Name of the target column.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: X_rf (features), y_rf (target)
    """
    logger.info("Preparing data for Random Forest...")
    if data.empty:
        logger.warning("Input data for RF preparation is empty. Returning empty structures.")
        return pd.DataFrame(), pd.Series(dtype=float) # Ensure correct types for empty return

    missing_features = [col for col in feature_cols if col not in data.columns]
    if missing_features:
         msg = f"RF Feature column(s) not found in data: {missing_features}. Available: {data.columns}"
         logger.error(msg)
         raise KeyError(msg)
    if target_col not in data.columns:
        msg = f"RF Target column '{target_col}' not found in data. Available: {data.columns}"
        logger.error(msg)
        raise KeyError(msg)

    X_rf = data[feature_cols].copy()
    y_rf = data[target_col].copy()

    # Drop rows with NaNs that might arise from alignment or earlier steps
    initial_len = len(X_rf)
    combined = pd.concat([X_rf, y_rf], axis=1)
    combined.dropna(inplace=True)
    rows_dropped = initial_len - len(combined)
    if rows_dropped > 0:
        logger.warning(f"Dropped {rows_dropped} rows with NaN values during RF data preparation.")

    X_rf = combined[feature_cols]
    y_rf = combined[target_col]

    logger.info(f"RF data preparation complete. X_rf shape: {X_rf.shape}, y_rf shape: {y_rf.shape}")
    return X_rf, y_rf


if __name__ == '__main__':
    # Basic logging setup for standalone test
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("Testing data_preprocessing module...")
    try:
        raw_df = load_data(config.RAW_DATA_FILE)
        processed_df = preprocess_data(raw_df, config.DATE_COL, config.TARGET_COL, config.ALL_FEATURE_COLS, config.RESAMPLE_FREQ)
        train_df, val_df, test_df = split_data_sequential(processed_df, config.TRAIN_SPLIT, config.VALIDATION_SPLIT)
        train_scaled, val_scaled, test_scaled, scaler = scale_data(train_df, val_df, test_df)

        logger.info("\n--- Testing LSTM Sequence Creation (Univariate Target) ---")
        X_train_lstm_uni, y_train_lstm_uni = create_sequences_lstm(
            train_scaled, config.SEQUENCE_LENGTH, config.TARGET_COL, config.ALL_FEATURE_COLS
        )
        logger.info(f"  Univariate LSTM: X_train shape: {X_train_lstm_uni.shape}, y_train shape: {y_train_lstm_uni.shape}")
        assert len(y_train_lstm_uni.shape) == 1

        logger.info("\n--- Testing LSTM Sequence Creation (Multivariate Target) ---")
        X_train_lstm_multi, y_train_lstm_multi = create_sequences_lstm(
            train_scaled, config.SEQUENCE_LENGTH, config.LSTM_FEATURE_TARGET_COLS, config.ALL_FEATURE_COLS
        )
        logger.info(f"  Multivariate LSTM: X_train shape: {X_train_lstm_multi.shape}, y_train shape: {y_train_lstm_multi.shape}")
        assert len(y_train_lstm_multi.shape) == 2
        assert y_train_lstm_multi.shape[1] == len(config.LSTM_FEATURE_TARGET_COLS)

        logger.info("\n--- Testing RF Data Preparation ---")
        X_train_rf, y_train_rf = prepare_data_for_rf(
            train_scaled, config.RF_FEATURE_COLS, config.TARGET_COL
        )
        logger.info(f"  RF Data: X_train shape: {X_train_rf.shape}, y_train shape: {y_train_rf.shape}")
        assert isinstance(X_train_rf, pd.DataFrame)
        assert isinstance(y_train_rf, pd.Series)
        assert not X_train_rf.isnull().values.any()
        assert not y_train_rf.isnull().values.any()

        logger.info("\nModule test sequences and RF data created successfully.")

    except (ImportError, FileNotFoundError, ValueError, KeyError, IndexError, ModuleNotFoundError, AssertionError) as e:
        logger.exception(f"\nError during standalone test: {e}") # Log exception traceback
        logger.info("\nHints for troubleshooting:")
        logger.info("  - Ensure you run this from the project root directory using 'python -m src.data_preprocessing'")
        logger.info(f"  - Ensure the raw data file exists at the location specified in config.py: '{config.RAW_DATA_FILE}'")
        logger.info(f"  - Ensure column names in config.py ({config.DATE_COL}, {config.TARGET_COL}, {config.ALL_FEATURE_COLS}) match your CSV.")
        logger.info(f"  - Ensure split ratios ({config.TRAIN_SPLIT}, {config.VALIDATION_SPLIT}) are valid (sum < 1.0).")
        logger.info(f"  - Ensure sequence length ({config.SEQUENCE_LENGTH}) is less than the length of train/val/test sets after preprocessing.")
        logger.info(f"  - Ensure target/feature column names for Approach 2 ({config.LSTM_FEATURE_TARGET_COLS}, {config.RF_FEATURE_COLS}) exist in the preprocessed data.")