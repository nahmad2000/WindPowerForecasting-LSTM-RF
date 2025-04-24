# src/evaluation.py
# --------------------
# Functions for evaluating model performance
# --------------------

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from typing import List, Dict, Tuple
import logging # Added logging

# Get logger for this module
logger = logging.getLogger(__name__)

# Import configuration relative to src
try:
    from . import config
except ImportError:
    logger.info("Attempting fallback import for config in evaluation.py...")
    try:
       import config
    except ModuleNotFoundError as e:
        logger.error(f"Cannot import 'config'. Ensure 'src' is in PYTHONPATH or run scripts as modules. Details: {e}")
        raise

# --- Individual Metric Functions ---

def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates MAPE, ignoring points where y_true is zero."""
    y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()
    if y_true.shape[0] == 0: return np.nan

    mask = y_true != 0
    if not np.all(mask):
        zero_count = np.sum(~mask)
        # logger.debug(f"Zero values ({zero_count}) found in y_true for MAPE calculation. These points will be ignored.")
        pass # Reduce log verbosity
    if np.sum(mask) == 0:
        logger.warning("All true values are zero or masked; cannot calculate MAPE.")
        return np.nan

    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates RMSE."""
    y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()
    if y_true.shape[0] == 0: return np.nan
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def index_of_agreement(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates the Index of Agreement (IA)."""
    y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()
    if y_true.shape[0] == 0: return np.nan
    mean_true = np.mean(y_true)
    numerator = np.sum((y_true - y_pred)**2)
    denominator = np.sum((np.abs(y_pred - mean_true) + np.abs(y_true - mean_true))**2)

    if denominator == 0:
        logger.warning("Denominator is zero in IA calculation.")
        return 1.0 if numerator == 0 else np.nan
    ia = 1.0 - (numerator / denominator)
    return float(ia)

def standard_deviation_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates the Standard Deviation of the Errors (SDE)."""
    y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()
    if y_true.shape[0] == 0: return np.nan
    errors = y_true - y_pred
    return float(np.std(errors))

# --- Core Evaluation Functions ---

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, metrics_list: List[str]) -> Dict[str, float]:
    """Calculates a dictionary of specified performance metrics."""
    results: Dict[str, float] = {}
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    if y_true.shape[0] == 0 or y_pred.shape[0] == 0:
         logger.warning("Empty arrays passed to calculate_metrics. Returning NaNs.")
         return {metric: np.nan for metric in metrics_list}

    if y_true.shape != y_pred.shape:
        msg = f"Input arrays must have the same shape. Got {y_true.shape} and {y_pred.shape}"
        logger.error(msg)
        raise ValueError(msg)

    logger.debug(f"Calculating metrics: {metrics_list}")
    metric_fns = {
        'MAE': mean_absolute_error,
        'MAPE': mean_absolute_percentage_error,
        'RMSE': root_mean_squared_error,
        'R2': r2_score,
        'IA': index_of_agreement,
        'SDE': standard_deviation_error
    }

    for metric in metrics_list:
        if metric in metric_fns:
            try:
                results[metric] = metric_fns[metric](y_true, y_pred)
                logger.debug(f"  {metric}: {results[metric]:.4f}")
            except Exception as e:
                logger.error(f"Error calculating {metric}: {e}")
                results[metric] = np.nan
        else:
             logger.warning(f"Metric '{metric}' requested but not implemented/recognized.")
             results[metric] = np.nan

    return results

def evaluate_model(model: tf.keras.Model, X_data: np.ndarray, y_data_scaled: np.ndarray,
                   scaler: MinMaxScaler, target_col_index: int,
                   metrics_list: List[str]
                   ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Makes predictions (scaled), inverse transforms, and evaluates the model."""

    logger.info(f"Starting evaluation on {X_data.shape[0]} samples...")
    if X_data.shape[0] == 0 or y_data_scaled.shape[0] == 0:
         logger.warning("Empty X_data or y_data_scaled passed to evaluate_model. Returning empty results.")
         return np.array([]), np.array([]), {metric: np.nan for metric in metrics_list}
    if X_data.shape[0] != y_data_scaled.shape[0]:
        msg = f"Mismatch shapes between X_data ({X_data.shape[0]}) and y_data_scaled ({y_data_scaled.shape[0]})"
        logger.error(msg)
        raise ValueError(msg)

    logger.info(f"Predicting with model...")
    y_pred_scaled = model.predict(X_data)
    logger.debug(f"Prediction shape (scaled): {y_pred_scaled.shape}")

    # --- Inverse transform ---
    logger.info("Inverse transforming predictions and actual values...")
    y_pred_scaled_flat = y_pred_scaled.flatten()
    y_data_scaled_flat = y_data_scaled.flatten()

    n_features = scaler.n_features_in_
    if target_col_index >= n_features:
         msg = f"target_col_index ({target_col_index}) is out of bounds for the scaler's {n_features} features."
         logger.error(msg)
         raise IndexError(msg)

    try:
        # Predictions
        dummy_pred_array = np.zeros((len(y_pred_scaled_flat), n_features))
        dummy_pred_array[:, target_col_index] = y_pred_scaled_flat
        y_pred_inverse = scaler.inverse_transform(dummy_pred_array)[:, target_col_index]
        logger.debug(f"Inverse transformed prediction shape: {y_pred_inverse.shape}")

        # Actual values
        dummy_actual_array = np.zeros((len(y_data_scaled_flat), n_features))
        dummy_actual_array[:, target_col_index] = y_data_scaled_flat
        y_true_inverse = scaler.inverse_transform(dummy_actual_array)[:, target_col_index]
        logger.debug(f"Inverse transformed actual shape: {y_true_inverse.shape}")
    except Exception as e:
        logger.exception("Error during inverse transformation:")
        raise

    # --- Calculate metrics ---
    logger.info("Calculating performance metrics on inverse transformed data...")
    metrics = calculate_metrics(y_true_inverse, y_pred_inverse, metrics_list)
    logger.info(f"Evaluation complete. Metrics: {metrics}")

    return y_true_inverse, y_pred_inverse, metrics


if __name__ == '__main__':
    # Basic logging setup for standalone test
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("Testing evaluation module...")
    y_true_test = np.array([100, 110, 120, 105, 95, 0, 115])
    y_pred_test = np.array([102, 108, 118, 107, 98, 5, 112])

    try:
        metrics_to_test = config.METRICS_TO_CALCULATE
    except NameError:
        logger.warning("config not found during test, using default metrics list.")
        metrics_to_test = ['MAE', 'RMSE', 'R2', 'MAPE']
    except Exception as e:
        logger.error(f"Error accessing config during test: {e}")
        metrics_to_test = ['MAE', 'RMSE']

    metrics_result = calculate_metrics(y_true_test, y_pred_test, metrics_to_test)
    logger.info("\nCalculated metrics on dummy data:")
    if metrics_result:
        for name, value in metrics_result.items():
            value_str = f"{value:.4f}" if not np.isnan(value) else "NaN"
            logger.info(f"  {name}: {value_str}")
    else:
        logger.info("  No metrics calculated.")

    logger.info("\n(Skipping full test of 'evaluate_model' in standalone execution)")
    logger.info("Evaluation module testing finished.")