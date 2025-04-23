# src/evaluation.py
# --------------------
# Functions for evaluating model performance
# --------------------

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler # Needed for type hint and inverse transform logic
import tensorflow as tf # Needed for type hinting the model

# Import configuration relative to src
try:
    from . import config
except ImportError:
    # Fallback for running the script directly
    import config

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates MAPE, ignoring points where y_true is zero."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if not np.all(mask):
      # print("Warning: Zero values found in y_true for MAPE calculation. These points will be ignored.")
      pass # Avoid printing too many warnings during evaluation loops
    if np.sum(mask) == 0:
        return np.nan # Avoid division by zero if all true values are zero or masked
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def index_of_agreement(y_true, y_pred):
    """Calculates the Index of Agreement (IA)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if len(y_true) == 0: return np.nan
    mean_true = np.mean(y_true)
    numerator = np.sum((y_true - y_pred)**2)
    denominator = np.sum((np.abs(y_pred - mean_true) + np.abs(y_true - mean_true))**2)
    if denominator == 0:
        # This happens if all predictions and true values are the same as the mean
        return 1.0 if numerator == 0 else np.nan
    return 1.0 - (numerator / denominator)

def standard_deviation_error(y_true, y_pred):
    """Calculates the Standard Deviation of Errors (SDE)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if len(y_true) == 0: return np.nan
    errors = y_true - y_pred
    return np.std(errors)

def calculate_metrics(y_true, y_pred, metrics_list):
    """Calculates a dictionary of specified performance metrics."""
    results = {}
    if len(y_true) == 0 or len(y_pred) == 0:
         print("Warning: Empty arrays passed to calculate_metrics.")
         return {metric: np.nan for metric in metrics_list} # Return NaNs for all requested metrics

    # Ensure inputs are numpy arrays
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    if len(y_true) != len(y_pred):
        raise ValueError(f"Input arrays must have the same length. Got {len(y_true)} and {len(y_pred)}")


    if 'MAE' in metrics_list:
        results['MAE'] = mean_absolute_error(y_true, y_pred)
    if 'MAPE' in metrics_list:
        results['MAPE'] = mean_absolute_percentage_error(y_true, y_pred)
    if 'RMSE' in metrics_list:
        results['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
    if 'R2' in metrics_list:
        results['R2'] = r2_score(y_true, y_pred)
    if 'IA' in metrics_list:
        results['IA'] = index_of_agreement(y_true, y_pred)
    if 'SDE' in metrics_list:
        results['SDE'] = standard_deviation_error(y_true, y_pred)

    # Add fallback for any requested but uncalculated metric
    for metric in metrics_list:
        if metric not in results:
            results[metric] = np.nan

    return results

def evaluate_model(model: tf.keras.Model, X_data: np.ndarray, y_data_scaled: np.ndarray,
                   scaler: MinMaxScaler, target_col_index: int,
                   metrics_list: list) -> tuple[np.ndarray, np.ndarray, dict]:
    """Makes predictions and evaluates the model on a given dataset.

    Args:
        model: Trained Keras model.
        X_data: Input sequence data (scaled).
        y_data_scaled: True target values (scaled, 1D array).
        scaler: Fitted Scikit-learn scaler object.
        target_col_index: The column index of the target variable in the original (pre-scaled) data.
        metrics_list: A list of strings naming the metrics to calculate.

    Returns:
        A tuple containing:
        - y_true_inverse: True target values (inverse transformed).
        - y_pred_inverse: Predicted target values (inverse transformed).
        - metrics: Dictionary of calculated performance metrics.
    """
    if X_data.shape[0] == 0:
         print("Warning: Empty X_data passed to evaluate_model.")
         return np.array([]), np.array([]), {metric: np.nan for metric in metrics_list}


    # Predict using the model
    y_pred_scaled = model.predict(X_data)

    # Inverse transform predictions
    # Ensure y_pred_scaled is flattened if model outputs shape (n, 1)
    y_pred_scaled_flat = y_pred_scaled.flatten()

    # Create a dummy array matching the scaler's expected input shape (n_samples, n_features)
    n_features = scaler.n_features_in_
    dummy_pred_array = np.zeros((len(y_pred_scaled_flat), n_features))
    dummy_pred_array[:, target_col_index] = y_pred_scaled_flat
    y_pred_inverse = scaler.inverse_transform(dummy_pred_array)[:, target_col_index]

    # Inverse transform actual scaled target values (which are already 1D)
    y_data_scaled_flat = y_data_scaled.flatten()
    dummy_actual_array = np.zeros((len(y_data_scaled_flat), n_features))
    dummy_actual_array[:, target_col_index] = y_data_scaled_flat
    y_true_inverse = scaler.inverse_transform(dummy_actual_array)[:, target_col_index]

    # Calculate metrics
    metrics = calculate_metrics(y_true_inverse, y_pred_inverse, metrics_list)

    return y_true_inverse, y_pred_inverse, metrics


if __name__ == '__main__':
    print("Testing evaluation module...")
    y_true_test = np.array([100, 110, 120, 105, 95, 0, 115]) # Added a zero for MAPE test
    y_pred_test = np.array([102, 108, 118, 107, 98, 5, 112])

    try:
        metrics = calculate_metrics(y_true_test, y_pred_test, config.METRICS_TO_CALCULATE)
        print("\nCalculated metrics on dummy data:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")
    except NameError:
        print("Warning: config not found, using default metrics for test.")
        metrics = calculate_metrics(y_true_test, y_pred_test, ['MAE', 'RMSE'])
        print("\nCalculated metrics on dummy data (default):")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")


    print("\nTesting specific functions:")
    print(f"  MAPE: {mean_absolute_percentage_error(y_true_test, y_pred_test):.4f}%")
    print(f"  IA: {index_of_agreement(y_true_test, y_pred_test):.4f}")
    print(f"  SDE: {standard_deviation_error(y_true_test, y_pred_test):.4f}")

    # Note: evaluate_model requires a trained model, scaler, and prepared data.
    # Skipping its full test here as it depends on other modules.
    print("\nEvaluation module testing finished.")