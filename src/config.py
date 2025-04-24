# filename: project/src/config.py
"""
# src/config.py
# --------------------
# Configuration parameters
# --------------------
"""
import os
import logging

# --- General Settings ---
APPROACH = 'direct' # Options: 'direct' or 'indirect' - Controls the workflow executed in main.py

# --- File Paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Assumes config.py is in src/

RAW_DATA_FILE = os.path.join(PROJECT_ROOT, 'data/wind_turbine_data.csv')
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data') # Directory for processed data
PROCESSED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'processed_data_hourly.csv') # Optional intermediate save

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results/')
MODEL_SAVE_DIR = os.path.join(RESULTS_DIR, 'saved_models/')
IMAGE_DIR = os.path.join(RESULTS_DIR, 'images/')
LOG_FILE = os.path.join(PROJECT_ROOT, 'project_workflow.log')

# --- Approach 1: Direct LSTM Output Paths ---
METRICS_LSTM_DIRECT_FILE = os.path.join(RESULTS_DIR, 'metrics_lstm_direct.csv')
BEST_LSTM_DIRECT_MODEL_FILE = os.path.join(MODEL_SAVE_DIR, 'best_lstm_direct_model.keras')
DIRECT_LEARNING_CURVES_PLOT = os.path.join(IMAGE_DIR, 'lstm_direct_learning_curves.png')
DIRECT_ACTUAL_VS_PREDICTED_PLOT = os.path.join(IMAGE_DIR, 'lstm_direct_actual_vs_predicted_test.png')

# --- Approach 2: Indirect (LSTM-Features -> RF-Power) Output Paths ---
METRICS_LSTM_INDIRECT_FILE = os.path.join(RESULTS_DIR, 'metrics_lstm_indirect.csv')
LSTM_FEATURE_MODEL_FILE = os.path.join(MODEL_SAVE_DIR, 'lstm_feature_model.keras') # LSTM predicts features
RF_POWER_MODEL_FILE = os.path.join(MODEL_SAVE_DIR, 'rf_power_model.joblib') # RF predicts power
INDIRECT_LSTM_LEARNING_CURVES_PLOT = os.path.join(IMAGE_DIR, 'lstm_indirect_learning_curves.png')
INDIRECT_RF_ACTUAL_VS_PREDICTED_PLOT = os.path.join(IMAGE_DIR, 'rf_indirect_actual_vs_predicted_test.png')

# --- Comparison Metrics ---
METRICS_COMPARISON_FILE = os.path.join(RESULTS_DIR, 'metrics_comparison.csv') # Compare final results

# --- Logging ---
LOG_LEVEL = logging.INFO # Level for logging (e.g., logging.DEBUG, logging.INFO)
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# --- Data Columns ---
DATE_COL: str = 'Date/Time'
TARGET_COL: str = 'LV ActivePower (kW)' # The final target we want to predict

# Features to initially consider for the models (used by preprocessor)
# Should include the TARGET_COL and any features needed by either approach
ALL_FEATURE_COLS: list[str] = [
    'LV ActivePower (kW)', # Target
    'Wind Speed (m/s)',
    'Theoretical_Power_Curve (KWh)',
    'Wind Direction (°)'
]

# --- Approach 2 Specific Features ---
# Features the LSTM will predict (these become inputs for the RF model)
# Based on report[cite: 49, 55]: wind speed, wind direction, and power curve
LSTM_FEATURE_TARGET_COLS: list[str] = [
    'Wind Speed (m/s)',
    'Theoretical_Power_Curve (KWh)',
    'Wind Direction (°)'
]
# Features used as input by the RF model (should match LSTM_FEATURE_TARGET_COLS)
RF_FEATURE_COLS: list[str] = LSTM_FEATURE_TARGET_COLS

# --- Preprocessing ---
RESAMPLE_FREQ: str = 'H' # Resampling frequency (e.g., 'H' for hourly, 'D' for daily)
# Number of past time steps to use as input features for predicting the next step (applies to both LSTMs)
SEQUENCE_LENGTH: int = 24 # Example: Use past 24 hours
# Proportions for splitting the data sequentially
TRAIN_SPLIT: float = 0.7
VALIDATION_SPLIT: float = 0.15 # Test split will be 1.0 - TRAIN_SPLIT - VALIDATION_SPLIT

# --- Modeling (LSTM - applies to both Direct Power and Indirect Feature LSTMs) ---
# Same config used for both LSTMs as per report [cite: 55]
LSTM_UNITS_L1: int = 100 # Number of units in the first LSTM layer
LSTM_UNITS_L2: int = 100 # Number of units in the second LSTM layer
EPOCHS: int = 50 # Maximum number of training epochs
BATCH_SIZE: int = 64 # Number of samples per gradient update
LEARNING_RATE: float = 0.001 # Learning rate for the Adam optimizer
# Number of epochs with no improvement on validation loss before stopping training
EARLY_STOPPING_PATIENCE: int = 10

# --- Modeling (Random Forest - for Approach 2) ---
# Hyperparameters from report [cite: 54]
RF_N_ESTIMATORS: int = 50
RF_MAX_DEPTH: int = 10
RF_MIN_SAMPLES_SPLIT: int = 10
RF_N_JOBS: int = -1 # Use all available CPU cores for training RF

# --- Evaluation Metrics ---
# List of metric names to calculate (must match functions/logic in evaluation.py)
# Apply to the final power prediction in both approaches
METRICS_TO_CALCULATE: list[str] = ['MAE', 'RMSE', 'R2', 'IA', 'SDE', 'MAPE'] # Added MAPE for comparison

# --- Plotting ---
PLOT_STYLE: str = 'seaborn-v0_8-darkgrid' # Matplotlib style for plots
FIG_SIZE: tuple[int, int] = (15, 6) # Default figure size (width, height)
FONT_SIZE: int = 12 # Base font size for plots
ACF_PACF_LAGS: int = 48 # Number of lags for ACF/PACF plots (e.g., 2 days for hourly data)