# src/config.py
# --------------------
# Configuration parameters
# --------------------
import os

# Data files
RAW_DATA_FILE = 'data/wind_turbine_data.csv'
PROCESSED_DATA_FILE = 'data/processed_data_hourly.csv' # Optional intermediate save

# Results/Model files
RESULTS_DIR = 'results/'
METRICS_LSTM_DIRECT_FILE = os.path.join(RESULTS_DIR, 'metrics_lstm_direct.csv')
METRICS_COMPARISON_FILE = os.path.join(RESULTS_DIR, 'metrics_comparison.csv')
MODEL_SAVE_DIR = os.path.join(RESULTS_DIR, 'saved_models/')
BEST_LSTM_MODEL_FILE = os.path.join(MODEL_SAVE_DIR, 'best_lstm_direct_model.h5')

# Image files
IMAGE_DIR = 'images/'

# Data Columns
DATE_COL = 'Date/Time'
TARGET_COL = 'LV ActivePower (kW)'
FEATURE_COLS = ['LV ActivePower (kW)', 'Wind Speed (m/s)', 'Theoretical_Power_Curve (KWh)', 'Wind Direction (°)'] # Features to use

# Preprocessing
RESAMPLE_FREQ = 'H' # Hourly resampling
SEQUENCE_LENGTH = 24 # Number of past hours to use for prediction (example)
TRAIN_SPLIT = 0.7
VALIDATION_SPLIT = 0.15 # Test split will be 1 - TRAIN_SPLIT - VALIDATION_SPLIT

# Modeling (LSTM Example)
LSTM_UNITS_L1 = 100
LSTM_UNITS_L2 = 100
EPOCHS = 50 # Adjust as needed
BATCH_SIZE = 64
LEARNING_RATE = 0.001 # Example for Adam optimizer
EARLY_STOPPING_PATIENCE = 10 # Patience for EarlyStopping callback

# Evaluation Metrics
METRICS_TO_CALCULATE = ['MAE', 'MAPE', 'RMSE', 'R2', 'IA', 'SDE']

# Plotting
PLOT_STYLE = 'seaborn-v0_8_darkgrid'
FIG_SIZE = (15, 6)
FONT_SIZE = 12