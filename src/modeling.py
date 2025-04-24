# filename: project/src/modeling.py
"""
# src/modeling.py
# --------------------
# Functions for building and training models (LSTM and Random Forest)
# --------------------
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestRegressor
import joblib # For saving/loading sklearn models
import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union # For type hinting
import logging # Added logging
import io # Added for capturing model summary

# Get logger for this module
logger = logging.getLogger(__name__)

# Import configuration relative to src
try:
    from . import config
except ImportError:
    logger.info("Attempting fallback import for config in modeling.py...")
    try:
       import config
    except ModuleNotFoundError as e:
        logger.error(f"Cannot import 'config'. Ensure 'src' is in PYTHONPATH or run scripts as modules. Details: {e}")
        raise

# --- LSTM Model Functions ---

def build_lstm_forecasting_model(input_shape: Tuple[int, int],
                                 output_units: int = 1,
                                 lstm_units_l1: int = 50,
                                 lstm_units_l2: int = 50,
                                 learning_rate: float = 0.001,
                                 model_name: str = "LSTM_Forecasting_Model") -> tf.keras.Model:
    """
    Builds a sequential LSTM model for forecasting.
    Handles both univariate (output_units=1) and multivariate (>1) targets.

    Args:
        input_shape (Tuple[int, int]): Shape of input data (sequence_length, n_features).
        output_units (int): Number of output units (number of target variables). Defaults to 1.
        lstm_units_l1 (int): Number of units in the first LSTM layer.
        lstm_units_l2 (int): Number of units in the second LSTM layer.
        learning_rate (float): Learning rate for the Adam optimizer.
        model_name (str): Name for the Keras model.

    Returns:
        tf.keras.Model: Compiled LSTM model.
    """
    logger.info(f"Building LSTM model '{model_name}'...")
    if not isinstance(input_shape, tuple) or len(input_shape) != 2:
        msg = f"input_shape must be a tuple of length 2 (sequence_length, n_features), got {input_shape}"
        logger.error(msg)
        raise ValueError(msg)
    if input_shape[0] <= 0 or input_shape[1] <= 0:
         msg = f"sequence_length and n_features in input_shape must be positive, got {input_shape}"
         logger.error(msg)
         raise ValueError(msg)
    if output_units <= 0:
        msg = f"output_units must be positive, got {output_units}"
        logger.error(msg)
        raise ValueError(msg)

    logger.debug(f"Model input shape: {input_shape}")
    logger.debug(f"Model output units: {output_units}")
    logger.debug(f"LSTM Layer 1 Units: {lstm_units_l1}, LSTM Layer 2 Units: {lstm_units_l2}")

    model = Sequential(name=model_name)
    # Input layer + First LSTM layer
    model.add(LSTM(lstm_units_l1, return_sequences=True, input_shape=input_shape, name="LSTM_Layer_1"))
    # Optional Dropout
    # model.add(Dropout(0.2, name="Dropout_1"))
    # Second LSTM layer
    model.add(LSTM(lstm_units_l2, return_sequences=False, name="LSTM_Layer_2"))
    # Optional Dropout
    # model.add(Dropout(0.2, name="Dropout_2"))
    # Output layer - number of units depends on target variables
    model.add(Dense(output_units, name="Output_Layer"))

    optimizer = Adam(learning_rate=learning_rate)
    # Use 'mae' as metric since it's often reported and interpretable
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    logger.info(f"LSTM Model '{model_name}' compiled successfully (Optimizer=Adam, Loss=MSE, Metrics=[MAE]).")

    # Log model summary
    summary_stream = io.StringIO()
    model.summary(print_fn=lambda x: summary_stream.write(x + '\n'), line_length=100)
    model_summary = summary_stream.getvalue()
    summary_stream.close()
    logger.info(f"Model Summary:\n{model_summary}")

    return model

# Custom callback using logging (can be used optionally)
class LoggingProgressCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        log_msg = f"Epoch {epoch+1}/{self.params['epochs']} - loss: {logs.get('loss'):.4f} - mae: {logs.get('mae'):.4f}"
        if 'val_loss' in logs:
            log_msg += f" - val_loss: {logs.get('val_loss'):.4f} - val_mae: {logs.get('val_mae'):.4f}"
        logger.info(log_msg)

def train_lstm_model(model: tf.keras.Model, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray, epochs: int, batch_size: int,
                     model_checkpoint_path: str, early_stopping_patience: int
                     ) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """Trains the LSTM model with validation data and saves the best weights."""

    logger.info(f"Preparing training for LSTM model: {model.name}")
    if X_train.shape[0] == 0 or y_train.shape[0] == 0:
        msg = "Cannot train model: Training data sequences are empty."
        logger.error(msg)
        raise ValueError(msg)
    if X_val.shape[0] == 0 or y_val.shape[0] == 0:
        logger.warning("Validation data sequences are empty. ModelCheckpoint/EarlyStopping based on 'val_loss' might default to 'loss'.")
        monitor_metric = 'loss' # Fallback metric if validation data is absent
        # Note: Early stopping might not be very effective without validation data.
    else:
        monitor_metric = 'val_loss'

    # Ensure the directory for saving the model exists
    try:
        model_dir = os.path.dirname(model_checkpoint_path)
        os.makedirs(model_dir, exist_ok=True)
        logger.debug(f"Ensured directory exists for model saving: {model_dir}")
    except OSError as e:
        logger.warning(f"Could not create directory for saving model: {e}. ModelCheckpoint might fail.")

    # --- Callbacks ---
    logger.info(f"Setting up training callbacks (Monitor: '{monitor_metric}')...")
    checkpoint = ModelCheckpoint(
        filepath=model_checkpoint_path, monitor=monitor_metric, save_best_only=True,
        save_weights_only=False, mode='min', verbose=0 # verbose=0 to avoid duplicate console output
    )
    early_stopping = EarlyStopping(
        monitor=monitor_metric, patience=early_stopping_patience,
        restore_best_weights=True, verbose=1
    )
    # logging_callback = LoggingProgressCallback() # Optional custom logger

    logger.info(f"Starting model training: Max Epochs={epochs}, Batch Size={batch_size}, EarlyStopping Patience={early_stopping_patience}")
    logger.info(f"Best model based on '{monitor_metric}' will be saved to: {model_checkpoint_path}")

    # Pass validation data even if empty; Keras handles it.
    history = model.fit(
        X_train, y_train,
        epochs=epochs, batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stopping], # Add logging_callback here if preferred
        verbose=2 # Keras verbose=2 gives one line per epoch
    )
    logger.info("Model training finished.")

    # --- Load Best Model State ---
    # EarlyStopping with restore_best_weights=True should have the best weights loaded
    # in the returned 'model' object. However, reloading from the saved file is safer
    # to ensure we have the state that was explicitly saved as 'best'.
    logger.info(f"Attempting to load best model state from {model_checkpoint_path}...")
    if os.path.exists(model_checkpoint_path):
        try:
            best_model = load_model(model_checkpoint_path)
            logger.info("Successfully loaded best model state saved during training.")
            return best_model, history
        except Exception as e:
            logger.warning(f"Could not load the best model from {model_checkpoint_path}. Error: {e}")
            logger.warning("Proceeding with the model state at the end of training (restored best weights if EarlyStopping triggered).")
            return model, history
    else:
        logger.warning(f"Model checkpoint file not found at {model_checkpoint_path}. Returning model state from end of training.")
        return model, history


# --- Random Forest Model Functions (for Approach 2) ---

def build_rf_model(n_estimators: int, max_depth: int, min_samples_split: int, n_jobs: int = -1, random_state: int = 42) -> RandomForestRegressor:
    """
    Builds a RandomForestRegressor model.

    Args:
        n_estimators (int): Number of trees in the forest.
        max_depth (int): Maximum depth of the trees.
        min_samples_split (int): Minimum number of samples required to split an internal node.
        n_jobs (int): Number of jobs to run in parallel. -1 means using all processors.
        random_state (int): Controls randomness for reproducibility.

    Returns:
        RandomForestRegressor: Untrained scikit-learn RandomForestRegressor model.
    """
    logger.info("Building Random Forest Regressor model...")
    logger.debug(f"Parameters: n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}, n_jobs={n_jobs}, random_state={random_state}")
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=n_jobs,
        # oob_score=True # Can be useful for evaluation without a separate validation set
    )
    logger.info("Random Forest model built.")
    return rf_model

def train_rf_model(model: RandomForestRegressor, X_train: Union[pd.DataFrame, np.ndarray], y_train: Union[pd.Series, np.ndarray],
                   model_save_path: str) -> RandomForestRegressor:
    """
    Trains the RandomForestRegressor model and saves it using joblib.

    Args:
        model (RandomForestRegressor): The scikit-learn model instance to train.
        X_train (Union[pd.DataFrame, np.ndarray]): Training features.
        y_train (Union[pd.Series, np.ndarray]): Training target variable.
        model_save_path (str): Path to save the trained model (.joblib).

    Returns:
        RandomForestRegressor: The trained model.
    """
    logger.info("Starting Random Forest model training...")
    if isinstance(X_train, pd.DataFrame):
        logger.debug(f"Training RF on DataFrame with shape {X_train.shape} and columns {X_train.columns.tolist()}")
    else:
        logger.debug(f"Training RF on NumPy array with shape {X_train.shape}")

    if len(X_train) == 0 or len(y_train) == 0:
        msg = "Cannot train RF model: Training data is empty."
        logger.error(msg)
        raise ValueError(msg)
    if len(X_train) != len(y_train):
        msg = f"Mismatch between X_train ({len(X_train)}) and y_train ({len(y_train)}) lengths."
        logger.error(msg)
        raise ValueError(msg)

    model.fit(X_train, y_train)
    logger.info("Random Forest model training complete.")
    # logger.info(f"OOB Score: {model.oob_score_:.4f}") # If oob_score=True

    # --- Save the trained model ---
    try:
        model_dir = os.path.dirname(model_save_path)
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model, model_save_path)
        logger.info(f"Trained Random Forest model saved to: {model_save_path}")
    except Exception as e:
        logger.error(f"Error saving Random Forest model to {model_save_path}: {e}")
        # Decide whether to raise the error or just warn
        # raise

    return model

def load_rf_model(model_load_path: str) -> Optional[RandomForestRegressor]:
    """
    Loads a trained Random Forest model from a joblib file.

    Args:
        model_load_path (str): Path to the saved model file (.joblib).

    Returns:
        Optional[RandomForestRegressor]: The loaded model, or None if loading fails.
    """
    logger.info(f"Loading Random Forest model from: {model_load_path}")
    if not os.path.exists(model_load_path):
        logger.error(f"Model file not found at {model_load_path}")
        return None
    try:
        model = joblib.load(model_load_path)
        logger.info("Random Forest model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading Random Forest model from {model_load_path}: {e}")
        return None


if __name__ == '__main__':
    # Basic logging setup for standalone test
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("Testing modeling module...")

    # --- LSTM Test ---
    logger.info("\n--- Testing LSTM Components ---")
    n_samples_test = 100
    try:
        # Use ALL_FEATURE_COLS count for input features to LSTM
        n_features_test = len(config.ALL_FEATURE_COLS)
        seq_len_test = config.SEQUENCE_LENGTH
    except NameError:
        logger.warning("config not found during LSTM test, using default values.")
        n_features_test = 4; seq_len_test = 24
    except Exception as e:
         logger.error(f"Error accessing config during LSTM test: {e}")
         n_features_test = 4; seq_len_test = 24

    X_train_lstm_dummy = np.random.rand(n_samples_test, seq_len_test, n_features_test)
    y_train_lstm_uni_dummy = np.random.rand(n_samples_test) # Univariate
    y_train_lstm_multi_dummy = np.random.rand(n_samples_test, 3) # Multivariate (e.g., 3 features)
    X_val_lstm_dummy = np.random.rand(n_samples_test // 2, seq_len_test, n_features_test)
    y_val_lstm_uni_dummy = np.random.rand(n_samples_test // 2)
    y_val_lstm_multi_dummy = np.random.rand(n_samples_test // 2, 3)
    input_shape_test = (X_train_lstm_dummy.shape[1], X_train_lstm_dummy.shape[2])

    try:
        lr = config.LEARNING_RATE if 'config' in locals() else 0.001
        u1 = config.LSTM_UNITS_L1 if 'config' in locals() else 50
        u2 = config.LSTM_UNITS_L2 if 'config' in locals() else 50

        logger.info("\nTesting build_lstm_forecasting_model (univariate)...")
        lstm_model_uni_test = build_lstm_forecasting_model(input_shape_test, output_units=1, lstm_units_l1=u1, lstm_units_l2=u2, learning_rate=lr, model_name="TestUniLSTM")

        logger.info("\nTesting build_lstm_forecasting_model (multivariate)...")
        lstm_model_multi_test = build_lstm_forecasting_model(input_shape_test, output_units=3, lstm_units_l1=u1, lstm_units_l2=u2, learning_rate=lr, model_name="TestMultiLSTM")
        assert lstm_model_multi_test.output_shape == (None, 3)

        dummy_lstm_path = os.path.join("results", "saved_models", "dummy_lstm_model.keras")
        patience = config.EARLY_STOPPING_PATIENCE if 'config' in locals() else 5
        batch = config.BATCH_SIZE if 'config' in locals() else 32

        logger.info("\nTesting train_lstm_model structure (will skip actual training)...")
        # _, _ = train_lstm_model(lstm_model_uni_test, X_train_lstm_dummy, y_train_lstm_uni_dummy, X_val_lstm_dummy, y_val_lstm_uni_dummy,
        #                    epochs=1, batch_size=batch, model_checkpoint_path=dummy_lstm_path,
        #                    early_stopping_patience=patience)
        logger.info("(Skipped actual LSTM training call in standalone test)")

    except NameError as e:
         logger.error(f"Error during LSTM standalone test (likely config issue): {e}")
    except Exception as e:
         logger.exception(f"An unexpected error occurred during LSTM standalone test: {e}")


    # --- Random Forest Test ---
    logger.info("\n--- Testing Random Forest Components ---")
    n_samples_rf = 200
    n_features_rf = 3
    X_train_rf_dummy = pd.DataFrame(np.random.rand(n_samples_rf, n_features_rf), columns=[f'F{i}' for i in range(n_features_rf)])
    y_train_rf_dummy = pd.Series(np.random.rand(n_samples_rf) * 100)
    dummy_rf_path = os.path.join("results", "saved_models", "dummy_rf_model.joblib")

    try:
        n_est = config.RF_N_ESTIMATORS if 'config' in locals() else 50
        m_dep = config.RF_MAX_DEPTH if 'config' in locals() else 10
        m_spl = config.RF_MIN_SAMPLES_SPLIT if 'config' in locals() else 10
        n_job = config.RF_N_JOBS if 'config' in locals() else -1

        logger.info("\nTesting build_rf_model...")
        rf_model_test = build_rf_model(n_estimators=n_est, max_depth=m_dep, min_samples_split=m_spl, n_jobs=n_job)
        assert rf_model_test is not None

        logger.info("\nTesting train_rf_model structure (will skip actual training)...")
        # trained_rf = train_rf_model(rf_model_test, X_train_rf_dummy, y_train_rf_dummy, dummy_rf_path)
        logger.info("(Skipped actual RF training call in standalone test)")
        # Manually create dummy file for load test
        os.makedirs(os.path.dirname(dummy_rf_path), exist_ok=True)
        joblib.dump(rf_model_test, dummy_rf_path) # Save the *untrained* model for loading test

        logger.info("\nTesting load_rf_model...")
        loaded_rf = load_rf_model(dummy_rf_path)
        assert loaded_rf is not None
        # Clean up dummy file
        if os.path.exists(dummy_rf_path): os.remove(dummy_rf_path)


    except NameError as e:
        logger.error(f"Error during RF standalone test (likely config issue): {e}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred during RF standalone test: {e}")

    logger.info("\nModeling module testing structural components finished.")