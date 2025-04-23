# src/modeling.py
# --------------------
# Functions for building and training models
# --------------------

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
import os
import numpy as np

# Import configuration relative to src
try:
    from . import config
except ImportError:
    # Fallback for running the script directly
    import config


def build_lstm_model(input_shape, lstm_units_l1, lstm_units_l2, learning_rate):
    """Builds a sequential LSTM model."""
    if not isinstance(input_shape, tuple) or len(input_shape) != 2:
        raise ValueError(f"input_shape must be a tuple of length 2 (sequence_length, n_features), got {input_shape}")

    model = Sequential(name="LSTM_Direct_Forecast_Model")
    model.add(LSTM(lstm_units_l1, return_sequences=True, input_shape=input_shape, name="LSTM_Layer_1"))
    # Optional: Add Dropout for regularization
    # model.add(Dropout(0.2, name="Dropout_1"))
    model.add(LSTM(lstm_units_l2, return_sequences=False, name="LSTM_Layer_2"))
    # Optional: Add Dropout
    # model.add(Dropout(0.2, name="Dropout_2"))
    model.add(Dense(1, name="Output_Layer")) # Output layer predicts a single value

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae']) # Use MSE for regression, add MAE for monitoring

    print("LSTM Model built successfully:")
    model.summary()
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, model_checkpoint_path, early_stopping_patience):
    """Trains the model with validation data and saves the best weights."""

    # Ensure the directory for saving the model exists
    os.makedirs(os.path.dirname(model_checkpoint_path), exist_ok=True)

    # Callback to save the best model based on validation loss
    checkpoint = ModelCheckpoint(
        filepath=model_checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False, # Save entire model for easier loading
        mode='min',
        verbose=1
    )

    # Callback for early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=early_stopping_patience, # Number of epochs with no improvement after which training will be stopped
        restore_best_weights=True, # Restore model weights from the epoch with the best val_loss
        verbose=1
    )

    print(f"\nStarting model training for max {epochs} epochs (Batch Size: {batch_size})...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stopping],
        verbose=1 # Set to 1 or 2 for progress updates, 0 for silent
    )
    print("Model training complete.")

    # Best model is automatically restored if early stopping triggered with restore_best_weights=True
    # If training completed all epochs without early stopping, the best model was saved by ModelCheckpoint
    # Loading it ensures consistency, though it might be redundant if early stopping restored it.
    print(f"Loading best model state from {model_checkpoint_path}")
    try:
        # Load the entire model saved by ModelCheckpoint
        best_model = tf.keras.models.load_model(model_checkpoint_path)
        print("Successfully loaded best model.")
        return best_model, history
    except Exception as e:
        print(f"Warning: Could not load model from {model_checkpoint_path}. Using model state from end of training. Error: {e}")
        return model, history


# Example of how you might structure the RF part (if needed for Approach 2)
# from sklearn.ensemble import RandomForestRegressor
#
# def build_rf_model(n_estimators=100, random_state=42, **kwargs):
#     """Builds a RandomForestRegressor model."""
#     rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1, **kwargs)
#     print("Random Forest Model initialized.")
#     return rf_model
#
# def train_rf_model(model, X_train, y_train):
#     """Trains a RandomForestRegressor model."""
#     # Ensure X_train is 2D for RF (e.g., reshape LSTM outputs/features)
#     if X_train.ndim != 2:
#          raise ValueError(f"Random Forest requires 2D input, got shape {X_train.shape}")
#     print("Training Random Forest model...")
#     model.fit(X_train, y_train)
#     print("Random Forest training complete.")
#     return model


if __name__ == '__main__':
    print("Testing modeling module...")
    # Dummy data for testing structure
    n_samples = 100
    # Use feature count from config if possible, otherwise default
    try:
        n_features = len(config.FEATURE_COLS)
        seq_len = config.SEQUENCE_LENGTH
    except NameError:
        print("Warning: config not found, using default values for test.")
        n_features = 4
        seq_len = 24

    X_train_dummy = np.random.rand(n_samples, seq_len, n_features)
    y_train_dummy = np.random.rand(n_samples)
    X_val_dummy = np.random.rand(n_samples // 2, seq_len, n_features)
    y_val_dummy = np.random.rand(n_samples // 2)

    input_s = (X_train_dummy.shape[1], X_train_dummy.shape[2]) # (sequence_length, n_features)

    try:
        lstm_model = build_lstm_model(input_s, config.LSTM_UNITS_L1, config.LSTM_UNITS_L2, config.LEARNING_RATE)
        print("\n(Skipping actual training in standalone test)")
        print("Modeling module testing structural components finished.")
    except NameError as e:
         print(f"Error during standalone test: {e}")
         print("Ensure config is available or default values are set.")
    except Exception as e:
         print(f"An unexpected error occurred: {e}")