# filename: project/main.py
# --------------------
# Main script to run the wind power forecasting workflow
# --------------------

import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import logging
import traceback
import joblib # For loading RF model

# --- Setup Project Path ---
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# --- Import Custom Modules ---
# Import config first to use its settings for logging etc.
try:
    import config
except ImportError as e:
    print(f"FATAL ERROR: Could not import 'config.py'. Ensure 'src/config.py' exists and is accessible.")
    print(f"Details: {e}")
    sys.exit(1)
except Exception as e:
    print(f"FATAL ERROR during config import: {e}")
    sys.exit(1)

# --- Setup Logging ---
# Configure logging based on config.py settings
try:
    # Ensure log directory exists
    os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)
    logging.basicConfig(level=config.LOG_LEVEL,
                        format=config.LOG_FORMAT,
                        handlers=[
                            logging.FileHandler(config.LOG_FILE, mode='w'), # Log to file, overwrite each run
                            logging.StreamHandler() # Log to console
                        ])
    logger = logging.getLogger(__name__) # Get logger for main script
    logger.info("Logging configured successfully.")
    # Suppress overly verbose logs from libraries if needed
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('tensorflow').setLevel(logging.WARNING) # Use WARNING instead of ERROR to see potentially important TF info
    logging.getLogger('h5py').setLevel(logging.WARNING) # Reduce h5py verbosity

except Exception as e:
     print(f"FATAL ERROR setting up logging: {e}")
     sys.exit(1)

# --- Import Other Custom Modules ---
try:
    logger.info("Importing project modules...")
    import data_preprocessing as dp
    import modeling as mdl
    import evaluation as evl
    import plotting as pl
    logger.info("Project modules imported successfully.")
except ImportError as e:
    logger.critical(f"Could not import project modules. Ensure 'src' directory exists and is accessible.")
    logger.critical(f"Details: {e}")
    sys.exit(1)
except Exception as e:
    logger.critical(f"An unexpected error occurred during module import: {e}")
    logger.exception("Import traceback:") # Log traceback
    sys.exit(1)


# --- Helper Function for Inverse Scaling ---
def inverse_transform_data(scaled_data: np.ndarray, scaler: dp.MinMaxScaler, feature_index: int) -> np.ndarray:
    """Inverse transforms scaled data for a specific feature."""
    n_features = scaler.n_features_in_
    if feature_index >= n_features:
        raise IndexError(f"feature_index ({feature_index}) out of bounds for scaler ({n_features} features).")

    # Create a dummy array with the same shape as the original data scaler was fit on
    dummy_array = np.zeros((len(scaled_data), n_features))
    # Place the scaled data into the correct column
    dummy_array[:, feature_index] = scaled_data.flatten()
    # Inverse transform
    inversed_array = scaler.inverse_transform(dummy_array)
    # Extract the column we care about
    return inversed_array[:, feature_index]

# --- Main Function ---
def main():
    """Main function to execute the forecasting workflow based on config.APPROACH."""
    logger.info(f"--- Starting Wind Power Forecasting Workflow (Approach: {config.APPROACH}) ---")

    # --- 1. Setup & Configuration ---
    logger.info("[Step 1/N] Setting up environment and configuration...") # N depends on approach
    try:
        pl.setup_plot_style()
        logger.info(f"Using data file: {config.RAW_DATA_FILE}")
        logger.info(f"Results will be saved in: {config.RESULTS_DIR}")
        logger.info(f"Logs will be saved to: {config.LOG_FILE}")
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
        os.makedirs(config.IMAGE_DIR, exist_ok=True)
        os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
        logger.debug("Result, model, image, and processed data directories ensured.")
    except Exception as e:
         logger.critical(f"Error during setup: {e}")
         logger.exception("Setup traceback:")
         sys.exit(1)

    # --- 2. Load Raw Data ---
    logger.info("[Step 2/N] Loading raw data...")
    try:
        raw_df = dp.load_data(config.RAW_DATA_FILE)
        if raw_df.empty:
            logger.critical("Loaded raw data is empty. Exiting.")
            sys.exit(1)
        logger.info("Raw data loaded successfully.")
        logger.debug(f"Raw data shape: {raw_df.shape}")
    except FileNotFoundError:
        logger.critical(f"Raw data file not found at {config.RAW_DATA_FILE}. Exiting.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Error loading data: {e}. Exiting.")
        logger.exception("Data loading traceback:")
        sys.exit(1)

    # --- 3. Preprocess Data ---
    logger.info("[Step 3/N] Preprocessing data...")
    try:
        processed_df = dp.preprocess_data(
            raw_df,
            config.DATE_COL,
            config.TARGET_COL,
            config.ALL_FEATURE_COLS, # Use all potential features initially
            config.RESAMPLE_FREQ
        )
        if processed_df.empty:
            logger.critical("Data preprocessing resulted in an empty DataFrame. Exiting.")
            sys.exit(1)
        logger.info("Data preprocessing complete.")
        logger.debug(f"Processed data shape: {processed_df.shape}")
        logger.debug(f"Processed data columns: {processed_df.columns.tolist()}")
    except Exception as e:
        logger.critical(f"Error during data preprocessing: {e}. Exiting.")
        logger.exception("Preprocessing traceback:")
        sys.exit(1)

    # --- 4. Exploratory Data Analysis (Plots) ---
    # Run EDA plots regardless of approach
    logger.info("[Step 4/N] Generating EDA plots...")
    try:
        pl.plot_time_series(processed_df, config.TARGET_COL, title=f'Time Series of {config.TARGET_COL}', ylabel=config.TARGET_COL, save_path=os.path.join(config.IMAGE_DIR, 'target_timeseries.png'))
        pl.plot_feature_distribution(processed_df, config.TARGET_COL, title=f'Distribution of {config.TARGET_COL}', save_path=os.path.join(config.IMAGE_DIR, 'target_distribution.png'))
        # Plot other key features if they exist
        for col in ['Wind Speed (m/s)', 'Theoretical_Power_Curve (KWh)', 'Wind Direction (°)']:
             if col in processed_df.columns:
                  pl.plot_feature_distribution(processed_df, col, title=f'Distribution of {col}', save_path=os.path.join(config.IMAGE_DIR, f'{col.replace(" ", "_").lower()}_distribution.png'))
        # Use ALL_FEATURE_COLS available in processed_df for correlation
        available_features = [col for col in config.ALL_FEATURE_COLS if col in processed_df.columns]
        pl.plot_correlation_heatmap(processed_df[available_features], title='Feature Correlation Heatmap', save_path=os.path.join(config.IMAGE_DIR, 'feature_correlation_heatmap.png'))
        pl.plot_acf_pacf(processed_df[config.TARGET_COL], lags=config.ACF_PACF_LAGS, title_prefix=config.TARGET_COL, save_path_prefix=os.path.join(config.IMAGE_DIR, f'{config.TARGET_COL.replace(" ", "_")}'))
        logger.info("EDA plots generated and saved.")
        plt.close('all')
    except Exception as e:
        logger.error(f"Error during EDA plotting: {e}")
        logger.exception("EDA plotting traceback:")
        # Continue workflow


    # --- 5. Common Data Preparation (Split & Scale) ---
    logger.info("[Step 5/N] Splitting and Scaling Data...")
    try:
        train_df, val_df, test_df = dp.split_data_sequential(
            processed_df, config.TRAIN_SPLIT, config.VALIDATION_SPLIT
        )
        if train_df.empty:
            logger.critical("Training data is empty after split, cannot proceed. Exiting.")
            sys.exit(1)
        # Scale data based on training set - scaler will be used for inverse transforms later
        train_scaled_df, val_scaled_df, test_scaled_df, scaler = dp.scale_data(
            train_df, val_df, test_df
        )
        # Get target column index AFTER scaling (needed for inverse transform)
        try:
            target_col_index = train_scaled_df.columns.get_loc(config.TARGET_COL)
            logger.debug(f"Target column '{config.TARGET_COL}' found at index {target_col_index} in scaled data.")
        except KeyError:
            logger.critical(f"Target column '{config.TARGET_COL}' not found after scaling. Check feature list and preprocessing steps.")
            sys.exit(1)

        # Get feature column indices for Approach 2 AFTER scaling
        try:
            lstm_target_feature_indices = [train_scaled_df.columns.get_loc(col) for col in config.LSTM_FEATURE_TARGET_COLS]
            rf_feature_indices = [train_scaled_df.columns.get_loc(col) for col in config.RF_FEATURE_COLS] # Should be same as above
            logger.debug(f"LSTM Feature Target indices: {lstm_target_feature_indices}")
            logger.debug(f"RF Feature indices: {rf_feature_indices}")
        except KeyError as e:
             logger.critical(f"Feature column '{e}' required for Approach 2 not found after scaling. Check config.LSTM_FEATURE_TARGET_COLS/RF_FEATURE_COLS.")
             sys.exit(1)


        logger.info("Data splitting and scaling complete.")
    except Exception as e:
        logger.critical(f"Error during data splitting or scaling: {e}. Exiting.")
        logger.exception("Data split/scale traceback:")
        sys.exit(1)


    # ==============================================================
    # --- Approach 1: Direct LSTM Forecasting ---
    # ==============================================================
    if config.APPROACH == 'direct':
        logger.info("--- Running Direct LSTM Forecasting Workflow ---")
        num_steps = 9 # Update total steps for logging

        # --- 6a. Create Sequences for Direct LSTM ---
        logger.info(f"[Step 6/{num_steps}] Creating sequences for Direct LSTM...")
        try:
            # Target is the power column, Features are all columns (incl. power)
            X_train, y_train = dp.create_sequences_lstm(
                train_scaled_df, config.SEQUENCE_LENGTH, config.TARGET_COL, config.ALL_FEATURE_COLS
            )
            X_val, y_val = dp.create_sequences_lstm(
                val_scaled_df, config.SEQUENCE_LENGTH, config.TARGET_COL, config.ALL_FEATURE_COLS
            )
            X_test, y_test = dp.create_sequences_lstm(
                test_scaled_df, config.SEQUENCE_LENGTH, config.TARGET_COL, config.ALL_FEATURE_COLS
            )
            logger.info(f"  Shapes: X_train={X_train.shape}, y_train={y_train.shape}")
            logger.info(f"  Shapes: X_val={X_val.shape}, y_val={y_val.shape}")
            logger.info(f"  Shapes: X_test={X_test.shape}, y_test={y_test.shape}")

            if X_train.shape[0] == 0:
                 logger.critical("No training sequences created. Check data length vs SEQUENCE_LENGTH. Exiting.")
                 sys.exit(1)

        except Exception as e:
            logger.critical(f"Error creating sequences for Direct LSTM: {e}. Exiting.")
            logger.exception("Direct LSTM sequence creation traceback:")
            sys.exit(1)

        # --- 7a. Build and Train Direct LSTM Model ---
        logger.info(f"[Step 7/{num_steps}] Building and training Direct LSTM model...")
        try:
            input_shape = (X_train.shape[1], X_train.shape[2])
            logger.info(f"  Building model with input shape: {input_shape}")
            direct_lstm_model = mdl.build_lstm_forecasting_model(
                input_shape=input_shape,
                output_units=1, # Direct power prediction
                lstm_units_l1=config.LSTM_UNITS_L1,
                lstm_units_l2=config.LSTM_UNITS_L2,
                learning_rate=config.LEARNING_RATE,
                model_name="Direct_LSTM_Power_Forecast"
            )

            logger.info("  Training Direct LSTM model...")
            direct_lstm_model, history = mdl.train_lstm_model(
                direct_lstm_model, X_train, y_train, X_val, y_val,
                epochs=config.EPOCHS,
                batch_size=config.BATCH_SIZE,
                model_checkpoint_path=config.BEST_LSTM_DIRECT_MODEL_FILE,
                early_stopping_patience=config.EARLY_STOPPING_PATIENCE
            )
            logger.info("Direct LSTM model training process finished.")

            logger.info("  Plotting learning curves...")
            pl.plot_learning_curves(
                history,
                title='Direct LSTM Model Learning Curves',
                save_path=config.DIRECT_LEARNING_CURVES_PLOT
            )
            plt.close('all')

        except Exception as e:
            logger.critical(f"Error during Direct LSTM model building or training: {e}. Exiting.")
            logger.exception("Direct LSTM Model build/train traceback:")
            sys.exit(1)

        # --- 8a. Evaluate Direct LSTM Model ---
        logger.info(f"[Step 8/{num_steps}] Evaluating Direct LSTM model performance...")
        try:
            logger.info("  Loading best saved Direct LSTM model for evaluation...")
            best_direct_lstm_model = direct_lstm_model # From train_lstm_model return

            metrics_results = []
            y_test_true_inv, y_test_pred_inv = np.array([]), np.array([]) # Initialize

            # Evaluate on Training Set
            if X_train.shape[0] > 0:
                logger.info("  Evaluating Direct LSTM on Training Set...")
                _, _, train_metrics = evl.evaluate_model( # Use evaluate_model as it handles inverse scaling
                     best_direct_lstm_model, X_train, y_train, scaler, target_col_index, config.METRICS_TO_CALCULATE
                 )
                train_metrics['Set'] = 'Training'
                metrics_results.append(train_metrics)
                logger.info(f"  Training Metrics: {train_metrics}")
            else: logger.warning("Skipping evaluation on empty Training Set.")

            # Evaluate on Validation Set
            if X_val.shape[0] > 0:
                logger.info("  Evaluating Direct LSTM on Validation Set...")
                _, _, val_metrics = evl.evaluate_model(
                     best_direct_lstm_model, X_val, y_val, scaler, target_col_index, config.METRICS_TO_CALCULATE
                 )
                val_metrics['Set'] = 'Validation'
                metrics_results.append(val_metrics)
                logger.info(f"  Validation Metrics: {val_metrics}")
            else: logger.warning("Skipping evaluation on empty Validation Set.")

            # Evaluate on Test Set
            if X_test.shape[0] > 0:
                logger.info("  Evaluating Direct LSTM on Test Set...")
                y_test_true_inv, y_test_pred_inv, test_metrics = evl.evaluate_model(
                     best_direct_lstm_model, X_test, y_test, scaler, target_col_index, config.METRICS_TO_CALCULATE
                 )
                test_metrics['Set'] = 'Test'
                metrics_results.append(test_metrics)
                logger.info(f"  Test Metrics: {test_metrics}")
            else:
                logger.warning("Skipping evaluation on empty Test Set.")


            # Save metrics if any were calculated
            if metrics_results:
                metrics_df = pd.DataFrame(metrics_results)
                cols_order = ['Set'] + [col for col in config.METRICS_TO_CALCULATE if col in metrics_df.columns]
                metrics_df = metrics_df[cols_order]

                logger.info(f"  Saving Direct LSTM metrics to {config.METRICS_LSTM_DIRECT_FILE}...")
                metrics_df.to_csv(config.METRICS_LSTM_DIRECT_FILE, index=False, float_format='%.4f')
                logger.info("Metrics saved.")
                logger.info(f"Evaluation Metrics Summary (Direct LSTM):\n{metrics_df.to_string(index=False)}")
            else:
                logger.warning("No metrics were calculated for Direct LSTM.")

        except Exception as e:
            logger.error(f"Error during Direct LSTM model evaluation: {e}")
            logger.exception("Direct LSTM Evaluation traceback:")

        # --- 9a. Visualize Direct LSTM Results ---
        logger.info(f"[Step 9/{num_steps}] Visualizing Direct LSTM results...")
        try:
            if y_test_true_inv.shape[0] > 0 and y_test_pred_inv.shape[0] > 0:
                # Get corresponding timestamps for the test set predictions
                # Index needs to account for sequence length offset
                test_start_iloc = train_scaled_df.shape[0] + val_scaled_df.shape[0] + config.SEQUENCE_LENGTH
                test_end_iloc = test_start_iloc + len(y_test_true_inv)
                # Use iloc on the original *processed* dataframe index
                test_timestamps = processed_df.index[test_start_iloc : test_end_iloc]


                if len(test_timestamps) == len(y_test_true_inv):
                     logger.info("  Plotting actual vs predicted for Test Set (Direct LSTM)...")
                     pl.plot_actual_vs_predicted(
                        y_true=y_test_true_inv,
                        y_pred=y_test_pred_inv,
                        data_index=test_timestamps,
                        title='Direct LSTM Forecast: Actual vs. Predicted Power (Test Set)',
                        ylabel=config.TARGET_COL,
                        save_path=config.DIRECT_ACTUAL_VS_PREDICTED_PLOT
                     )
                     plt.close('all')
                else:
                     logger.warning(f"Length mismatch between test timestamps ({len(test_timestamps)}) and test predictions ({len(y_test_true_inv)}). Cannot plot actual vs predicted accurately.")
            else:
                 logger.warning("Test set predictions for Direct LSTM are empty or missing. Skipping actual vs predicted plot.")

            logger.info("Direct LSTM result visualization complete.")
        except Exception as e:
             logger.error(f"Error during Direct LSTM result visualization: {e}")
             logger.exception("Direct LSTM Visualization traceback:")

    # ==============================================================
    # --- Approach 2: Indirect LSTM-Features -> RF-Power ---
    # ==============================================================
    elif config.APPROACH == 'indirect':
        logger.info("--- Running Indirect (LSTM-Features -> RF-Power) Workflow ---")
        num_steps = 11 # Update total steps for logging

        # --- 6b. Create Sequences for Feature LSTM ---
        logger.info(f"[Step 6/{num_steps}] Creating sequences for Feature LSTM...")
        try:
            # Target = Feature columns, Features = All columns
            X_train_feat, y_train_feat = dp.create_sequences_lstm(
                train_scaled_df, config.SEQUENCE_LENGTH, config.LSTM_FEATURE_TARGET_COLS, config.ALL_FEATURE_COLS
            )
            X_val_feat, y_val_feat = dp.create_sequences_lstm(
                val_scaled_df, config.SEQUENCE_LENGTH, config.LSTM_FEATURE_TARGET_COLS, config.ALL_FEATURE_COLS
            )
            X_test_feat, y_test_feat = dp.create_sequences_lstm( # y_test_feat are the TRUE future features (scaled)
                test_scaled_df, config.SEQUENCE_LENGTH, config.LSTM_FEATURE_TARGET_COLS, config.ALL_FEATURE_COLS
            )
            logger.info(f"  Shapes: X_train_feat={X_train_feat.shape}, y_train_feat={y_train_feat.shape}")
            logger.info(f"  Shapes: X_val_feat={X_val_feat.shape}, y_val_feat={y_val_feat.shape}")
            logger.info(f"  Shapes: X_test_feat={X_test_feat.shape}, y_test_feat={y_test_feat.shape}") # y_test_feat = true scaled features

            if X_train_feat.shape[0] == 0:
                 logger.critical("No training sequences created for Feature LSTM. Check data length vs SEQUENCE_LENGTH. Exiting.")
                 sys.exit(1)

        except Exception as e:
            logger.critical(f"Error creating sequences for Feature LSTM: {e}. Exiting.")
            logger.exception("Feature LSTM sequence creation traceback:")
            sys.exit(1)

        # --- 7b. Build and Train Feature LSTM Model ---
        logger.info(f"[Step 7/{num_steps}] Building and training Feature LSTM model...")
        try:
            input_shape = (X_train_feat.shape[1], X_train_feat.shape[2])
            output_units = len(config.LSTM_FEATURE_TARGET_COLS) # Multivariate output
            logger.info(f"  Building Feature LSTM model with input shape: {input_shape}, output units: {output_units}")
            feature_lstm_model = mdl.build_lstm_forecasting_model(
                input_shape=input_shape,
                output_units=output_units,
                lstm_units_l1=config.LSTM_UNITS_L1,
                lstm_units_l2=config.LSTM_UNITS_L2,
                learning_rate=config.LEARNING_RATE,
                model_name="Feature_LSTM_Forecast"
            )

            logger.info("  Training Feature LSTM model...")
            feature_lstm_model, history_lstm_feat = mdl.train_lstm_model(
                feature_lstm_model, X_train_feat, y_train_feat, X_val_feat, y_val_feat,
                epochs=config.EPOCHS,
                batch_size=config.BATCH_SIZE,
                model_checkpoint_path=config.LSTM_FEATURE_MODEL_FILE,
                early_stopping_patience=config.EARLY_STOPPING_PATIENCE
            )
            logger.info("Feature LSTM model training process finished.")

            logger.info("  Plotting Feature LSTM learning curves...")
            pl.plot_learning_curves(
                history_lstm_feat,
                title='Feature LSTM Model Learning Curves',
                save_path=config.INDIRECT_LSTM_LEARNING_CURVES_PLOT
            )
            plt.close('all')

        except Exception as e:
            logger.critical(f"Error during Feature LSTM model building or training: {e}. Exiting.")
            logger.exception("Feature LSTM Model build/train traceback:")
            sys.exit(1)

        # --- 8b. Prepare Data and Train Random Forest Model ---
        logger.info(f"[Step 8/{num_steps}] Preparing data and training Random Forest model...")
        try:
            logger.info("  Preparing data for RF (using scaled historical data)...")
            # Use the *original* scaled train/val dataframes, not sequences
            X_train_rf, y_train_rf = dp.prepare_data_for_rf(
                 train_scaled_df, config.RF_FEATURE_COLS, config.TARGET_COL
             )
            # We don't strictly need validation RF data unless doing hyperparameter tuning here
            # X_val_rf, y_val_rf = dp.prepare_data_for_rf(
            #     val_scaled_df, config.RF_FEATURE_COLS, config.TARGET_COL
            # )

            if X_train_rf.empty or y_train_rf.empty:
                 logger.critical("RF training data is empty after preparation. Exiting.")
                 sys.exit(1)

            logger.info("  Building RF model...")
            rf_model = mdl.build_rf_model(
                n_estimators=config.RF_N_ESTIMATORS,
                max_depth=config.RF_MAX_DEPTH,
                min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
                n_jobs=config.RF_N_JOBS
            )

            logger.info("  Training RF model...")
            rf_model = mdl.train_rf_model(
                rf_model, X_train_rf, y_train_rf,
                model_save_path=config.RF_POWER_MODEL_FILE
            )
            logger.info("RF model training and saving complete.")

        except Exception as e:
            logger.critical(f"Error during RF model data preparation or training: {e}. Exiting.")
            logger.exception("RF Model prep/train traceback:")
            sys.exit(1)


        # --- 9b. Indirect Prediction Workflow (Test Set) ---
        logger.info(f"[Step 9/{num_steps}] Performing indirect prediction on Test Set...")
        try:
            if X_test_feat.shape[0] > 0:
                logger.info("  Loading trained models for indirect prediction...")
                # Load the best Feature LSTM model
                try:
                     best_feature_lstm_model = load_model(config.LSTM_FEATURE_MODEL_FILE)
                     logger.info(f"  Loaded Feature LSTM model from: {config.LSTM_FEATURE_MODEL_FILE}")
                except Exception as e:
                     logger.error(f"Could not load Feature LSTM model from {config.LSTM_FEATURE_MODEL_FILE}. Using model from end of training. Error: {e}")
                     best_feature_lstm_model = feature_lstm_model # Fallback

                # Load the trained RF model
                rf_power_model = mdl.load_rf_model(config.RF_POWER_MODEL_FILE)
                if rf_power_model is None:
                    logger.critical("Failed to load RF Power model. Cannot perform indirect prediction. Exiting.")
                    sys.exit(1)

                # Step A: Predict future features using LSTM
                logger.info("  Predicting future features using LSTM on test sequences (X_test_feat)...")
                y_pred_features_scaled = best_feature_lstm_model.predict(X_test_feat) # Shape: (n_samples, n_features_lstm_target)
                logger.debug(f"  Predicted scaled features shape: {y_pred_features_scaled.shape}")

                # Ensure predicted features align with RF input features
                if y_pred_features_scaled.shape[1] != len(config.RF_FEATURE_COLS):
                     msg = f"Mismatch between LSTM predicted features ({y_pred_features_scaled.shape[1]}) and RF expected features ({len(config.RF_FEATURE_COLS)})"
                     logger.error(msg)
                     raise ValueError(msg)

                # We need the *actual* power values corresponding to the test sequences for evaluation later
                # These are derived from the test_scaled_df, offset by sequence length
                test_actual_power_scaled = test_scaled_df[config.TARGET_COL].values[config.SEQUENCE_LENGTH:]
                logger.debug(f"  Actual scaled power shape for test set: {test_actual_power_scaled.shape}")

                # Make sure lengths match prediction outputs
                if len(test_actual_power_scaled) != len(y_pred_features_scaled):
                     logger.warning(f"Length mismatch between actual power ({len(test_actual_power_scaled)}) and predicted features ({len(y_pred_features_scaled)}) on test set. Truncating to shortest.")
                     min_len = min(len(test_actual_power_scaled), len(y_pred_features_scaled))
                     test_actual_power_scaled = test_actual_power_scaled[:min_len]
                     y_pred_features_scaled = y_pred_features_scaled[:min_len]
                     # Also need to adjust y_test_feat if using it for comparison later
                     y_test_feat = y_test_feat[:min_len]


                # Step B: Prepare features for RF model (use LSTM predicted features)
                # The RF model expects a 2D array (samples, features)
                # The order of columns in y_pred_features_scaled *must* match the order expected by the RF model
                # (which is determined by config.RF_FEATURE_COLS during RF training data prep).
                # Assuming the LSTM output columns correspond directly to RF_FEATURE_COLS.
                X_test_rf_input = pd.DataFrame(y_pred_features_scaled, columns=config.RF_FEATURE_COLS)
                logger.debug(f"  Prepared RF input features shape: {X_test_rf_input.shape}")

                # Step C: Predict power using RF model
                logger.info("  Predicting final power using RF model with LSTM-predicted features...")
                y_pred_power_scaled = rf_power_model.predict(X_test_rf_input) # Output is scaled power prediction
                logger.debug(f"  Predicted scaled power shape (RF output): {y_pred_power_scaled.shape}")

                # Step D: Inverse Transform Results
                logger.info("  Inverse transforming actual power and predicted power...")
                y_test_true_inv = inverse_transform_data(test_actual_power_scaled, scaler, target_col_index)
                y_test_pred_inv = inverse_transform_data(y_pred_power_scaled, scaler, target_col_index)
                logger.debug(f"  Inverse transformed actual power shape: {y_test_true_inv.shape}")
                logger.debug(f"  Inverse transformed predicted power shape: {y_test_pred_inv.shape}")

            else:
                logger.warning("Test set sequences for Feature LSTM are empty. Skipping indirect prediction.")
                y_test_true_inv, y_test_pred_inv = np.array([]), np.array([]) # Ensure defined

        except Exception as e:
            logger.error(f"Error during Indirect Prediction Workflow: {e}")
            logger.exception("Indirect Prediction traceback:")
            y_test_true_inv, y_test_pred_inv = np.array([]), np.array([]) # Reset on error


        # --- 10b. Evaluate Indirect Approach ---
        logger.info(f"[Step 10/{num_steps}] Evaluating Indirect (RF) model performance...")
        try:
            if y_test_true_inv.shape[0] > 0 and y_test_pred_inv.shape[0] > 0:
                logger.info("  Calculating metrics for Indirect Approach (on Test Set)...")
                metrics_indirect = evl.calculate_metrics(y_test_true_inv, y_test_pred_inv, config.METRICS_TO_CALCULATE)
                metrics_indirect['Set'] = 'Test_Indirect' # Add identifier
                metrics_indirect['Approach'] = 'Indirect'

                # Save metrics
                metrics_df_indirect = pd.DataFrame([metrics_indirect]) # Create DataFrame from dict
                 # Reorder columns
                cols_order = ['Approach', 'Set'] + [col for col in config.METRICS_TO_CALCULATE if col in metrics_df_indirect.columns]
                metrics_df_indirect = metrics_df_indirect[cols_order]


                logger.info(f"  Saving Indirect approach metrics to {config.METRICS_LSTM_INDIRECT_FILE}...")
                metrics_df_indirect.to_csv(config.METRICS_LSTM_INDIRECT_FILE, index=False, float_format='%.4f')
                logger.info("Metrics saved.")
                logger.info(f"Evaluation Metrics Summary (Indirect Approach - Test Set):\n{metrics_df_indirect.to_string(index=False)}")
            else:
                 logger.warning("No test predictions available for Indirect Approach. Skipping evaluation.")

        except Exception as e:
            logger.error(f"Error during Indirect approach evaluation: {e}")
            logger.exception("Indirect Evaluation traceback:")


        # --- 11b. Visualize Indirect Results ---
        logger.info(f"[Step 11/{num_steps}] Visualizing Indirect results...")
        try:
            if y_test_true_inv.shape[0] > 0 and y_test_pred_inv.shape[0] > 0:
                # Get corresponding timestamps for the test set predictions
                # Need to match the length of y_test_true_inv / y_test_pred_inv after potential truncation
                test_start_iloc = train_scaled_df.shape[0] + val_scaled_df.shape[0] + config.SEQUENCE_LENGTH
                test_end_iloc = test_start_iloc + len(y_test_true_inv) # Use length after potential truncation
                test_timestamps = processed_df.index[test_start_iloc : test_end_iloc]

                if len(test_timestamps) == len(y_test_true_inv):
                     logger.info("  Plotting actual vs predicted for Test Set (Indirect Approach)...")
                     pl.plot_actual_vs_predicted(
                        y_true=y_test_true_inv,
                        y_pred=y_test_pred_inv,
                        data_index=test_timestamps,
                        title='Indirect Forecast (LSTM-RF): Actual vs. Predicted Power (Test Set)',
                        ylabel=config.TARGET_COL,
                        save_path=config.INDIRECT_RF_ACTUAL_VS_PREDICTED_PLOT
                     )
                     plt.close('all')
                else:
                     logger.warning(f"Length mismatch between test timestamps ({len(test_timestamps)}) and indirect predictions ({len(y_test_true_inv)}). Cannot plot actual vs predicted accurately.")
            else:
                 logger.warning("Test set predictions for Indirect approach are empty or missing. Skipping actual vs predicted plot.")

            logger.info("Indirect approach result visualization complete.")
        except Exception as e:
             logger.error(f"Error during Indirect approach result visualization: {e}")
             logger.exception("Indirect Visualization traceback:")

    # --- Invalid Approach ---
    else:
        logger.critical(f"Invalid APPROACH specified in config.py: '{config.APPROACH}'. Must be 'direct' or 'indirect'. Exiting.")
        sys.exit(1)


    # --- N. Final Conclusion ---
    logger.info("[Step N/N] Workflow finished.")
    logger.info(f"Results saved in: {config.RESULTS_DIR}")
    logger.info(f"Full logs available at: {config.LOG_FILE}")
    logger.info("-------------------------------------------------")


# --- Main Execution Guard ---
if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        logger.info(f"Workflow exited with status {e.code}.")
    except Exception as e:
         # Catch any unexpected errors in main execution not caught earlier
         logger.critical(f"An unexpected critical error occurred in main execution: {e}")
         logger.exception("Main execution traceback:")
         sys.exit(1) # Ensure non-zero exit code on critical failure