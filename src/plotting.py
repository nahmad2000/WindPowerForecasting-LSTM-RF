# src/plotting.py
# --------------------
# Functions for creating visualizations
# --------------------

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from typing import Optional, Union # For type hinting
import logging # Added logging
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf # Added for ACF/PACF

# Get logger for this module
logger = logging.getLogger(__name__)

# Import configuration relative to src
try:
    from . import config
except ImportError:
    logger.info("Attempting fallback import for config in plotting.py...")
    try:
       import config
    except ModuleNotFoundError as e:
        logger.error(f"Cannot import 'config'. Ensure 'src' is in PYTHONPATH or run scripts as modules. Details: {e}")
        raise


def setup_plot_style():
    """Sets the plot style and font sizes defined in config."""
    try:
        plt.style.use(config.PLOT_STYLE)
        logger.info(f"Using plot style: {config.PLOT_STYLE}")
    except Exception as e:
        logger.warning(f"Could not apply plot style '{config.PLOT_STYLE}'. Using default. Error: {e}")
        plt.style.use('default')

    # Apply font size configurations
    try:
        plt.rcParams.update({'font.size': config.FONT_SIZE})
        plt.rcParams['axes.labelsize'] = config.FONT_SIZE
        plt.rcParams['axes.titlesize'] = config.FONT_SIZE + 1
        plt.rcParams['xtick.labelsize'] = config.FONT_SIZE - 1
        plt.rcParams['ytick.labelsize'] = config.FONT_SIZE - 1
        plt.rcParams['legend.fontsize'] = config.FONT_SIZE - 1
        plt.rcParams['figure.titlesize'] = config.FONT_SIZE + 2
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['savefig.facecolor'] = 'white'
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.linestyle'] = '--'
        plt.rcParams['grid.alpha'] = 0.6
    except AttributeError as e:
         logger.warning(f"Missing plotting configuration in config.py? Error: {e}")
    except Exception as e:
         logger.warning(f"Error applying plot style configurations: {e}")


def plot_time_series(df: pd.DataFrame, column: str, title: str, ylabel: str,
                     save_path: Optional[str] = None):
    """Plots a specific time series column from a DataFrame."""
    if df.empty:
        logger.warning(f"DataFrame is empty. Skipping plot '{title}'.")
        return
    if column not in df.columns:
        logger.warning(f"Column '{column}' not found in DataFrame for plotting '{title}'. Available columns: {df.columns}")
        return

    setup_plot_style()
    fig, ax = plt.subplots(figsize=config.FIG_SIZE)
    ax.plot(df.index, df[column], label=column, linewidth=1.5)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.legend()

    try:
        locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
    except Exception as e:
        logger.warning(f"Could not apply auto date formatting: {e}")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=30, ha='right')

    plt.tight_layout()

    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving plot to {save_path}: {e}")
    plt.show()
    # plt.close(fig) # Explicitly close figure to free memory

def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray,
                             data_index: Union[pd.Index, np.ndarray],
                             title: str, ylabel: str,
                             save_path: Optional[str] = None):
    """Plots actual vs predicted values over time or index."""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    if y_true.shape[0] == 0 or y_pred.shape[0] == 0:
         logger.warning(f"Empty data arrays provided for actual vs predicted plot '{title}'. Skipping.")
         return
    if y_true.shape != y_pred.shape:
         logger.warning(f"Shape mismatch for actual ({y_true.shape}) vs predicted ({y_pred.shape}) in plot '{title}'. Skipping.")
         return
    if len(data_index) != len(y_true):
         logger.warning(f"Index length ({len(data_index)}) does not match data length ({len(y_true)}) in plot '{title}'. Plotting against sample number.")
         plot_index = np.arange(len(y_true))
         xlabel = "Sample Index"
         time_axis = False
    else:
         plot_index = data_index
         xlabel = "Time" if isinstance(data_index, pd.DatetimeIndex) else "Index"
         time_axis = isinstance(data_index, pd.DatetimeIndex)

    setup_plot_style()
    fig, ax = plt.subplots(figsize=config.FIG_SIZE)

    ax.plot(plot_index, y_true, label='Actual', linewidth=2, color='royalblue', alpha=0.9)
    ax.plot(plot_index, y_pred, label='Predicted', linestyle='--', linewidth=1.5, color='darkorange', alpha=0.8)

    ax.set_title(title, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    if time_axis:
        try:
            locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
            formatter = mdates.ConciseDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
        except Exception as e:
            logger.warning(f"Could not apply auto date formatting for time axis: {e}")
            plt.xticks(rotation=30, ha='right')

    plt.tight_layout()

    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        except Exception as e:
             logger.error(f"Error saving plot to {save_path}: {e}")
    plt.show()
    # plt.close(fig)

def plot_learning_curves(history: tf.keras.callbacks.History, title: str = "Model Training History",
                         save_path: Optional[str] = None):
    """Plots training & validation loss curves from Keras History object."""
    if not hasattr(history, 'history') or not isinstance(history.history, dict) or 'loss' not in history.history:
        logger.warning(f"Invalid or incomplete Keras History object passed to plot_learning_curves for '{title}'. Skipping plot.")
        return

    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    history_df = pd.DataFrame(history.history)
    epochs_ran = range(1, len(history_df['loss']) + 1)

    ax.plot(epochs_ran, history_df['loss'], label='Training Loss', color='mediumblue', marker='.', markersize=5, linestyle='-')

    if 'val_loss' in history_df:
        ax.plot(epochs_ran, history_df['val_loss'], label='Validation Loss', color='darkorange', marker='.', markersize=5, linestyle='--')
        min_val_loss_epoch = history_df['val_loss'].idxmin() + 1
        min_val_loss = history_df['val_loss'].min()
        ax.axvline(min_val_loss_epoch, color='red', linestyle=':', linewidth=1.5, label=f'Best Epoch ({min_val_loss_epoch})')
        ax.set_title(f"{title}\n(Min Validation Loss: {min_val_loss:.4f} at Epoch {min_val_loss_epoch})", fontweight='bold')
        logger.info(f"Minimum validation loss ({min_val_loss:.4f}) occurred at epoch {min_val_loss_epoch}")
    else:
         ax.set_title(title, fontweight='bold')

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    if save_path:
         try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
         except Exception as e:
             logger.error(f"Error saving plot to {save_path}: {e}")
    plt.show()
    # plt.close(fig)

def plot_feature_distribution(df: pd.DataFrame, column: str, title: str,
                              save_path: Optional[str] = None):
    """Plots the distribution of a feature using a histogram and KDE."""
    if df.empty:
        logger.warning(f"DataFrame is empty. Skipping distribution plot '{title}'.")
        return
    if column not in df.columns:
        logger.warning(f"Column '{column}' not found in DataFrame for distribution plot '{title}'. Available columns: {df.columns}")
        return

    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    try:
        sns.histplot(df[column], kde=True, ax=ax, bins=40, color='steelblue', stat='density', edgecolor='k', linewidth=0.5)
        mean_val = df[column].mean()
        median_val = df[column].median()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle=':', linewidth=1.5, label=f'Median: {median_val:.2f}')
        ax.legend()
        ylabel = "Density"
    except Exception as e:
         logger.warning(f"Could not plot distribution with KDE for '{column}'. Error: {e}. Plotting basic histogram.")
         sns.histplot(df[column], ax=ax, bins=40, color='steelblue')
         ylabel = "Frequency"


    ax.set_title(title, fontweight='bold')
    ax.set_xlabel(column)
    ax.set_ylabel(ylabel)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        except Exception as e:
             logger.error(f"Error saving plot to {save_path}: {e}")
    plt.show()
    # plt.close(fig)

def plot_correlation_heatmap(df: pd.DataFrame, title: str, save_path: Optional[str] = None):
    """Plots the correlation heatmap for the DataFrame columns."""
    if df.empty:
        logger.warning(f"DataFrame is empty. Skipping correlation heatmap '{title}'.")
        return
    if df.shape[1] < 2:
        logger.warning(f"DataFrame has less than 2 columns ({df.shape[1]}). Skipping correlation heatmap '{title}'.")
        return

    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 8)) # Adjust size for heatmap
    try:
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax, annot_kws={"size": config.FONT_SIZE - 2})
        ax.set_title(title, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
    except Exception as e:
        logger.error(f"Error generating correlation heatmap: {e}")
        plt.close(fig) # Close the figure if error occurs
        return

    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        except Exception as e:
             logger.error(f"Error saving plot to {save_path}: {e}")
    plt.show()
    # plt.close(fig)


def plot_acf_pacf(series: pd.Series, lags: int, title_prefix: str, save_path_prefix: Optional[str] = None):
    """Plots the ACF and PACF for a given time series."""
    if series.empty:
        logger.warning(f"Series is empty. Skipping ACF/PACF plot for '{title_prefix}'.")
        return
    if not isinstance(series, pd.Series):
        logger.warning(f"Input must be a pandas Series for ACF/PACF plot. Got {type(series)}. Skipping plot.")
        return

    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(config.FIG_SIZE[0], config.FIG_SIZE[1] * 0.6)) # Two plots side-by-side

    try:
        # ACF Plot
        plot_acf(series, lags=lags, ax=axes[0], title=f'{title_prefix} - Autocorrelation (ACF)')
        axes[0].grid(True, linestyle='--', alpha=0.6)

        # PACF Plot
        plot_pacf(series, lags=lags, ax=axes[1], title=f'{title_prefix} - Partial Autocorrelation (PACF)', method='ywm') # Specify method
        axes[1].grid(True, linestyle='--', alpha=0.6)

        fig.suptitle(f'ACF and PACF for {series.name}' if series.name else title_prefix, fontsize=config.FONT_SIZE + 2, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle

        if save_path_prefix:
             acf_save_path = f"{save_path_prefix}_acf_pacf.png"
             try:
                os.makedirs(os.path.dirname(acf_save_path), exist_ok=True)
                plt.savefig(acf_save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to {acf_save_path}")
             except Exception as e:
                 logger.error(f"Error saving ACF/PACF plot to {acf_save_path}: {e}")
        plt.show()
        # plt.close(fig)
    except Exception as e:
        logger.error(f"Error generating ACF/PACF plots for '{title_prefix}': {e}")
        plt.close(fig) # Ensure figure is closed on error


if __name__ == '__main__':
    # Basic logging setup for standalone test
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("Testing plotting module...")
    setup_plot_style()

    # Dummy data for plot structure testing
    # Use a future date range to avoid confusion with real data
    dates_test = pd.to_datetime(pd.date_range(start='2025-01-01', periods=100, freq='H'))

    # Create initial DataFrame with independent columns first
    dummy_data_test = pd.DataFrame({
        'Value1': np.random.rand(100) * 1000 + 500 + np.sin(np.linspace(0, 20, 100))*100,
        'Value2': np.random.rand(100) * 500 + 100 - np.cos(np.linspace(0, 10, 100))*50
    }, index=dates_test)

    # Now add Value3 which depends on Value1 from the created DataFrame
    dummy_data_test['Value3'] = (np.random.rand(100)-0.5)*200 + dummy_data_test['Value1'] * 0.1

    # Now dummy_data_test has all three columns correctly defined
    y_true_dummy_test = dummy_data_test['Value1'].iloc[10:] + np.random.randn(90)*50
    y_pred_dummy_test = dummy_data_test['Value1'].iloc[10:] + np.random.randn(90)*80

    # --- Test plotting functions ---
    logger.info("\nTesting plot_time_series...")
    plot_time_series(dummy_data_test, 'Value1', 'Dummy Time Series', 'Value Unit')

    logger.info("\nTesting plot_actual_vs_predicted...")
    plot_actual_vs_predicted(y_true_dummy_test.values, y_pred_dummy_test.values, dummy_data_test.index[10:],
                             'Dummy Actual vs Predicted', 'Value Unit')

    logger.info("\nTesting plot_learning_curves...")
    # Simulate a Keras history object
    dummy_hist_data = {'loss': np.logspace(0, -2, 20), 'val_loss': np.logspace(0.1, -1.8, 20) + np.random.rand(20)*0.1}
    class MockHistory: history = dummy_hist_data
    plot_learning_curves(MockHistory())

    logger.info("\nTesting plot_feature_distribution...")
    plot_feature_distribution(dummy_data_test, 'Value2', 'Dummy Feature Distribution')

    logger.info("\nTesting plot_correlation_heatmap...")
    plot_correlation_heatmap(dummy_data_test, 'Dummy Feature Correlation')

    logger.info("\nTesting plot_acf_pacf...")
    # Use config.ACF_PACF_LAGS if config is available, else default
    lags_test = config.ACF_PACF_LAGS if 'config' in locals() and hasattr(config, 'ACF_PACF_LAGS') else 48
    plot_acf_pacf(dummy_data_test['Value1'], lags=lags_test, title_prefix="Dummy Value1")

    logger.info("\nPlotting module testing finished.")
    plt.close('all') # Close all figures after testing