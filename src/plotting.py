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

# Import configuration relative to src
try:
    from . import config
except ImportError:
    # Fallback for running the script directly
    import config

def setup_plot_style():
    """Sets the plot style and font sizes."""
    try:
        plt.style.use(config.PLOT_STYLE)
    except:
        print(f"Warning: Plot style '{config.PLOT_STYLE}' not found. Using default.")
        plt.style.use('default')

    plt.rcParams.update({'font.size': config.FONT_SIZE})
    plt.rcParams['axes.labelsize'] = config.FONT_SIZE
    plt.rcParams['axes.titlesize'] = config.FONT_SIZE + 1 # Slightly larger title
    plt.rcParams['xtick.labelsize'] = config.FONT_SIZE - 2
    plt.rcParams['ytick.labelsize'] = config.FONT_SIZE - 2
    plt.rcParams['legend.fontsize'] = config.FONT_SIZE - 1 # Slightly larger legend
    plt.rcParams['figure.titlesize'] = config.FONT_SIZE + 2
    plt.rcParams['figure.facecolor'] = 'white' # Ensure white background for saving
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'


def plot_time_series(df, column, title, ylabel, save_path=None):
    """Plots a specific time series column."""
    if column not in df.columns:
        print(f"Warning: Column '{column}' not found in DataFrame for plotting.")
        return
    if df.empty:
        print("Warning: DataFrame is empty. Skipping plot.")
        return

    setup_plot_style()
    fig, ax = plt.subplots(figsize=config.FIG_SIZE)
    ax.plot(df.index, df[column], label=column, linewidth=1.5)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.legend()
    # Improve date formatting - adjust frequency based on time range if needed
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) # Adjust format as needed
    plt.xticks(rotation=30, ha='right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        except Exception as e:
            print(f"Error saving plot to {save_path}: {e}")
    plt.show()

def plot_actual_vs_predicted(y_true, y_pred, data_index, title, ylabel, save_path=None):
    """Plots actual vs predicted values over time."""
    if len(y_true) == 0 or len(y_pred) == 0:
         print("Warning: Empty data arrays provided for actual vs predicted plot.")
         return

    setup_plot_style()
    fig, ax = plt.subplots(figsize=config.FIG_SIZE)

    # Ensure index aligns with the predictions/true values
    if len(data_index) != len(y_true) or len(data_index) != len(y_pred):
         print(f"Warning: Index length ({len(data_index)}) does not match data lengths (True: {len(y_true)}, Pred: {len(y_pred)}). Plotting against sequence number.")
         plot_index = np.arange(len(y_true))
         xlabel = "Sample Index"
         time_axis = False
    elif isinstance(data_index, pd.DatetimeIndex):
         plot_index = data_index
         xlabel = "Time"
         time_axis = True
    else:
         plot_index = data_index
         xlabel = "Index"
         time_axis = False


    ax.plot(plot_index, y_true, label='Actual', linewidth=2, color='royalblue')
    ax.plot(plot_index, y_pred, label='Predicted', linestyle='--', linewidth=1.5, color='darkorange', alpha=0.9)

    ax.set_title(title, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    if time_axis: # Format x-axis if it's datetime
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        plt.xticks(rotation=30, ha='right')

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        except Exception as e:
             print(f"Error saving plot to {save_path}: {e}")
    plt.show()

def plot_learning_curves(history, title="Model Training History", save_path=None):
    """Plots training & validation loss curves."""
    if not hasattr(history, 'history') or 'loss' not in history.history:
        print("Warning: Invalid history object passed to plot_learning_curves.")
        return

    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 5)) # Smaller figure for loss curves

    history_df = pd.DataFrame(history.history)
    epochs_ran = range(1, len(history_df['loss']) + 1)

    ax.plot(epochs_ran, history_df['loss'], label='Training Loss', color='mediumblue', marker='o', markersize=4, linestyle='-')
    if 'val_loss' in history_df:
        ax.plot(epochs_ran, history_df['val_loss'], label='Validation Loss', color='darkorange', marker='x', markersize=5, linestyle='--')

    ax.set_title(title, fontweight='bold')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.legend()
    # Find minimum validation loss epoch if available
    if 'val_loss' in history_df:
        min_val_loss_epoch = history_df['val_loss'].idxmin() + 1
        min_val_loss = history_df['val_loss'].min()
        ax.axvline(min_val_loss_epoch, color='red', linestyle=':', linewidth=1.5, label=f'Best Epoch ({min_val_loss_epoch})')
        ax.legend() # Update legend to include vline label
        print(f"Minimum validation loss ({min_val_loss:.4f}) at epoch {min_val_loss_epoch}")


    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_path:
         try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
         except Exception as e:
             print(f"Error saving plot to {save_path}: {e}")
    plt.show()

def plot_feature_distribution(df, column, title, save_path=None):
    """Plots the distribution of a feature using a histogram and KDE."""
    if column not in df.columns:
        print(f"Warning: Column '{column}' not found in DataFrame for distribution plot.")
        return
    if df.empty:
        print("Warning: DataFrame is empty. Skipping distribution plot.")
        return

    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df[column], kde=True, ax=ax, bins=30, color='steelblue', stat='density')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel(column)
    ax.set_ylabel("Density")
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        except Exception as e:
             print(f"Error saving plot to {save_path}: {e}")
    plt.show()


if __name__ == '__main__':
    print("Testing plotting module...")
    setup_plot_style() # Set style for dummy plots

    # Dummy data for plot structure testing
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=100, freq='H'))
    dummy_data = pd.DataFrame({
        'Value1': np.random.rand(100) * 1000 + 500,
        'Value2': np.random.rand(100) * 500 + 100
    }, index=dates)
    y_true_dummy = dummy_data['Value1'].iloc[10:] + np.random.randn(90)*50
    y_pred_dummy = dummy_data['Value1'].iloc[10:] + np.random.randn(90)*80

    # Test time series plot
    print("\nTesting plot_time_series...")
    plot_time_series(dummy_data, 'Value1', 'Dummy Time Series', 'Value Unit')

    # Test actual vs predicted plot
    print("\nTesting plot_actual_vs_predicted...")
    plot_actual_vs_predicted(y_true_dummy.values, y_pred_dummy.values, dummy_data.index[10:],
                             'Dummy Actual vs Predicted', 'Value Unit')

    # Test learning curves plot
    print("\nTesting plot_learning_curves...")
    dummy_history = {'loss': np.logspace(0, -2, 20), 'val_loss': np.logspace(0.1, -1.8, 20) + np.random.rand(20)*0.1}
    dummy_hist_obj = type('obj', (object,), {'history': dummy_history}) # Mock history object
    plot_learning_curves(dummy_hist_obj)

    # Test distribution plot
    print("\nTesting plot_feature_distribution...")
    plot_feature_distribution(dummy_data, 'Value2', 'Dummy Feature Distribution')

    print("\nPlotting module testing finished.")