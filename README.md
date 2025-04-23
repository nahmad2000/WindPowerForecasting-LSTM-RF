# Wind Power Forecasting using LSTM

This project focuses on forecasting wind turbine power generation using time-series analysis, specifically employing an LSTM (Long Short-Term Memory) neural network.

## Project Structure

The repository is organized as follows:

```

WindPowerForecasting/

├── data/

│ └── wind_turbine_data.csv # Renamed raw data (originally T1.csv)

├── notebooks/

│ └── 01_Data_Analysis_and_Modeling.ipynb # Main notebook for analysis & modeling

├── results/

│ ├── metrics_lstm_direct.csv # Performance metrics for the LSTM model

│ ├── metrics_comparison.csv # Comparison metrics (if multiple models implemented)

│ └── saved_models/

│ └── best_lstm_direct_model.h5 # Saved trained model weights

├── images/ # Saved plots from the analysis

│ ├── target_timeseries.png

│ ├── target_distribution.png

│ ├── windspeed_distribution.png

│ ├── lstm_learning_curves.png

│ └── lstm_actual_vs_predicted_test.png

│ └── ... (other generated plots)

├── src/ # Source code modules

│ ├── init.py # Makes src a Python package

│ ├── config.py # Configuration parameters

│ ├── data_preprocessing.py # Data loading & preprocessing functions

│ ├── modeling.py # Model building & training functions

│ ├── evaluation.py # Evaluation metric functions

│ └── plotting.py # Plotting functions

├── README.md # This file

└── requirements.txt # Project dependencies

```

## Dataset

The dataset used is sourced from Kaggle ([Link to Kaggle Dataset - *replace if available*](https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset)) and contains time-series data for a wind turbine, including:
* `Date/Time`: Timestamp of the recording.
* `LV ActivePower (kW)`: The target variable (power generated).
* `Wind Speed (m/s)`: Measured wind speed.
* `Theoretical_Power_Curve (KWh)`: Theoretical power output based on wind speed.
* `Wind Direction (°)`: Wind direction.

The raw data (`T1.csv`) should be placed in the `data/` directory and renamed to `wind_turbine_data.csv`.

## Methodology

The primary approach demonstrated in the main notebook (`01_Data_Analysis_and_Modeling.ipynb`) is **Direct Forecasting using LSTM**:

1.  **Preprocessing**: The raw data is loaded, datetime index is set, data is resampled to an hourly frequency (configurable in `src/config.py`), missing values handled (ffill/bfill), and features are scaled using `MinMaxScaler`.
2.  **Sequence Creation**: The time series data is transformed into overlapping sequences suitable for LSTM input (using past `N` hours to predict the next hour, where `N` is `SEQUENCE_LENGTH` in `config.py`).
3.  **LSTM Modeling**: A sequential LSTM model with two LSTM layers followed by a Dense output layer is built using Keras/TensorFlow. Architecture details are in `src/modeling.py`.
4.  **Training**: The model is trained on the training dataset and validated on a separate validation set. `ModelCheckpoint` saves the best model based on validation loss, and `EarlyStopping` prevents overfitting.
5.  **Evaluation**: The trained model's performance is evaluated on the training, validation, and test sets using metrics like MAE, MAPE, RMSE, R2, IA, and SDE (calculated in `src/evaluation.py`).

*(Future work could include implementing an indirect approach, potentially using Random Forest with LSTM features).*

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd WindPowerForecasting
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Ensure you have the correct version of TensorFlow (`tensorflow` or `tensorflow-cpu`) installed as specified/chosen in `requirements.txt`.*

## Usage

1.  Ensure the raw data file (`T1.csv` from your original upload) is renamed to `wind_turbine_data.csv` and placed in the `data/` directory.
2.  Launch Jupyter Lab or Jupyter Notebook from the **project root directory** (`WindPowerForecasting/`):
    ```bash
    jupyter lab
    # or
    jupyter notebook
    ```
3.  Navigate into the `notebooks/` directory within the Jupyter interface.
4.  Open and run the `01_Data_Analysis_and_Modeling.ipynb` notebook cell by cell.

The notebook will execute the entire workflow, from data loading to evaluation, saving results (metrics CSVs, model file) to the `results/` directory and plots to the `images/` directory.

## Results Highlights

The performance of the final LSTM model on the **test set** is summarized below (update these values after running the notebook):

| Metric      | Value   |
|-------------|---------|
| MAE         | [Value] |
| MAPE (%)    | [Value] |
| RMSE        | [Value] |
| R2          | [Value] |
| IA          | [Value] |
| SDE         | [Value] |

**Actual vs Predicted Power (Test Set):**

![LSTM Actual vs Predicted Test Set](images/lstm_actual_vs_predicted_test.png)
*(This image will be generated in the `images/` folder when you run the notebook)*

**Learning Curves:**

![LSTM Learning Curves](images/lstm_learning_curves.png)
*(This image will be generated in the `images/` folder when you run the notebook)*

