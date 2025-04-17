# Power Time-Series Forecasting
This project focuses on wind turbine power forecasting, employing two distinct approaches: direct forecasting using Long Short-Term Memory (LSTM) networks and indirect forecasting using a combination of Random Forest Regression and LSTM. The objective is to assess and compare the performance of these two methodologies.


## Dataset
The project utilizes the "Wind Turbine Scada Dataset," accessible at [Kaggle](https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset/). This dataset contains various features, such as wind speed, wind direction, power curve, and generated power. It comprises 50,530 samples recorded at 10-minute intervals from January 1, 2018, to December 31, 2018. The dataset originates from a wind turbine's SCADA system actively generating power in Turkey.


## Dataset Processing:
- Sampling Rate: The original dataset's 10-minute interval is adjusted to a 1-hour interval.
- Splitting Dataset: A train/valid/test split with a ratio of 0.7/0.1/0.2 is applied.
- Min-Max Standardization: The dataset is standardized using Min-Max scaling to ensure that all features have the same scale (This is performed on the input features of the regression model).


## First Approach: Direct Forecasting (LSTM-based)
In this approach, the LSTM model is exclusively used for univariate forecasting of the "Generated Power" feature.


### Hyperparameters:
- Optimizer: Adam
- Learning Rate: 0.001
- Epochs: 100


### Model Configuration:
- Input window size: 5


### Training Process:
- Model checkpoints are saved for the best model.
- Performance metrics, including MAE, MAPE, RMSE, R2, IA, and SDE, are computed for training, validation, and test sets.
- Results are saved in the "RNN_results_Approach_1.csv" CSV file.
- Plots comparing ground truth and predictions are generated for different sample sizes (2-days and 2-weeks) and saved as "results_plot_Approach_1.png."



## Second Approach (Indirect Forecasting)
This approach combines a regression model (Random Forest Regression) and a forecasting model (LSTM) to achieve multivariate prediction.


### Regression Model:
- Inputs: Wind speed, wind direction, power curve
- Output: Generated power
- The model is trained with the same split ratio and without shuffling for comparison purposes.


### Forecasting Model:
- LSTM is utilized for multivariate prediction of wind speed, wind direction, and power curve.
- The LSTM model is configured identically to the one used in the first approach.


### Combined Forecasting:
- LSTM predictions are used as the "test set" for the trained random forest regression model to predict the corresponding "generated power" variable.


### Performance Metrics:
- Metrics are computed for both direct and indirect forecasting approaches.
- Results are saved in the "Combined_Model_Results.csv" CSV file.
- Plots comparing the predictions of both approaches are generated and saved as "combined_results_subplots.png."



## Results & Plots

### Rsults for Approach 1:
![results_plot_Approach_1](https://github.com/ahmad2246/EE491-Project/assets/130466567/a62e6658-dbc4-494c-8c09-b5b1cab41c3d)

![results_1](https://github.com/ahmad2246/EE491-Project/assets/130466567/62af0c67-b635-4178-9a92-04a57a0f2f1d)



### Results for Approach 1 vs Approach 2:
![combined_results_subplots](https://github.com/ahmad2246/EE491-Project/assets/130466567/6b18d6fd-4e77-4c55-835f-1cb8c1a29118)

![results_2](https://github.com/ahmad2246/EE491-Project/assets/130466567/bfc951a0-0acf-43f4-b6b8-9db37ef65571)



## Usage
1. Ensure the required libraries are installed.

> * tensorflow
> * numpy
> * pandas
> * matplotlib
> * statsmodels
> * sickit-learn

You can run the following code to install them:

```
pip install tensorflow numpy pandas matplotlib statsmodels scikit-learn
```

2. Run the code in the provided Jupyter Notebook (FINAL.ipynb).


## Files
* FINAL.ipynb: The main Jupyter Notebook containing the code.
* RNN_results_Approach_1.csv: CSV file containing results for the first approach.
* Combined_Model_Results.csv: CSV file containing results for the combined models.
* results_plot_Approach_1.png: Plot of results for the first approach.
* combined_results_subplots.png: Combined plot comparing results from both approaches.