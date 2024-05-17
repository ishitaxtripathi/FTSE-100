# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 14:55:22 2024

@author: ishita
"""
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

def arima_prep(file_path, sheet_name):
    # Load data
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    print('Data loaded successfully.')

    # Preprocess data
    data['DATE'] = pd.to_datetime(data['DATE'], format='%Y-%m')

    return data

def arima_analysis(data):
    # Plot original data
    plt.figure(figsize=(12, 6))
    plt.plot(data['DATE'], data['NUMBER'], label='Original Data', color='blue')
    plt.title('Average Weekly Earning Over Time')
    plt.xlabel('Date')
    plt.ylabel('Average Weekly Earning')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # ADF and KPSS tests before differencing
    adf_before_diff = adfuller(data['NUMBER'])
    kpss_before_diff = kpss(data['NUMBER'])

    print('ADF Test (Before differencing):')
    print('ADF Statistic:', adf_before_diff[0])
    print('p-value:', adf_before_diff[1])
    print('Critical Values:', adf_before_diff[4])

    print('\nKPSS Test (Before differencing):')
    print('KPSS Statistic:', kpss_before_diff[0])
    print('p-value:', kpss_before_diff[1])
    print('Critical Values:', kpss_before_diff[3])

    # Differencing the data
    data_diff = data['NUMBER'].diff().dropna()

    # Plot differenced data
    plt.figure(figsize=(12, 6))
    plt.plot(data['DATE'][1:], data_diff, label='1st Differenced Data', color='purple')
    plt.title('1st Differenced Average Weekly Earning')
    plt.xlabel('Date')
    plt.ylabel('1st Differenced Average Weekly Earning')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # ADF and KPSS tests after differencing
    adf_after_diff = adfuller(data_diff)
    kpss_after_diff = kpss(data_diff)

    print('ADF Test (After differencing):')
    print('ADF Statistic:', adf_after_diff[0])
    print('p-value:', adf_after_diff[1])
    print('Critical Values:', adf_after_diff[4])

    print('\nKPSS Test (After differencing):')
    print('KPSS Statistic:', kpss_after_diff[0])
    print('p-value:', kpss_after_diff[1])
    print('Critical Values:', kpss_after_diff[3])

def arima_modeling(data):
    # Plot original data
    plt.figure(figsize=(12, 6))
    plt.plot(data['DATE'], data['NUMBER'], label='Original Data', color='blue')

    # Create ARIMA model
    arima_model = auto_arima(data['NUMBER'], suppress_warnings=True, 
                             seasonal=True, stepwise=True, trace=True, m=12)
        
    print(arima_model.summary())

    # In-sample prediction
    arima_pred = arima_model.predict_in_sample()

    # Calculate metrics
    rmse = np.sqrt(np.mean((data['NUMBER'] - arima_pred)**2))
    mape = np.mean(np.abs((data['NUMBER'] - arima_pred) / data['NUMBER'])) * 100

    # Display metrics
    print('Auto ARIMA RMSE:', rmse)
    print('Auto ARIMA MAPE:', mape)

    # Plot in-sample predictions
    plt.plot(data['DATE'], arima_pred, label='Auto ARIMA Predictions', linestyle=':', color='red')

    # Forecast future values
    arima_forecast_values = arima_model.predict(n_periods=12)
    print('Values forecasted successfully.')

    # Create forecast DataFrame
    forecast_df = pd.DataFrame({'Date': pd.date_range(start='2023-12-01', periods=12, freq='M'),
                                'Auto ARIMA Forecast': arima_forecast_values})

    # Plot forecasted values
    plt.plot(forecast_df['Date'], forecast_df['Auto ARIMA Forecast'], label='Auto ARIMA Forecast', linestyle='--', color='green')

    # Display RMSE and MAPE on the plot
    plt.text(data['DATE'].iloc[-1], data['NUMBER'].max(), f'RMSE: {rmse:.2f}\nMAPE: {mape:.2f}%', color='black',
             bbox=dict(facecolor='white', alpha=0.5))

    plt.title('Combined Plot of Predicted and Forecasted Values')
    plt.xlabel('Date')
    plt.ylabel('Average Weekly Earning')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def arima_modeling_cv(data, n_splits=5):
    # Plot original data
    plt.figure(figsize=(12, 6))
    plt.plot(data['DATE'], data['NUMBER'], label='Original Data', color='blue')

    # Initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Perform cross-validation
    for train_index, test_index in tscv.split(data['NUMBER']):
        train_data, test_data = data['NUMBER'][train_index], data['NUMBER'][test_index]

        # Create ARIMA model
        arima_model = auto_arima(train_data, seasonal=True, m=12)

        # In-sample prediction
        arima_pred = arima_model.predict_in_sample()

        # Out-of-sample prediction
        arima_forecast_values = arima_model.predict(n_periods=len(test_data))

        # Calculate RMSE for each fold
        rmse = np.sqrt(mean_squared_error(test_data, arima_forecast_values))
        print(f'Fold RMSE: {rmse:.2f}')

        # Plot in-sample predictions
        plt.plot(data['DATE'][train_index], arima_pred, linestyle=':', color='red', alpha=0.5)

        # Plot out-of-sample predictions
        plt.plot(data['DATE'][test_index], arima_forecast_values, linestyle='--', color='green', alpha=0.5)

    # Display RMSE on the plot
    plt.text(data['DATE'].iloc[-1], data['NUMBER'].max(), f'Average RMSE: {np.mean(rmse):.2f}', color='black',
             bbox=dict(facecolor='white', alpha=0.5))

    plt.title('Combined Plot of Predicted and Forecasted Values with Cross-Validation')
    plt.xlabel('Date')
    plt.ylabel('Average Weekly Earning')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

file_path = 'Original_35469722.xls'
sheet_name = 'WE'
#arima analyse
weekly_earning = arima_prep(file_path, sheet_name)
arima_analysis(weekly_earning)

# forecast
arima_modeling(weekly_earning)

# Cross-validated forecast
arima_modeling_cv(weekly_earning)

