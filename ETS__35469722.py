#-*- coding: utf-8 -*-
"""
Created on Wed Feb 8 04:04:34 2024

@author: ishit
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from scipy.stats import boxcox, zscore, norm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import TimeSeriesSplit
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

def exponential_smoothing_prep(excel_file_path, sheet_name):
    # Read and preprocess data
    data = pd.read_excel(excel_file_path, sheet_name=sheet_name)
    data['DATE'] = pd.to_datetime(data['DATE'], errors='coerce', format='%Y-%m')
    data.set_index('DATE', inplace=True)
    data['NUMBER'] = pd.to_numeric(data['NUMBER'], errors='coerce')
    data['NUMBER'].interpolate(method='time', inplace=True)
        
    # Display sum of missing values
    sum_of_na = data['NUMBER'].isna().sum()
    print(f"Sum of missing values in {sheet_name}: {sum_of_na}")

    # Descriptive statistics
    descriptive_stats = data['NUMBER'].describe()
    print(f"Descriptive Statistics for {sheet_name}:\n{descriptive_stats}")

    # Data visualization - Histogram
    plt.figure(figsize=(8, 6))
    plt.hist(data['NUMBER'], bins=20, edgecolor='black')
    plt.title(f'Histogram of NUMBER - {sheet_name}')
    plt.xlabel('Number')
    plt.ylabel('Frequency')
    plt.show()
    
  
    # Stationarity analysis
    adf_result = adfuller(data['NUMBER'])
    stationarity_info = {
        'ADF Statistic': adf_result[0],
        'p-value': adf_result[1],
        'Critical Values': adf_result[4]
    }
    print(f"Stationarity Analysis for {sheet_name}:\n{stationarity_info}")

    return data

def exponential_smoothing_analysis(data, sheet_name):
    # ACF and PACF Analysis
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plot_acf(data['NUMBER'], lags=40, ax=plt.gca(), title=f'Autocorrelation Function (ACF) - {sheet_name}')
    plt.subplot(2, 1, 2)
    plot_pacf(data['NUMBER'], lags=40, ax=plt.gca(), title=f'Partial Autocorrelation Function (PACF) - {sheet_name}')
    plt.show()

    # Line Graph with Year as Color
    data['Year'] = data.index.year
    data['Month'] = data.index.month

    plt.figure(figsize=(12, 6))
    for year, subset in data.groupby('Year'):
        plt.plot(subset['Month'], subset['NUMBER'], label=f'Year {year}', marker='o')

    plt.title(f'Line Graph with Year as Color - {sheet_name}')
    plt.xlabel('Month')
    plt.ylabel('Number')
    plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

    decomposition_result = seasonal_decompose(data['NUMBER'], model='additive', period=12)

    # Display the components (trend, seasonal, residual)
    plt.figure(figsize=(12, 8))
    plt.subplot(4, 1, 1)
    plt.plot(decomposition_result.trend, label='Trend')
    plt.title('Trend')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(decomposition_result.seasonal, label='Seasonal')
    plt.title('Seasonal')
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(decomposition_result.resid, label='Residual')
    plt.title('Residual')
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(data['NUMBER'], label='Original')
    plt.title('Original Data')
    plt.legend()

    plt.tight_layout()
    plt.show() 
    
def exponential_smoothing_modeling(data, sheet_name):
    # Define the valid options
    valid_options = {'add', 'mul'}
    
    # Initialize variables to store the best AIC and corresponding option
    best_aic = float('inf')
    best_option = None

    # Iterate over valid options
    for option in valid_options:
        # Run Exponential Smoothing with the current option for both trend and seasonality
        model = ExponentialSmoothing(data['NUMBER'], trend=option, seasonal=option, seasonal_periods=12, freq='MS')
        result = model.fit()
        # Get the AIC for the current option
        current_aic = result.aic

        # Update the best option if the current AIC is lower
        if current_aic < best_aic:
            best_aic = current_aic
            best_option = option

   # Print or return the best option and AIC
    print(f"Best option: '{best_option}' with AIC: {best_aic}")

   # Use the best option to run the final Exponential Smoothing
    hw_model= ExponentialSmoothing(data['NUMBER'], trend=best_option, seasonal=best_option, seasonal_periods=12, freq='MS')    
    fitted_model = hw_model.fit(optimized='powell')

    # Generate forecast index with the original frequency
    forecast_index = pd.date_range(start=data.index[-1], periods=12 + 1, freq='MS')[1:]
       
    # Fitted Values and Forecast
    fitted_values = fitted_model.fittedvalues
    forecast_values = fitted_model.forecast(steps=12)
    
    # Plot and evaluate forecast performance for Holt-Winters Smoothing
    residuals = data['NUMBER'] - fitted_values
    residual_std = residuals.std()

    # Confidence intervals calculated
    confidence_intervals = pd.DataFrame({
        'lower': forecast_values - 1.96 * residual_std,
        'upper': forecast_values + 1.96 * residual_std
    }, index=forecast_values.index)
 
    # Calculate and print performance metrics
    mape = np.mean(np.abs((data['NUMBER'].iloc[1:] - fitted_values) / data['NUMBER'].iloc[1:])) * 100
    rmse = np.sqrt(np.mean((data['NUMBER'].iloc[1:] - fitted_values) ** 2))
    mae = np.mean(np.abs(data['NUMBER'].iloc[1:] - fitted_values))
    mse = np.mean((data['NUMBER'].iloc[1:] - fitted_values) ** 2)

    print(f'MAPE: {mape}, RMSE: {rmse}, MAE: {mae}, MSE: {mse}')
    print("AIC",fitted_model.aic)
    # Retrieve the alpha values
    alpha_values = fitted_model.params['smoothing_level']

    print(f"Alpha Value: {alpha_values}")
    #plot
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['NUMBER'], label=f'Original - {sheet_name}', color='blue')
    plt.plot(data.index, fitted_values, label=f'Fitted Values - {sheet_name}', linestyle='--', color='red')
    plt.plot(data.index, residuals, label=f'Residuals - {sheet_name}')
    plt.plot(forecast_index, forecast_values, label=f'Forecast - {sheet_name}', linestyle='--', color='green')
    plt.fill_between(forecast_index,confidence_intervals.iloc[:, 0], confidence_intervals.iloc[:, 1], color='lightgreen', label='95% Confidence Interval')
    
    plt.title(f'Exponential Smoothing Forecast and Cross-Validation - {sheet_name} MAPE: {mape:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, MSE: {mse:.2f}')
    plt.xlabel('Year')
    plt.ylabel('Pounds')
    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc='upper left', mode='expand', ncol=2)
    plt.show()

    return forecast_index, forecast_values

def cross_validation(data, sheet_name):
    # Specify the number of splits for time series cross-validation
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Initialize a list to store the cross-validation results
    cv_results = []
    
    # Loop through the time series cross-validation splits
    for train_index, test_index in tscv.split(data):
        train_data, test_data = data.iloc[train_index], data.iloc[test_index]
        
        # Fit Holt-Winters additive model
        model = ExponentialSmoothing(train_data['NUMBER'], trend='add', seasonal='add', seasonal_periods=12, freq='MS')
        model_fit = model.fit()

        # Make predictions on the test set
        predictions = model_fit.forecast(len(test_data))
        rmse = np.sqrt(np.mean((test_data['NUMBER'].iloc[1:] - predictions) ** 2))
               
        # Append the results to the list
        cv_results.append(rmse)
    
    # Convert the list to a pandas Series for easy analysis
    cv_results = pd.Series(cv_results, name='Root Mean Squared Error')
    cv_mean= np.mean(cv_results)
    # Print or store the cross-validation results
    print(f'Cross-validation results for {sheet_name}:\n{cv_mean:.2f}')
    
    return cv_results

excel_file_path = 'Original_35469722.xls'
sheet_names = ['WE','RS','PI','TO']

# Create a dictionary to store forecasted values
forecast_dict = {}

for sheet_name in sheet_names:
    # Prep data
    data= exponential_smoothing_prep(excel_file_path, sheet_name)
 
    # Analyse data
    decomposition_result = exponential_smoothing_analysis(data, sheet_name)
    # Apply Holt-Winters Exponential Smoothing model
    forecast_index, forecast_values = exponential_smoothing_modeling(data, sheet_name)
    #cross validation
    cv_results = cross_validation(data, sheet_name)  
    # Store forecast values with index as one of the columns
    forecast_dict[sheet_name] = {'Index': forecast_index, 'Forecast': forecast_values}
    
# Convert the dictionary to a DataFrame
forecast_df = pd.DataFrame({k: v['Forecast'] for k, v in forecast_dict.items()})
    
# Add the index column to the DataFrame
forecast_df['DATE'] = forecast_dict[sheet_names[0]]['Index']
forecast_df.set_index('DATE', inplace=True)
#Display the resulting DataFrame
print(forecast_df)

    