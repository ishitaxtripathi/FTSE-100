# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 19:34:59 2024

@author: ishita
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import numpy as np
from math import sqrt
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
from scipy.stats import boxcox
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from ETS__35469722 import forecast_df


# Suppress all warnings
warnings.filterwarnings("ignore")

DATE_START = '2000-01-01'
DATE_END = '2023-12-01'
DATE_FORECAST_START = '2024-01-01'
DATE_FORECAST_END = '2024-01-01'
sns.set(style="whitegrid")

def regression_prep(file_path, sheet_name):
    try:
        # Load data
        data = pd.read_excel(file_path, sheet_name=sheet_name)

        # Clean data
        data['DATE'] = pd.to_datetime(data['DATE'], errors='coerce', format='%Y-%m')
        data.set_index('DATE', inplace=True)
        data = data[data.index >= DATE_START]
        data.ffill(inplace=True)
        data['Month'] = data.index.month
        data['Year'] = data.index.year

        # Calculate and display correlation matrix with heatmap
        correlation_matrix = data.iloc[:, :5].corr()
        print("Correlation Matrix:")
        print(correlation_matrix)
        plt.figure(figsize=(10, 8))
        with sns.axes_style("whitegrid"):
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix Heatmap')
        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.show()
        
        # Compute Covariance Matrix
        covariance_matrix = data.iloc[:, :5].cov()
        print("Covariance Matrix:")
        print(covariance_matrix)
        
        # Create Heatmap for Correlation Matrix
        plt.figure(figsize=(10, 8))
        with sns.axes_style("whitegrid"):
            sns.heatmap(covariance_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Cocariance Matrix Heatmap')
        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.show()

        # Plot seasonality
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=data.iloc[:, :5])
        plt.title('Seasonality Plot of First Five Columns')
        plt.xlabel('Date')
        plt.ylabel('Values')
        plt.show()

        # Pair plot with selected variables
        columns_to_plot = data.columns[:5]
        data_to_plot = data[columns_to_plot].reset_index(drop=True)
        sns.pairplot(data_to_plot)
        plt.title('Pair Plot of Selected Variables')
        plt.show()

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
        
    return data
 
def multiple_linear_regression(data, features, target):
    X = data[features].values
    y = data[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    cv_scores=cv_scores.mean()
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X, y)
    
    X_ols = sm.add_constant(X)
    ols_model = OLS(y, X_ols).fit()
    # Extract F-statistic and p-values
    f_statistic = ols_model.fvalue
    p_values = ols_model.pvalues

    print("\nMultiple Linear Regression:")
    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)
    print("R^2 Score:", r2_score(y_test, predictions))
    print("Root Mean Squared Error:", sqrt(mean_squared_error(y_test, predictions)))
    print("Cross-Validated RMSE Scores:", sqrt(-cv_scores))
    print("f-statistic:", f_statistic)
    print("p-VALUE:", p_values)
    print("Feature Importances:", rf_model.feature_importances_)

    plt.scatter(y_test, predictions)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted Values - ML")
    plt.show()

    # Confidence Intervals
    conf_int = np.percentile(predictions, [2.5, 97.5])
    print("Confidence Intervals:")
    print(conf_int)

    return model, conf_int

def multiple_linear_regression_forecast(data, features, target, forecast_data):
    X = data[features].values
    y = data[target].values  
    model = LinearRegression()
    model.fit(X, y)

    # Predict 'OPEN' values for the forecast period
    X_forecast = forecast_data[features].values
    forecasted_open = model.predict(X_forecast)

    # Add the forecasted 'OPEN' values to the forecast DataFrame
    forecast_data['OPEN'] = forecasted_open

    # Confidence Intervals
    conf_int = np.percentile(forecasted_open, [2.5, 97.5])
    print("Confidence Intervals for Forecasted Data:")
    print(conf_int)

    return model, forecast_data, conf_int

# Combined data

file_path = 'Original_35469722.xls'
sheet_name = 'FTSE'
cleaned_data= regression_prep(file_path, sheet_name)

# Removed Box-Cox transformation and any transformation

# Define features for the multiple linear regression
explanatory_variables = ['Month','Year','WE', 'RS', 'PI', 'TO']
target_variable = 'OPEN'
date= 'DATE'
# Multiple Linear Regression model
ml_model, ml_conf_int = multiple_linear_regression(cleaned_data, explanatory_variables, target_variable)

plt.figure(figsize=(12, 6))

# Scatter plot of actual values
plt.scatter(cleaned_data.index, cleaned_data[target_variable], label='Actual Values', alpha=0.5)

# Plot Multiple Linear Regression line
ml_predictions = ml_model.predict(cleaned_data[explanatory_variables].values)

# Filter out infinite values from predictions
ml_predictions = np.where(np.isfinite(ml_predictions), ml_predictions, np.nan)

plt.plot(cleaned_data.index, ml_predictions, label='ML Regression', color='blue')

plt.title(f'Actual Values and Regression Lines for {target_variable}')
plt.xlabel('Date')
plt.ylabel(target_variable)
plt.legend()
plt.show()

forecast_df['Month'] = forecast_df.index.month
forecast_df['Year'] = forecast_df.index.year

combined_data = pd.concat([cleaned_data, forecast_df])

# Check if there is any forecast data
if not forecast_df.empty:
    print("Moving with selected Model :")
    # Separate historical and forecasted data
    historical_data = combined_data.loc[DATE_START:DATE_END].copy()
    forecast_data = combined_data.loc[DATE_END:].copy()
    
    # Multiple Linear Regression for forecasting
    ml_forecast_model, forecast_data, ml_forecast_conf_int = multiple_linear_regression_forecast(
        historical_data, explanatory_variables, target_variable, forecast_data)

     # Plotting with Confidence Intervals
    plt.figure(figsize=(12, 6))
    plt.plot(cleaned_data.index, cleaned_data['OPEN'], label='Actual Data', color='blue')

    # Plot predicted data from historical period
    ml_predictions = ml_forecast_model.predict(historical_data[explanatory_variables].values)
    plt.plot(historical_data.index, ml_predictions, linestyle='dashed', label='Predicted Data (ML)', color='red')

    # Plot forecasted data
    plt.plot(forecast_data.index, forecast_data['OPEN'], linestyle='dashed', label='Forecasted Data (ML)', color='green')

    # Plot confidence intervals for the forecasted data
    lower_bound = ml_forecast_conf_int[0]
    upper_bound = ml_forecast_conf_int[1]
    plt.fill_between(forecast_data.index, lower_bound, upper_bound, color='green', alpha=0.2, label='Confidence Intervals (ML)')

    plt.xlabel('Date')
    plt.ylabel('OPEN')
    plt.legend()
    plt.show()
    with pd.ExcelWriter('Forecasted__35469722.xls', engine='openpyxl') as writer:
        forecast_data.to_excel(writer, sheet_name='results', index=True)
else:
    print("No forecast data available.")
