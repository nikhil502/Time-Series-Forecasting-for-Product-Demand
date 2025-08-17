**Time-Series Sales Forecasting for Product Demand**
**Overview**
This project implements time-series forecasting for daily sales data using multiple models: ARIMA/SARIMA, XGBoost, Holt-Winters (Exponential Smoothing), and Prophet (with additive and multiplicative seasonality). The goal is to predict future sales based on historical data, leveraging statistical and machine learning approaches. The project includes data preprocessing, feature engineering, model training, hyperparameter tuning, evaluation, and visualization.
Dataset
The dataset (sales.csv) contains daily sales data from October 1, 2021, to September 30, 2022 (365 days). It includes:

**Columns:**
Date: Date of sales (datetime).
Sales: Daily sales values (float).


Source: Simulated sales data with time-series features, hosted on Kaggle.
Characteristics: No missing values or duplicates, with potential weekly seasonality and an upward trend toward the end of the period.

Models
The project implements the following forecasting models:

**ARIMA/SARIMA:**
Uses pmdarima.auto_arima to automatically select optimal parameters.
Models non-seasonal (ARIMA) and seasonal (SARIMA) patterns, with a weekly seasonal period (m=7).


**XGBoost:**
A machine learning model with feature engineering (lag features: 1-day and 7-day lags, 7-day rolling mean).
Hyperparameter tuning performed using GridSearchCV with parameters like n_estimators, max_depth, and learning_rate.


**Holt-Winters (Exponential Smoothing):**
Uses statsmodels.tsa.holtwinters.ExponentialSmoothing with additive trend and seasonality (weekly period).
Captures trends and seasonal patterns in a computationally efficient manner.


**Prophet:**
Implements Facebook’s Prophet with both additive and multiplicative seasonality modes.
Configured with weekly and daily seasonality, suitable for capturing complex patterns.



**Evaluation**

Metrics: Root Mean Squared Error (RMSE) and Mean Absolute Percentage Error (MAPE) are used to evaluate model performance. RMSE measures prediction error magnitude, while MAPE provides error as a percentage of actual values, offering insight into relative accuracy.
Train-Test Split: 80% training (292 days), 20% testing (73 days).
Visualization: Plots compare actual vs. forecasted sales for each model, with confidence intervals for ARIMA/SARIMA and Prophet.

**Project Structure**

time-series-sales-forecasting (1).ipynb: Main Jupyter Notebook with data loading, preprocessing, model implementations, and visualizations.
sales.csv: Dataset containing daily sales data (located in /kaggle/input/simulated-sales-data-with-timeseries-features/).
README.md: This file, providing an overview and instructions.

Requirements
To run the project, install the required Python libraries:
pip install pandas numpy matplotlib seaborn statsmodels pmdarima prophet xgboost scikit-learn

**Data Preprocessing:**
Loads and cleans the dataset, converting Date to datetime and dropping unnecessary columns.
Checks for missing values and duplicates (none found).


**Feature Engineering (for XGBoost):**
Creates lag features (1-day, 7-day) and a 7-day rolling mean.


**Model Training:**
ARIMA/SARIMA: Automatically selects parameters using auto_arima.
XGBoost: Trains with tuned hyperparameters (e.g., n_estimators=100, max_depth=5).
Holt-Winters: Fits with additive trend and seasonality.
Prophet: Trains with additive and multiplicative seasonality modes.


**Evaluation and Visualization:**
Computes RMSE and MAPE for each model on the test set.
Generates plots comparing actual vs. forecasted sales.



**Results**

ARIMA/SARIMA: Captures weekly seasonality but may struggle with complex trends.
XGBoost: Performs well with proper feature engineering; tuned parameters improve accuracy.
Holt-Winters: Effective for stable seasonal patterns, computationally lightweight.
Prophet: Robust for handling seasonality; multiplicative mode may outperform additive if seasonality scales with trend.

**ARIMA : RMSE = 2.19, MAPE = 4.34%

SARIMA: RMSE = 1.6, MAPE = 3.1%

XGBoost: RMSE = 2.10, MAPE = 4%

Holt-Winters: RMSE = 0.92, MAPE = 1.8%

Prophet (Additive): RMSE = 0.94, MAPE  = 1.84%

Prophet (Multiplicative): RMSE =1.53, MAPE = 3.02%**


Visualizations show forecasts aligning with test data, with Prophet and XGBoost often capturing trends effectively.

**Future Improvements**

Feature Engineering: Add more features (e.g., day of week, month, holidays) for XGBoost and Prophet.
Cross-Validation: Implement TimeSeriesSplit for robust model evaluation.
Ensemble: Combine forecasts from multiple models for improved accuracy.
Seasonality Tuning: Test different seasonal periods (e.g., monthly) for SARIMA and Holt-Winters.
Component Analysis: Use Prophet’s plot_components or seasonal decomposition to analyze trends and seasonality.

**Conclusion**

• Analyzed daily sales and decomposed into level, trend, seasonality, performing ADF test to check stationarity

• Forecasted 90-day sales with Holt’s Winter, SARIMA, ARIMA, and Prophet using lags and date-based features

• Attained MAPE of 1.8% with Holt-Winters, 1.84% with Prophet and surpassing others in forecasting accuracy

License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

Dataset sourced from Kaggle.
Built with Python libraries: pandas, numpy, matplotlib, seaborn, statsmodels, pmdarima, prophet, xgboost, scikit-learn.
