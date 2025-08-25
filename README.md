# Stock Price Prediction Using Machine Learning

## Project Overview

This project aims to predict stock prices using machine learning models based on a 20-year dataset of daily stock data for 10 different stocks. The dataset includes key features such as open, high, low, close prices, adjusted close, volume, and stock split information.

Multiple ML models were trained and evaluated to identify the best approach for stock price prediction, focusing primarily on engineered features and standard regression techniques without explicit time series modeling.

---

## Dataset

- **Source:** 20 years of daily stock market data for 10 stocks.
- **Entries:** 21,887 rows spanning 1999 to 2021.
- **Features:**
  - `symbol`: Stock ticker symbol.
  - `date`: Trading date.
  - `open`, `high`, `low`, `close`, `close_adjusted`: Daily price data.
  - `volume`: Trading volume.
  - `split_coefficient`: Stock split information.
  - Additional engineered features such as spread, price change, percentage change, ratios, and binary split flags.
  - One-hot encoded columns for each stock symbol.

---

## Exploratory Data Analysis (EDA)

- Dataset contains no missing values.
- Prices and volumes exhibit strong right-skewed distributions with significant outliers, especially for certain stocks.
- Price columns (`open`, `high`, `low`, `close`) show near-perfect correlations indicating redundancy.
- Volume is weakly correlated with price features.
- Some trading days have zero or non-positive values in price and volume, handled by filtering.
- Stock splits are rare events in the dataset.

---

## Data Preprocessing & Feature Engineering

- Removed records with zero or negative prices or volumes to improve data quality.
- Engineered new features:
  - `spread`: Difference between high and low prices.
  - `price_change`: Difference between close and open prices.
  - `pct_change`: Percent return per day.
  - Price ratios such as `high_low_ratio` and `close_open_ratio`.
  - Log-transformed volume (`log_volume`) to reduce skew.
  - `split_flag` to indicate stock split events.
- One-hot encoded categorical feature `symbol`.
- Dropped redundant price columns (`open`, `high`, `low`, `close`) keeping `close_adjusted` as target.
- Applied RobustScaler to normalize numerical features against outliers.

---

## Models Trained

- Random Forest Regressor
- XGBoost Regressor
- K-Nearest Neighbors Regressor
- Decision Tree Regressor
- Support Vector Regressor (SVR)
- Linear Regression
- AdaBoost Regressor

Each model was trained on 80% of the data and tested on the remaining 20%. Metrics calculated include RMSE, MAE, R-squared, and MAPE.

---

## Results & Comparative Analysis

| Model             | RMSE   | MAE    | R-squared | MAPE (%) |
| ----------------- | ------ | ------ | --------- | -------- |
| Random Forest     | 0.1897 | 0.0758 | 0.9395    | 117.37   |
| XGBoost           | 0.1969 | 0.0905 | 0.9348    | 133.63   |
| KNN               | 0.2659 | 0.1231 | 0.8811    | 137.46   |
| Decision Tree     | 0.2686 | 0.1006 | 0.8786    | 189.57   |
| SVR               | 0.3471 | 0.1793 | 0.7973    | 267.81   |
| Linear Regression | 0.5871 | 0.4280 | 0.4201    | 558.86   |
| AdaBoost          | 0.5126 | 0.4419 | 0.5579    | 659.47   |

- Random Forest delivered the best balance of accuracy and explained variance, making it the recommended model.
- XGBoost closely followed and warrants further hyperparameter tuning for improvement.
- Linear Regression and AdaBoost showed weakest performance, reflecting the complex, nonlinear nature of stock price movements.
- Metrics indicate strong absolute error performance but inflated MAPE values due to small price values; thus, focus more on RMSE, MAE, and R-squared.

---
