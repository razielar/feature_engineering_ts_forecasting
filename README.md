# Feature Engineering for Time Series Forecasting

Repo from the course is placed [here](https://github.com/trainindata/feature-engineering-for-time-series-forecasting).

1. [Tabularizing Time Series](#one)
2. [Multi-step Forecasting](#two)
3. [Time series Decomposition](#three)
4. [Missing Data Imputation](#four)
5. [Outliers](#five)
6. [Lag features](#six)
7. [Window features](#seven)
8. [Trend features](#eight)
9. [Seasonality Features](#nine)
10. [Date & Time features](#ten)
11. [Categorical features](#eleven)

## 1) <a id='one'></a> Tabularizing Time Series 

### Feature engineering

<div align="center">
<img src="https://github.com/razielar/feature_engineering_ts_forecasting/blob/main/img/features_classification_schema.png" alt="logo"></img>
</div>

### Tabularizing Time Series

<div align="center">
<img src="https://github.com/razielar/feature_engineering_ts_forecasting/blob/main/img/tabularize_ts_ml.png" alt="logo"></img>
</div>

## 2) <a id='two'></a> Multi-step Forecasting

### ML workflow

<div align="center">
<img src="https://github.com/razielar/feature_engineering_ts_forecasting/blob/main/img/ml_workflow.png" alt="logo"></img>
</div>

## 3) <a id='three'></a> Time series Decomposition

### Multiplicative time series example

Air passangers dataset increases as the trend increases, that is the **variance gets larger** as the **trend increases**.

<div align="center">
<img src="https://github.com/razielar/feature_engineering_ts_forecasting/blob/main/img/example_multiplicative_ts.png" alt="logo"></img>
</div>

### MSTL Model

<div align="center">
<img src="https://github.com/razielar/feature_engineering_ts_forecasting/blob/main/img/mstl_model.png" alt="logo"></img>
</div>

**MSTL Frequency**

| Data | Day | Week | Year |
|----------|----------|----------|----------|
| Daily  |  | 7 | 365.25 |
| Hourly | 24 | 168 (24*7) | 8766 |

## 4) <a id='four'></a> Missing Data Imputation

* **1)** Forward filling (last observation carried forward): better than backwards filling.
* **2)** Linear interpolation: better than spline interpolation, because is simpler.
* **3)** Spline interpolation: could dramatically disrupt the time series without EDA.
* **4)** `Seasonal decomposition and linear interpolation`
    * **4.1)** Linear interpolation.
    * **4.2)** Use `STL` or `MSTL` to obtain seasonality. **NOTE**: STL & MSTL assume an `additive model` remember to transform the data.
    * **4.3)** De-seasonalise the original time series.
    * **4.4)** Linear interpolation on the de-seasonalised data.
    * **4.5)** Add the seasonal component back to the imputed de-seasonalised data.

## 5) <a id='five'></a> Outliers

### Outliers in time series data

The outlier classification in time series data comes from [Blazquez-Garcia *et al.* paper](https://arxiv.org/pdf/2002.04236.pdf)

<div align="center">
<img src="https://github.com/razielar/feature_engineering_ts_forecasting/blob/main/img/outlier_clf.png" alt="logo"></img>
</div>

### Estimation methods to identify outliers

* **1)** Rolling mean (mean & std)
* **2)** Rolling median (median & MAD: Median Absolute Deviation)
* **3)** LOWESS residuals
* **4)** STL residuals

## 6) <a id='six'></a> Lag features

<div align="center">
<img src="https://github.com/razielar/feature_engineering_ts_forecasting/blob/main/img/lag_selection_methods.png" alt="logo"></img>
</div>

### Autocorrelation function (ACF)

### Partial autocorrelation function (PACF)

Measures how correlated a $yt$ is with itself at lags: $y{_t-k}$ **after removing** the **effect of intermediate lags**, by substracting the linear impact by a linear regression.  
In practice, we use `ywmle` (Yuke-Walker maximum likelihood estimation) instead of linear regression.

### Cross correlation function (CCF)

Measure how correlated $y_t$ is with another variable at some lag: $x{_t-k}$.

## 7) <a id='seven'></a> Window features

### Rolling Window features

### Expanding Window features

### Weighted Expanding/Rolling Window features

## 8) <a id='eight'></a> Trend features

### Piecewise linear regression

<div align="center">
<img src="https://github.com/razielar/feature_engineering_ts_forecasting/blob/main/img/piecewise_linear_regression.png" alt="logo"></img>
</div>

## 9) <a id='nine'></a> Seasonality Features

* **Seasonality**: A pattern or effect that repeats with **a fixed frequency** (frequency = 1/period) over time.

* **Cyclical patterns**: A pattern or effect that repeats **without a fixed frequency** over time.

| Features to capture seasonality and cyclical patterns |
|----------|----------|
| **Seasonality**                                   | **Cyclical patterns**  |
|---------------------------------------------------|------------------|
| 1. Lag features                                   | 1. Lag features  |
| 2. Calendar features (*aka* datetime features)    |                  |
| 3. Seasonal dummies                               |                  |
| 4. Fourier features                               |                  |



## 10) <a id='ten'></a> Date & Time features

## 11) <a id='eleven'></a> Categorical features


