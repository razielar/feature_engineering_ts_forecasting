# Feature Engineering for Time Series Forecasting

Repo from the course is placed [here](https://github.com/trainindata/feature-engineering-for-time-series-forecasting).

1. [Tabularizing Time Series](#one)
2. [Multi-step Forecasting](#two)
3. [Time series Decomposition](#three)
4. [Missing Data Imputation](#four)
5. [Outliers](#five)

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

### MSTL Frequency

| Data | Day | Week | Year |
|----------|----------|----------|----------|
| Daily  |  | 7 | 365.25 |
| Hourly | 24 | 168 (24*7) | 8766 |

## 4) <a id='four'></a> Missing Data Imputation

* **1)** Forward filling (last observation carried forward): better than backwards filling.
* **2)** Backward filling (next observation carried backward): avoid to use it could introduce `data leakage`.
* **3)** Linear interpolation: better than spline interpolation, because is simpler.
* **4)** Spline interpolation: could dramatically disrupt the time series without EDA.
* **5)** `Seasonal decomposition and linear interpolation`
    * **5.1)** Linear interpolation.
    * **5.2)** Use `STL` or `MSTL` to obtain seasonality. **NOTE**: STL & MSTL assume an `additive` model remember to transform the data.
    * **5.3)** De-seasonalise the original time series.
    * **5.4)** Linear interpolation on the de-seasonalised data.
    * **5.5)** Add the seasonal component back to the imputed de-seasonalised data.

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

