# Time Series Analysis

In this repo I will compare and predict the market capitalisation of GM and Tesla using time series analysis techniques

Methods & Techniques Use:
* API
* Moving Average
* Regression
* ARIMA
* Prophet/FBProphet
* LSTM

## Data Gathering and exploration

I obtained the data for the stock price of GM & Tesla using Quandl API

<p align="left">
  <img src="https://github.com/joekryan/time_series_analysis/blob/main/images/gm_tesla_close_comp.png">
</p>

As you can see, the price for the stocks is different by an order of magnitude, so it is better to compare the market cap. This data is not provided by Quandl API but can be computed by multiplying the average number of shares outstanding in each year by the share price on a given day. This doesn't give an exact figure, but it will be good enough to demonstrate time series analysis.

<p align="left">
  <img src="https://github.com/joekryan/time_series_analysis/blob/main/images/gm_tesla_mcap_comp.png">
</p>

## Analysis
### 1. Moving Average
The simplest way to predict future values is just to take the average of a set of previously observed values. The predicted closing price for each day will be the average of a set of previously observed values. Instead of using the simple average, this is the moving average, which uses the latest set of values for each prediction. I.e. for each subsequent step, the predicted values are taken into consideration while removing the oldest observed value from the set.

<p align="left">
  <img src="https://github.com/joekryan/time_series_analysis/blob/main/images/tesla_moving_average_pred.png">
</p>

As you can see, this is not very accurate but it gives us a good baseline to compare against

### 2. Regression

Regression uses input variables to create a formula that predicts an output variable (market cap in this case). To improve the performance of the regression model, I used fastai' add_datepart library to add features to the model by expanding datetime into year,	month, week, day, day of week, day of year,	Is_month_end, Is_month_start, Is_quarter_end, Is_quarter_start, Is_year_end and	Is_year_start.

<p align="left">
  <img src="https://github.com/joekryan/time_series_analysis/blob/main/images/tesla_regression_pred.png">
</p>

This is still not particularly accurate, however it does begin to capture some of the up and down nature of market cap fluctuation.

### 3. ARIMA

ARIMA (Auto Regressive Integrated Moving Average) models take into account the past values to predict the future values.There are three important hyperparameters in ARIMA:

* p (past values used for forecasting the next value)
* q (past forecast errors used to predict the future values)
* d (order of differencing)

In this case I am using an Auto ARIMA model, where these hyperparameters are automatically selected to minimise error.

<p align="left">
  <img src="https://github.com/joekryan/time_series_analysis/blob/main/images/tesla_arima_pred.png">
</p>

This follows the general trend a bit closer, but we can do better

### 4. SARIMAX

SARIMAX is similar to ARIMA, it stands for Seasonal AutoRegressive Integrated Moving Averages with eXogenous regressor. A problem with ARIMA is that it does not support seasonal data (i.e. time series with a repeating cycle).

<p align="left">
  <img src="https://github.com/joekryan/time_series_analysis/blob/main/images/tesla_sarima_pred.png">
</p>

In this case, I have used a SARIMAX model that predicts one month ahead. As you can see, it fits the data very well, however, what if we want to predict more than one month ahead?

### 5. Prophet

Prophet, designed and pioneered by Facebook, is a time series forecasting library that requires no data preprocessing and is extremely simple to implement. The input for Prophet is a dataframe with two columns: date and target

<p align="left">
  <img src="https://github.com/joekryan/time_series_analysis/blob/main/images/tesla_prophet_pred.png">
</p>

This does not perform very well as Prophet, like ARIMA/SARIMA, tries  to capture the trend and seasonality from past data. This works very well for cases like predicting inventory or sales, but for stocks or market cap it is less good as the seasonality and trend is much weaker. 

However, it is still worth exploring the other features of Prophet to show it's power and explore the seasonal trends withing the data.

<p align="left">
  <img src="https://github.com/joekryan/time_series_analysis/blob/main/images/gm_tesla_mcap_prophet_pred.png">
</p>

<p align="left">
  <img src="https://github.com/joekryan/time_series_analysis/blob/main/images/tesla_forecast.png">
</p>
<p align="left">
  <img src="https://github.com/joekryan/time_series_analysis/blob/main/images/gm_forecast.png">
</p>

You can see from these trend decomposition diagrams that Tesla market cap tends to increase during the summer and decrease during the winter, whereas GM has the opposite trend. You can also see that the market cap does not increase over the weekend (as the stock markets are shut).

### 6. LSTM

Finally, I will use a LSTM (Long Short Term Memory) Neural Network. LSTMs have an edge over conventional feed-forward neural networks and RNN when it comes to time series predictions. This is because of their property of selectively remembering patterns for long durations of time. Essentially they can remeber useful information while forgetting everything else.

<p align="left">
  <img src="https://github.com/joekryan/time_series_analysis/blob/main/images/tesla_mcap_lstm_pred.png">
</p>

This prediction is extremely accurate, even up to a year out. However, LSTMs, like all neural networks, are prone to overfitting.

## Conclusion

Time series analyses based on trend and seasonality are of limited use when it comes to predicting market cap over anything other than the short term. For longer term predictions, neural nets like LSTM perform much better. However, you must be wary of overfitting your neural net.
