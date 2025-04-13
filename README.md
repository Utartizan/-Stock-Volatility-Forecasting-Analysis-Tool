# Stock Volatility Forecasting Tool
![image](https://i.ibb.co/qLW8VCVN/u-1126137439-3929098091-fm-253-app-138-f-JPEG-1.jpg)

This is a Streamlit app that forecasts stock volatility using a GARCH(1, 1) model. It pulls historical stock data from Yahoo Finance, calculates historical volatility, predicts future volatility, and runs diagnostic tests to check the model’s reliability. You’ll get a plot with historical (blue) and forecasted (red dashed) volatility, plus some key metrics and test results.

## What It Does

- **Data Input**: Enter a stock ticker (e.g., "AAPL"), pick a start date, and set a forecast horizon (up to 100 days).
- **Volatility Plot**: Shows historical volatility (21-day rolling, annualised) and GARCH-forecasted volatility.
- **Stock Plot**: Displays a mini stock plot for the users selected stock ticker on the sidebar.
- **Diagnostics**: Runs four tests to see if the GARCH model fits the stock’s volatility well:
  - **ARCH-LM Test**: Checks for leftover volatility clustering.
  - **Sign Bias Test**: Looks at whether price drops spike volatility more than price gains.
  - **Ljung-Box Test**: Tests for patterns in the model’s errors.
  - **GARCH Stability Check**: Ensures volatility forecasts don’t spiral out of control.
- **Metrics**: Displays the latest historical volatility and the first day of the forecast.

## How to Use It

1. **Run the App**:
   - Clone this repo: `git clone <https://github.com/Utartizan/Stock-Volatility-Forecasting-Analysis-Tool/blob/main/README.md>`
   - Navigate to the directory: `cd QuantStockVolatilityforecast`
   - Install dependencies (see below).
   - Run the command on a Python terminal: `streamlit run Core.py`
2. **Inputs**:
   - **Ticker**: Type a stock symbol (e.g., "AAPL" for Apple).
   - **Start Date**: Pick a date (default is Jan 1, 2020).
   - **Forecast Days**: Slide between 1 and 100 days (default is 10).
3. **Output**:
   - Check the plot for historical (blue) and forecasted (red dashed) volatility. (To be fixed)
   - Scroll down for the model summary, diagnostic tests, and volatility metrics.

## Setup

You’ll need Python 3.8+ and the following packages. Install them with:

```bash
pip install streamlit pandas numpy yfinance arch plotly statsmodels
