import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from arch import arch_model
import plotly.graph_objects as go
from datetime import datetime, timedelta
from statsmodels.stats.diagnostic import het_arch
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm

st.title("📈 Stock Volatility Forecasting Analysis Tool")

st.sidebar.header("⚙️ Input Parameters")
ticker = st.sidebar.text_input("Stock Ticker (e.g., AAPL)", "AAPL").upper()

startDate = st.sidebar.date_input("Start Date", datetime(2020, 1, 1))

forecastDays = st.sidebar.slider("Forecast Days", 1, 100, 10)

@st.cache_data
def fetch_data(ticker, start_date):
    end_date = datetime.now()
    try:
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)
        if data.empty:
            raise ValueError("No data returned.")
        if 'Adj Close' not in data.columns:
            st.error("No data returned.")
            return None
        return data['Adj Close']
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

price_data = fetch_data(ticker, startDate)
if price_data is not None:
    returns = 100 * np.log(price_data / price_data.shift(1)).dropna()  # In percentage

    hist_vol = returns.rolling(window=21).std() * np.sqrt(252)  # 21-day rolling, annualized
    hist_vol = hist_vol.dropna()

    try:
        model = arch_model(returns, vol='Garch', p=1, q=1, dist='Normal', rescale=True, mean='zero')
        garch_fit = model.fit(disp='off')

        forecast = garch_fit.forecast(horizon=forecastDays)
        forecast_variance = forecast.variance.iloc[-1].values  # Variance forecast
        forecast_vol = np.sqrt(forecast_variance) * np.sqrt(252)  # Annualized volatility in %

        last_date = returns.index[-1]
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1),
                                       periods=forecastDays, freq='B')  # Business days

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_vol.index, y=hist_vol,
                                 mode='lines', name='Historical Volatility',
                                 line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_vol,
                                 mode='lines', name='GARCH Forecast',
                                 line=dict(color='red', dash='dash')))
        fig.update_layout(
            title=f"Volatility Forecast for {ticker}",
            xaxis_title="Date",
            yaxis_title="Annualized Volatility (%)",
            legend=dict(x=0, y=1),
            template="plotly_white"
        )
        st.plotly_chart(fig)

        st.subheader("Model Summary")
        st.write(garch_fit.summary())  # Restored to original formatted output

        # Diagnostic Tests
        st.subheader("GARCH Model Diagnostic Tests")

        # 1. ARCH-LM Test
        resid = garch_fit.resid
        std_resid = resid / garch_fit.conditional_volatility
        arch_lm_test = het_arch(std_resid, maxlag=10)
        arch_lm_data = {
            "Metric": ["Test Statistic", "P-value", "Interpretation"],
            "Value": [
                f"{arch_lm_test[0]:.4f}",
                f"{arch_lm_test[1]:.4f}",
                "Low p-value (< 0.05) suggests remaining ARCH effects."
            ]
        }
        st.write("Model Diagnostics are a set of statistical tests that evaluate how well the GARCH(1,1) model captures the stock’s volatility patterns. These tests check for issues like remaining volatility clustering (ARCH-LM), asymmetric responses to price changes (Sign Bias), patterns in model errors (Ljung-Box), and whether volatility forecasts are stable over time (Stability Check).")
        st.table(pd.DataFrame(arch_lm_data))

        # 2. Sign Bias Test
        neg_shock = (std_resid < 0).astype(int)
        pos_shock = (std_resid > 0).astype(int)
        sign_bias_data = pd.DataFrame({
            'std_resid_sq': std_resid[1:]**2,
            'neg_shock': neg_shock[:-1],
            'pos_shock': pos_shock[:-1],
            'const': 1
        }).dropna()
        X = sign_bias_data[['const', 'neg_shock', 'pos_shock']]
        y = sign_bias_data['std_resid_sq']
        sign_bias_model = sm.OLS(y, X).fit()
        sign_bias_results = pd.DataFrame({
            "Coefficient": sign_bias_model.params,
            "Std Error": sign_bias_model.bse,
            "P-value": sign_bias_model.pvalues
        }).round(4)
        st.subheader("**Sign Bias Test**")
        st.write("The Sign Bias Test checks whether positive and negative stock returns (price changes) have different impacts on future volatility. In financial markets, negative returns (price drops) often lead to larger volatility spikes than positive returns (price increases) of the same size, a phenomenon called asymmetry. This test examines if such asymmetry exists in the GARCH model's residuals.")
        st.write("What it means for stock volatility: If the test shows significant coefficients for negative or positive shocks (p-value < 0.05), it suggests the GARCH(1,1) model may not fully capture this asymmetry. For example, a significant negative shock coefficient indicates that price drops increase volatility more than price rises, which is common in stocks due to panic selling or risk aversion. In such cases, a more advanced model like EGARCH might better capture this behavior, improving volatility forecasts.")
        st.table(sign_bias_results)
        st.markdown("_Interpretation: Significant coefficients (p < 0.05) for neg_shock or pos_shock suggest asymmetric effects._")

        # 3. Ljung-Box Test
        lb_test = acorr_ljungbox(std_resid, lags=[10], return_df=True)
        lb_data = {
            "Metric": ["Test Statistic (lag 10)", "P-value (lag 10)", "Interpretation"],
            "Value": [
                f"{lb_test['lb_stat'].iloc[0]:.4f}",
                f"{lb_test['lb_pvalue'].iloc[0]:.4f}",
                "Low p-value (< 0.05) indicates autocorrelation in residuals."
            ]
        }
        st.subheader("**Ljung-Box Test for Autocorrelation**")
        st.write("What it means for stock volatility: If the test shows significant coefficients for negative or positive shocks (p-value < 0.05), it suggests the GARCH(1,1) model may not fully capture this asymmetry. For example, a significant negative shock coefficient indicates that price drops increase volatility more than price rises, which is common in stocks due to panic selling or risk aversion. In such cases, a more advanced model like EGARCH might better capture this behavior, improving volatility forecasts.")
        st.table(pd.DataFrame(lb_data))



        # 4. Stability Check
        params = garch_fit.params
        stability_data = {"Metric": [], "Value": []}
        if 'alpha[1]' in params and 'beta[1]' in params:
            persistence = params['alpha[1]'] + params['beta[1]']
            stability_data["Metric"].extend(["Persistence (α + β)", "Result"])
            stability_data["Value"].extend([
                f"{persistence:.4f}",
                "Stationary (α + β < 1)" if persistence < 1 else "Non-stationary (α + β ≥ 1)"
            ])
        else:
            stability_data["Metric"].append("Result")
            stability_data["Value"].append("Unable to compute persistence.")
        st.subheader("**GARCH Stability Check**")
        st.write("The GARCH Stability Check evaluates whether the GARCH(1,1) model’s parameters (α and β) produce stable volatility forecasts. It calculates persistence, defined as α + β, where a value less than 1 indicates that volatility shocks will gradually diminish over time, ensuring the model remains reliable.")
        st.table(pd.DataFrame(stability_data))

        latest_hist_vol = hist_vol[-1] if not hist_vol.empty else np.nan
        first_forecast_vol = forecast_vol[0]
        st.metric("Latest Historical Volatility", f"{latest_hist_vol:.2f}%")
        st.metric(f"{forecastDays}-Day Forecast Volatility (Day 1)", f"{first_forecast_vol:.2f}%")

    except Exception as e:
        st.error(f"Error fitting GARCH model: {e}")

else:
    st.warning("Please enter a valid ticker to proceed.")

st.sidebar.subheader("Instructions")
st.sidebar.write("""
- Enter a stock ticker (e.g., 'AAPL' for Apple).
- Select a start date for historical data.
- Adjust the forecast horizon to predict future volatility.
- View the plot and metrics below.
""")