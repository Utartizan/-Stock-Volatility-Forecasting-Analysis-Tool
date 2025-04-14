import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from arch import arch_model
import plotly.express as px
from datetime import datetime, timedelta
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
import statsmodels.api as sm

# Title
st.title("📈 Stock Volatility Forecasting with GARCH Models")


# Data fetching function
@st.cache_data
def fetch_data(ticker, start_date):
    end_date = datetime.now()
    try:
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)
        if data.empty or 'Adj Close' not in data.columns:
            st.error(f"No data for {ticker}. Please check the ticker.")
            return None
        adj_close = data['Adj Close']
        if isinstance(adj_close, pd.DataFrame):
            adj_close = adj_close.iloc[:, 0]
        return adj_close
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None


# Sidebar inputs
st.sidebar.header("⚙️ Input Parameters")
ticker = st.sidebar.text_input("Stock Ticker (e.g., AAPL)", "MSFT").upper()
start_date = st.sidebar.date_input("Start Date", datetime(2024, 1, 1))
forecast_days = st.sidebar.slider("Forecast Days", 1, 30, 10)
model_type = st.sidebar.selectbox("GARCH Model Type", ["GARCH", "TGARCH"])

# Fetch data
price_data = fetch_data(ticker, start_date)

# Sidebar: Historical price plot
if price_data is not None:
    price_values = price_data.values.flatten() if price_data.values.ndim > 1 else price_data.values
    price_df = pd.DataFrame({'Date': price_data.index, 'Price': price_values})
    fig_price = px.line(
        price_df,
        x='Date',
        y='Price',
        title=f"📈 Historical Price for {ticker}",
        labels={'Price': 'Price (USD)', 'Date': 'Date'},
        template='plotly'
    )
    fig_price.update_traces(line=dict(width=2))
    fig_price.update_layout(
        showlegend=False,
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_rangeslider_visible=True
    )
    st.sidebar.plotly_chart(fig_price, use_container_width=True)

# Sidebar: Instructions and info
st.sidebar.subheader("Instructions")
st.sidebar.write("""
- Enter a stock ticker (e.g., 'AAPL').
- Select a start date and forecast horizon.
- Choose a GARCH model to analyze volatility.
- View results and diagnostics.
""")
st.sidebar.subheader("About")
st.sidebar.write("""
- Created by **Ubayd Knight** (Final Year Student, University of Huddersfield)
- Powered by **Streamlit**, **yfinance**, **arch**, **plotly**, and **statsmodels**.
""")

# Main panel: Volatility analysis
if price_data is not None:
    # Calculate returns
    returns = 100 * np.log(price_data / price_data.shift(1)).dropna()
    returns_values = returns.values

    # Check data sufficiency
    if len(returns_values) < 100:
        st.warning("Data too short (<100 points). Try a longer date range for stable model fitting.")
    elif np.std(returns_values) < 1e-6:
        st.warning("Returns have near-zero variance. Try a different ticker or period.")
    else:
        # Historical volatility
        hist_vol = returns.rolling(window=21).std() * np.sqrt(252)
        hist_vol = hist_vol.dropna()

        try:
            # Configure GARCH model
            if model_type == "TGARCH":
                model = arch_model(returns_values, vol="GARCH", p=1, o=1, q=1, power=1.0, dist="Normal", rescale=True,
                                   mean="zero")
            else:  # GARCH
                model = arch_model(returns_values, vol="Garch", p=1, q=1, dist="Normal", rescale=True, mean="zero")

            # Fit model
            try:
                garch_fit = model.fit(disp='off', options={'maxiter': 100, 'ftol': 1e-6})
            except Exception as fit_error:
                st.error(f"Model fitting failed: {fit_error}. Try a different ticker or date range.")
                st.stop()

            # Forecast volatility
            if model_type == "TGARCH":
                forecast_horizon = 1
                st.info("Note: TGARCH supports only 1-day forecasts due to model complexity.")
            else:
                forecast_horizon = forecast_days

            forecast = garch_fit.forecast(horizon=forecast_horizon, reindex=False)
            forecast_variance = forecast.variance.iloc[-1].values
            forecast_vol = np.sqrt(forecast_variance) * np.sqrt(252)

            # Create forecast dates
            last_date = returns.index[-1]
            forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_horizon, freq='B')

            # Combine historical and forecast volatility
            hist_df = pd.DataFrame({
                'Date': hist_vol.index,
                'Volatility': hist_vol.values,
                'Type': 'Historical'
            })
            forecast_df = pd.DataFrame({
                'Date': forecast_dates,
                'Volatility': forecast_vol,
                'Type': 'Forecast'
            })
            vol_df = pd.concat([hist_df, forecast_df], ignore_index=True)

            # Volatility plot
            fig_vol = px.line(
                vol_df,
                x='Date',
                y='Volatility',
                color='Type',
                title=f"{model_type} Volatility Forecast for {ticker}",
                color_discrete_map={'Historical': '#1f77b4', 'Forecast': '#ff7f0e'},
                labels={'Volatility': 'Annualized Volatility (%)', 'Date': 'Date'},
                template='plotly_dark'
            )
            fig_vol.update_traces(line=dict(width=2), selector=dict(name='Historical'))
            fig_vol.update_traces(line=dict(width=2, dash='dash'), selector=dict(name='Forecast'))
            fig_vol.update_layout(
                legend=dict(x=0.01, y=0.99),
                xaxis_rangeslider_visible=True,
                yaxis=dict(range=[0, max(vol_df['Volatility'].max() * 1.1, 10)])
            )
            st.plotly_chart(fig_vol, use_container_width=True)

            # Model info
            with st.expander("About GARCH Models"):
                st.write("""
                            - **GARCH**: Models symmetric volatility clustering.
                            - **TGARCH**: Emphasizes threshold effects for negative shocks.
                            """)

            # Volatility metrics
            st.subheader("Volatility Metrics")
            st.metric("Latest Historical Volatility - (stock’s recent price volatility based on the past 52 days)", f"{hist_vol.iloc[-1]:.2f}%")
            st.metric(f"Day 1 Forecast Volatility - (Volatility predicted for the next trading day)", f"{forecast_vol[0]:.2f}%")


            # Model summary as table
            st.subheader("Model Summary")
            # Change: Display summary as a clean DataFrame table like diagnostics
            summary_df = pd.DataFrame({
                "Parameter": garch_fit.params.index,
                "Value": [f"{x:.4f}" for x in garch_fit.params.values],
                "Std Error": [f"{x:.4f}" for x in garch_fit.std_err.values],
                "P-value": [f"{x:.4f}" for x in garch_fit.pvalues.values]
            })
            st.write("Model parameters and significance:")
            st.table(summary_df)

            # Diagnostics
            st.subheader("Model Diagnostics")
            resid = garch_fit.resid
            std_resid = resid / garch_fit.conditional_volatility

            # ARCH-LM Test
            arch_lm_test = het_arch(std_resid, nlags=10)
            st.write("**ARCH-LM Test**")
            st.write(
                "Checks for remaining volatility clustering. Low p-value (<0.05) suggests model may need refinement.")
            st.table({
                "Metric": ["Statistic", "P-value"],
                "Value": [f"{arch_lm_test[0]:.4f}", f"{arch_lm_test[1]:.4f}"]
            })

            # Sign Bias Test
            neg_shock = (std_resid < 0).astype(int)
            pos_shock = (std_resid > 0).astype(int)
            sign_bias_data = pd.DataFrame({
                'std_resid_sq': std_resid[1:] ** 2,
                'Negative Shock': neg_shock[:-1],
                'Positive Shock': pos_shock[:-1],
                'Constant': 1
            }).dropna()
            X = sign_bias_data[['Constant', 'Negative Shock', 'Positive Shock']]
            y = sign_bias_data['std_resid_sq']
            sign_bias_model = sm.OLS(y, X).fit()
            st.write("**Sign Bias Test**")
            st.write("Tests for asymmetric shock effects. Significant p-values indicate asymmetry.")
            st.table(pd.DataFrame({
                "Coefficient": sign_bias_model.params.index,
                "Value": sign_bias_model.params.values.round(4),
                "P-value": sign_bias_model.pvalues.round(4)
            }))

            # Ljung-Box Test
            lb_test = acorr_ljungbox(std_resid, lags=[10], return_df=True)
            st.write("**Ljung-Box Test**")
            st.write("Checks for residual autocorrelation. Low p-value (<0.05) suggests model misses patterns.")
            st.table({
                "Metric": ["Statistic", "P-value"],
                "Value": [f"{lb_test['lb_stat'].iloc[0]:.4f}", f"{lb_test['lb_pvalue'].iloc[0]:.4f}"]
            })

            # Stability Check
            params = garch_fit.params
            persistence = params.get('alpha[1]', 0) + params.get('beta[1]', 0)
            if model_type == "TGARCH":
                persistence = "N/A (model-specific)"
                stability = "Check model summary for asymmetry terms."
            else:
                stability = "Stationary" if persistence < 1 else "Non-stationary"
            st.write("**Stability Check**")
            st.write("Ensures model reliability. Persistence <1 indicates stable volatility.")
            st.table({
                "Metric": ["Persistence", "Result"],
                "Value": [str(persistence), stability]
            })





        except Exception as e:
            st.error(f"Error fitting model: {e}")
else:
    st.warning("Please enter a valid ticker to proceed.")