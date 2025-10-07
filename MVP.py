import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock Probability Explorer", layout="centered")
st.title("📈 Stock Probability Explorer")
st.markdown("""
*Enter a ticker to see the probability distribution of where the price might go in 5 days.*  
⚠️ **Not financial advice. For educational purposes only.**
""")

ticker = st.text_input("Enter stock ticker (e.g. AAPL, TSLA, MSFT)", value="AAPL").upper()
forecast_days = 5

if ticker:
    try:
        data = yf.download(ticker, period="1y", progress=False)
        if data.empty:
            st.error("No data found for this ticker. Try another (e.g. AAPL, GOOG, SPY).")
        else:
            close_prices = data['Close']
            close_prices = close_prices.dropna()
            
            if len(close_prices) < 30:
                st.error("Not enough price data. Try a more liquid ticker.")
            else:
                current_price = float(close_prices.iloc[-1])
                returns = close_prices.pct_change().dropna()
                vol = float(returns.std())
                
                if vol == 0 or np.isnan(vol):
                    st.error("Volatility is zero or undefined. Try another ticker.")
                else:
                    # Monte Carlo — только numpy, без pandas
                    np.random.seed(42)
                    daily_returns = np.random.normal(loc=0.0, scale=vol, size=(10000, forecast_days))
                    cumulative_returns = np.cumsum(daily_returns, axis=1)
                    final_cumulative = cumulative_returns[:, -1]
                    future_prices = current_price * np.exp(final_cumulative)
                    
                    # Вероятности
                    up_0_5 = np.mean((future_prices > current_price) & (future_prices <= current_price * 1.05))
                    up_5_10 = np.mean((future_prices > current_price * 1.05) & (future_prices <= current_price * 1.10))
                    down_0_4 = np.mean(future_prices < current_price * 0.96)
                    extreme = np.mean((future_prices > current_price * 1.10) | (future_prices < current_price * 0.90))
                    
                    st.subheader(f"Current price: ${current_price:.2f}")
                    st.write(f"**5-day outlook for {ticker}:**")
                    st.write(f"- 📈 {up_0_5:.0%} chance: +0% to +5%")
                    st.write(f"- 📈 {up_5_10:.0%} chance: +5% to +10%")
                    st.write(f"- 📉 {down_0_4:.0%} chance: down to -4%")
                    st.write(f"- ⚠️ {extreme:.0%} chance: extreme move (>±10%)")
                    
                    # График
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.hist(future_prices, bins=100, density=True, alpha=0.7, color='steelblue')
                    ax.axvline(current_price, color='red', linestyle='--', label='Current Price')
                    ax.set_xlabel('Future Price ($)')
                    ax.set_ylabel('Density')
                    ax.legend()
                    ax.grid(True, linestyle='--', alpha=0.5)
                    st.pyplot(fig)
                    
                    st.caption("Model: Monte Carlo simulation based on 1-year historical volatility. Drift = 0.")
                    
    except Exception as e:
        st.error(f"Error processing {ticker}. Try a major ticker like AAPL, MSFT, or SPY.")
