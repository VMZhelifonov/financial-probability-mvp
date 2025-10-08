import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm

st.set_page_config(page_title="Advanced Stochastic Stock Forecaster", layout="centered")
st.title("📈 Advanced Stochastic Stock Forecaster")
st.markdown("""
*Professional-grade probabilistic forecasting using Heston and SABR stochastic volatility models.*  
⚠️ **Not financial advice. For educational/research purposes only.**
""")

ticker = st.text_input("Enter stock ticker (e.g. AAPL, TSLA, MSFT)", value="AAPL").upper()
forecast_days = st.slider("Forecast horizon (days)", min_value=1, max_value=30, value=5)
model_choice = st.selectbox("Stochastic model", ["Heston", "SABR", "GBM (Baseline)"])

if ticker:
    try:
        data = yf.download(ticker, period="2y", progress=False)  # больше данных для калибровки
        if data.empty:
            st.error("No data found. Try a liquid ticker (e.g. AAPL, SPY).")
        else:
            close_prices = data['Close'].dropna()
            if len(close_prices) < 60:
                st.error("Not enough data. Need ≥60 days.")
            else:
                current_price = float(close_prices.iloc[-1])
                log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
                sigma_hist = float(log_returns.std() * np.sqrt(252))  # annualized vol

                # ----------------------------
                # 1. GBM (Baseline)
                # ----------------------------
                def simulate_gbm(S0, vol, T, n_paths=20000, seed=42):
                    np.random.seed(seed)
                    dt = 1/252
                    steps = int(T * 252)
                    Z = np.random.randn(n_paths, steps)
                    logS = np.log(S0) + np.cumsum(-0.5 * vol**2 * dt + vol * np.sqrt(dt) * Z, axis=1)
                    return np.exp(logS[:, -1])

                # ----------------------------
                # 2. Heston Model
                # dS/S = sqrt(v) dW1
                # dv = κ(θ - v)dt + ξ sqrt(v) dW2
                # corr(dW1, dW2) = ρ
                # ----------------------------
                def calibrate_heston(log_returns):
    # Убедимся, что данные валидны
    if len(log_returns) < 20:
        # Возвращаем разумные дефолтные параметры
        sigma2 = 0.04  # 20% годовой волатильности → var = 0.2^2 = 0.04
        return np.array([2.0, sigma2, 0.3, -0.5, sigma2])
    
    # Очистка от NaN/inf
    log_returns = log_returns[np.isfinite(log_returns)]
    if len(log_returns) < 20:
        sigma2 = 0.04
        return np.array([2.0, sigma2, 0.3, -0.5, sigma2])

    try:
        hist_var = np.var(log_returns) * 252
        hist_var = np.clip(hist_var, 1e-4, 1.0)  # ограничим разумными пределами
        kurt_hist = ((np.mean((log_returns - np.mean(log_returns))**4) /
                     (np.var(log_returns)**2)) - 3) if np.var(log_returns) > 1e-8 else 0.0
        kurt_hist = np.clip(kurt_hist, 0.0, 20.0)
    except:
        hist_var = 0.04
        kurt_hist = 3.0

    # Начальные параметры как np.array
    x0 = np.array([2.0, hist_var, 0.3, -0.5, hist_var])

    def loss(params):
        kappa, theta, xi, rho, v0 = params
        # Проверка границ
        if (kappa <= 0.01 or theta <= 1e-4 or xi <= 0.01 or v0 <= 1e-4 or 
            abs(rho) >= 0.999 or kappa > 50 or theta > 1.0 or xi > 5.0):
            return 1e6
        try:
            var_model = theta
            kurt_model = max(0.0, 3 * xi**2 / (kappa * theta + 1e-8))
            err_var = (var_model - hist_var) ** 2
            err_kurt = (kurt_model - kurt_hist) ** 2
            return err_var + err_kurt
        except:
            return 1e6

    try:
        res = minimize(loss, x0, method='L-BFGS-B',
                       bounds=[(0.1, 20), (1e-4, 1.0), (0.01, 2.0), (-0.99, -0.01), (1e-4, 1.0)],
                       options={'maxiter': 100})
        if res.success and np.all(np.isfinite(res.x)):
            return np.array(res.x, dtype=np.float64)
        else:
            return x0.astype(np.float64)
    except:
        return x0.astype(np.float64)

                    

                def simulate_heston(S0, kappa, theta, xi, rho, v0, T, n_paths=20000, seed=42):
                    np.random.seed(seed)
                    dt = 1/252
                    n_steps = int(T * 252)
                    S = np.full(n_paths, S0)
                    v = np.full(n_paths, v0)

                    for _ in range(n_steps):
                        Z1 = np.random.randn(n_paths)
                        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.randn(n_paths)
                        # Full Truncation Scheme (prevents negative variance)
                        v = np.maximum(v, 0)
                        S *= np.exp(np.sqrt(v) * np.sqrt(dt) * Z1 - 0.5 * v * dt)
                        v += kappa * (theta - v) * dt + xi * np.sqrt(v) * np.sqrt(dt) * Z2
                    return S

                # ----------------------------
                # 3. SABR Model (for short-term)
                # dF = α F^β dW1
                # dα = ν α dW2
                # ----------------------------
                def calibrate_sabr(log_returns, current_price):
                    # Calibrate ν (vol-of-vol) and β (elasticity)
                    # Fix β = 0.5 (common for equities), calibrate ν to match historical vol change
                    beta = 0.5
                    # Estimate vol-of-vol from rolling volatility changes
                    window = 10
                    rolling_vol = log_returns.rolling(window).std()
                    vol_changes = np.diff(rolling_vol.dropna())
                    if len(vol_changes) < 5:
                        nu = 0.5
                    else:
                        nu = np.std(vol_changes) * np.sqrt(252) / (rolling_vol.mean() + 1e-8)
                        nu = np.clip(nu, 0.1, 2.0)
                    alpha0 = sigma_hist  # initial volatility
                    return alpha0, beta, nu

                def simulate_sabr(F0, alpha0, beta, nu, T, n_paths=20000, seed=42):
                    np.random.seed(seed)
                    dt = 1/252
                    n_steps = int(T * 252)
                    F = np.full(n_paths, F0)
                    alpha = np.full(n_paths, alpha0)

                    for _ in range(n_steps):
                        Z1 = np.random.randn(n_paths)
                        Z2 = np.random.randn(n_paths)  # assume uncorrelated for simplicity
                        F += alpha * (F ** beta) * np.sqrt(dt) * Z1
                        alpha *= np.exp(-0.5 * nu**2 * dt + nu * np.sqrt(dt) * Z2)
                        F = np.maximum(F, 1e-8)  # avoid negative prices
                    return F

                # ----------------------------
                # Run selected model
                # ----------------------------
                T = forecast_days / 252.0

                if model_choice == "Heston":
                    kappa, theta, xi, rho, v0 = calibrate_heston(log_returns)
                    future_prices = simulate_heston(current_price, kappa, theta, xi, rho, v0, T)
                    model_desc = f"Heston (κ={kappa:.2f}, θ={theta:.4f}, ξ={xi:.2f}, ρ={rho:.2f})"
                elif model_choice == "SABR":
                    alpha0, beta, nu = calibrate_sabr(log_returns, current_price)
                    future_prices = simulate_sabr(current_price, alpha0, beta, nu, T)
                    model_desc = f"SABR (α₀={alpha0:.3f}, β={beta:.1f}, ν={nu:.2f})"
                else:  # GBM
                    future_prices = simulate_gbm(current_price, sigma_hist, T)
                    model_desc = f"GBM (σ={sigma_hist:.2%})"

                # Compute probabilities
                up_0_5 = np.mean((future_prices > current_price) & (future_prices <= current_price * 1.05))
                up_5_10 = np.mean((future_prices > current_price * 1.05) & (future_prices <= current_price * 1.10))
                down_0_4 = np.mean(future_prices < current_price * 0.96)
                extreme = np.mean((future_prices > current_price * 1.10) | (future_prices < current_price * 0.90))

                st.subheader(f"Current price: ${current_price:.2f}")
                st.write(f"**{forecast_days}-day outlook for {ticker} ({model_choice}):**")
                st.write(f"- 📈 {up_0_5:.0%} chance: +0% to +5%")
                st.write(f"- 📈 {up_5_10:.0%} chance: +5% to +10%")
                st.write(f"- 📉 {down_0_4:.0%} chance: down to -4%")
                st.write(f"- ⚠️ {extreme:.0%} chance: extreme move (>±10%)")

                # Plot
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(future_prices, bins=120, density=True, alpha=0.7, color='steelblue', edgecolor='none')
                ax.axvline(current_price, color='red', linestyle='--', linewidth=2, label='Current Price')
                ax.set_xlabel('Future Price ($)')
                ax.set_ylabel('Probability Density')
                ax.set_title(f'{model_choice} Forecast Distribution')
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.5)
                st.pyplot(fig)

                st.caption(f"Model: {model_desc} | Calibration: 2-year historical data | Paths: 20,000")

    except Exception as e:
        st.error(f"Error: {str(e)}. Try a major ticker like AAPL, MSFT, or SPY.")

