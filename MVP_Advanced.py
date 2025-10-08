import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

st.set_page_config(page_title="Advanced Stochastic Stock Forecaster", layout="centered")
st.title("📈 Advanced Stochastic Stock Forecaster")
st.markdown("""
*Professional-grade probabilistic forecasting with scenario paths.*  
⚠️ **Not financial advice. For educational/research purposes only.**
""")

ticker = st.text_input("Enter stock ticker (e.g. AAPL, TSLA, MSFT)", value="AAPL").upper()
forecast_days = st.slider("Forecast horizon (days)", min_value=1, max_value=30, value=5)
model_choice = st.selectbox("Stochastic model", [
    "Heston", 
    "SABR", 
    "GBM (Baseline)",
    "Double Exp Jump-Diffusion",
    "Regime-Switching Heston"
])

if ticker:
    try:
        data = yf.download(ticker, period="2y", progress=False)
        if data.empty:
            st.error("No data found. Try a liquid ticker (e.g. AAPL, SPY).")
        else:
            close_prices = data['Close'].dropna()
            if len(close_prices) < 60:
                st.error("Not enough data. Need ≥60 days.")
            else:
                current_price = float(close_prices.iloc[-1])
                log_prices = np.log(close_prices.values)
                log_returns = np.diff(log_prices)
                log_returns = log_returns[np.isfinite(log_returns)]
                if len(log_returns) < 20:
                    sigma_hist = 0.2
                else:
                    sigma_hist = float(np.std(log_returns) * np.sqrt(252))

                # ----------------------------
                # GBM
                # ----------------------------
                def simulate_gbm_paths(S0, vol, T, n_paths=20000, n_steps=None, seed=42):
                    if n_steps is None:
                        n_steps = int(T * 252)
                    np.random.seed(seed)
                    dt = 1/252
                    if n_steps == 0:
                        return np.full((n_paths, 1), S0)
                    Z = np.random.randn(n_paths, n_steps)
                    logS = np.log(S0) + np.cumsum(-0.5 * vol**2 * dt + vol * np.sqrt(dt) * Z, axis=1)
                    S = np.exp(np.hstack([np.full((n_paths, 1), S0), logS]))
                    return S

                # ----------------------------
                # Heston
                # ----------------------------
                def calibrate_heston(log_returns):
                    if len(log_returns) < 20:
                        return np.array([2.0, 0.04, 0.3, -0.5, 0.04])
                    log_returns = log_returns[np.isfinite(log_returns)]
                    if len(log_returns) < 20:
                        return np.array([2.0, 0.04, 0.3, -0.5, 0.04])
                    try:
                        daily_var = np.var(log_returns)
                        hist_var = daily_var * 252 if daily_var > 1e-8 else 0.04
                        hist_var = np.clip(hist_var, 1e-4, 1.0)
                        kurt_hist = ((np.mean((log_returns - np.mean(log_returns))**4) / (daily_var**2)) - 3) if daily_var > 1e-8 else 0.0
                        kurt_hist = np.clip(kurt_hist, 0.0, 20.0)
                    except:
                        hist_var = 0.04
                        kurt_hist = 3.0
                    x0 = np.array([2.0, hist_var, 0.3, -0.5, hist_var], dtype=np.float64)
                    def loss(params):
                        kappa, theta, xi, rho, v0 = params
                        if not (0.1 <= kappa <= 20 and 1e-4 <= theta <= 1.0 and 0.01 <= xi <= 2.0 and -0.99 <= rho <= -0.01 and 1e-4 <= v0 <= 1.0):
                            return 1e6
                        try:
                            kurt_model = 3 * xi**2 / (kappa * theta + 1e-8)
                            return (theta - hist_var)**2 + (kurt_model - kurt_hist)**2
                        except:
                            return 1e6
                    try:
                        res = minimize(loss, x0, method='L-BFGS-B',
                                       bounds=[(0.1, 20), (1e-4, 1.0), (0.01, 2.0), (-0.99, -0.01), (1e-4, 1.0)],
                                       options={'maxiter': 100})
                        if res.success and np.all(np.isfinite(res.x)):
                            return np.array(res.x, dtype=np.float64)
                        else:
                            return x0
                    except:
                        return x0

                def simulate_heston_paths(S0, kappa, theta, xi, rho, v0, T, n_paths=20000, n_steps=None, seed=42):
                    if n_steps is None:
                        n_steps = int(T * 252)
                    np.random.seed(seed)
                    dt = 1/252
                    if n_steps == 0:
                        return np.full((n_paths, 1), S0)
                    S = np.full((n_paths, n_steps+1), S0, dtype=np.float64)
                    v = np.full(n_paths, v0, dtype=np.float64)
                    sqrt_dt = np.sqrt(dt)
                    for t in range(1, n_steps+1):
                        Z1 = np.random.randn(n_paths)
                        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.randn(n_paths)
                        v = np.maximum(v, 0.0)
                        S[:, t] = S[:, t-1] * np.exp(np.sqrt(v) * sqrt_dt * Z1 - 0.5 * v * dt)
                        v += kappa * (theta - v) * dt + xi * np.sqrt(v) * sqrt_dt * Z2
                    return np.maximum(S, 1e-8)

                # ----------------------------
                # SABR
                # ----------------------------
                def calibrate_sabr(log_returns, current_price):
                    beta = 0.5
                    if len(log_returns) < 20:
                        nu = 0.5
                        alpha0 = 0.2
                    else:
                        window = min(10, len(log_returns) // 2)
                        if window < 5:
                            nu = 0.5
                        else:
                            rolling_vol = []
                            for i in range(len(log_returns) - window + 1):
                                vol = np.std(log_returns[i:i+window])
                                if np.isfinite(vol):
                                    rolling_vol.append(vol)
                            if len(rolling_vol) < 5:
                                nu = 0.5
                            else:
                                rolling_vol = np.array(rolling_vol)
                                vol_changes = np.diff(rolling_vol)
                                if len(vol_changes) < 5 or np.std(vol_changes) == 0:
                                    nu = 0.5
                                else:
                                    nu = (np.std(vol_changes) * np.sqrt(252)) / (np.mean(rolling_vol) + 1e-8)
                                    nu = np.clip(nu, 0.1, 2.0)
                        alpha0 = np.std(log_returns) * np.sqrt(252)
                        alpha0 = np.clip(alpha0, 0.01, 2.0)
                    return alpha0, beta, nu

                def simulate_sabr_paths(F0, alpha0, beta, nu, T, n_paths=20000, n_steps=None, seed=42):
                    if n_steps is None:
                        n_steps = int(T * 252)
                    np.random.seed(seed)
                    dt = 1/252
                    if n_steps == 0:
                        return np.full((n_paths, 1), F0)
                    F = np.full((n_paths, n_steps+1), F0, dtype=np.float64)
                    alpha = np.full(n_paths, alpha0, dtype=np.float64)
                    sqrt_dt = np.sqrt(dt)
                    for t in range(1, n_steps+1):
                        Z1 = np.random.randn(n_paths)
                        Z2 = np.random.randn(n_paths)
                        F[:, t] = F[:, t-1] + alpha * (np.maximum(F[:, t-1], 1e-8) ** beta) * sqrt_dt * Z1
                        alpha *= np.exp(-0.5 * nu**2 * dt + nu * sqrt_dt * Z2)
                        F[:, t] = np.maximum(F[:, t], 1e-8)
                    return F

                # ----------------------------
                # Double Exponential Jump-Diffusion
                # ----------------------------
                def calibrate_kou(log_returns):
                    if len(log_returns) < 50:
                        return 0.1, 3.0, 3.0, 0.4
                    threshold = np.percentile(np.abs(log_returns), 98)
                    jump_mask = np.abs(log_returns) > threshold
                    if np.sum(jump_mask) < 5:
                        return 0.1, 3.0, 3.0, 0.4
                    jump_returns = log_returns[jump_mask]
                    λ = len(jump_returns) / len(log_returns) * 252
                    λ = np.clip(λ, 0.01, 2.0)
                    down_jumps = -jump_returns[jump_returns < 0]
                    up_jumps = jump_returns[jump_returns > 0]
                    η1 = 1.0 / (np.mean(down_jumps) + 1e-8) if len(down_jumps) > 0 else 3.0
                    η2 = 1.0 / (np.mean(up_jumps) + 1e-8) if len(up_jumps) > 0 else 3.0
                    η1 = np.clip(η1, 1.0, 10.0)
                    η2 = np.clip(η2, 1.0, 10.0)
                    p = len(up_jumps) / (len(jump_returns) + 1e-8) if len(jump_returns) > 0 else 0.4
                    p = np.clip(p, 0.1, 0.9)
                    return λ, η1, η2, p

                def simulate_kou_paths(S0, vol, λ, η1, η2, p, T, n_paths=20000, n_steps=None, seed=42):
                    if n_steps is None:
                        n_steps = int(T * 252)
                    np.random.seed(seed)
                    dt = 1/252
                    if n_steps == 0:
                        return np.full((n_paths, 1), S0)
                    S = np.full((n_paths, n_steps+1), S0, dtype=np.float64)
                    for t in range(1, n_steps+1):
                        Z = np.random.randn(n_paths)
                        S[:, t] = S[:, t-1] * np.exp(-0.5 * vol**2 * dt + vol * np.sqrt(dt) * Z)
                        N = np.random.poisson(λ * dt, n_paths)
                        total_jump = np.zeros(n_paths)
                        for i in range(n_paths):
                            if N[i] > 0:
                                U = np.random.rand(N[i])
                                signs = np.where(U < p, 1, -1)
                                magnitudes = np.where(signs > 0,
                                                     np.random.exponential(1/η2, N[i]),
                                                     -np.random.exponential(1/η1, N[i]))
                                total_jump[i] = np.sum(magnitudes)
                        S[:, t] *= np.exp(total_jump)
                        S[:, t] = np.maximum(S[:, t], 1e-8)
                    return S

                # ----------------------------
                # Regime-Switching Heston
                # ----------------------------
                def simulate_regime_switching_heston_paths(S0, T, n_paths=20000, n_steps=None, seed=42):
                    if n_steps is None:
                        n_steps = int(T * 252)
                    np.random.seed(seed)
                    dt = 1/252
                    if n_steps == 0:
                        return np.full((n_paths, 1), S0)
                    params0 = [3.0, 0.02, 0.2, -0.3, 0.02]
                    params1 = [1.5, 0.10, 0.6, -0.8, 0.10]
                    P = np.array([[0.98, 0.02], [0.10, 0.90]])
                    S = np.full((n_paths, n_steps+1), S0, dtype=np.float64)
                    v = np.full(n_paths, params0[4], dtype=np.float64)
                    regime = np.zeros(n_paths, dtype=int)
                    sqrt_dt = np.sqrt(dt)
                    for t in range(1, n_steps+1):
                        rand = np.random.rand(n_paths)
                        switch_to_1 = (regime == 0) & (rand < P[0,1])
                        switch_to_0 = (regime == 1) & (rand < P[1,0])
                        regime[switch_to_1] = 1
                        regime[switch_to_0] = 0
                        kappa = np.where(regime == 0, params0[0], params1[0])
                        theta = np.where(regime == 0, params0[1], params1[1])
                        xi = np.where(regime == 0, params0[2], params1[2])
                        rho = np.where(regime == 0, params0[3], params1[3])
                        Z1 = np.random.randn(n_paths)
                        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.randn(n_paths)
                        v = np.maximum(v, 0.0)
                        S[:, t] = S[:, t-1] * np.exp(np.sqrt(v) * sqrt_dt * Z1 - 0.5 * v * dt)
                        v += kappa * (theta - v) * dt + xi * np.sqrt(v) * sqrt_dt * Z2
                    return np.maximum(S, 1e-8)

                # ----------------------------
                # Run selected model (with paths)
                # ----------------------------
                T = forecast_days / 252.0
                n_steps = forecast_days  # 1 step per day

                if model_choice == "Heston":
                    params = calibrate_heston(log_returns)
                    if not (isinstance(params, np.ndarray) and params.shape == (5,)):
                        params = np.array([2.0, 0.04, 0.3, -0.5, 0.04])
                    kappa, theta, xi, rho, v0 = params
                    all_paths = simulate_heston_paths(current_price, kappa, theta, xi, rho, v0, T, n_paths=20000, n_steps=n_steps)
                    model_desc = f"Heston (κ={kappa:.2f}, θ={theta:.4f}, ξ={xi:.2f}, ρ={rho:.2f})"

                elif model_choice == "SABR":
                    alpha0, beta, nu = calibrate_sabr(log_returns, current_price)
                    all_paths = simulate_sabr_paths(current_price, alpha0, beta, nu, T, n_paths=20000, n_steps=n_steps)
                    model_desc = f"SABR (α₀={alpha0:.3f}, β={beta:.1f}, ν={nu:.2f})"

                elif model_choice == "GBM (Baseline)":
                    all_paths = simulate_gbm_paths(current_price, sigma_hist, T, n_paths=20000, n_steps=n_steps)
                    model_desc = f"GBM (σ={sigma_hist:.2%})"

                elif model_choice == "Double Exp Jump-Diffusion":
                    λ, η1, η2, p = calibrate_kou(log_returns)
                    all_paths = simulate_kou_paths(current_price, sigma_hist, λ, η1, η2, p, T, n_paths=20000, n_steps=n_steps)
                    model_desc = f"Kou Jump (λ={λ:.2f}, η₁={η1:.1f}, η₂={η2:.1f}, p={p:.2f})"

                elif model_choice == "Regime-Switching Heston":
                    all_paths = simulate_regime_switching_heston_paths(current_price, T, n_paths=20000, n_steps=n_steps)
                    model_desc = "Regime-Switching Heston (Calm ↔ Crisis)"

                else:
                    all_paths = simulate_gbm_paths(current_price, sigma_hist, T, n_paths=20000, n_steps=n_steps)
                    model_desc = "Fallback GBM"

                # Final prices
                future_prices = all_paths[:, -1]
                future_prices = future_prices[np.isfinite(future_prices)]
                if len(future_prices) == 0:
                    st.error("Simulation produced no valid prices.")
                    st.stop()

                # ----------------------------
                # Compute probabilities (correctly)
                # ----------------------------
                p0 = current_price
                p_up5 = p0 * 1.05
                p_up10 = p0 * 1.10
                p_down5 = p0 * 0.95
                p_down10 = p0 * 0.90

                prob_up_0_5 = np.mean((future_prices > p0) & (future_prices <= p_up5))
                prob_up_5_10 = np.mean((future_prices > p_up5) & (future_prices <= p_up10))
                prob_down_0_5 = np.mean((future_prices >= p_down5) & (future_prices < p0))
                prob_down_5_10 = np.mean((future_prices >= p_down10) & (future_prices < p_down5))
                prob_extreme = np.mean((future_prices > p_up10) | (future_prices < p_down10))

                total = prob_up_0_5 + prob_up_5_10 + prob_down_0_5 + prob_down_5_10 + prob_extreme
                if abs(total - 1.0) > 1e-3:
                    st.warning(f"⚠️ Probability sum = {total:.4f}")

                down_0_10 = prob_down_0_5 + prob_down_5_10

                # ----------------------------
                # Find one path per scenario
                # ----------------------------
                def find_path_in_range(paths, lower, upper):
                    final = paths[:, -1]
                    mask = (final > lower) & (final <= upper)
                    if np.any(mask):
                        idx = np.where(mask)[0][0]
                        return paths[idx]
                    else:
                        # Fallback: closest path
                        distances = np.abs(final - (lower + upper) / 2)
                        idx = np.argmin(distances)
                        return paths[idx]

                path_up_5_10 = find_path_in_range(all_paths, p_up5, p_up10)
                path_up_0_5 = find_path_in_range(all_paths, p0, p_up5)
                path_down_0_5 = find_path_in_range(all_paths, p_down5, p0)
                path_down_5_10 = find_path_in_range(all_paths, p_down10, p_down5)

                # ----------------------------
                # Plot 1: Distribution
                # ----------------------------
                fig1, ax1 = plt.subplots(figsize=(8, 3.5))
                ax1.hist(future_prices, bins=120, density=True, alpha=0.7, color='steelblue', edgecolor='none')
                ax1.axvline(current_price, color='red', linestyle='--', linewidth=2, label='Current Price')
                ax1.set_xlabel('Future Price ($)')
                ax1.set_ylabel('Density')
                ax1.set_title(f'{model_choice} Forecast Distribution')
                ax1.legend()
                ax1.grid(True, linestyle='--', alpha=0.5)
                st.pyplot(fig1)

                # ----------------------------
                # Plot 2: Scenario paths + last 7 days
                # ----------------------------
                last_7_days = close_prices.iloc[-7:]
                days = np.arange(-6, 1)  # -6, -5, ..., 0 (today)
                forecast_days_arr = np.arange(1, forecast_days + 1)
                all_time = np.concatenate([days, forecast_days_arr])

                fig2, ax2 = plt.subplots(figsize=(8, 4))
                # Plot historical
                ax2.plot(days, last_7_days.values, 'o-', color='black', label='Last 7 Days', linewidth=2, markersize=4)

                # Plot scenarios
                full_path_up_5_10 = np.concatenate([last_7_days.values[-1:], path_up_5_10[1:]])
                full_path_up_0_5 = np.concatenate([last_7_days.values[-1:], path_up_0_5[1:]])
                full_path_down_0_5 = np.concatenate([last_7_days.values[-1:], path_down_0_5[1:]])
                full_path_down_5_10 = np.concatenate([last_7_days.values[-1:], path_down_5_10[1:]])

                ax2.plot(forecast_days_arr, full_path_up_5_10[1:], 'o--', color='green', label='+5% to +10%', linewidth=2, markersize=4)
                ax2.plot(forecast_days_arr, full_path_up_0_5[1:], 'o--', color='blue', label='+0% to +5%', linewidth=2, markersize=4)
                ax2.plot(forecast_days_arr, full_path_down_0_5[1:], 'o--', color='orange', label='-5% to 0%', linewidth=2, markersize=4)
                ax2.plot(forecast_days_arr, full_path_down_5_10[1:], 'o--', color='red', label='-10% to -5%', linewidth=2, markersize=4)

                ax2.set_xlabel('Days (0 = today)')
                ax2.set_ylabel('Price ($)')
                ax2.set_title('Scenario Paths (Next {} Days)'.format(forecast_days))
                ax2.axvline(0, color='gray', linestyle=':', linewidth=1)
                ax2.legend()
                ax2.grid(True, linestyle='--', alpha=0.5)
                st.pyplot(fig2)

                # ----------------------------
                # Output probabilities
                # ----------------------------
                st.subheader(f"Current price: ${current_price:.2f}")
                st.write(f"**{forecast_days}-day outlook for {ticker} ({model_choice}):**")
                st.write(f"- 📈 {prob_up_0_5:.0%} chance: +0% to +5%")
                st.write(f"- 📈 {prob_up_5_10:.0%} chance: +5% to +10%")
                st.write(f"- 📉 {down_0_10:.0%} chance: down to -10%")
                st.write(f"- ⚠️ {prob_extreme:.0%} chance: extreme move (>±10%)")

                st.caption(f"Model: {model_desc} | Calibration: 2-year historical data | Paths: 20,000")

    except Exception as e:
        st.error(f"Error: {str(e)}. Try a major ticker like AAPL, MSFT, or SPY.")
