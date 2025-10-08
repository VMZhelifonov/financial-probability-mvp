import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

st.set_page_config(page_title="Advanced Stochastic Stock Forecaster", layout="centered")
st.title("📈 Advanced Stochastic Stock Forecaster")
st.markdown("""
*Professional-grade probabilistic forecasting with jumps and regime-switching.*  
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
                def simulate_gbm(S0, vol, T, n_paths=20000, seed=42):
                    np.random.seed(seed)
                    dt = 1/252
                    steps = int(T * 252)
                    if steps == 0:
                        return np.full(n_paths, S0)
                    Z = np.random.randn(n_paths, steps)
                    logS = np.log(S0) + np.cumsum(-0.5 * vol**2 * dt + vol * np.sqrt(dt) * Z, axis=1)
                    return np.exp(logS[:, -1])

                # ----------------------------
                # Heston (robust)
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

                def simulate_heston(S0, kappa, theta, xi, rho, v0, T, n_paths=20000, seed=42):
                    np.random.seed(seed)
                    dt = 1/252
                    n_steps = int(T * 252)
                    if n_steps == 0:
                        return np.full(n_paths, S0)
                    S = np.full(n_paths, S0, dtype=np.float64)
                    v = np.full(n_paths, v0, dtype=np.float64)
                    sqrt_dt = np.sqrt(dt)
                    for _ in range(n_steps):
                        Z1 = np.random.randn(n_paths)
                        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.randn(n_paths)
                        v = np.maximum(v, 0.0)
                        S *= np.exp(np.sqrt(v) * sqrt_dt * Z1 - 0.5 * v * dt)
                        v += kappa * (theta - v) * dt + xi * np.sqrt(v) * sqrt_dt * Z2
                    return S

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
                            rolling_vol = np.array([np.std(log_returns[i:i+window]) for i in range(len(log_returns)-window+1)])
                            vol_changes = np.diff(rolling_vol)
                            if len(vol_changes) < 5 or np.std(vol_changes) == 0:
                                nu = 0.5
                            else:
                                nu = (np.std(vol_changes) * np.sqrt(252)) / (np.mean(rolling_vol) + 1e-8)
                                nu = np.clip(nu, 0.1, 2.0)
                        alpha0 = np.std(log_returns) * np.sqrt(252)
                        alpha0 = np.clip(alpha0, 0.01, 2.0)
                    return alpha0, beta, nu

                def simulate_sabr(F0, alpha0, beta, nu, T, n_paths=20000, seed=42):
                    np.random.seed(seed)
                    dt = 1/252
                    n_steps = int(T * 252)
                    if n_steps == 0:
                        return np.full(n_paths, F0)
                    F = np.full(n_paths, F0, dtype=np.float64)
                    alpha = np.full(n_paths, alpha0, dtype=np.float64)
                    sqrt_dt = np.sqrt(dt)
                    for _ in range(n_steps):
                        Z1 = np.random.randn(n_paths)
                        Z2 = np.random.randn(n_paths)
                        F += alpha * (np.maximum(F, 1e-8) ** beta) * sqrt_dt * Z1
                        alpha *= np.exp(-0.5 * nu**2 * dt + nu * sqrt_dt * Z2)
                        F = np.maximum(F, 1e-8)
                    return F

                # ----------------------------
                # Double Exponential Jump-Diffusion (Kou, 2002)
                # ----------------------------
                def calibrate_kou(log_returns):
                    # Estimate jump intensity and parameters from historical outliers
                    if len(log_returns) < 50:
                        return 0.1, 3.0, 3.0, 0.4  # λ, η1, η2, p
                    # Assume jumps occur in worst/best 2% of returns
                    threshold = np.percentile(np.abs(log_returns), 98)
                    jump_mask = np.abs(log_returns) > threshold
                    if np.sum(jump_mask) < 5:
                        return 0.1, 3.0, 3.0, 0.4
                    jump_returns = log_returns[jump_mask]
                    λ = len(jump_returns) / len(log_returns) * 252  # annualized intensity
                    λ = np.clip(λ, 0.01, 2.0)
                    # Separate up/down jumps
                    down_jumps = -jump_returns[jump_returns < 0]
                    up_jumps = jump_returns[jump_returns > 0]
                    η1 = 1.0 / (np.mean(down_jumps) + 1e-8) if len(down_jumps) > 0 else 3.0
                    η2 = 1.0 / (np.mean(up_jumps) + 1e-8) if len(up_jumps) > 0 else 3.0
                    η1 = np.clip(η1, 1.0, 10.0)
                    η2 = np.clip(η2, 1.0, 10.0)
                    p = len(up_jumps) / (len(jump_returns) + 1e-8) if len(jump_returns) > 0 else 0.4
                    p = np.clip(p, 0.1, 0.9)
                    return λ, η1, η2, p

                def simulate_kou(S0, vol, λ, η1, η2, p, T, n_paths=20000, seed=42):
                    np.random.seed(seed)
                    dt = 1/252
                    n_steps = int(T * 252)
                    if n_steps == 0:
                        return np.full(n_paths, S0)
                    S = np.full(n_paths, S0, dtype=np.float64)
                    for _ in range(n_steps):
                        # Diffusion part
                        Z = np.random.randn(n_paths)
                        S *= np.exp(-0.5 * vol**2 * dt + vol * np.sqrt(dt) * Z)
                        # Jump part
                        N = np.random.poisson(λ * dt, n_paths)  # number of jumps
                        total_jump = np.zeros(n_paths)
                        for i in range(n_paths):
                            if N[i] > 0:
                                # Each jump: with prob p — up (exp with η2), else down (exp with η1)
                                U = np.random.rand(N[i])
                                signs = np.where(U < p, 1, -1)
                                magnitudes = np.where(signs > 0,
                                                     np.random.exponential(1/η2, N[i]),
                                                     -np.random.exponential(1/η1, N[i]))
                                total_jump[i] = np.sum(magnitudes)
                        S *= np.exp(total_jump)
                        S = np.maximum(S, 1e-8)
                    return S

                # ----------------------------
                # Regime-Switching Heston
                # Regime 0: calm, Regime 1: crisis
                # ----------------------------
                def simulate_regime_switching_heston(S0, T, n_paths=20000, seed=42):
                    np.random.seed(seed)
                    dt = 1/252
                    n_steps = int(T * 252)
                    if n_steps == 0:
                        return np.full(n_paths, S0)
                    
                    # Calibrate two regimes
                    # Regime 0 (calm): low vol, low vol-of-vol
                    params0 = [3.0, 0.02, 0.2, -0.3, 0.02]  # κ, θ, ξ, ρ, v0
                    # Regime 1 (crisis): high vol, high vol-of-vol, strong leverage
                    params1 = [1.5, 0.10, 0.6, -0.8, 0.10]
                    
                    # Transition matrix: P[0->1] = 0.02, P[1->0] = 0.1
                    P = np.array([[0.98, 0.02],
                                  [0.10, 0.90]])
                    
                    S = np.full(n_paths, S0, dtype=np.float64)
                    v = np.full(n_paths, params0[4], dtype=np.float64)  # start in calm
                    regime = np.zeros(n_paths, dtype=int)  # 0 = calm
                    
                    sqrt_dt = np.sqrt(dt)
                    for _ in range(n_steps):
                        # Update regime
                        rand = np.random.rand(n_paths)
                        switch_to_1 = (regime == 0) & (rand < P[0,1])
                        switch_to_0 = (regime == 1) & (rand < P[1,0])
                        regime[switch_to_1] = 1
                        regime[switch_to_0] = 0
                        
                        # Get parameters per path
                        kappa = np.where(regime == 0, params0[0], params1[0])
                        theta = np.where(regime == 0, params0[1], params1[1])
                        xi = np.where(regime == 0, params0[2], params1[2])
                        rho = np.where(regime == 0, params0[3], params1[3])
                        
                        Z1 = np.random.randn(n_paths)
                        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.randn(n_paths)
                        v = np.maximum(v, 0.0)
                        S *= np.exp(np.sqrt(v) * sqrt_dt * Z1 - 0.5 * v * dt)
                        v += kappa * (theta - v) * dt + xi * np.sqrt(v) * sqrt_dt * Z2
                    return S

                # ----------------------------
                # Run selected model
                # ----------------------------
                T = forecast_days / 252.0

                if model_choice == "Heston":
                    params = calibrate_heston(log_returns)
                    if not (isinstance(params, np.ndarray) and params.shape == (5,)):
                        params = np.array([2.0, 0.04, 0.3, -0.5, 0.04])
                    kappa, theta, xi, rho, v0 = params
                    future_prices = simulate_heston(current_price, kappa, theta, xi, rho, v0, T)
                    model_desc = f"Heston (κ={kappa:.2f}, θ={theta:.4f}, ξ={xi:.2f}, ρ={rho:.2f})"

                elif model_choice == "SABR":
                    alpha0, beta, nu = calibrate_sabr(log_returns, current_price)
                    future_prices = simulate_sabr(current_price, alpha0, beta, nu, T)
                    model_desc = f"SABR (α₀={alpha0:.3f}, β={beta:.1f}, ν={nu:.2f})"

                elif model_choice == "GBM (Baseline)":
                    future_prices = simulate_gbm(current_price, sigma_hist, T)
                    model_desc = f"GBM (σ={sigma_hist:.2%})"

                elif model_choice == "Double Exp Jump-Diffusion":
                    λ, η1, η2, p = calibrate_kou(log_returns)
                    future_prices = simulate_kou(current_price, sigma_hist, λ, η1, η2, p, T)
                    model_desc = f"Kou Jump (λ={λ:.2f}, η₁={η1:.1f}, η₂={η2:.1f}, p={p:.2f})"

                elif model_choice == "Regime-Switching Heston":
                    future_prices = simulate_regime_switching_heston(current_price, T)
                    model_desc = "Regime-Switching Heston (Calm ↔ Crisis)"

                else:
                    future_prices = simulate_gbm(current_price, sigma_hist, T)
                    model_desc = "Fallback GBM"

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
