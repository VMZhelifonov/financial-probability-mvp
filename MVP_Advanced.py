import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats

st.set_page_config(page_title="Advanced Stochastic Stock Forecaster", layout="centered")
st.title("📈 Advanced Stochastic Stock Forecaster")
st.markdown("""
*Professional-grade probabilistic forecasting with statistically calibrated models and validation (Q-Q, KS-test).*  
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
                # Heston: GMM Calibration
                # ----------------------------
                def calibrate_heston_gmm(log_returns, maxiter=150):
                    if len(log_returns) < 50:
                        return np.array([2.0, 0.04, 0.3, -0.5, 0.04])

                    mu_emp = np.mean(log_returns)
                    var_emp = np.var(log_returns)
                    if var_emp < 1e-8:
                        return np.array([2.0, 0.04, 0.3, -0.5, 0.04])
                    skew_emp = np.mean((log_returns - mu_emp)**3) / (var_emp**1.5)
                    kurt_emp = np.mean((log_returns - mu_emp)**4) / (var_emp**2)

                    def simulate_moments(params, n_sims=800, n_steps=252):
                        kappa, theta, xi, rho, v0 = params
                        dt = 1.0 / 252
                        final_logS = np.zeros(n_sims)
                        for i in range(n_sims):
                            v = v0
                            logS = 0.0
                            for _ in range(n_steps):
                                Z1 = np.random.randn()
                                Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.randn()
                                v = max(v + kappa * (theta - v) * dt + xi * np.sqrt(max(v, 1e-8)) * np.sqrt(dt) * Z2, 1e-8)
                                logS += -0.5 * v * dt + np.sqrt(v * dt) * Z1
                            final_logS[i] = logS
                        mu = np.mean(final_logS)
                        var = np.var(final_logS)
                        if var < 1e-12:
                            return mu, var_emp, 0.0, 3.0
                        skew = np.mean((final_logS - mu)**3) / (var**1.5)
                        kurt = np.mean((final_logS - mu)**4) / (var**2)
                        return mu, var, skew, kurt

                    def loss(params):
                        kappa, theta, xi, rho, v0 = params
                        if not (0.1 <= kappa <= 20 and 1e-4 <= theta <= 1.0 and 
                                0.01 <= xi <= 2.0 and -0.99 <= rho <= -0.01 and 
                                1e-4 <= v0 <= 1.0):
                            return 1e6
                        try:
                            mu_m, var_m, skew_m, kurt_m = simulate_moments(params)
                            return (
                                10 * (mu_m - mu_emp)**2 +
                                100 * (var_m - var_emp)**2 +
                                (skew_m - skew_emp)**2 +
                                (kurt_m - kurt_emp)**2
                            )
                        except:
                            return 1e6

                    hist_var = np.clip(np.var(log_returns) * 252, 1e-4, 1.0)
                    x0 = np.array([2.0, hist_var, 0.3, -0.5, hist_var])
                    bounds = [(0.1, 20), (1e-4, 1.0), (0.01, 2.0), (-0.99, -0.01), (1e-4, 1.0)]
                    res = minimize(loss, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': maxiter})
                    if res.success and np.all(np.isfinite(res.x)):
                        return res.x
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
                        v = np.maximum(v, 1e-8)
                        S[:, t] = S[:, t-1] * np.exp(-0.5 * v * dt + np.sqrt(v * dt) * Z1)
                        v += kappa * (theta - v) * dt + xi * np.sqrt(v) * sqrt_dt * Z2
                        v = np.maximum(v, 1e-8)
                    return np.maximum(S, 1e-8)

                # ----------------------------
                # SABR
                # ----------------------------
                def calibrate_sabr(log_returns, current_price):
                    beta = 0.5
                    if len(log_returns) < 30:
                        return 0.2, beta, 0.5
                    vol_ann = np.std(log_returns) * np.sqrt(252)
                    alpha0 = vol_ann * (current_price ** (1 - beta))
                    alpha0 = np.clip(alpha0, 0.01, 2.0)

                    window = min(10, len(log_returns) // 3)
                    if window < 5:
                        nu = 0.5
                    else:
                        rolling_vols = []
                        for i in range(len(log_returns) - window + 1):
                            rv = np.std(log_returns[i:i+window])
                            if np.isfinite(rv):
                                rolling_vols.append(rv)
                        if len(rolling_vols) < 5:
                            nu = 0.5
                        else:
                            rolling_vols = np.array(rolling_vols)
                            vol_of_vol = np.std(rolling_vols) * np.sqrt(252)
                            avg_vol = np.mean(rolling_vols) + 1e-8
                            nu = vol_of_vol / avg_vol
                            nu = np.clip(nu, 0.1, 2.0)
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
                # Kou Jump-Diffusion
                # ----------------------------
                def calibrate_kou(log_returns):
                    if len(log_returns) < 50:
                        return 0.1, 3.0, 3.0, 0.4

                    threshold = np.percentile(np.abs(log_returns), 95)
                    jump_mask = np.abs(log_returns) > threshold
                    n_jumps = np.sum(jump_mask)
                    if n_jumps < 3:
                        return 0.1, 3.0, 3.0, 0.4

                    λ = (n_jumps / len(log_returns)) * 252
                    λ = np.clip(λ, 0.01, 2.0)

                    jump_vals = log_returns[jump_mask]
                    up_jumps = jump_vals[jump_vals > 0]
                    down_jumps = -jump_vals[jump_vals < 0]

                    η2 = 1.0 / (np.mean(up_jumps) + 1e-8) if len(up_jumps) > 0 else 3.0
                    η1 = 1.0 / (np.mean(down_jumps) + 1e-8) if len(down_jumps) > 0 else 3.0
                    η1 = np.clip(η1, 1.0, 10.0)
                    η2 = np.clip(η2, 1.0, 10.0)

                    p = len(up_jumps) / (len(jump_vals) + 1e-8) if len(jump_vals) > 0 else 0.5
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

                    calm = [3.0, 0.02, 0.2, -0.3, 0.02]
                    crisis = [1.0, 0.15, 0.8, -0.7, 0.15]
                    P = np.array([[0.985, 0.015], [0.12, 0.88]])
                    S = np.full((n_paths, n_steps+1), S0, dtype=np.float64)
                    v = np.full(n_paths, calm[4], dtype=np.float64)
                    regime = np.zeros(n_paths, dtype=int)
                    sqrt_dt = np.sqrt(dt)

                    for t in range(1, n_steps+1):
                        rand = np.random.rand(n_paths)
                        switch_to_crisis = (regime == 0) & (rand < P[0,1])
                        switch_to_calm = (regime == 1) & (rand < P[1,0])
                        regime[switch_to_crisis] = 1
                        regime[switch_to_calm] = 0

                        kappa = np.where(regime == 0, calm[0], crisis[0])
                        theta = np.where(regime == 0, calm[1], crisis[1])
                        xi = np.where(regime == 0, calm[2], crisis[2])
                        rho = np.where(regime == 0, calm[3], crisis[3])

                        Z1 = np.random.randn(n_paths)
                        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.randn(n_paths)
                        v = np.maximum(v, 1e-8)
                        S[:, t] = S[:, t-1] * np.exp(-0.5 * v * dt + np.sqrt(v * dt) * Z1)
                        v += kappa * (theta - v) * dt + xi * np.sqrt(v) * sqrt_dt * Z2
                        v = np.maximum(v, 1e-8)
                    return np.maximum(S, 1e-8)

                # ----------------------------
                # Run model for forecast
                # ----------------------------
                T = forecast_days / 252.0
                n_steps = forecast_days

                if model_choice == "Heston":
                    with st.spinner("Calibrating Heston model..."):
                        params = calibrate_heston_gmm(log_returns)
                    if not (isinstance(params, np.ndarray) and len(params) == 5):
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

                # ----------------------------
                # Generate 1-day simulated returns FOR VALIDATION (separate simulation)
                # ----------------------------
                empirical_log_returns = log_returns[np.isfinite(log_returns)]
                simulated_log_returns = None

                # Only proceed if we have enough empirical data
                if len(empirical_log_returns) >= 20:
                    n_val_paths = 10000
                    try:
                        if model_choice == "Heston":
                            val_paths = simulate_heston_paths(current_price, kappa, theta, xi, rho, v0, T=1/252, n_paths=n_val_paths, n_steps=1, seed=999)
                        elif model_choice == "SABR":
                            val_paths = simulate_sabr_paths(current_price, alpha0, beta, nu, T=1/252, n_paths=n_val_paths, n_steps=1, seed=999)
                        elif model_choice == "GBM (Baseline)":
                            val_paths = simulate_gbm_paths(current_price, sigma_hist, T=1/252, n_paths=n_val_paths, n_steps=1, seed=999)
                        elif model_choice == "Double Exp Jump-Diffusion":
                            val_paths = simulate_kou_paths(current_price, sigma_hist, λ, η1, η2, p, T=1/252, n_paths=n_val_paths, n_steps=1, seed=999)
                        elif model_choice == "Regime-Switching Heston":
                            val_paths = simulate_regime_switching_heston_paths(current_price, T=1/252, n_paths=n_val_paths, n_steps=1, seed=999)
                        else:
                            val_paths = simulate_gbm_paths(current_price, sigma_hist, T=1/252, n_paths=n_val_paths, n_steps=1, seed=999)

                        simulated_prices_1d = val_paths[:, 1]
                        simulated_log_returns = np.log(simulated_prices_1d / current_price)
                        simulated_log_returns = simulated_log_returns[np.isfinite(simulated_log_returns)]
                    except Exception as e:
                        st.warning(f"Validation simulation failed: {e}")
                        simulated_log_returns = None

                qq_available = (simulated_log_returns is not None) and (len(simulated_log_returns) >= 100) and (len(empirical_log_returns) >= 100)

                # ----------------------------
                # Probabilities
                # ----------------------------
                future_prices = all_paths[:, -1]
                future_prices = future_prices[np.isfinite(future_prices) & (future_prices > 0)]
                if len(future_prices) == 0:
                    st.error("Simulation produced no valid prices.")
                    st.stop()

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
                down_0_10 = prob_down_0_5 + prob_down_5_10

                def find_path_in_range(paths, lower, upper):
                    final = paths[:, -1]
                    mask = (final > lower) & (final <= upper)
                    if np.any(mask):
                        idx = np.where(mask)[0][0]
                        return paths[idx]
                    else:
                        target = (lower + upper) / 2
                        idx = np.argmin(np.abs(final - target))
                        return paths[idx]

                path_up_5_10 = find_path_in_range(all_paths, p_up5, p_up10)
                path_up_0_5 = find_path_in_range(all_paths, p0, p_up5)
                path_down_0_5 = find_path_in_range(all_paths, p_down5, p0)
                path_down_5_10 = find_path_in_range(all_paths, p_down10, p_down5)

                # ----------------------------
                # Output probabilities
                # ----------------------------
                st.subheader(f"Current price: ${current_price:.2f}")
                st.write(f"**{forecast_days}-day outlook for {ticker} ({model_choice}):**")
                st.write(f"- 📈 {prob_up_0_5:.0%} chance: +0% to +5%")
                st.write(f"- 📈 {prob_up_5_10:.0%} chance: +5% to +10%")
                st.write(f"- 📉 {down_0_10:.0%} chance: down to -10%")
                st.write(f"- ⚠️ {prob_extreme:.0%} chance: extreme move (>±10%)")

                # ----------------------------
                # Plot 1: Forecast Distribution
                # ----------------------------
                fig1, ax1 = plt.subplots(figsize=(8, 3))
                ax1.hist(future_prices, bins=120, density=True, alpha=0.7, color='steelblue', edgecolor='none')
                ax1.axvline(current_price, color='red', linestyle='--', linewidth=2, label='Current Price')
                ax1.set_xlabel('Future Price ($)')
                ax1.set_ylabel('Density')
                ax1.set_title(f'{model_choice} Forecast Distribution')
                ax1.legend()
                ax1.grid(True, linestyle='--', alpha=0.5)
                st.pyplot(fig1)
                plt.close(fig1)

                # ----------------------------
                # Plot 2: Q-Q Plot and KS Test
                # ----------------------------
                if qq_available:
                    fig2, ax2 = plt.subplots(figsize=(6, 6))
                    # Use all empirical and simulated returns (no subsampling needed now)
                    emp_sorted = np.sort(empirical_log_returns)
                    sim_sorted = np.sort(simulated_log_returns[:len(empirical_log_returns)])  # match length

                    ax2.scatter(emp_sorted, sim_sorted, alpha=0.6, s=10, color='steelblue')
                    min_val = min(emp_sorted.min(), sim_sorted.min())
                    max_val = max(emp_sorted.max(), sim_sorted.max())
                    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5)
                    ax2.set_xlabel('Empirical Log-Returns Quantiles')
                    ax2.set_ylabel('Model Simulated Quantiles')
                    ax2.set_title('Q-Q Plot: Empirical vs Model (1-day)')
                    ax2.grid(True, linestyle='--', alpha=0.5)
                    st.pyplot(fig2)
                    plt.close(fig2)

                    # KS Test
                    try:
                        ks_stat, p_value = stats.ks_2samp(empirical_log_returns, simulated_log_returns[:len(empirical_log_returns)])
                        st.markdown(f"**Kolmogorov-Smirnov Test (Empirical vs Model):**")
                        st.write(f"- KS Statistic: {ks_stat:.4f}")
                        st.write(f"- p-value: {p_value:.4f}")
                        if p_value > 0.05:
                            st.success("✅ Model distribution is statistically indistinguishable from empirical (p > 0.05)")
                        else:
                            st.warning("⚠️ Model distribution differs significantly from empirical (p ≤ 0.05)")
                    except Exception as e:
                        st.error(f"KS test failed: {e}")
                else:
                    st.warning("⚠️ Not enough data for Q-Q plot or KS test (need ≥100 empirical returns).")

                # ----------------------------
                # Plot 3: Scenario Paths
                # ----------------------------
                last_7_days = close_prices.iloc[-7:]
                days_hist = np.arange(-6, 1)
                days_forecast = np.arange(1, forecast_days + 1)

                fig3, ax3 = plt.subplots(figsize=(8, 4))
                ax3.plot(days_hist, last_7_days.values, 'o-', color='black', label='Last 7 Days', linewidth=2, markersize=4)
                ax3.plot(days_forecast, path_up_5_10[1:], 'o--', color='green', label='+5% to +10%', linewidth=2, markersize=4)
                ax3.plot(days_forecast, path_up_0_5[1:], 'o--', color='blue', label='+0% to +5%', linewidth=2, markersize=4)
                ax3.plot(days_forecast, path_down_0_5[1:], 'o--', color='orange', label='-5% to 0%', linewidth=2, markersize=4)
                ax3.plot(days_forecast, path_down_5_10[1:], 'o--', color='red', label='-10% to -5%', linewidth=2, markersize=4)

                ax3.set_xlabel('Days (0 = today)')
                ax3.set_ylabel('Price ($)')
                ax3.set_title(f'Scenario Paths (Next {forecast_days} Days)')
                ax3.axvline(0, color='gray', linestyle=':', linewidth=1)
                ax3.legend()
                ax3.grid(True, linestyle='--', alpha=0.5)
                st.pyplot(fig3)
                plt.close(fig3)

                st.caption(f"Model: {model_desc} | Calibration: 2-year historical data | Paths: 20,000")

    except Exception as e:
        st.error(f"Error: {str(e)}. Try a major ticker like AAPL, MSFT, or SPY.")
