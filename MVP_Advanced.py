import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from statsmodels.tsa.stattools import adfuller
from scipy import stats

st.set_page_config(page_title="Advanced Stochastic Stock Forecaster", layout="centered")
st.title("📈 Advanced Stochastic Stock Forecaster")
st.markdown("""
*Professional-grade probabilistic forecasting with model validation.*  
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
seed_input = st.number_input("Random seed (0 = random)", min_value=0, value=0, step=1)
use_seed = None if seed_input == 0 else int(seed_input)

if ticker:
    try:
        data = yf.download(ticker, period="2y", progress=False)
        if data.empty:
            st.error("No data found. Try a liquid ticker (e.g. AAPL, SPY).")
        else:
            close_prices = data['Close'].dropna()
            if close_prices.empty:
                st.error("Price data contains only missing values. Try a major ticker like AAPL or SPY.")
            elif len(close_prices) < 60:
                st.error("Not enough data. Need ≥60 days of valid prices.")
            else:
                current_price = float(close_prices.iloc[-1])
                log_prices = np.log(close_prices.values)
                log_returns = np.diff(log_prices)
                log_returns = log_returns[np.isfinite(log_returns)]
                if len(log_returns) < 20:
                    sigma_hist = 0.2
                    mu_hist = 0.0
                else:
                    sigma_hist = float(np.std(log_returns) * np.sqrt(252))
                    mu_hist = float(np.mean(log_returns) * 252)

                # ADF Test & Trend
                adf_result = adfuller(log_prices)
                is_stationary = adf_result[1] < 0.05
                trend_annualized = 0.0
                if len(log_prices) >= 2:
                    n_trend = min(60, len(log_prices))
                    if n_trend >= 2:
                        try:
                            x_trend = np.arange(n_trend)
                            slope, _, _, _, _ = stats.linregress(x_trend, log_prices[-n_trend:])
                            if np.isfinite(slope):
                                trend_annualized = slope * 252
                        except Exception:
                            trend_annualized = 0.0

                st.caption(f"📊 ADF p-value: {adf_result[1]:.3f} → {'Stationary' if is_stationary else 'Non-stationary (trend present)'}")
                st.caption(f"📈 Estimated annualized trend: {trend_annualized:.2%}")

                # ----------------------------
                # GBM with drift
                # ----------------------------
                def simulate_gbm_paths(S0, mu, vol, T, n_paths=20000, n_steps=None, seed=None):
                    if n_steps is None:
                        n_steps = int(T * 252)
                    if seed is not None:
                        np.random.seed(seed)
                    dt = 1/252
                    if n_steps == 0:
                        return np.full((n_paths, 1), S0)
                    Z = np.random.randn(n_paths, n_steps)
                    logS = np.log(S0) + np.cumsum((mu - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * Z, axis=1)
                    S = np.exp(np.hstack([np.full((n_paths, 1), S0), logS]))
                    return S

                # ----------------------------
                # Heston Calibration
                # ----------------------------
                def calibrate_heston(log_returns):
                    if len(log_returns) < 50:
                        return np.array([2.0, 0.04, 0.3, -0.5, 0.04])
                    log_returns = log_returns[np.isfinite(log_returns)]
                    if len(log_returns) < 50:
                        return np.array([2.0, 0.04, 0.3, -0.5, 0.04])
                    
                    daily_var = np.var(log_returns)
                    hist_var = daily_var * 252
                    hist_var = np.clip(hist_var, 1e-4, 1.0)
                    
                    squared = log_returns**2
                    autocorr = 0.0
                    if len(squared) > 2:
                        try:
                            autocorr = np.corrcoef(squared[:-1], squared[1:])[0,1]
                            if not np.isfinite(autocorr):
                                autocorr = 0.0
                        except:
                            autocorr = 0.0
                    autocorr = np.clip(autocorr, 0.0, 0.9)
                    
                    x0 = np.array([2.0, hist_var, 0.3, -0.5, hist_var], dtype=np.float64)
                    
                    def loss(params):
                        kappa, theta, xi, rho, v0 = params
                        if not (0.1 <= kappa <= 20 and 1e-4 <= theta <= 1.0 and 0.01 <= xi <= 2.0 and -0.99 <= rho <= 0.0 and 1e-4 <= v0 <= 1.0):
                            return 1e6
                        try:
                            theoretical_ac = np.exp(-kappa / 252)
                            var_error = (theta - hist_var)**2
                            ac_error = (theoretical_ac - autocorr)**2
                            return var_error + 10 * ac_error
                        except:
                            return 1e6
                    
                    try:
                        res = minimize(loss, x0, method='L-BFGS-B',
                                       bounds=[(0.1, 20), (1e-4, 1.0), (0.01, 2.0), (-0.99, 0.0), (1e-4, 1.0)],
                                       options={'maxiter': 100})
                        if res.success and np.all(np.isfinite(res.x)):
                            return np.array(res.x, dtype=np.float64)
                        else:
                            return x0
                    except:
                        return x0

                # ----------------------------
                # Heston with Full Truncation
                # ----------------------------
                def simulate_heston_paths(S0, kappa, theta, xi, rho, v0, T, n_paths=20000, n_steps=None, seed=None):
                    if n_steps is None:
                        n_steps = int(T * 252)
                    if seed is not None:
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
                        v_plus = np.maximum(v, 0.0)
                        S[:, t] = S[:, t-1] * np.exp(-0.5 * v_plus * dt + np.sqrt(v_plus) * sqrt_dt * Z1)
                        v = v_plus + kappa * (theta - v_plus) * dt + xi * np.sqrt(v_plus) * sqrt_dt * Z2
                        v = np.maximum(v, 0.0)
                    return np.maximum(S, 1e-8)

                # ----------------------------
                # SABR Calibration and Simulation
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
                                if np.isfinite(vol) and vol > 0:
                                    rolling_vol.append(vol)
                            if len(rolling_vol) < 5:
                                nu = 0.5
                            else:
                                rolling_vol = np.array(rolling_vol)
                                vol_changes = np.diff(rolling_vol)
                                if len(vol_changes) < 2 or np.std(vol_changes) == 0:
                                    nu = 0.5
                                else:
                                    nu = (np.std(vol_changes) * np.sqrt(252)) / (np.mean(rolling_vol) + 1e-8)
                                    nu = np.clip(nu, 0.1, 2.0)
                        alpha0 = np.std(log_returns) * np.sqrt(252)
                        alpha0 = np.clip(alpha0, 0.01, 2.0)
                    return alpha0, beta, nu

                def simulate_sabr_paths(F0, alpha0, beta, nu, T, n_paths=20000, n_steps=None, seed=None):
                    if n_steps is None:
                        n_steps = int(T * 252)
                    if seed is not None:
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
                # Kou Jump-Diffusion Calibration
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

                # ----------------------------
                # Kou Jump-Diffusion with drift
                # ----------------------------
                def simulate_kou_paths(S0, mu, vol, λ, η1, η2, p, T, n_paths=20000, n_steps=None, seed=None):
                    if n_steps is None:
                        n_steps = int(T * 252)
                    if seed is not None:
                        np.random.seed(seed)
                    dt = 1/252
                    if n_steps == 0:
                        return np.full((n_paths, 1), S0)
                    S = np.full((n_paths, n_steps+1), S0, dtype=np.float64)
                    for t in range(1, n_steps+1):
                        Z = np.random.randn(n_paths)
                        S[:, t] = S[:, t-1] * np.exp((mu - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * Z)
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
                # Regime-Switching Heston with Full Truncation
                # ----------------------------
                def simulate_regime_switching_heston_paths(S0, T, n_paths=20000, n_steps=None, seed=None):
                    if n_steps is None:
                        n_steps = int(T * 252)
                    if seed is not None:
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
                        v_plus = np.maximum(v, 0.0)
                        S[:, t] = S[:, t-1] * np.exp(-0.5 * v_plus * dt + np.sqrt(v_plus) * sqrt_dt * Z1)
                        v = v_plus + kappa * (theta - v_plus) * dt + xi * np.sqrt(v_plus) * sqrt_dt * Z2
                        v = np.maximum(v, 0.0)
                    return np.maximum(S, 1e-8)

                # ----------------------------
                # Run selected model
                # ----------------------------
                T = forecast_days / 252.0
                n_steps = forecast_days

                if model_choice == "Heston":
                    params = calibrate_heston(log_returns)
                    if not (isinstance(params, np.ndarray) and params.shape == (5,)):
                        params = np.array([2.0, 0.04, 0.3, -0.5, 0.04])
                    kappa, theta, xi, rho, v0 = params
                    all_paths = simulate_heston_paths(current_price, kappa, theta, xi, rho, v0, T, n_paths=20000, n_steps=n_steps, seed=use_seed)
                    model_desc = f"Heston (κ={kappa:.2f}, θ={theta:.4f}, ξ={xi:.2f}, ρ={rho:.2f})"

                elif model_choice == "SABR":
                    alpha0, beta, nu = calibrate_sabr(log_returns, current_price)
                    all_paths = simulate_sabr_paths(current_price, alpha0, beta, nu, T, n_paths=20000, n_steps=n_steps, seed=use_seed)
                    model_desc = f"SABR (α₀={alpha0:.3f}, β={beta:.1f}, ν={nu:.2f})"

                elif model_choice == "GBM (Baseline)":
                    all_paths = simulate_gbm_paths(current_price, mu_hist, sigma_hist, T, n_paths=20000, n_steps=n_steps, seed=use_seed)
                    model_desc = f"GBM (μ={mu_hist:.2%}, σ={sigma_hist:.2%})"

                elif model_choice == "Double Exp Jump-Diffusion":
                    λ, η1, η2, p = calibrate_kou(log_returns)
                    all_paths = simulate_kou_paths(current_price, mu_hist, sigma_hist, λ, η1, η2, p, T, n_paths=20000, n_steps=n_steps, seed=use_seed)
                    model_desc = f"Kou Jump (λ={λ:.2f}, η₁={η1:.1f}, η₂={η2:.1f}, p={p:.2f})"

                elif model_choice == "Regime-Switching Heston":
                    all_paths = simulate_regime_switching_heston_paths(current_price, T, n_paths=20000, n_steps=n_steps, seed=use_seed)
                    model_desc = "Regime-Switching Heston (Calm ↔ Crisis)"

                else:
                    all_paths = simulate_gbm_paths(current_price, mu_hist, sigma_hist, T, n_paths=20000, n_steps=n_steps, seed=use_seed)
                    model_desc = "Fallback GBM"

                # 🔴 КРИТИЧЕСКАЯ ПРОВЕРКА: не пустой ли all_paths?
                if all_paths.size == 0 or all_paths.shape[0] == 0:
                    st.error("Simulation returned empty paths. Try a different model or ticker.")
                    st.stop()

                future_prices = all_paths[:, -1]
                future_prices = future_prices[np.isfinite(future_prices)]
                if len(future_prices) == 0:
                    st.error("Simulation produced no valid prices.")
                    st.stop()

                # Dynamic thresholds
                vol_forecast = sigma_hist * np.sqrt(T)
                p_up2σ = current_price * np.exp(2 * vol_forecast)
                p_down2σ = current_price * np.exp(-2 * vol_forecast)
                p_up1σ = current_price * np.exp(vol_forecast)
                p_down1σ = current_price * np.exp(-vol_forecast)

                prob_up_0_1σ = np.mean((future_prices > current_price) & (future_prices <= p_up1σ))
                prob_up_1σ_2σ = np.mean((future_prices > p_up1σ) & (future_prices <= p_up2σ))
                prob_down_0_1σ = np.mean((future_prices >= p_down1σ) & (future_prices < current_price))
                prob_down_1σ_2σ = np.mean((future_prices >= p_down2σ) & (future_prices < p_down1σ))
                prob_extreme = np.mean((future_prices > p_up2σ) | (future_prices < p_down2σ))

                # Model Validation: Q-Q Plot & KS Test
                empirical_quantiles = np.quantile(log_returns, [0.1, 0.25, 0.5, 0.75, 0.9])
                simulated_returns = np.log(all_paths[:, -1] / all_paths[:, 0])
                simulated_returns = simulated_returns[np.isfinite(simulated_returns)]
                if len(simulated_returns) == 0:
                    model_quantiles = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
                else:
                    model_quantiles = np.quantile(simulated_returns, [0.1, 0.25, 0.5, 0.75, 0.9])

                ks_ok = False
                ks_p = 0.0
                try:
                    emp_std = np.std(log_returns)
                    sim_std = np.std(simulated_returns) if len(simulated_returns) > 1 else 0.0
                    if emp_std > 1e-8 and sim_std > 1e-8:
                        emp_norm = (log_returns - np.mean(log_returns)) / emp_std
                        sim_norm = (simulated_returns - np.mean(simulated_returns)) / sim_std
                        ks_stat, ks_p = stats.kstest(emp_norm, 'norm')
                        ks_ok = ks_p > 0.05
                except Exception:
                    ks_ok = False
                    ks_p = 0.0

                # Output
                st.subheader(f"Current price: ${current_price:.2f}")
                st.write(f"**{forecast_days}-day outlook for {ticker} ({model_choice}):**")
                st.write(f"- 📈 {prob_up_0_1σ:.0%} chance: +0σ to +1σ (${current_price:.2f} → ${p_up1σ:.2f})")
                st.write(f"- 📈 {prob_up_1σ_2σ:.0%} chance: +1σ to +2σ")
                st.write(f"- 📉 {prob_down_0_1σ + prob_down_1σ_2σ:.0%} chance: down to -2σ")
                st.write(f"- ⚠️ {prob_extreme:.0%} chance: extreme move (>±2σ)")

                if not ks_ok:
                    st.warning(f"⚠️ Model fit may be poor (KS test p={ks_p:.3f} < 0.05)")

                # Plot 1: Distribution
                fig1, ax1 = plt.subplots(figsize=(8, 3))
                ax1.hist(future_prices, bins=100, density=True, alpha=0.7, color='steelblue', edgecolor='none')
                ax1.axvline(current_price, color='red', linestyle='--', linewidth=2, label='Current Price')
                ax1.set_xlabel('Future Price ($)')
                ax1.set_ylabel('Density')
                ax1.set_title(f'{model_choice} Forecast Distribution')
                ax1.legend()
                ax1.grid(True, linestyle='--', alpha=0.5)
                st.pyplot(fig1)

                # Plot 2: Q-Q Plot
                fig2, ax2 = plt.subplots(figsize=(5, 5))
                ax2.scatter(empirical_quantiles, model_quantiles, c='red', s=50)
                lims = [min(empirical_quantiles.min(), model_quantiles.min()),
                        max(empirical_quantiles.max(), model_quantiles.max())]
                ax2.plot(lims, lims, 'k--', alpha=0.75, linewidth=1)
                ax2.set_xlabel('Empirical Quantiles (log-returns)')
                ax2.set_ylabel('Model Quantiles')
                ax2.set_title('Q-Q Plot: Model vs Historical')
                ax2.grid(True, linestyle='--', alpha=0.5)
                st.pyplot(fig2)

                st.caption(f"Model: {model_desc} | Seed: {'random' if use_seed is None else use_seed} | KS p-value: {ks_p:.3f}")

    except Exception as e:
        st.error(f"Error: {str(e)}. Try a major ticker like AAPL, MSFT, or SPY.")
