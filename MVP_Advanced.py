import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import io

# ----------------------------
# Вспомогательная функция для доверительного интервала Уилсона
# ----------------------------
def wilson_confidence_interval(count, nobs, alpha=0.05):
    """
    Calculate Wilson confidence interval for a proportion.
    Returns (lower_bound, upper_bound).
    """
    if nobs == 0:
        return 0.0, 0.0
    if count == 0:
        return 0.0, 1.0 - (alpha / 2) ** (1 / nobs)
    if count == nobs:
        return (alpha / 2) ** (1 / nobs), 1.0

    z = 1.96  # For 95% CI
    p = count / nobs
    denominator = 1 + z**2 / nobs
    centre_adjusted_probability = p + z**2 / (2 * nobs)
    adjusted_standard_deviation = np.sqrt((p * (1 - p) + z**2 / (4 * nobs)) / nobs)

    lower_bound = (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator
    upper_bound = (centre_adjusted_probability + z * adjusted_standard_deviation) / denominator

    return np.clip(lower_bound, 0.0, 1.0), np.clip(upper_bound, 0.0, 1.0)


# ----------------------------
# КЭШИРОВАНИЕ ДАННЫХ И МОДЕЛЕЙ
# ----------------------------
@st.cache_data(ttl=3600)
def fetch_stock_data(ticker):
    data = yf.download(ticker, period="2y", progress=False)
    return data

@st.cache_data(ttl=3600)
def validate_ticker(ticker):
    """Returns (is_valid: bool, name: str or error_msg: str)"""
    if not ticker or not isinstance(ticker, str):
        return False, "Ticker is empty"
    try:
        info = yf.Ticker(ticker).info
        if 'regularMarketPrice' in info or 'currentPrice' in info or 'longName' in info:
            name = info.get('longName') or info.get('shortName') or ticker
            return True, name
        else:
            return False, "No valid market data found"
    except Exception:
        return False, "Invalid ticker or connection error"

@st.cache_data(ttl=3600)
def calibrate_and_simulate(
    ticker, forecast_days, model_choice, n_paths=5000, seed=42
):
    # Получаем дивидендную доходность
    try:
        ticker_info = yf.Ticker(ticker).info
        div_yield = ticker_info.get('dividendYield', 0.0)
        if not isinstance(div_yield, (int, float)) or div_yield is None:
            div_yield = 0.0
        div_yield = float(div_yield)
    except:
        div_yield = 0.0

    risk_free = 0.05  # Фиксированная безрисковая ставка

    data = fetch_stock_data(ticker)
    if data.empty:
        return None, None, None, None, "No historical data"

    close_prices = data['Close'].dropna()
    if len(close_prices) < 60:
        return None, None, None, None, "Not enough historical data (need ≥60 days)"

    current_price = float(close_prices.iloc[-1])
    log_prices = np.log(close_prices.values)
    log_returns = np.diff(log_prices)
    log_returns = log_returns[np.isfinite(log_returns)]
    
    # Улучшенная оценка волатильности
    sigma_hist = float(np.std(log_returns) * np.sqrt(252)) if len(log_returns) >= 20 else 0.2
    if len(log_returns) >= 30:
        realized_vol = np.std(log_returns[-30:]) * np.sqrt(252)
        sigma_hist = 0.7 * sigma_hist + 0.3 * realized_vol

    T = forecast_days / 252.0
    # Адаптивная временная дискретизация
    n_steps = max(forecast_days, 10)
    dt = T / n_steps

    # ----------------------------
    # GBM с рыночными параметрами
    # ----------------------------
    def simulate_gbm_paths(S0, vol, T, n_paths=5000, n_steps=None, seed=42, div_yield=0.0, risk_free=0.05):
        if n_steps is None:
            n_steps = max(int(T * 252), 10)
        dt = T / n_steps
        np.random.seed(seed)
        if n_steps == 0:
            return np.full((n_paths, 1), S0)
        Z = np.random.randn(n_paths, n_steps)
        # Исправленный дрейф с учётом дивидендов и безрисковой ставки
        drift = (risk_free - div_yield - 0.5 * vol**2) * dt
        diffusion = vol * np.sqrt(dt) * Z
        logS = np.log(S0) + np.cumsum(drift + diffusion, axis=1)
        S = np.exp(np.hstack([np.full((n_paths, 1), S0), logS]))
        return np.maximum(S, 1e-8)

    # ----------------------------
    # Heston с Full Truncation
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

    def simulate_heston_paths(S0, kappa, theta, xi, rho, v0, T, n_paths=5000, n_steps=None, seed=42, div_yield=0.0, risk_free=0.05):
        if n_steps is None:
            n_steps = max(int(T * 252), 10)
        dt = T / n_steps
        np.random.seed(seed)
        if n_steps == 0:
            return np.full((n_paths, 1), S0)
        S = np.full((n_paths, n_steps+1), S0, dtype=np.float64)
        v = np.full(n_paths, v0, dtype=np.float64)
        for t in range(1, n_steps+1):
            Z1 = np.random.randn(n_paths)
            Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.randn(n_paths)
            # Full Truncation Scheme
            v_plus = np.maximum(v, 0.0)
            # Исправленный дрейф для цены
            drift = (risk_free - div_yield - 0.5 * v_plus) * dt
            diffusion = np.sqrt(v_plus * dt) * Z1
            S[:, t] = S[:, t-1] * np.exp(drift + diffusion)
            # Обновление волатильности
            v = v + kappa * (theta - v_plus) * dt + xi * np.sqrt(v_plus * dt) * Z2
        return np.maximum(S, 1e-8)

    # ----------------------------
    # SABR с логнормальной формой
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

    def simulate_sabr_paths(F0, alpha0, beta, nu, T, n_paths=5000, n_steps=None, seed=42):
        if n_steps is None:
            n_steps = max(int(T * 252), 10)
        dt = T / n_steps
        np.random.seed(seed)
        if n_steps == 0:
            return np.full((n_paths, 1), F0)
        F = np.full((n_paths, n_steps+1), F0, dtype=np.float64)
        alpha = np.full(n_paths, alpha0, dtype=np.float64)
        for t in range(1, n_steps+1):
            Z1 = np.random.randn(n_paths)
            Z2 = np.random.randn(n_paths)
            # Логнормальная форма SABR
            F_prev = np.maximum(F[:, t-1], 1e-8)
            dlogF = -0.5 * (alpha**2) * (F_prev**(2*beta - 2)) * dt + alpha * (F_prev**(beta - 1)) * np.sqrt(dt) * Z1
            F[:, t] = F_prev * np.exp(dlogF)
            # Обновление альфы
            alpha *= np.exp(-0.5 * nu**2 * dt + nu * np.sqrt(dt) * Z2)
        return np.maximum(F, 1e-8)

    # ----------------------------
    # Double Exponential Jump-Diffusion с рыночными параметрами
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

    def simulate_kou_paths(S0, vol, λ, η1, η2, p, T, n_paths=5000, n_steps=None, seed=42, div_yield=0.0, risk_free=0.05):
        if n_steps is None:
            n_steps = max(int(T * 252), 10)
        dt = T / n_steps
        np.random.seed(seed)
        if n_steps == 0:
            return np.full((n_paths, 1), S0)
        S = np.full((n_paths, n_steps+1), S0, dtype=np.float64)
        for t in range(1, n_steps+1):
            Z = np.random.randn(n_paths)
            # Исправленный дрейф с учётом дивидендов
            drift = (risk_free - div_yield - 0.5 * vol**2) * dt
            S[:, t] = S[:, t-1] * np.exp(drift + vol * np.sqrt(dt) * Z)
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
    # Regime-Switching Heston (оставим без изменений, так как он внутренний)
    # ----------------------------
    def simulate_regime_switching_heston_paths(S0, T, n_paths=5000, n_steps=None, seed=42):
        if n_steps is None:
            n_steps = max(int(T * 252), 10)
        dt = T / n_steps
        np.random.seed(seed)
        if n_steps == 0:
            return np.full((n_paths, 1), S0)
        params0 = [3.0, 0.02, 0.2, -0.3, 0.02]
        params1 = [1.5, 0.10, 0.6, -0.8, 0.10]
        P = np.array([[0.98, 0.02], [0.10, 0.90]])
        S = np.full((n_paths, n_steps+1), S0, dtype=np.float64)
        v = np.full(n_paths, params0[4], dtype=np.float64)
        regime = np.zeros(n_paths, dtype=int)
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
            S[:, t] = S[:, t-1] * np.exp(-0.5 * v_plus * dt + np.sqrt(v_plus * dt) * Z1)
            v = v + kappa * (theta - v_plus) * dt + xi * np.sqrt(v_plus * dt) * Z2
        return np.maximum(S, 1e-8)

    # ----------------------------
    # Run selected model
    # ----------------------------
    if model_choice == "Heston":
        params = calibrate_heston(log_returns)
        if not (isinstance(params, np.ndarray) and params.shape == (5,)):
            params = np.array([2.0, 0.04, 0.3, -0.5, 0.04])
        kappa, theta, xi, rho, v0 = params
        all_paths = simulate_heston_paths(current_price, kappa, theta, xi, rho, v0, T, n_paths=n_paths, n_steps=n_steps, div_yield=div_yield, risk_free=risk_free)
        model_desc = f"Heston (κ={kappa:.2f}, θ={theta:.4f}, ξ={xi:.2f}, ρ={rho:.2f})"

    elif model_choice == "SABR":
        alpha0, beta, nu = calibrate_sabr(log_returns, current_price)
        all_paths = simulate_sabr_paths(current_price, alpha0, beta, nu, T, n_paths=n_paths, n_steps=n_steps)
        model_desc = f"SABR (α₀={alpha0:.3f}, β={beta:.1f}, ν={nu:.2f})"

    elif model_choice == "GBM (Baseline)":
        all_paths = simulate_gbm_paths(current_price, sigma_hist, T, n_paths=n_paths, n_steps=n_steps, div_yield=div_yield, risk_free=risk_free)
        model_desc = f"GBM (σ={sigma_hist:.2%})"

    elif model_choice == "Double Exp Jump-Diffusion":
        λ, η1, η2, p = calibrate_kou(log_returns)
        all_paths = simulate_kou_paths(current_price, sigma_hist, λ, η1, η2, p, T, n_paths=n_paths, n_steps=n_steps, div_yield=div_yield, risk_free=risk_free)
        model_desc = f"Kou Jump (λ={λ:.2f}, η₁={η1:.1f}, η₂={η2:.1f}, p={p:.2f})"

    elif model_choice == "Regime-Switching Heston":
        all_paths = simulate_regime_switching_heston_paths(current_price, T, n_paths=n_paths, n_steps=n_steps)
        model_desc = "Regime-Switching Heston (Calm ↔ Crisis)"

    else:
        all_paths = simulate_gbm_paths(current_price, sigma_hist, T, n_paths=n_paths, n_steps=n_steps, div_yield=div_yield, risk_free=risk_free)
        model_desc = "Fallback GBM"

    future_prices = all_paths[:, -1]
    # Защита от численных ошибок
    future_prices = future_prices[np.isfinite(future_prices) & (future_prices > 0)]
    if len(future_prices) == 0:
        return None, None, None, None, "Simulation produced no valid prices"

    return all_paths, future_prices, current_price, model_desc, None


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Advanced Stochastic Stock Forecaster", layout="centered")
st.title("📈 Advanced Stochastic Stock Forecaster")
st.markdown("""
*Professional-grade probabilistic forecasting with scenario paths.*  
⚠️ **Not financial advice. For educational/research purposes only.**
""")

ticker_input = st.text_input("Enter stock ticker (e.g. AAPL, TSLA, MSFT)", value="AAPL").upper().strip()
forecast_days = st.slider("Forecast horizon (days)", min_value=1, max_value=30, value=5)
model_choice = st.selectbox("Stochastic model", [
    "Heston", 
    "SABR", 
    "GBM (Baseline)",
    "Double Exp Jump-Diffusion",
    "Regime-Switching Heston"
])

# Новое поле: целевая цена
target_price_input = st.text_input("🎯 Target price (optional)", value="")
compare_with_spy = st.checkbox("📊 Compare with S&P 500 (SPY)")

run_button = st.button("🚀 Run Forecast")

if run_button:
    if not ticker_input:
        st.warning("Please enter a stock ticker.")
    else:
        with st.spinner("Validating ticker..."):
            is_valid, name_or_error = validate_ticker(ticker_input)

        if not is_valid:
            st.error(f"❌ Invalid ticker '{ticker_input}': {name_or_error}. Try AAPL, MSFT, SPY, or BTC-USD.")
        else:
            st.success(f"✅ Valid ticker: **{name_or_error}** ({ticker_input})")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("Running simulation...")

            try:
                all_paths, future_prices, current_price, model_desc, error = calibrate_and_simulate(
                    ticker_input, forecast_days, model_choice, n_paths=5000
                )
                progress_bar.progress(100)
                status_text.text("Simulation complete!")
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Simulation failed: {str(e)}")
                st.stop()

            if error:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Simulation error: {error}")
            elif all_paths is None:
                progress_bar.empty()
                status_text.empty()
                st.error("Unexpected error during simulation.")
            else:
                progress_bar.empty()
                status_text.empty()

                # ----------------------------
                # Target price probability
                # ----------------------------
                target_price = None
                if target_price_input.strip():
                    try:
                        target_price = float(target_price_input.strip())
                        if target_price <= 0:
                            st.warning("Target price must be positive.")
                        else:
                            count_target = np.sum(future_prices >= target_price)
                            nobs = len(future_prices)
                            prob_above = count_target / nobs if nobs > 0 else 0.0
                            ci_low, ci_up = wilson_confidence_interval(count_target, nobs)
                            st.info(f"🎯 Probability that **{ticker_input} ≥ ${target_price:.2f}** in {forecast_days} days: **{prob_above:.1%}** (95% CI: {ci_low:.1%}–{ci_up:.1%})")
                    except ValueError:
                        st.warning("Invalid target price. Please enter a number.")

                # ----------------------------
                # Compute base probabilities with CIs
                # ----------------------------
                p0 = current_price
                p_up5 = p0 * 1.05
                p_up10 = p0 * 1.10
                p_down5 = p0 * 0.95
                p_down10 = p0 * 0.90

                nobs = len(future_prices)
                # Up 0-5%
                count_up_0_5 = np.sum((future_prices > p0) & (future_prices <= p_up5))
                prob_up_0_5 = count_up_0_5 / nobs if nobs > 0 else 0.0
                ci_up_0_5_low, ci_up_0_5_up = wilson_confidence_interval(count_up_0_5, nobs)

                # Up 5-10%
                count_up_5_10 = np.sum((future_prices > p_up5) & (future_prices <= p_up10))
                prob_up_5_10 = count_up_5_10 / nobs if nobs > 0 else 0.0
                ci_up_5_10_low, ci_up_5_10_up = wilson_confidence_interval(count_up_5_10, nobs)

                # Down 0-5%
                count_down_0_5 = np.sum((future_prices >= p_down5) & (future_prices < p0))
                prob_down_0_5 = count_down_0_5 / nobs if nobs > 0 else 0.0
                ci_down_0_5_low, ci_down_0_5_up = wilson_confidence_interval(count_down_0_5, nobs)

                # Down 5-10%
                count_down_5_10 = np.sum((future_prices >= p_down10) & (future_prices < p_down5))
                prob_down_5_10 = count_down_5_10 / nobs if nobs > 0 else 0.0
                ci_down_5_10_low, ci_down_5_10_up = wilson_confidence_interval(count_down_5_10, nobs)

                # Extreme
                count_extreme = np.sum((future_prices > p_up10) | (future_prices < p_down10))
                prob_extreme = count_extreme / nobs if nobs > 0 else 0.0
                ci_extreme_low, ci_extreme_up = wilson_confidence_interval(count_extreme, nobs)

                down_0_10 = prob_down_0_5 + prob_down_5_10

                # ----------------------------
                # Find representative paths
                # ----------------------------
                def find_path_in_range(paths, lower, upper):
                    final = paths[:, -1]
                    mask = (final > lower) & (final <= upper)
                    if np.any(mask):
                        idx = np.where(mask)[0][0]
                        return paths[idx]
                    else:
                        distances = np.abs(final - (lower + upper) / 2)
                        idx = np.argmin(distances)
                        return paths[idx]

                path_up_5_10 = find_path_in_range(all_paths, p_up5, p_up10)
                path_up_0_5 = find_path_in_range(all_paths, p0, p_up5)
                path_down_0_5 = find_path_in_range(all_paths, p_down5, p0)
                path_down_5_10 = find_path_in_range(all_paths, p_down10, p_down5)

                # ----------------------------
                # Display results
                # ----------------------------
                st.subheader(f"Current price: ${current_price:.2f}")
                st.write(f"**{forecast_days}-day outlook for {name_or_error} ({ticker_input}) using {model_choice}:**")
                st.write(f"- 📈 {prob_up_0_5:.0%} (95% CI: {ci_up_0_5_low:.0%}–{ci_up_0_5_up:.0%}) chance: +0% to +5%")
                st.write(f"- 📈 {prob_up_5_10:.0%} (95% CI: {ci_up_5_10_low:.0%}–{ci_up_5_10_up:.0%}) chance: +5% to +10%")
                st.write(f"- 📉 {down_0_10:.0%} chance: down to -10%")
                st.write(f"- ⚠️ {prob_extreme:.0%} (95% CI: {ci_extreme_low:.0%}–{ci_extreme_up:.0%}) chance: extreme move (>±10%)")
                
                # Ожидаемая доходность
                expected_price = np.mean(future_prices)
                expected_return_pct = (expected_price / current_price - 1) * 100
                st.write(f"- 💡 **Expected return: {expected_return_pct:+.2f}%** in {forecast_days} days")

                # Plot 1: Distribution
                fig1, ax1 = plt.subplots(figsize=(8, 3.5))
                ax1.hist(future_prices, bins=120, density=True, alpha=0.7, color='steelblue', edgecolor='none')
                ax1.axvline(current_price, color='red', linestyle='--', linewidth=2, label='Current Price')
                if target_price and target_price > 0:
                    ax1.axvline(target_price, color='purple', linestyle='-.', linewidth=2, label=f'Target ${target_price:.2f}')
                ax1.set_xlabel('Future Price ($)')
                ax1.set_ylabel('Density')
                ax1.set_title(f'{model_choice} Forecast Distribution')
                ax1.legend()
                ax1.grid(True, linestyle='--', alpha=0.5)
                st.pyplot(fig1)

                # Plot 2: Scenario paths
                data = fetch_stock_data(ticker_input)
                close_prices = data['Close'].dropna()
                last_7_days = close_prices.iloc[-7:]
                days_hist = np.arange(-6, 1)
                days_forecast = np.arange(1, forecast_days + 1)

                fig2, ax2 = plt.subplots(figsize=(8, 4))
                ax2.plot(days_hist, last_7_days.values, 'o-', color='black', label='Last 7 Days', linewidth=2, markersize=4)
                ax2.plot(days_forecast, path_up_5_10[1:], 'o--', color='green', label='+5% to +10%', linewidth=2, markersize=4)
                ax2.plot(days_forecast, path_up_0_5[1:], 'o--', color='blue', label='+0% to +5%', linewidth=2, markersize=4)
                ax2.plot(days_forecast, path_down_0_5[1:], 'o--', color='orange', label='-5% to 0%', linewidth=2, markersize=4)
                ax2.plot(days_forecast, path_down_5_10[1:], 'o--', color='red', label='-10% to -5%', linewidth=2, markersize=4)

                ax2.set_xlabel('Days (0 = today)')
                ax2.set_ylabel('Price ($)')
                ax2.set_title('Scenario Paths (Next {} Days)'.format(forecast_days))
                ax2.axvline(0, color='gray', linestyle=':', linewidth=1)
                ax2.legend()
                ax2.grid(True, linestyle='--', alpha=0.5)
                st.pyplot(fig2)

                st.caption(f"Model: {model_desc} | Calibration: 2-year historical data | Paths: 5,000")

                # ----------------------------
                # Сравнение с SPY (если выбрано)
                # ----------------------------
                if compare_with_spy:
                    st.markdown("---")
                    st.subheader("🆚 Comparison with S&P 500 (SPY)")
                    with st.spinner("Simulating SPY scenarios..."):
                        spy_paths, spy_future, spy_price, _, _ = calibrate_and_simulate(
                            "SPY", forecast_days, "GBM (Baseline)", n_paths=2000
                        )
                    if spy_paths is not None:
                        spy_expected = (np.mean(spy_future) / spy_price - 1) * 100
                        st.write(f"**SPY current: ${spy_price:.2f}** | Expected return: **{spy_expected:+.2f}%** in {forecast_days} days")

                        # Plot SPY distribution
                        fig3, ax3 = plt.subplots(figsize=(8, 3))
                        ax3.hist(spy_future, bins=80, density=True, alpha=0.6, color='gray', label='SPY')
                        ax3.hist(future_prices, bins=80, density=True, alpha=0.6, color='steelblue', label=ticker_input)
                        ax3.axvline(spy_price, color='black', linestyle='--', label='SPY Today')
                        ax3.axvline(current_price, color='red', linestyle='--', label=f'{ticker_input} Today')
                        ax3.set_xlabel('Future Price ($)')
                        ax3.set_ylabel('Density')
                        ax3.set_title('Forecast Distribution: {} vs SPY'.format(ticker_input))
                        ax3.legend()
                        ax3.grid(True, linestyle='--', alpha=0.5)
                        st.pyplot(fig3)
                    else:
                        st.warning("Could not simulate SPY.")

                # ----------------------------
                # Экспорт в CSV
                # ----------------------------
                st.markdown("---")
                st.subheader("📥 Export Forecast Data")
                # Создаём DataFrame: каждая строка — один сценарий, столбцы — дни (0 = сегодня, 1...forecast_days)
                df_export = pd.DataFrame(all_paths)
                df_export.columns = [f"Day_{i}" for i in range(forecast_days + 1)]
                csv_buffer = io.StringIO()
                df_export.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()

                st.download_button(
                    label="💾 Download Forecast Scenarios (CSV)",
                    data=csv_data,
                    file_name=f"{ticker_input}_forecast_{forecast_days}d.csv",
                    mime="text/csv"
                )

else:
    st.info("👆 Adjust parameters and click **'Run Forecast'** to start.")
