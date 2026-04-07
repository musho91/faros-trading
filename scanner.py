# ==============================================================================
# FAROS v7.0 - INSTITUTIONAL QUANT SUITE
# Autor: Juan Arroyo | SG Consulting Group
# Core: Navier-Stokes + Future Alpha Integration (TAI-ACF v3.0)
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta
from physics_engine import FarosPhysics

# Instancia Física
fisica = FarosPhysics()

# --- CONFIGURACIÓN VISUAL (ESTILO BLOOMBERG/INSTITUCIONAL) ---
st.set_page_config(page_title="FAROS Institutional", page_icon="🏛️", layout="wide")
st.markdown("""
<style>
    .stApp { background-color: #0a0e1a; color: #e0e6f0; }
    h1, h2, h3 { color: #c9d6f0 !important; font-family: 'Helvetica Neue', sans-serif; letter-spacing: 0.05em; }
    .metric-card {
        background: linear-gradient(135deg, #111827, #1e293b);
        border: 1px solid #1e3a5f;
        padding: 16px 20px;
        border-radius: 8px;
        text-align: center;
    }
    .regime-tag {
        font-weight: bold;
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 0.85em;
        letter-spacing: 0.08em;
    }
    .stDataFrame { background-color: #111827; }
    section[data-testid="stSidebar"] { background-color: #070b14; border-right: 1px solid #1e3a5f; }
    .stProgress > div > div { background-color: #1d4ed8; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# BASE DE DATOS MAESTRA
# ==============================================================================
ASSET_DB = {
    "NVIDIA Corp (NVDA)": "NVDA",
    "Palantir Tech (PLTR)": "PLTR",
    "Tesla Inc (TSLA)": "TSLA",
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Apple Inc (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Amazon (AMZN)": "AMZN",
    "Alphabet (GOOGL)": "GOOGL",
    "Meta Platforms (META)": "META",
    "S&P 500 ETF (SPY)": "SPY",
    "Nasdaq 100 (QQQ)": "QQQ",
    "Russell 2000 (IWM)": "IWM",
    "Coinbase (COIN)": "COIN",
    "MicroStrategy (MSTR)": "MSTR",
    "D-Wave Quantum (QBTS)": "QBTS",
    "IonQ Inc (IONQ)": "IONQ",
    "C3.ai (AI)": "AI",
}

def get_ticker_list(selection, manual_input):
    final_list = [ASSET_DB[item] for item in selection if item in ASSET_DB]
    if manual_input:
        extras = [x.strip().upper() for x in manual_input.split(',') if x.strip()]
        final_list.extend(extras)
    return list(set(final_list))

# ==============================================================================
# DATA FEED
# ==============================================================================
@st.cache_data(ttl=600)
def fetch_market_data(ticker, period="1y"):
    try:
        df = yf.Ticker(ticker).history(period=period)
        return df if not df.empty else pd.DataFrame()
    except:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_global_context(profile):
    spy = fetch_market_data("SPY", "1y")
    if spy.empty:
        return "#6b7280", 0, "Data Feed Offline", pd.DataFrame()
    try:
        metrics = fisica.calcular_metricas_completas(spy, profile)
        if metrics is None:
            return "#6b7280", 0, "Insufficient Data", spy
        re_pct = metrics.reynolds_pct if metrics.reynolds_pct > 0 else 50.0
        psi = metrics.psi
        regime = metrics.regime
        msg = f"{regime}  |  Re: {re_pct:.0f}%ile"
        color = "#16a34a" if "ACCUMULATION" in regime else ("#dc2626" if "BREAK" in regime else "#d97706")
        return color, psi, msg, spy
    except Exception:
        return "#6b7280", 0, "Calculation Error", spy

def signal_from_regime(psi, regime):
    if "ACCUMULATION" in regime and psi >= 50:
        return "✅ BUY / LONG", "#16a34a"
    elif "MOMENTUM" in regime:
        return "🚀 STRONG BUY", "#2563eb"
    elif "CONSOLIDATION" in regime:
        return "⏸ HOLD / NEUTRAL", "#6b7280"
    else:
        return "⛔ SELL / CASH", "#dc2626"

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown("## 🏛️ FAROS")
    st.caption("**TAI-ACF Framework v3.0**")
    st.markdown("---")
    risk_profile = st.select_slider(
        "Investment Profile",
        options=["Conservador", "Growth", "Quantum"],
        value="Growth"
    )
    st.markdown("---")
    app_mode = st.radio("MODULES", [
        "🤖 QUANT ANALYST",
        "💼 PORTFOLIO BUILDER",
        "🔍 ALPHA SCANNER",
        "⏳ BACKTEST LAB",
        "🔮 ORACLE PROJECTIONS",
    ])
    st.markdown("---")
    c_glob, psi_glob, msg_glob, _ = get_global_context(risk_profile)
    st.markdown("**Global Context — S&P 500**")
    st.markdown(
        f"<div style='background:{c_glob};color:white;padding:8px;border-radius:6px;"
        f"text-align:center;font-size:0.78em;letter-spacing:0.06em;'>{msg_glob}</div>",
        unsafe_allow_html=True
    )
    st.metric("Market Health (Ψ)", f"{psi_glob:.0f} / 100")

# ==============================================================================
# MÓDULO 1 — QUANT ANALYST
# ==============================================================================
if app_mode == "🤖 QUANT ANALYST":
    st.header("Quantitative Analyst")
    st.caption("Structural analysis powered by Navier-Stokes Financial Hydrodynamics.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for chat in st.session_state.chat_history:
        st.chat_message(chat["role"]).markdown(chat["content"])

    if prompt := st.chat_input("Enter ticker (e.g. PLTR, NVDA, BTC-USD)..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Running TAI-ACF engine..."):
                # Extrae el token que más parece un ticker (1-7 chars, solo letras y guión)
                # Así "analiza PLTR" o "dame NVDA" funciona correctamente
                tokens = prompt.upper().replace("$", "").split()
                ticker = next(
                    (t for t in reversed(tokens)
                     if t.replace("-", "").isalpha() and 1 <= len(t) <= 7),
                    tokens[-1]
                )
                df = fetch_market_data(ticker, "2y")

                if not df.empty:
                    metrics = fisica.calcular_metricas_completas(df, risk_profile)

                    if metrics:
                        last_price = df['Close'].iloc[-1]
                        signal, sig_color = signal_from_regime(metrics.psi, metrics.regime)

                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Last Price", f"${last_price:,.2f}")
                        c2.metric("Governance Ψ", f"{metrics.psi:.1f}/100")
                        c3.metric("Reynolds Pct.", f"{metrics.reynolds_pct:.0f}%ile")
                        c4.metric("Future Alpha", f"{metrics.future_score:.1f}/100")

                        st.markdown(
                            f"<div style='background:{sig_color};color:white;padding:10px;"
                            f"border-radius:6px;text-align:center;font-size:1.1em;"
                            f"letter-spacing:0.1em;margin:12px 0;'><b>{signal}</b></div>",
                            unsafe_allow_html=True
                        )

                        with st.expander("🔬 Physical Metrics (TAI-ACF)"):
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("Regime", metrics.regime)
                            m2.metric("Shannon Entropy", f"{metrics.shannon_entropy:.3f}")
                            m3.metric("α-flow", f"{metrics.alpha_flow:.3f}")
                            m4.metric("Z-Score", f"{metrics.z_score_price:.2f}")

                            mu_col, rho_col = st.columns(2)
                            mu_col.metric("Viscosity (µ)", f"{metrics.viscosity_mu:.5f}")
                            rho_col.metric("Density (ρ)", f"{metrics.density_rho:.2f}")

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=df.index, y=df['Close'],
                            mode='lines', name=ticker,
                            line=dict(color='#3b82f6', width=2)
                        ))
                        sma50 = df['Close'].rolling(50).mean()
                        sma200 = df['Close'].rolling(200).mean()
                        fig.add_trace(go.Scatter(x=df.index, y=sma50, name='SMA 50',
                                                  line=dict(color='#f59e0b', width=1, dash='dot')))
                        fig.add_trace(go.Scatter(x=df.index, y=sma200, name='SMA 200',
                                                  line=dict(color='#ef4444', width=1, dash='dot')))
                        fig.update_layout(
                            paper_bgcolor='#0a0e1a', plot_bgcolor='#111827',
                            font_color='#e0e6f0', height=350,
                            margin=dict(l=10, r=10, t=30, b=10),
                            legend=dict(bgcolor='#111827')
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        response_md = (
                            f"**{ticker}** — Regime: `{metrics.regime}` | "
                            f"Ψ: `{metrics.psi:.1f}` | Signal: {signal}"
                        )
                    else:
                        response_md = f"⚠️ Insufficient data to compute TAI-ACF metrics for **{ticker}**."
                else:
                    response_md = f"❌ No market data found for **{ticker}**. Check the ticker symbol."

                st.markdown(response_md)
                st.session_state.chat_history.append({"role": "assistant", "content": response_md})

# ==============================================================================
# MÓDULO 2 — PORTFOLIO BUILDER
# ==============================================================================
elif app_mode == "💼 PORTFOLIO BUILDER":
    st.header("Portfolio Builder")
    st.caption("Optimal capital allocation weighted by Governance Score Ψ.")

    col_sel, col_man = st.columns([2, 1])
    with col_sel:
        selection = st.multiselect(
            "Select Assets:", list(ASSET_DB.keys()),
            default=["NVIDIA Corp (NVDA)", "Palantir Tech (PLTR)", "Apple Inc (AAPL)", "Bitcoin (BTC)"]
        )
    with col_man:
        manual = st.text_input("Add tickers manually (comma-separated):", "")

    capital = st.number_input("Portfolio Capital (USD):", min_value=1000, value=100000, step=1000)

    if st.button("Build Portfolio", type="primary"):
        tickers = get_ticker_list(selection, manual)
        if not tickers:
            st.warning("Select at least one asset.")
        else:
            scores = {}
            log_data = []
            prog = st.progress(0)

            for i, t in enumerate(tickers):
                df = fetch_market_data(t, "1y")
                if not df.empty:
                    re, future, psi, regime = fisica.calcular_metricas_institucionales(df, risk_profile)
                    if psi > 20:
                        scores[t] = psi
                    log_data.append({
                        "Asset": t,
                        "Price": f"${df['Close'].iloc[-1]:,.2f}",
                        "Regime": regime,
                        "Future Alpha": round(future, 1),
                        "Governance Ψ": round(psi, 1),
                        "Investable": "✅" if psi > 20 else "⛔"
                    })
                prog.progress((i + 1) / len(tickers))

            if scores:
                total = sum(scores.values())
                weights = {t: s / total for t, s in scores.items()}

                fig = go.Figure(go.Pie(
                    labels=list(weights.keys()),
                    values=[round(w * 100, 1) for w in weights.values()],
                    hole=0.45,
                    marker=dict(colors=px.colors.sequential.Blues_r[:len(weights)])
                ))
                fig.update_layout(
                    paper_bgcolor='#0a0e1a', font_color='#e0e6f0',
                    title="Capital Allocation by Ψ Weight",
                    height=380, margin=dict(t=40, b=10)
                )
                st.plotly_chart(fig, use_container_width=True)

                alloc_df = pd.DataFrame([
                    {"Ticker": t, "Weight": f"{w*100:.1f}%", "USD Allocation": f"${w*capital:,.0f}"}
                    for t, w in weights.items()
                ])
                st.dataframe(alloc_df, use_container_width=True, hide_index=True)
            else:
                st.warning("No investable assets found (Ψ > 20). Consider moving to CASH.")

            with st.expander("🔎 Full Quantitative Analysis"):
                st.dataframe(pd.DataFrame(log_data), use_container_width=True, hide_index=True)

# ==============================================================================
# MÓDULO 3 — ALPHA SCANNER
# ==============================================================================
elif app_mode == "🔍 ALPHA SCANNER":
    st.header("Institutional Alpha Scanner")
    st.caption("Scans the universe for assets with high Governance Score Ψ.")

    defaults = ["NVIDIA Corp (NVDA)", "Palantir Tech (PLTR)", "Tesla Inc (TSLA)", "Bitcoin (BTC)"]
    sel_scan = st.multiselect("Universe:", list(ASSET_DB.keys()), default=defaults)

    if st.button("Run Scanner", type="primary"):
        tickers = get_ticker_list(sel_scan, "")
        results = []
        prog = st.progress(0)

        for i, t in enumerate(tickers):
            df = fetch_market_data(t, "1y")
            if not df.empty:
                try:
                    metrics = fisica.calcular_metricas_completas(df, risk_profile)
                    if metrics:
                        results.append({
                            "Ticker": t,
                            "Last Price": df['Close'].iloc[-1],
                            "Regime": metrics.regime,
                            "Reynolds %ile": round(metrics.reynolds_pct, 0),
                            "Shannon H": round(metrics.shannon_entropy, 3),
                            "α-flow": round(metrics.alpha_flow, 3),
                            "Future Alpha": round(metrics.future_score, 1),
                            "Ψ Score": round(metrics.psi, 1),
                        })
                except:
                    pass
            prog.progress((i + 1) / len(tickers))

        if results:
            df_res = pd.DataFrame(results).sort_values("Ψ Score", ascending=False)

            def color_regime(val):
                if 'ACCUMULATION' in str(val): return 'background-color: #14532d; color: #86efac'
                if 'MOMENTUM' in str(val): return 'background-color: #1e3a8a; color: #93c5fd'
                if 'BREAK' in str(val): return 'background-color: #7f1d1d; color: #fca5a5'
                return 'background-color: #1f2937; color: #9ca3af'

            styled = df_res.style\
                .map(color_regime, subset=['Regime'])\
                .format({"Last Price": "${:.2f}", "Future Alpha": "{:.1f}", "Ψ Score": "{:.1f}"})

            st.dataframe(styled, use_container_width=True, hide_index=True)

            fig = px.scatter(
                df_res, x="Future Alpha", y="Ψ Score",
                color="Regime", size="Ψ Score", hover_name="Ticker",
                title="Alpha Map — Future Potential vs. Governance Score",
                color_discrete_map={
                    "INSTITUTIONAL ACCUMULATION": "#16a34a",
                    "HIGH MOMENTUM": "#2563eb",
                    "STRUCTURAL BREAK": "#dc2626",
                    "CONSOLIDATION": "#6b7280",
                    "DISTRIBUTION/BEAR": "#9a3412",
                }
            )
            fig.update_layout(paper_bgcolor='#0a0e1a', plot_bgcolor='#111827',
                               font_color='#e0e6f0', height=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data returned for selected assets.")

# ==============================================================================
# MÓDULO 4 — BACKTEST LAB
# ==============================================================================
elif app_mode == "⏳ BACKTEST LAB":
    st.header("Historical Validation Lab")
    st.caption("Structural discipline vs. reactive management.")

    c1, c2, c3 = st.columns(3)
    tck = c1.text_input("Ticker:", "NVDA").upper()
    years = c2.selectbox("Period:", ["1y", "2y", "5y"], index=1)
    vol_thresh = c3.slider("Vol. Filter (daily):", 0.01, 0.06, 0.025, 0.005)

    if st.button("Run Simulation", type="primary"):
        df = fetch_market_data(tck, years)
        if not df.empty:
            df['Ret'] = df['Close'].pct_change()
            df['SMA50'] = df['Close'].rolling(50).mean()
            df['SMA200'] = df['Close'].rolling(200).mean()
            vol = df['Ret'].rolling(20).std()

            signal = np.where((df['SMA50'] > df['SMA200']) & (vol < vol_thresh), 1, 0)
            if risk_profile == "Quantum":
                strong_trend = df['Close'] > df['SMA50'] * 1.1
                signal = np.where(strong_trend, 1, signal)

            df['Signal'] = pd.Series(signal, index=df.index).shift(1).fillna(0)
            df['Strategy'] = (1 + df['Ret'] * df['Signal']).cumprod()
            df['BuyHold'] = (1 + df['Ret']).cumprod()

            perf_s = (df['Strategy'].iloc[-1] - 1) * 100
            perf_bh = (df['BuyHold'].iloc[-1] - 1) * 100

            strat_returns = df['Ret'] * df['Signal']
            sharpe = (strat_returns.mean() / strat_returns.std() * np.sqrt(252)) if strat_returns.std() > 0 else 0

            m1, m2, m3 = st.columns(3)
            m1.metric("FAROS Strategy", f"{perf_s:,.1f}%", delta=f"{perf_s - perf_bh:.1f}% vs B&H")
            m2.metric("Buy & Hold", f"{perf_bh:,.1f}%")
            m3.metric("Sharpe Ratio", f"{sharpe:.2f}")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['BuyHold'], name='Buy & Hold',
                                      line=dict(color='#6b7280', dash='dash')))
            fig.add_trace(go.Scatter(x=df.index, y=df['Strategy'], name='FAROS Strategy',
                                      line=dict(color='#3b82f6', width=2)))
            fig.update_layout(
                paper_bgcolor='#0a0e1a', plot_bgcolor='#111827',
                font_color='#e0e6f0', height=400,
                title=f"Equity Curve — {tck} ({years})",
                yaxis_title="Growth of $1"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Data unavailable for this ticker.")

# ==============================================================================
# MÓDULO 5 — ORACLE PROJECTIONS
# ==============================================================================
elif app_mode == "🔮 ORACLE PROJECTIONS":
    st.header("Future Price Projections (Monte Carlo)")
    st.caption("Structural drift projection with regime-adjusted parameters.")

    c1, c2 = st.columns(2)
    t_input = c1.text_input("Ticker:", "NVDA").upper()
    h_days = c2.slider("Projection Horizon (Days):", 30, 365, 252)

    if st.button("Generate Projection", type="primary"):
        df = fetch_market_data(t_input, "2y")
        if not df.empty:
            log_ret = np.log(df['Close'] / df['Close'].shift(1)).dropna()
            mu = log_ret.mean() * 252
            sigma = log_ret.std() * np.sqrt(252)
            last_price = df['Close'].iloc[-1]

            re, future, psi, regime = fisica.calcular_metricas_institucionales(df, risk_profile)

            if "MOMENTUM" in regime or "ACCUMULATION" in regime:
                mu = max(0.15, mu)

            dt = 1 / 252
            N = 1000
            paths = np.zeros((h_days, N))
            paths[0] = last_price

            for step in range(1, h_days):
                rand = np.random.standard_normal(N)
                paths[step] = paths[step - 1] * np.exp(
                    (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rand
                )

            p95 = np.percentile(paths[-1], 95)
            p50 = np.percentile(paths[-1], 50)
            p05 = np.percentile(paths[-1], 5)

            col_bull, col_base, col_bear = st.columns(3)
            col_bull.metric("🟢 Bull Case (P95)", f"${p95:,.2f}", f"+{((p95/last_price)-1)*100:.0f}%")
            col_base.metric("🔵 Base Case (P50)", f"${p50:,.2f}", f"+{((p50/last_price)-1)*100:.0f}%")
            col_bear.metric("🔴 Bear Case (P5)", f"${p05:,.2f}", f"{((p05/last_price)-1)*100:.0f}%")

            fig = go.Figure()
            for i in range(min(80, N)):
                fig.add_trace(go.Scatter(
                    y=paths[:, i], mode='lines',
                    line=dict(color='rgba(59,130,246,0.08)', width=1),
                    showlegend=False
                ))
            fig.add_trace(go.Scatter(y=np.percentile(paths, 95, axis=1), mode='lines',
                                      name='Bull (P95)', line=dict(color='#16a34a', dash='dash', width=2)))
            fig.add_trace(go.Scatter(y=np.percentile(paths, 50, axis=1), mode='lines',
                                      name='Base (P50)', line=dict(color='#3b82f6', width=2)))
            fig.add_trace(go.Scatter(y=np.percentile(paths, 5, axis=1), mode='lines',
                                      name='Bear (P5)', line=dict(color='#ef4444', dash='dash', width=2)))

            fig.update_layout(
                paper_bgcolor='#0a0e1a', plot_bgcolor='#111827',
                font_color='#e0e6f0', height=440,
                title=f"Monte Carlo — {t_input} | {h_days} Days | {regime}",
                yaxis_title="Price (USD)",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.info(f"Regime: **{regime}** | Governance Ψ: **{psi:.1f}** | Volatility: **{sigma*100:.1f}%** annualized")
        else:
            st.error("Data unavailable.")
