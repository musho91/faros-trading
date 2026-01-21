# ==============================================================================
# FAROS v7.0 - INSTITUTIONAL QUANT SUITE
# Autor: Juan Arroyo | SG Consulting Group
# Core: Navier-Stokes + Future Alpha Integration
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
    .stApp { background-color: #f0f2f6; color: #212529; }
    h1, h2, h3 { color: #0f172a !important; font-family: 'Helvetica', sans-serif; }
    .metric-container { background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); text-align: center; }
    .status-tag { font-weight: bold; padding: 4px 8px; border-radius: 4px; font-size: 0.9em; }
    .stDataFrame { background-color: white; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 0. BASE DE DATOS MAESTRA
# ==============================================================================
ASSET_DB = {
    "NVIDIA Corp (NVDA)": "NVDA", "Palantir Tech (PLTR)": "PLTR", "Tesla Inc (TSLA)": "TSLA",
    "Bitcoin (BTC)": "BTC-USD", "Ethereum (ETH)": "ETH-USD",
    "Apple Inc (AAPL)": "AAPL", "Microsoft (MSFT)": "MSFT", "Amazon (AMZN)": "AMZN",
    "Alphabet (GOOGL)": "GOOGL", "Meta Platforms (META)": "META",
    "S&P 500 ETF (SPY)": "SPY", "Nasdaq 100 (QQQ)": "QQQ", "Russell 2000 (IWM)": "IWM",
    "Coinbase (COIN)": "COIN", "MicroStrategy (MSTR)": "MSTR",
    "D-Wave Quantum (QBTS)": "QBTS", "IonQ Inc (IONQ)": "IONQ", "C3.ai (AI)": "AI"
}

def get_ticker_list(selection, manual_input):
    final_list = []
    # Recuperación segura desde el DB
    for item in selection:
        if item in ASSET_DB:
            final_list.append(ASSET_DB[item])
    
    # Input manual limpio
    if manual_input:
        extras = [x.strip().upper() for x in manual_input.split(',') if x.strip()]
        final_list.extend(extras)
    
    return list(set(final_list))

# ==============================================================================
# 1. DATA FEED (ROBUSTO)
# ==============================================================================
@st.cache_data(ttl=600)
def fetch_market_data(ticker, period="1y"):
    try:
        df = yf.Ticker(ticker).history(period=period)
        return df if not df.empty else pd.DataFrame()
    except: return pd.DataFrame()

@st.cache_data(ttl=300)
def get_global_context(profile):
    spy = fetch_market_data("SPY", "6mo")
    if spy.empty: return "NEUTRAL", 0, "Data Feed Offline", pd.DataFrame()
    
    # Análisis usando el perfil de riesgo seleccionado
    re, future, psi, regime = fisica.calcular_metricas_institucionales(spy, profile)
    
    msg = f"{regime} (Stability Idx: {re:.0f})"
    color = "#198754" if "ACCUMULATION" in regime else ("#dc3545" if "BREAK" in regime else "#fd7e14")
    
    return color, psi, msg, spy

# ==============================================================================
# 2. ALGORITMOS DE ASIGNACIÓN
# ==============================================================================
def strategic_allocation(tickers, risk_profile):
    portfolio = {}
    analysis_log = []
    
    for t in tickers:
        df = fetch_market_data(t, "1y")
        if not df.empty:
            re, future, psi, regime = fisica.calcular_metricas_institucionales(df, risk_profile)
            
            # Lógica Institucional:
            # - psi (Governance Score) ya incluye el Alpha Futuro (40%).
            # - Si psi > 20, asignamos peso. Si es menor, cortamos por riesgo.
            
            weight_score = psi if psi > 20 else 0
            
            portfolio[t] = weight_score
            analysis_log.append({
                "Asset": t, "Price": df['Close'].iloc[-1], 
                "Regime": regime, "Future Alpha": future, 
                "Governance (CAS)": psi
            })
    
    # Normalización de pesos
    total_score = sum(portfolio.values())
    final_weights = {}
    
    if total_score > 0:
        for t, score in portfolio.items():
            if score > 0:
                final_weights[t] = score / total_score
    else:
        return "ALL_CASH", pd.DataFrame(analysis_log)
        
    return final_weights, pd.DataFrame(analysis_log)

# ==============================================================================
# 3. INTERFAZ DE USUARIO (DASHBOARD)
# ==============================================================================
with st.sidebar:
    st.title("🏛️ FAROS Inst.")
    st.caption("**Quantitative Asset Management**")
    
    st.markdown("### Risk Parameters")
    risk_profile = st.select_slider("Investment Profile", options=["Conservador", "Growth", "Quantum"], value="Growth")
    
    st.markdown("---")
    app_mode = st.radio("MODULES:", ["🤖 QUANT ANALYST", "🌎 MACRO FRACTAL", "💼 PORTFOLIO BUILDER", "🔍 ALPHA SCANNER", "⏳ BACKTEST LAB", "🔮 ORACLE PROJECTIONS"])
    
    st.markdown("---")
    # Global Context
    c_glob, psi_glob, msg_glob, _ = get_global_context(risk_profile)
    st.markdown("**Global Context (S&P 500):**")
    st.markdown(f"<div style='background-color:{c_glob}; color:white; padding:8px; border-radius:5px; text-align:center; font-size:0.8em;'>{msg_glob}</div>", unsafe_allow_html=True)
    st.metric("Market Health (CAS)", f"{psi_glob:.0f}/100")

# --- QUANT ANALYST ---
if app_mode == "🤖 QUANT ANALYST":
    st.header("Quantitative Analyst (AI)")
    st.caption("Real-time structural analysis and future projections.")
    
    if "chat_history" not in st.session_state: st.session_state.chat_history = []
    
    for chat in st.session_state.chat_history:
        st.chat_message(chat["role"]).markdown(chat["content"])
        
    if prompt := st.chat_input("Enter ticker (e.g., PLTR, BTC-USD)..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Running Navier-Stokes & Monte Carlo models..."):
                ticker = prompt.upper().replace("$", "").split()[0]
                df = fetch_market_data(ticker, "2y")
                
                if not df.empty:
                    re, future, psi, regime = fisica.calcular_metricas_institucionales(df, risk_profile)
                    last_price = df['Close'].iloc[-1]
                    
                    # Interpretación Profesional
                    if "ACCUMULATION" in regime: 
                        sentiment = "✅ BUY / LONG"
                        color = "green"
                    elif "HIGH MOMENTUM" in regime: 
                        sentiment = "🚀 STRONG BUY (High Vol)"
                        color = "#0d6efd"
                    elif "CONSOLIDATION" in regime: 
                        sentiment = "HOLD / NEUTRAL"
                        color = "gray"
                    else: 
                        sentiment = "⛔ SELL / CASH"
                        color = "red"

                    response = f"""
                    ### 📊 Asset Report: {ticker}
                    **Price:** ${last_price:,.2f} | **CAS (Score):** {psi:.0f}/100
                    
                    **Structural Regime:** <span style='color:{color}'>**{regime}**</span>
                    * **Stability Index:** {re:.0f} (Lower is better for pure stability)
                    * **Future Alpha (1Y):** {future:.0f}/100 (Projected Drift Influence)
                    
                    **Institutional Verdict:** **{sentiment}**
                    """
                    st.markdown(response, unsafe_allow_html=True)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                else:
                    st.error("Ticker not found or data unavailable.")

# --- PORTFOLIO BUILDER ---
elif app_mode == "💼 PORTFOLIO BUILDER":
    st.header("Strategic Portfolio Allocation")
    st.caption("Uses 'Future Alpha' integration to weigh high-growth assets correctly.")
    
    c1, c2 = st.columns([2, 1])
    with c1:
        # Default keys deben coincidir EXACTAMENTE con ASSET_DB
        default_assets = ["NVIDIA Corp (NVDA)", "Palantir Tech (PLTR)"]
        sel_assets = st.multiselect("Select Assets:", list(ASSET_DB.keys()), default=default_assets)
    with c2:
        man_assets = st.text_input("Manual Input (Comma separated):", "BTC-USD")
        
    if st.button("⚖️ Optimize Allocation"):
        target_list = get_ticker_list(sel_assets, man_assets)
        
        if not target_list:
            st.warning("Please select at least one asset.")
        else:
            with st.spinner("Calculating Risk Parity & Future Drift..."):
                weights, df_log = strategic_allocation(target_list, risk_profile)
                
                if weights == "ALL_CASH":
                    st.error("⚠️ RISK ALERT: All selected assets are in STRUCTURAL BREAK regime. Recommended 100% CASH.")
                    st.dataframe(df_log)
                else:
                    st.success("Optimization Complete.")
                    
                    col_res1, col_res2 = st.columns([1, 1])
                    
                    with col_res1:
                        st.subheader("Target Weights")
                        df_w = pd.DataFrame(list(weights.items()), columns=['Ticker', 'Weight'])
                        df_w['Weight'] = df_w['Weight'].map("{:.1%}".format)
                        st.dataframe(df_w, hide_index=True, use_container_width=True)
                        
                    with col_res2:
                        st.subheader("Allocation Chart")
                        fig = px.pie(names=weights.keys(), values=weights.values(), hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with st.expander("🔎 View Quantitative Logic (Why this allocation?)"):
                        st.dataframe(df_log.style.format({"Price": "${:.2f}", "Future Alpha": "{:.1f}", "Governance (CAS)": "{:.1f}"}))

# --- ALPHA SCANNER ---
elif app_mode == "🔍 ALPHA SCANNER":
    st.header("Institutional Alpha Scanner")
    st.caption("Scans the universe for assets with high CAS (Capital Allocation Score).")
    
    defaults = ["NVIDIA Corp (NVDA)", "Palantir Tech (PLTR)", "Tesla Inc (TSLA)", "Bitcoin (BTC)"]
    sel_scan = st.multiselect("Universe:", list(ASSET_DB.keys()), default=defaults)
    
    if st.button("Run Scanner"):
        tickers = get_ticker_list(sel_scan, "")
        results = []
        
        # Barra de progreso
        prog_bar = st.progress(0)
        
        for i, t in enumerate(tickers):
            df = fetch_market_data(t, "1y")
            if not df.empty:
                # Manejo de errores por activo individual
                try:
                    re, future, psi, regime = fisica.calcular_metricas_institucionales(df, risk_profile)
                    results.append({
                        "Ticker": t, "Last Price": df['Close'].iloc[-1],
                        "Regime": regime, "Future Alpha": future,
                        "CAS Score": psi
                    })
                except: pass
            prog_bar.progress((i + 1) / len(tickers))
            
        if results:
            df_res = pd.DataFrame(results).sort_values("CAS Score", ascending=False)
            
            # Formato condicional
            def color_regime(val):
                color = 'white'
                if 'ACCUMULATION' in val: color = '#d1e7dd' # Greenish
                elif 'MOMENTUM' in val: color = '#cff4fc' # Blueish
                elif 'BREAK' in val: color = '#f8d7da' # Reddish
                return f'background-color: {color}'

            st.dataframe(df_res.style.applymap(color_regime, subset=['Regime']).format({"Last Price": "${:.2f}", "Future Alpha": "{:.0f}", "CAS Score": "{:.0f}"}), use_container_width=True)
            
            # Scatter Plot Profesional
            fig = px.scatter(df_res, x="Future Alpha", y="CAS Score", color="Regime", size="CAS Score", hover_name="Ticker",
                             title="Alpha Map: Future Potential vs. Governance Score",
                             color_discrete_map={"INSTITUTIONAL ACCUMULATION": "green", "HIGH MOMENTUM": "blue", "STRUCTURAL BREAK": "red", "CONSOLIDATION": "gray"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data returned for selected assets.")

# --- BACKTEST LAB ---
elif app_mode == "⏳ BACKTEST LAB":
    st.header("Historical Validation Lab")
    
    c1, c2 = st.columns(2)
    tck = c1.text_input("Ticker:", "PLTR").upper()
    years = c2.selectbox("Period:", ["1y", "2y", "5y"], index=1)
    
    if st.button("Run Simulation"):
        df = fetch_market_data(tck, years)
        if not df.empty:
            # Cálculo Vectorizado Simplificado para Backtest Rápido
            df['Ret'] = df['Close'].pct_change()
            df['SMA50'] = df['Close'].rolling(50).mean()
            df['SMA200'] = df['Close'].rolling(200).mean()
            
            # Señal Híbrida (Trend + Volatility Filter)
            # Compramos si SMA50 > SMA200 (Tendencia)
            # Y filtramos si la volatilidad explota (Crash filter)
            
            vol = df['Ret'].rolling(20).std()
            vol_threshold = 0.04 if risk_profile == "Quantum" else 0.025
            
            # Lógica: Estar dentro si hay tendencia alcista Y la volatilidad es aceptable para el perfil
            signal = np.where((df['SMA50'] > df['SMA200']) & (vol < vol_threshold), 1, 0)
            
            # Excepción Quantum: Si hay tendencia muy fuerte, ignorar volatilidad
            if risk_profile == "Quantum":
                strong_trend = (df['Close'] > df['SMA50'] * 1.1)
                signal = np.where(strong_trend, 1, signal)
            
            df['Signal'] = pd.Series(signal, index=df.index).shift(1).fillna(0)
            
            # Equity Curves
            df['Strategy'] = (1 + df['Ret'] * df['Signal']).cumprod()
            df['BuyHold'] = (1 + df['Ret']).cumprod()
            
            perf_strat = (df['Strategy'].iloc[-1] - 1) * 100
            perf_bh = (df['BuyHold'].iloc[-1] - 1) * 100
            
            m1, m2 = st.columns(2)
            m1.metric("FAROS Strategy", f"{perf_strat:,.1f}%", delta=f"{perf_strat - perf_bh:.1f}% vs B&H")
            m2.metric("Buy & Hold", f"{perf_bh:,.1f}%")
            
            st.line_chart(df[['BuyHold', 'Strategy']])
        else:
            st.error("Data unavailable.")

# --- ORACLE ---
elif app_mode == "🔮 ORACLE PROJECTIONS":
    st.header("Future Price Projections (Monte Carlo)")
    st.caption("Projects price cones based on current regime and structural drift.")
    
    c1, c2 = st.columns(2)
    t = c1.text_input("Ticker:", "NVDA").upper()
    h_days = c2.slider("Projection Horizon (Days):", 30, 365, 365)
    
    if st.button("Generate Projection"):
        df = fetch_market_data(t, "2y")
        if not df.empty:
            last_price = df['Close'].iloc[-1]
            
            # Obtener Drift Estructural (Tendencia de largo plazo)
            log_ret = np.log(df['Close'] / df['Close'].shift(1))
            mu = log_ret.mean() * 252
            sigma = log_ret.std() * np.sqrt(252)
            
            # Ajuste Institucional
            # Si el activo está en "HIGH MOMENTUM", proyectamos continuidad del drift positivo
            re, future, psi, regime = fisica.calcular_metricas_institucionales(df, risk_profile)
            
            if "MOMENTUM" in regime or "ACCUMULATION" in regime:
                mu = max(0.15, mu) # Piso de crecimiento para activos fuertes
            
            # Simulación
            T = h_days / 365
            dt = 1 / 365
            N = 1000 # Simulaciones
            
            S0 = last_price
            paths = np.zeros((h_days, N))
            paths[0] = S0
            
            for t_step in range(1, h_days):
                rand = np.random.standard_normal(N)
                paths[t_step] = paths[t_step - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rand)
            
            # Percentiles
            final_dist = paths[-1]
            p95 = np.percentile(final_dist, 95) # Bull Case
            p50 = np.percentile(final_dist, 50) # Base Case
            p05 = np.percentile(final_dist, 5)  # Bear Case
            
            st.subheader(f"Price Targets ({h_days} Days)")
            
            col_bull, col_base, col_bear = st.columns(3)
            col_bull.metric("🟢 Optimistic (Bull)", f"${p95:,.2f}", f"+{((p95/S0)-1)*100:.0f}%")
            col_base.metric("🔵 Base Case", f"${p50:,.2f}", f"+{((p50/S0)-1)*100:.0f}%")
            col_bear.metric("🔴 Structural Support", f"${p05:,.2f}", f"{((p05/S0)-1)*100:.0f}%")
            
            # Gráfico de Cono
            fig = go.Figure()
            # Muestreo aleatorio de caminos
            for i in range(50):
                fig.add_trace(go.Scatter(y=paths[:, i], mode='lines', line=dict(color='gray', width=0.5), opacity=0.1, showlegend=False))
            
            fig.add_trace(go.Scatter(y=np.percentile(paths, 95, axis=1), mode='lines', name='Bull Case', line=dict(color='green', dash='dash')))
            fig.add_trace(go.Scatter(y=np.percentile(paths, 50, axis=1), mode='lines', name='Base Case', line=dict(color='blue', width=2)))
            fig.add_trace(go.Scatter(y=np.percentile(paths, 5, axis=1), mode='lines', name='Bear Case', line=dict(color='red', dash='dash')))
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"Analysis based on **{regime}** regime. Volatility assumption: {sigma*100:.1f}%.")
            
        else:
            st.error("Data unavailable.")
