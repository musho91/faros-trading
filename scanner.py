# ==============================================================================
# FAROS v4.0 - THE MASTER SUITE (OPTIMIZED)
# Autor: Juan Arroyo | SG Consulting Group
# Motor Físico: v4.0 (Super-Laminar Logic)
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import requests_cache
from datetime import datetime, timedelta
from physics_engine import FarosPhysics 

# --- CONFIGURACIÓN ANTI-BLOQUEO YAHOO ---
session = requests_cache.CachedSession('yfinance.cache')
session.headers['User-agent'] = 'my-program/1.0'

# Instancia Física
fisica = FarosPhysics()

# --- CONFIGURACIÓN VISUAL ---
st.set_page_config(page_title="FAROS v4.0", page_icon="📡", layout="wide")
st.markdown("""
<style>
    .stApp { background-color: #FFFFFF; color: #111; }
    h1, h2, h3 { color: #0E1117 !important; } 
    .metric-card { background-color: #f8f9fa; border: 1px solid #ddd; padding: 15px; border-radius: 10px; text-align: center; }
    .status-badge { padding: 5px 10px; border-radius: 5px; font-weight: bold; color: white; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 0. BASE DE DATOS
# ==============================================================================
# Claves EXACTAS para evitar errores de Streamlit
ASSET_DB = {
    "PALANTIR (PLTR)": "PLTR", "NVIDIA (NVDA)": "NVDA", "D-WAVE (QBTS)": "QBTS", 
    "TESLA (TSLA)": "TSLA", "APPLE (AAPL)": "AAPL", "MICROSOFT (MSFT)": "MSFT", 
    "AMAZON (AMZN)": "AMZN", "GOOGLE (GOOGL)": "GOOGL", "META (META)": "META",
    "BITCOIN (BTC-USD)": "BTC-USD", "ETHEREUM (ETH-USD)": "ETH-USD", 
    "S&P 500 (SPY)": "SPY", "NASDAQ 100 (QQQ)": "QQQ", "RUSSELL 2000 (IWM)": "IWM",
    "COINBASE (COIN)": "COIN", "MICROSTRATEGY (MSTR)": "MSTR",
    "NETFLIX (NFLX)": "NFLX", "DISNEY (DIS)": "DIS", "VISA (V)": "V"
}

def get_ticker_symbol(option_key):
    return ASSET_DB.get(option_key, option_key)

def get_ecuador_time():
    return (datetime.utcnow() - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S")

# ==============================================================================
# 1. FUNCIONES CORE (CACHED)
# ==============================================================================

@st.cache_data(ttl=600) # Cache de 10 minutos para evitar rate limits
def download_data(ticker, period="1y"):
    try:
        # Usamos session para evitar bloqueos
        dat = yf.Ticker(ticker, session=session).history(period=period)
        return dat
    except: return pd.DataFrame()

@st.cache_data(ttl=300)
def get_market_status():
    try:
        spy = download_data("SPY", "6mo")
        if spy.empty: return "UNKNOWN", 0, "Error Data", pd.DataFrame()
        re, h, psi, estado = fisica.calcular_hidrodinamica(spy)
        
        msg = f"Re: {re:.0f} ({estado})"
        status = "LIQUID"
        if "TURBULENTO" in estado: status = "GAS"
        elif "TRANSICION" in estado: status = "WARNING"
        
        return status, h, msg, spy
    except: return "UNKNOWN", 0, "Offline", pd.DataFrame()

# ==============================================================================
# 2. LOGICA DE NEGOCIO
# ==============================================================================

def analyze_portfolio_allocation(selected_assets, manual_input):
    tickers = [ASSET_DB[k] for k in selected_assets]
    if manual_input:
        tickers += [x.strip().upper() for x in manual_input.split(',')]
    tickers = list(set(tickers))
    
    allocations = {}
    valid_tickers = []
    
    for t in tickers:
        hist = download_data(t, "6mo")
        if not hist.empty:
            _, _, psi, estado = fisica.calcular_hidrodinamica(hist)
            # Regla de Asignación:
            # Si es Laminar o Super-Laminar -> Psi completo
            # Si es Turbulento -> 0 (Cash)
            weight = psi if "TURBULENTO" not in estado else 0
            allocations[t] = weight
            valid_tickers.append(t)
            
    total_score = sum(allocations.values())
    final_weights = {}
    
    if total_score > 0:
        for t in valid_tickers:
            final_weights[t] = allocations[t] / total_score
    else:
        return "⚠️ SISTEMA EN MODO PROTECCIÓN (CASH 100%)", {}
        
    return final_weights, allocations

# ==============================================================================
# 3. INTERFAZ FRONT-END
# ==============================================================================

with st.sidebar:
    st.title("📡 FAROS v4.0")
    st.caption("**Optimized Physics Engine**")
    app_mode = st.radio("SISTEMA:", ["🤖 ANALISTA IA", "🌎 MACRO FRACTAL", "💼 GESTIÓN PORTAFOLIOS", "🔍 SCANNER FÍSICO", "⏳ BACKTEST LAB", "🔮 ORÁCULO"])
    
    st.markdown("---")
    m_stat, _, m_msg, _ = get_market_status()
    color = "green" if m_stat == "LIQUID" else "red"
    st.markdown(f"**Mercado Global (SPY):**")
    st.markdown(f"<span style='color:{color}; font-weight:bold'>{m_msg}</span>", unsafe_allow_html=True)

# --- ANALISTA IA ---
if app_mode == "🤖 ANALISTA IA":
    st.title("Analista Hidrodinámico v4")
    if "messages" not in st.session_state: st.session_state.messages = []
    
    for msg in st.session_state.messages: st.chat_message(msg["role"]).markdown(msg["content"])
    
    if prompt := st.chat_input("Consulta activo (ej: NVDA)"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Procesando física..."):
                t = prompt.upper().split()[0] # Simple extraction
                hist = download_data(t, "1y")
                if not hist.empty:
                    re, h, psi, estado = fisica.calcular_hidrodinamica(hist)
                    price = hist['Close'].iloc[-1]
                    
                    color_st = "green" if psi > 50 else "red"
                    res = f"""
                    ### 🔬 Diagnóstico: {t}
                    **Precio:** ${price:.2f} | **Gobernanza:** {psi:.0f}%
                    
                    **Estado Físico:** <span style='color:{color_st}'>**{estado}**</span>
                    * **Reynolds:** {re:.0f} (Fricción: {h:.2f} bits)
                    
                    **Recomendación:** {"COMPRAR/MANTENER" if psi > 50 else "VENDER/CASH"}
                    """
                    st.markdown(res, unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": res})
                else:
                    st.error("No encontré datos. Intenta con el Ticker exacto.")

# --- MACRO ---
elif app_mode == "🌎 MACRO FRACTAL":
    st.title("Sandbox Soberano")
    c_map = {"USA": "SPY", "EUROPA": "VGK", "CHINA": "MCHI", "BRASIL": "EWZ", "MEXICO": "EWW"}
    sel = st.selectbox("Jurisdicción", list(c_map.keys()))
    
    if st.button("Simular"):
        t = c_map[sel]
        df = download_data(t, "1y")
        if not df.empty:
            re, h, psi, estado = fisica.calcular_hidrodinamica(df)
            
            c1, c2 = st.columns([1,2])
            with c1:
                st.metric("Score Macro (Ψ)", f"{psi:.0f}/100")
                st.metric("Reynolds", f"{re:.0f}")
            with c2:
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number", value = re,
                    title = {'text': f"Turbulencia ({sel})"},
                    gauge = {'axis': {'range': [None, 6000]}, 
                             'steps': [{'range': [0, 2500], 'color': "lightgreen"},
                                       {'range': [2500, 5000], 'color': "yellow"},
                                       {'range': [5000, 6000], 'color': "red"}]}
                ))
                st.plotly_chart(fig, use_container_width=True)

# --- PORTAFOLIO ---
elif app_mode == "💼 GESTIÓN PORTAFOLIOS":
    st.title("Gestión de Activos (Ψ)")
    
    # 1. SELECCIÓN DE ACTIVOS (Restaurada)
    with st.expander("📝 Configuración de Cartera", expanded=True):
        col1, col2 = st.columns(2)
        # Fix: Claves por defecto deben existir en ASSET_DB
        defaults = ["PALANTIR (PLTR)", "NVIDIA (NVDA)", "BITCOIN (BTC-USD)"] 
        sel = col1.multiselect("Activos:", list(ASSET_DB.keys()), default=defaults)
        manual = col2.text_input("Manual (Separado por comas):", "COIN")
        
        if st.button("⚖️ Auto-Balanceo Físico"):
            weights, raw_scores = analyze_portfolio_allocation(sel, manual)
            st.session_state['alloc'] = weights
            st.session_state['scores'] = raw_scores

    # 2. RESULTADOS
    if 'alloc' in st.session_state:
        alloc = st.session_state['alloc']
        raw = st.session_state['scores']
        
        if isinstance(alloc, str):
            st.error(alloc) # Mensaje de Cash
        else:
            st.subheader("Distribución Óptima Sugerida")
            df_w = pd.DataFrame(list(alloc.items()), columns=['Ticker', 'Peso Ideal'])
            df_w['Peso Ideal'] = df_w['Peso Ideal'].apply(lambda x: f"{x*100:.1f}%")
            df_w['Score Físico'] = df_w['Ticker'].map(raw).map("{:.0f}".format)
            
            c1, c2 = st.columns([1, 2])
            with c1: st.dataframe(df_w, hide_index=True)
            with c2: 
                fig = px.pie(names=alloc.keys(), values=alloc.values(), title="Asignación de Capital")
                st.plotly_chart(fig, use_container_width=True)

# --- SCANNER ---
elif app_mode == "🔍 SCANNER FÍSICO":
    st.title("Scanner de Navier-Stokes")
    # Fix: Default values must match keys
    defaults = ["NVIDIA (NVDA)", "TESLA (TSLA)", "BITCOIN (BTC-USD)"]
    sel = st.multiselect("Universo:", list(ASSET_DB.keys()), default=defaults)
    
    if st.button("Escanear"):
        tickers = [ASSET_DB[k] for k in sel]
        data = []
        progress = st.progress(0)
        
        for i, t in enumerate(tickers):
            df = download_data(t, "6mo")
            if not df.empty:
                re, h, psi, estado = fisica.calcular_hidrodinamica(df)
                data.append({
                    "Ticker": t, "Price": df['Close'].iloc[-1],
                    "Reynolds": re, "Psi": psi, "Status": estado,
                    "Entropy": h
                })
            progress.progress((i+1)/len(tickers))
            
        df_res = pd.DataFrame(data).sort_values("Psi", ascending=False)
        st.dataframe(df_res.style.apply(lambda x: ['background-color: #ffcccc' if "TURBULENTO" in v else 'background-color: #c8e6c9' for v in x], subset=['Status']))
        
        # Grafico
        if not df_res.empty:
            fig = px.scatter(df_res, x="Reynolds", y="Psi", color="Status", text="Ticker", title="Mapa de Fases")
            st.plotly_chart(fig)

# --- BACKTEST ---
elif app_mode == "⏳ BACKTEST LAB":
    st.title("Validación Histórica (v4)")
    tck = st.text_input("Activo:", "NVDA").upper()
    
    if st.button("Ejecutar Simulación"):
        df = download_data(tck, "2y")
        if not df.empty:
            # Lógica Vectorizada (Aproximación v4)
            df['Ret'] = df['Close'].pct_change()
            
            # Variables Físicas
            df['Spread'] = (df['High']-df['Low'])/df['Close']
            df['Viscosity'] = df['Spread'].rolling(3).mean()
            df['Velocity'] = df['Ret'].abs().rolling(3).mean()
            df['Density'] = df['Volume'] / df['Volume'].rolling(14).mean()
            df['L'] = df['Close'].rolling(14).std() / df['Close']
            
            K = 150000
            df['Re'] = (df['Density'] * df['Velocity'] * df['L'] / (df['Viscosity'] + 0.0001)) * K
            
            # Tendencia
            df['Trend'] = (df['Close'] - df['Close'].shift(14)) / df['Close'].shift(14)
            
            # LÓGICA DE SEÑAL MEJORADA (SUPER-LAMINAR)
            # 1. Comprar si es Laminar (<2500)
            # 2. MANTENER si es Turbulento PERO la tendencia es fuerte (>10%) -> Super Laminar
            # 3. VENDER si es Turbulento Y la tendencia es débil o negativa
            
            cond_buy = (df['Re'] < 2500)
            cond_hold = (df['Re'] > 2500) & (df['Trend'] > 0.10) # Regla de Oro v4
            
            df['Signal'] = np.where(cond_buy | cond_hold, 1, 0)
            df['Signal'] = df['Signal'].shift(1) # Delay 1 dia
            
            df['Strat'] = df['Ret'] * df['Signal']
            df['Cum_Strat'] = (1 + df['Strat']).cumprod()
            df['Cum_BH'] = (1 + df['Ret']).cumprod()
            
            ret_bh = (df['Cum_BH'].iloc[-1]-1)*100
            ret_st = (df['Cum_Strat'].iloc[-1]-1)*100
            
            c1, c2 = st.columns(2)
            c1.metric("Buy & Hold", f"{ret_bh:.1f}%")
            c2.metric("FAROS v4", f"{ret_st:.1f}%", delta=f"{ret_st-ret_bh:.1f}%")
            
            st.line_chart(df[['Cum_BH', 'Cum_Strat']])
            
            with st.expander("Ver Datos"):
                st.dataframe(df[['Close', 'Re', 'Trend', 'Signal']].tail(20))

# --- ORACULO ---
elif app_mode == "🔮 ORÁCULO":
    st.title("Proyección Monte Carlo")
    t = st.text_input("Activo:", "BTC-USD").upper()
    days = st.slider("Días a proyectar:", 30, 365, 90)
    
    if st.button("Consultar Oráculo"):
        hist = download_data(t, "1y")
        if not hist.empty:
            re, _, _, estado = fisica.calcular_hidrodinamica(hist)
            last_price = hist['Close'].iloc[-1]
            
            # Volatilidad base
            daily_vol = hist['Close'].pct_change().std()
            # Ajuste por Turbulencia Actual
            if "TURBULENTO" in estado: daily_vol *= 1.5
            
            # Simulación
            sims = 1000
            paths = np.zeros((days, sims))
            paths[0] = last_price
            drift = hist['Close'].pct_change().mean()
            
            for d in range(1, days):
                shock = np.random.normal(0, daily_vol, sims)
                paths[d] = paths[d-1] * np.exp(drift + shock)
            
            # Bandas
            p90 = np.percentile(paths, 90, axis=1)
            p50 = np.percentile(paths, 50, axis=1)
            p10 = np.percentile(paths, 10, axis=1)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=p90, name="Techo (Optimista)", line=dict(color='green', dash='dash')))
            fig.add_trace(go.Scatter(y=p50, name="Tendencia Central", line=dict(color='blue')))
            fig.add_trace(go.Scatter(y=p10, name="Suelo (Riesgo)", line=dict(color='red', dash='dash')))
            
            st.plotly_chart(fig)
            st.info(f"Estado Inicial: {estado} (Re: {re:.0f}). Volatilidad ajustada por física.")
