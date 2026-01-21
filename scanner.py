# ==============================================================================
# FAROS v4.1 - STABLE CLOUD EDITION
# Autor: Juan Arroyo | SG Consulting Group
# Fixes: Cache nativo, manejo de errores KeyDB, Backtest lógico
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

# --- CONFIGURACIÓN VISUAL ---
st.set_page_config(page_title="FAROS v4.1", page_icon="📡", layout="wide")
st.markdown("""
<style>
    .stApp { background-color: #FFFFFF; color: #111; }
    h1, h2, h3 { color: #0E1117 !important; } 
    .metric-card { background-color: #f8f9fa; border: 1px solid #ddd; padding: 15px; border-radius: 10px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 0. BASE DE DATOS Y UTILIDADES
# ==============================================================================
ASSET_DB = {
    "PALANTIR (PLTR)": "PLTR", "NVIDIA (NVDA)": "NVDA", "D-WAVE (QBTS)": "QBTS", 
    "TESLA (TSLA)": "TSLA", "APPLE (AAPL)": "AAPL", "MICROSOFT (MSFT)": "MSFT", 
    "AMAZON (AMZN)": "AMZN", "GOOGLE (GOOGL)": "GOOGL", "META (META)": "META",
    "BITCOIN (BTC-USD)": "BTC-USD", "ETHEREUM (ETH-USD)": "ETH-USD", 
    "S&P 500 (SPY)": "SPY", "NASDAQ 100 (QQQ)": "QQQ", "RUSSELL 2000 (IWM)": "IWM",
    "COINBASE (COIN)": "COIN", "MICROSTRATEGY (MSTR)": "MSTR",
    "NETFLIX (NFLX)": "NFLX", "DISNEY (DIS)": "DIS", "VISA (V)": "V"
}

def get_tickers_from_selection(selection, manual_input):
    # Recuperación segura de claves
    selected = []
    for k in selection:
        if k in ASSET_DB:
            selected.append(ASSET_DB[k])
    
    if manual_input:
        manual_list = [x.strip().upper() for x in manual_input.split(',')]
        selected.extend(manual_list)
    
    return list(set(selected))

# ==============================================================================
# 1. GESTOR DE DATOS (FIXED: STREAMLIT CACHE)
# ==============================================================================

@st.cache_data(ttl=900) # 15 minutos de caché en RAM
def download_data(ticker, period="1y"):
    """Descarga datos usando la caché nativa de Streamlit (Cloud Friendly)"""
    try:
        # Añadimos .copy() para evitar problemas de fragmentación de pandas
        df = yf.Ticker(ticker).history(period=period).copy()
        if df.empty:
            return pd.DataFrame()
        # Reset index para asegurar que 'Date' no sea problema
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=600)
def get_market_status():
    try:
        spy = download_data("SPY", "6mo")
        if spy.empty: return "UNKNOWN", 0, "Error Data (SPY)", pd.DataFrame()
        
        re, h, psi, estado = fisica.calcular_hidrodinamica(spy)
        
        msg = f"Re: {re:.0f} ({estado})"
        status_code = "LIQUID"
        
        if "TURBULENTO" in estado: status_code = "GAS"
        elif "TRANSICION" in estado: status_code = "WARNING"
        elif "SUPER" in estado: status_code = "SUPER"
        
        return status_code, h, msg, spy
    except: return "UNKNOWN", 0, "Offline", pd.DataFrame()

# ==============================================================================
# 2. LÓGICA DE NEGOCIO
# ==============================================================================

def analyze_portfolio_allocation(selected_assets, manual_input):
    tickers = get_tickers_from_selection(selected_assets, manual_input)
    allocations = {}
    
    for t in tickers:
        hist = download_data(t, "6mo")
        if not hist.empty:
            _, _, psi, estado = fisica.calcular_hidrodinamica(hist)
            # Regla v4.1: Si es Super-Laminar, ignoramos riesgo Reynolds
            if "TURBULENTO" in estado:
                weight = 0
            else:
                weight = psi if psi > 0 else 0
                
            allocations[t] = weight
            
    total_score = sum(allocations.values())
    final_weights = {}
    
    if total_score > 0:
        for t, w in allocations.items():
            if w > 0: # Solo mostramos activos con peso > 0
                final_weights[t] = w / total_score
    else:
        return "⚠️ SISTEMA EN MODO PROTECCIÓN (CASH 100%)", {}
        
    return final_weights, allocations

# ==============================================================================
# 3. INTERFAZ
# ==============================================================================

with st.sidebar:
    st.title("📡 FAROS v4.1")
    st.caption("**Stable Cloud Edition**")
    app_mode = st.radio("SISTEMA:", ["🤖 ANALISTA IA", "🌎 MACRO FRACTAL", "💼 GESTIÓN PORTAFOLIOS", "🔍 SCANNER FÍSICO", "⏳ BACKTEST LAB", "🔮 ORÁCULO"])
    
    st.markdown("---")
    m_stat, _, m_msg, _ = get_market_status()
    
    color_map = {"LIQUID": "green", "SUPER": "#00FF00", "WARNING": "orange", "GAS": "red", "UNKNOWN": "gray"}
    st.markdown(f"**Mercado Global (SPY):**")
    st.markdown(f"<span style='color:{color_map.get(m_stat, 'gray')}; font-weight:bold'>{m_msg}</span>", unsafe_allow_html=True)

# --- ANALISTA IA ---
if app_mode == "🤖 ANALISTA IA":
    st.title("Analista Hidrodinámico")
    
    if "messages" not in st.session_state: 
        st.session_state.messages = [{"role": "assistant", "content": "Sistema listo. Ingresa un ticker (ej: PLTR)."}]
    
    for msg in st.session_state.messages: 
        st.chat_message(msg["role"]).markdown(msg["content"])
    
    if prompt := st.chat_input("Consulta activo..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Conectando con motor físico..."):
                # Extracción simple del ticker
                t = prompt.upper().replace("$","").replace("ANALIZA","").strip().split()[0]
                hist = download_data(t, "1y")
                
                if not hist.empty:
                    re, h, psi, estado = fisica.calcular_hidrodinamica(hist)
                    price = hist['Close'].iloc[-1]
                    
                    color_st = "red" if "TURBULENTO" in estado else "green"
                    
                    res = f"""
                    ### 🔬 Diagnóstico: {t}
                    **Precio:** ${price:.2f} | **Gobernanza:** {psi:.0f}%
                    
                    **Estado Físico:** <span style='color:{color_st}'>**{estado}**</span>
                    * **Reynolds:** {re:.0f}
                    * **Entropía:** {h:.2f} bits
                    
                    **Recomendación:** {"🔴 VENDER/ESPERAR" if psi < 40 else "🟢 COMPRAR/MANTENER"}
                    """
                    st.markdown(res, unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": res})
                else:
                    err_msg = f"No pude descargar datos para '{t}'. Verifica el símbolo."
                    st.error(err_msg)
                    st.session_state.messages.append({"role": "assistant", "content": err_msg})

# --- MACRO ---
elif app_mode == "🌎 MACRO FRACTAL":
    st.title("Sandbox Soberano")
    c_map = {"USA": "SPY", "EUROPA": "VGK", "CHINA": "MCHI", "BRASIL": "EWZ", "MEXICO": "EWW"}
    sel = st.selectbox("Jurisdicción", list(c_map.keys()))
    
    if st.button("Simular Economía"):
        t = c_map[sel]
        df = download_data(t, "1y")
        if not df.empty:
            re, h, psi, estado = fisica.calcular_hidrodinamica(df)
            
            c1, c2 = st.columns([1,2])
            with c1:
                st.metric("Score Macro (Ψ)", f"{psi:.0f}/100")
                st.metric("Reynolds", f"{re:.0f}")
                st.info(f"Estado: {estado}")
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
    st.title("Gobernanza de Capital (Ψ)")
    
    with st.expander("📝 Configuración", expanded=True):
        col1, col2 = st.columns(2)
        sel = col1.multiselect("Activos:", list(ASSET_DB.keys()), default=["PALANTIR (PLTR)", "NVIDIA (NVDA)"])
        manual = col2.text_input("Manual (Separado por comas):", "COIN")
        
        if st.button("⚖️ Auto-Balanceo Físico"):
            weights, raw_scores = analyze_portfolio_allocation(sel, manual)
            st.session_state['alloc'] = weights
            st.session_state['scores'] = raw_scores

    if 'alloc' in st.session_state:
        alloc = st.session_state['alloc']
        if isinstance(alloc, str):
            st.error(alloc)
        else:
            df_w = pd.DataFrame(list(alloc.items()), columns=['Ticker', 'Peso Ideal'])
            df_w['Peso Ideal'] = df_w['Peso Ideal'].apply(lambda x: f"{x*100:.1f}%")
            
            c1, c2 = st.columns([1, 2])
            with c1: st.dataframe(df_w, hide_index=True)
            with c2: 
                fig = px.pie(names=alloc.keys(), values=alloc.values(), title="Asignación Óptima")
                st.plotly_chart(fig, use_container_width=True)

# --- SCANNER ---
elif app_mode == "🔍 SCANNER FÍSICO":
    st.title("Scanner de Navier-Stokes")
    defaults = ["PALANTIR (PLTR)", "NVIDIA (NVDA)", "BITCOIN (BTC-USD)"]
    sel = st.multiselect("Universo:", list(ASSET_DB.keys()), default=defaults)
    
    if st.button("Escanear"):
        tickers = get_tickers_from_selection(sel, "")
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
            
        if data:
            df_res = pd.DataFrame(data).sort_values("Psi", ascending=False)
            st.dataframe(df_res.style.apply(lambda x: ['background-color: #ffcccc' if "TURBULENTO" in v else 'background-color: #c8e6c9' for v in x], subset=['Status']))
            
            fig = px.scatter(df_res, x="Reynolds", y="Psi", color="Status", text="Ticker", title="Mapa de Fases")
            st.plotly_chart(fig)
        else:
            st.warning("No se pudo obtener datos para los activos seleccionados.")

# --- BACKTEST ---
elif app_mode == "⏳ BACKTEST LAB":
    st.title("Validación Histórica (v4.1)")
    tck = st.text_input("Activo:", "NVDA").upper()
    
    if st.button("Simular"):
        df = download_data(tck, "2y")
        if not df.empty:
            # Física Vectorizada para velocidad
            df['Ret'] = df['Close'].pct_change()
            df['Spread'] = (df['High']-df['Low'])/df['Close']
            df['Viscosity'] = df['Spread'].rolling(3).mean().fillna(0.01)
            df['Velocity'] = df['Ret'].abs().rolling(3).mean().fillna(0)
            df['Density'] = df['Volume'] / df['Volume'].rolling(14).mean().fillna(1)
            df['L'] = df['Close'].rolling(14).std() / df['Close']
            
            # Constante calibrada
            K = 150000
            df['Re'] = (df['Density'] * df['Velocity'] * df['L'] / df['Viscosity']) * K
            df['Trend'] = (df['Close'] - df['Close'].shift(14)) / df['Close'].shift(14)
            
            # Lógica Super-Laminar: Comprar si Re < 2500 OR (Re > 2500 y Tendencia > 10%)
            cond_buy = (df['Re'] < 2500)
            cond_super = (df['Re'] > 2500) & (df['Trend'] > 0.10)
            
            df['Signal'] = np.where(cond_buy | cond_super, 1, 0)
            df['Signal'] = df['Signal'].shift(1)
            
            df['Strat'] = df['Ret'] * df['Signal']
            df['Cum_Strat'] = (1 + df['Strat']).cumprod()
            df['Cum_BH'] = (1 + df['Ret']).cumprod()
            
            ret_bh = (df['Cum_BH'].iloc[-1]-1)*100
            ret_st = (df['Cum_Strat'].iloc[-1]-1)*100
            
            c1, c2 = st.columns(2)
            c1.metric("Buy & Hold", f"{ret_bh:.1f}%")
            c2.metric("FAROS", f"{ret_st:.1f}%", delta=f"{ret_st-ret_bh:.1f}%")
            
            st.line_chart(df[['Cum_BH', 'Cum_Strat']])
        else:
            st.error("Datos insuficientes.")

# --- ORACULO ---
elif app_mode == "🔮 ORÁCULO":
    st.title("Proyección Monte Carlo")
    t = st.text_input("Activo:", "BTC-USD").upper()
    days = st.slider("Días:", 30, 365, 90)
    
    if st.button("Proyectar"):
        hist = download_data(t, "1y")
        if not hist.empty:
            re, _, _, estado = fisica.calcular_hidrodinamica(hist)
            last_price = hist['Close'].iloc[-1]
            
            daily_vol = hist['Close'].pct_change().std()
            # Penalización por turbulencia
            if "TURBULENTO" in estado: daily_vol *= 1.5
            
            sims = 1000
            paths = np.zeros((days, sims))
            paths[0] = last_price
            drift = hist['Close'].pct_change().mean()
            
            for d in range(1, days):
                shock = np.random.normal(0, daily_vol, sims)
                paths[d] = paths[d-1] * np.exp(drift + shock)
            
            p90 = np.percentile(paths, 90, axis=1)
            p50 = np.percentile(paths, 50, axis=1)
            p10 = np.percentile(paths, 10, axis=1)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=p90, name="Optimista", line=dict(color='green', dash='dash')))
            fig.add_trace(go.Scatter(y=p50, name="Central", line=dict(color='blue')))
            fig.add_trace(go.Scatter(y=p10, name="Riesgo", line=dict(color='red', dash='dash')))
            
            st.plotly_chart(fig)
            st.info(f"Estado Inicial: {estado} (Re: {re:.0f})")
        else:
            st.error("Sin datos.")
