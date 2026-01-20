# ==============================================================================
# FAROS v3.1 - THE CALIBRATED SUITE
# Autor: Juan Arroyo | SG Consulting Group & Emporium
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta
from physics_engine import FarosPhysics 

# Instancia Global
fisica = FarosPhysics()

# --- CONFIGURACIÓN VISUAL ---
st.set_page_config(page_title="FAROS v3.1 | Calibrated", page_icon="📡", layout="wide")
st.markdown("""
<style>
    .stApp { background-color: #FFFFFF; color: #111; }
    h1, h2, h3 { color: #0E1117 !important; } 
    .stExpander { border: 1px solid #ddd; background-color: #f8f9fa; border-radius: 8px; }
    .global-status { padding: 15px; border-radius: 8px; margin-bottom: 20px; font-weight: bold; text-align: center; border: 1px solid #ddd; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
    .macro-card { padding: 20px; background-color: #f8f9fa; border-radius: 10px; border: 1px solid #ddd; margin-bottom: 20px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 0. BASE DE DATOS & UTILIDADES
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
    selected = [ASSET_DB[k] for k in selection]
    if manual_input:
        manual_list = [x.strip().upper() for x in manual_input.split(',')]
        selected.extend(manual_list)
    return list(set(selected)) 

def get_ecuador_time():
    return (datetime.utcnow() - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S (Quito/EC)")

# ==============================================================================
# 1. CORE v3.1: INTEGRACIÓN FÍSICA
# ==============================================================================

@st.cache_data(ttl=300)
def get_market_status():
    try:
        spy = yf.Ticker("SPY").history(period="6mo")
        if spy.empty: return "UNKNOWN", 0, "Error Data", pd.DataFrame()
        re, h, psi, estado = fisica.calcular_hidrodinamica(spy)
        msg = f"Re: {re:.0f} ({estado})"
        if estado == "TURBULENTO": return "GAS", h, "CRASH ALERT: " + msg, spy
        elif estado == "TRANSICION": return "WARNING", h, "VISCOSO: " + msg, spy
        else: return "LIQUID", h, "NOMINAL: " + msg, spy
    except: return "UNKNOWN", 0, "Desconectado", pd.DataFrame()

def calculate_beta(ticker_hist, market_hist):
    try:
        df = pd.DataFrame({'Asset': ticker_hist['Close'].pct_change(), 'Market': market_hist['Close'].pct_change()}).dropna()
        if df.empty: return 1.0
        cov = df.cov().iloc[0, 1]
        var = df['Market'].var()
        return cov / var if var != 0 else 1.0
    except: return 1.0

# ==============================================================================
# 2. INTELIGENCIA (IA & TAI ENGINE)
# ==============================================================================

def extract_ticker(user_input):
    words = user_input.upper().replace('?', '').replace('.', '').split()
    known_tickers = list(ASSET_DB.values()) + ["PLTR", "NVDA", "BTC", "ETH", "SPY"]
    for w in words:
        if w in known_tickers: return w
        if 2 <= len(w) <= 6 and w.isalpha(): return w 
    return None

def generate_faros_insight(ticker):
    try:
        ticker = ticker.upper()
        hist = yf.Ticker(ticker).history(period="1y")
        if hist.empty: return f"⚠️ No encontré datos para **{ticker}**."
        
        re, h, psi, estado = fisica.calcular_hidrodinamica(hist)
        curr_price = hist['Close'].iloc[-1]
        
        if estado == "TURBULENTO":
            emoji = "⛔"; rec = "CASH / SALIDA"
            desc = f"Reynolds Crítico ({re:.0f}). Riesgo de ruina activado."
        elif estado == "TRANSICION":
            emoji = "⚠️"; rec = "PRECAUCIÓN"
            desc = f"Viscosidad alta (Re {re:.0f}). Mercado inestable."
        else: 
            if psi > 60: emoji = "🚀"; rec = "COMPRA FUERTE"; desc = "Flujo Laminar puro."
            elif psi > 20: emoji = "✅"; rec = "ACUMULAR"; desc = "Tendencia saludable."
            else: emoji = "🧊"; rec = "ESPERAR"; desc = "Sin inercia."

        return f"""
        ### 📡 Hidrodinámica TAI: {ticker}
        **Precio:** ${curr_price:.2f} | **Gobernanza Ψ:** {psi:.0f}%
        **Estado:** {emoji} **{estado}** (Re: {re:.0f})
        > {desc}
        **Veredicto:** {rec}
        """
    except Exception as e: return f"Error: {str(e)}"

def calculate_tai_weights(tickers):
    scores = {}; valid_tickers = []
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(period="6mo")
            if len(hist) > 50:
                _, _, psi, estado = fisica.calcular_hidrodinamica(hist)
                weight_score = 0 if estado == "TURBULENTO" else psi
                scores[t] = weight_score
                valid_tickers.append(t)
        except: pass
    total = sum(scores.values())
    w_str = ""
    if total > 0: 
        for t in valid_tickers: w_str += f"{t}, {scores[t]/total:.2f}\n"
    else: w_str = "SISTEMA EN CASH (PROTECCIÓN)"
    return w_str

# ==============================================================================
# 3. MÓDULOS DE ANÁLISIS
# ==============================================================================

def analyze_country(country_name):
    proxies = {"USA": "SPY", "MEXICO": "EWW", "EUROPA": "VGK", "CHINA": "MCHI", "BRASIL": "EWZ"}
    t = proxies.get(country_name)
    if not t: return None
    try:
        etf = yf.Ticker(t).history(period="1y")
        re, h, psi, estado = fisica.calcular_hidrodinamica(etf)
        return {"Name": country_name, "Ticker": t, "Price": etf['Close'].iloc[-1], "Status": estado, "Reynolds": re, "Score": psi}
    except: return None

def analyze_portfolio(holdings):
    _, _, _, spy_hist = get_market_status()
    results = []; tickers = [t for t in holdings.keys() if t != 'CASH']
    if not tickers: return None, None, None
    
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(period="1y")
            if not hist.empty:
                beta = calculate_beta(hist, spy_hist)
                re, h, psi, estado = fisica.calcular_hidrodinamica(hist)
                action = "VENDER" if estado == "TURBULENTO" else ("AUMENTAR" if psi > 60 else "MANTENER")
                results.append({"Ticker": t, "Weight": holdings[t], "Reynolds": re, "Psi": psi, "Status": estado, "Action": action, "Beta": beta})
        except: pass
    
    if not results: return None, pd.DataFrame(), {"Beta":0, "Psi":0}
    df_res = pd.DataFrame(results)
    port_psi = (df_res['Psi'] * df_res['Weight']).sum()
    return df_res, None, {"Beta": df_res['Beta'].mean(), "Psi": port_psi}

@st.cache_data(ttl=300)
def get_live_data(tickers_list):
    m_status, _, m_msg, _ = get_market_status()
    data_list = []
    for ticker in tickers_list:
        try:
            hist = yf.Ticker(ticker).history(period="6mo")
            if len(hist) > 20:
                re, h, psi, estado = fisica.calcular_hidrodinamica(hist)
                signal = "VENTA" if estado == "TURBULENTO" else ("COMPRA" if psi > 60 else "NEUTRAL")
                narrative = f"Reynolds: {re:.0f}"
                category = "danger" if estado == "TURBULENTO" else ("success" if psi > 60 else "warning")
                data_list.append({"Ticker": ticker, "Price": hist['Close'].iloc[-1], "Signal": signal, "Category": category, "Narrative": narrative, "Entropy": h, "Reynolds": re, "Psi": psi, "Status": estado})
        except: pass
    return pd.DataFrame(data_list).sort_values('Psi', ascending=False) if data_list else pd.DataFrame(), m_status, 0, m_msg

# ==============================================================================
# 4. ORÁCULO DE MONTE CARLO (Regresado de v2.0)
# ==============================================================================
def run_oracle_sim(ticker, days):
    try:
        hist = yf.Ticker(ticker).history(period="1y")
        if len(hist) < 50: return None
        
        # 1. Obtener física actual
        re, _, _, estado = fisica.calcular_hidrodinamica(hist)
        
        # 2. Configurar Monte Carlo
        last_price = hist['Close'].iloc[-1]
        daily_vol = hist['Close'].pct_change().std()
        
        # AJUSTE FÍSICO: Si hay turbulencia, aumentamos la volatilidad proyectada (Cisne Negro)
        if estado == "TURBULENTO": daily_vol *= 2.0 
        elif estado == "TRANSICION": daily_vol *= 1.5
        
        # Drift (Tendencia media)
        drift = hist['Close'].pct_change().mean()
        
        sims = 500
        paths = np.zeros((days, sims))
        paths[0] = last_price
        
        for t in range(1, days):
            shock = np.random.normal(0, daily_vol, sims)
            paths[t] = paths[t-1] * np.exp(drift + shock)
            
        return paths, re, estado
    except: return None

# ==============================================================================
# 5. FRONT-END
# ==============================================================================

with st.sidebar:
    st.title("📡 FAROS v3.1")
    st.caption("**Calibrated Physics Engine**")
    app_mode = st.radio("SISTEMA:", ["🤖 ANALISTA IA", "🌎 MACRO FRACTAL", "💼 GESTIÓN PORTAFOLIOS", "🔍 SCANNER FÍSICO", "⏳ BACKTEST LAB", "🔮 ORÁCULO"])

if app_mode == "🤖 ANALISTA IA":
    st.title("Analista Hidrodinámico")
    if "messages" not in st.session_state: st.session_state.messages = []
    for msg in st.session_state.messages: st.chat_message(msg["role"]).markdown(msg["content"])
    if prompt := st.chat_input("Consulta activo (ej: TSLA)"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)
        t = extract_ticker(prompt)
        res = generate_faros_insight(t) if t else "Ticker no detectado."
        st.chat_message("assistant").markdown(res)
        st.session_state.messages.append({"role": "assistant", "content": res})

elif app_mode == "🌎 MACRO FRACTAL":
    st.title("Sandbox Soberano")
    c = st.selectbox("País", ["USA", "EUROPA", "CHINA", "BRASIL", "MEXICO"])
    if st.button("Simular"):
        d = analyze_country(c)
        if d:
            fig = go.Figure(go.Indicator(mode="gauge+number", value=d['Reynolds'], title={'text': f"Turbulencia ({d['Name']})"}, gauge={'axis': {'range': [None, 6000]}, 'steps': [{'range': [0, 2300], 'color': "lightgreen"}, {'range': [2300, 4000], 'color': "yellow"}, {'range': [4000, 6000], 'color': "red"}]}))
            st.plotly_chart(fig)
            st.metric("Gobernanza (Ψ)", f"{d['Score']:.0f}/100")

elif app_mode == "💼 GESTIÓN PORTAFOLIOS":
    st.title("Gobernanza de Capital")
    assets = st.text_area("Cartera (Ticker, Peso):", "PLTR, 0.4\nNVDA, 0.4\nCASH, 0.2")
    if st.button("Analizar"):
        h = {l.split(',')[0].strip(): float(l.split(',')[1]) for l in assets.split('\n') if ',' in l}
        df, _, m = analyze_portfolio(h)
        if df is not None:
            c1, c2 = st.columns(2)
            c1.metric("Ψ Global", f"{m['Psi']:.0f}%"); c2.metric("Beta", f"{m['Beta']:.2f}")
            st.dataframe(df)

elif app_mode == "🔍 SCANNER FÍSICO":
    st.title("Scanner Calibrado")
    sel = st.multiselect("Activos", list(ASSET_DB.keys()), default=["BTC-USD", "NVDA", "SPY"])
    if st.button("Escanear"):
        t = get_tickers_from_selection(sel, "")
        df, m_s, _, m_msg = get_live_data(t)
        st.info(f"Contexto Global: {m_msg}")
        if not df.empty:
            fig = px.scatter(df, x="Reynolds", y="Psi", color="Status", text="Ticker", color_discrete_map={"TURBULENTO":"red", "LAMINAR":"green", "TRANSICION":"orange"})
            fig.add_vline(x=4000, line_dash="dash", line_color="red")
            st.plotly_chart(fig)
            st.dataframe(df)

elif app_mode == "⏳ BACKTEST LAB":
    st.title("Validación Histórica (Calibrada)")
    tck = st.text_input("Activo:", "BTC-USD").upper()
    
    if st.button("Simular"):
        try:
            df = yf.Ticker(tck).history(period="2y")
            # --- CALCULO MANUAL DE FÍSICA PARA BACKTEST ---
            # Usamos la misma lógica que PhysicsEngine pero vectorizada para velocidad
            df['Ret'] = df['Close'].pct_change()
            df['V'] = df['Ret'].abs().rolling(3).mean() # Velocidad
            df['Rho'] = df['Volume'] / df['Volume'].rolling(14).mean() # Densidad
            df['Mu'] = ((df['High']-df['Low'])/df['Close']).rolling(3).mean() # Viscosidad
            df['L'] = df['Close'].rolling(14).std() / df['Close'] # Longitud Normalizada
            
            # Constante calibrada K
            K = 250000 
            df['Re'] = (df['Rho'] * df['V'] * df['L'] / (df['Mu'] + 0.0001)) * K
            
            # Señales
            df['Signal'] = np.where(df['Re'] < 2300, 1, 0) # Entrar en Laminar
            df['Signal'] = np.where(df['Re'] > 4000, 0, df['Signal']) # Salir en Turbulento
            df['Signal'] = df['Signal'].shift(1) # Evitar Lookahead bias
            
            df['Strat'] = df['Ret'] * df['Signal']
            df['Cum_Strat'] = (1 + df['Strat']).cumprod()
            df['Cum_BH'] = (1 + df['Ret']).cumprod()
            
            st.line_chart(df[['Cum_Strat', 'Cum_BH']])
            st.metric("Retorno FAROS", f"{(df['Cum_Strat'].iloc[-1]-1)*100:.1f}%")
        except Exception as e: st.error(str(e))

elif app_mode == "🔮 ORÁCULO":
    st.title("Proyección Monte Carlo (Física)")
    o_t = st.text_input("Activo", "NVDA").upper()
    d = st.slider("Días", 30, 365, 90)
    
    if st.button("Proyectar"):
        res = run_oracle_sim(o_t, d)
        if res:
            paths, re, est = res
            fig = go.Figure()
            # Muestra 50 caminos aleatorios
            for i in range(min(50, paths.shape[1])): 
                fig.add_trace(go.Scatter(y=paths[:, i], line=dict(color='gray', width=0.5), opacity=0.2, showlegend=False))
            
            # Media y Percentiles
            median = np.median(paths, axis=1)
            p95 = np.percentile(paths, 95, axis=1)
            p05 = np.percentile(paths, 5, axis=1)
            
            fig.add_trace(go.Scatter(y=median, line=dict(color='blue', width=2), name='Tendencia Central'))
            fig.add_trace(go.Scatter(y=p95, line=dict(color='green', dash='dash'), name='Optimista'))
            fig.add_trace(go.Scatter(y=p05, line=dict(color='red', dash='dash'), name='Piso Riesgo'))
            
            st.plotly_chart(fig)
            st.metric("Estado Inicial", est, f"Re: {re:.0f}")
            if est == "TURBULENTO": st.error("⚠️ ALERTA: La proyección incluye volatilidad expandida por turbulencia actual.")
