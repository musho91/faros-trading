# ==============================================================================
# FAROS v6.0 - DEEP FLOW EDITION
# Autor: Juan Arroyo | SG Consulting Group
# Mejoras: Flexibilidad Temporal, Macro Fix, Oráculo Extendido
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta
from physics_engine import FarosPhysics 

fisica = FarosPhysics()

st.set_page_config(page_title="FAROS v6.0", page_icon="📡", layout="wide")
st.markdown("""
<style>
    .stApp { background-color: #FFFFFF; color: #111; }
    h1, h2, h3 { color: #0E1117 !important; } 
    .metric-card { background-color: #f8f9fa; border: 1px solid #ddd; padding: 15px; border-radius: 10px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 0. DATOS
# ==============================================================================
ASSET_DB = {
    "NVIDIA (NVDA)": "NVDA", "PALANTIR (PLTR)": "PLTR", "TESLA (TSLA)": "TSLA",
    "BITCOIN (BTC-USD)": "BTC-USD", "ETHEREUM (ETH-USD)": "ETH-USD",
    "APPLE (AAPL)": "AAPL", "MICROSOFT (MSFT)": "MSFT", "AMAZON (AMZN)": "AMZN",
    "GOOGLE (GOOGL)": "GOOGL", "META (META)": "META",
    "S&P 500 (SPY)": "SPY", "NASDAQ 100 (QQQ)": "QQQ", "RUSSELL 2000 (IWM)": "IWM",
    "COINBASE (COIN)": "COIN", "MICROSTRATEGY (MSTR)": "MSTR",
    "D-WAVE (QBTS)": "QBTS", "IONQ (IONQ)": "IONQ", "C3.AI (AI)": "AI"
}

def get_tickers(sel, manual):
    t = [ASSET_DB[k] for k in sel if k in ASSET_DB]
    if manual: t += [x.strip().upper() for x in manual.split(',')]
    return list(set(t))

@st.cache_data(ttl=900)
def download_data(ticker, period="2y", interval="1d"):
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval).copy()
        return df
    except: return pd.DataFrame()

# ==============================================================================
# 1. CORE LOGIC
# ==============================================================================
@st.cache_data(ttl=600)
def get_market_status(risk_p):
    spy = download_data("SPY", "1y") # Miramos 1 año para contexto real
    if spy.empty: return "UNKNOWN", "Offline"
    re, _, psi, est = fisica.calcular_hidrodinamica(spy, risk_p, "Mediano")
    return est, f"{est} (Re: {re:.0f})"

# ==============================================================================
# 2. INTERFAZ
# ==============================================================================
with st.sidebar:
    st.title("📡 FAROS v6.0")
    st.caption("**Deep Flow Engine**")
    
    st.markdown("### 🎚️ Calibración")
    risk_profile = st.select_slider("Perfil de Riesgo", options=["Conservador", "Growth", "Quantum"], value="Growth")
    time_horizon = st.selectbox("Perspectiva Temporal", ["Corto (Trading)", "Mediano (Swing)", "Largo (Inversión)"], index=1)
    
    # Mapeo de horizonte texto a param
    h_map = {"Corto (Trading)":"Corto", "Mediano (Swing)":"Mediano", "Largo (Inversión)":"Largo"}
    h_param = h_map[time_horizon]

    st.markdown("---")
    app_mode = st.radio("MÓDULOS:", ["🤖 ANALISTA IA", "🌎 MACRO FRACTAL", "💼 PORTAFOLIO", "🔍 SCANNER", "⏳ BACKTEST FLEX", "🔮 ORÁCULO"])
    
    st.markdown("---")
    # Status Global
    est, msg = get_market_status(risk_profile)
    c_stat = "green" if "LÍQUIDO" in est or "PLASMA" in est else "red"
    st.markdown(f"**Mercado (SPY):** <span style='color:{c_stat}'>**{msg}**</span>", unsafe_allow_html=True)

# --- ANALISTA ---
if app_mode == "🤖 ANALISTA IA":
    st.title("Analista de Flujo Profundo")
    if "chat" not in st.session_state: st.session_state.chat = []
    
    for m in st.session_state.chat: st.chat_message(m["role"]).markdown(m["content"])
    
    if p := st.chat_input("Analizar activo (ej: MSTR)..."):
        st.session_state.chat.append({"role":"user", "content":p})
        st.chat_message("user").markdown(p)
        
        with st.chat_message("assistant"):
            with st.spinner(f"Analizando física a {time_horizon}..."):
                t = p.upper().split()[0].replace("$","")
                # Descargamos más data para tener contexto
                df = download_data(t, "2y" if h_param=="Largo" else "1y")
                
                if not df.empty:
                    re, _, psi, est = fisica.calcular_hidrodinamica(df, risk_profile, h_param)
                    last = df['Close'].iloc[-1]
                    
                    # Interpretación Flexible
                    if "PLASMA" in est: 
                        rec = "🚀 MANTENER/COMPRAR (High Growth)"
                        why = "Alta turbulencia pero tendencia dominante."
                    elif "LÍQUIDO" in est:
                        rec = "✅ COMPRAR/ACUMULAR"
                        why = "Flujo limpio y estable."
                    elif "SÓLIDO" in est:
                        rec = "🧊 ESPERAR"
                        why = "Sin momentum."
                    else:
                        rec = "⛔ VENDER/CASH"
                        why = "Caos peligroso sin dirección."

                    res = f"""
                    ### 🔬 Diagnóstico: {t}
                    **Precio:** ${last:,.2f} | **Gobernanza:** {psi:.0f}%
                    
                    **Estado:** **{est}** (Re: {re:.0f})
                    > {why}
                    
                    **Veredicto:** {rec}
                    """
                    st.markdown(res)
                    st.session_state.chat.append({"role":"assistant", "content":res})
                else: st.error("Ticker no encontrado.")

# --- MACRO (ARREGLADO) ---
elif app_mode == "🌎 MACRO FRACTAL":
    st.title("Tablero Macroeconómico")
    # Proxies más robustos que no fallan en Yahoo
    macros = {
        "USA (S&P 500)": "SPY", 
        "USA (Dólar Index)": "UUP", # Cambiado de DX-Y a UUP
        "MUNDO (All World)": "VT",
        "EMERGENTES": "EEM",
        "CHINA": "MCHI",
        "EUROPA": "VGK"
    }
    
    sel_m = st.selectbox("Región / Indicador", list(macros.keys()))
    
    if st.button("Escanear Macro"):
        t = macros[sel_m]
        df = download_data(t, "2y")
        if not df.empty:
            re, _, psi, est = fisica.calcular_hidrodinamica(df, risk_profile, h_param)
            
            c1, c2 = st.columns([1,2])
            with c1:
                st.metric("Salud (Ψ)", f"{psi:.0f}/100")
                st.info(f"Estado: {est}")
            with c2:
                fig = px.line(df, y="Close", title=f"Tendencia {sel_m}")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Datos macro no disponibles temporalmente.")

# --- PORTAFOLIO ---
elif app_mode == "💼 PORTAFOLIO":
    st.title("Gestión de Cartera Flexible")
    
    with st.expander("Activos", expanded=True):
        c1, c2 = st.columns(2)
        sel = c1.multiselect("Tickers:", list(ASSET_DB.keys()), default=["NVIDIA (NVDA)", "BITCOIN (BTC-USD)"])
        man = c2.text_input("Manual:", "")
        
        if st.button("⚖️ Calcular Pesos"):
            ts = get_tickers(sel, man)
            res = {}
            for t in ts:
                df = download_data(t, "1y")
                if not df.empty:
                    # Usamos el perfil seleccionado para ser mas valientes
                    _, _, psi, est = fisica.calcular_hidrodinamica(df, risk_profile, h_param)
                    res[t] = psi
            
            tot = sum(res.values())
            if tot > 0:
                final = {k: v/tot for k,v in res.items() if v > 0}
                st.session_state['port'] = final
            else: st.session_state['port'] = "CASH"

    if 'port' in st.session_state:
        p = st.session_state['port']
        if p == "CASH": st.warning("Mercado peligroso. Recomendación: 100% Efectivo.")
        else:
            df_p = pd.DataFrame(list(p.items()), columns=['Activo', 'Peso'])
            df_p['Peso'] = df_p['Peso'].map("{:.1%}".format)
            
            c1, c2 = st.columns(2)
            c1.dataframe(df_p, hide_index=True)
            c2.plotly_chart(px.pie(names=p.keys(), values=p.values(), title="Asignación"), use_container_width=True)

# --- BACKTEST FLEXIBLE ---
elif app_mode == "⏳ BACKTEST FLEX":
    st.title("Laboratorio de Validación")
    
    c1, c2, c3 = st.columns(3)
    tck = c1.text_input("Activo:", "NVDA").upper()
    periodo = c2.selectbox("Historial:", ["1y", "2y", "5y", "10y", "max"], index=2)
    intervalo = c3.selectbox("Velas:", ["1d", "1wk"], index=0)
    
    if st.button("Ejecutar Simulación"):
        df = download_data(tck, periodo, intervalo)
        
        if not df.empty:
            # Lógica Vectorizada (Aprox Physics)
            df['Ret'] = df['Close'].pct_change()
            
            # Medias Móviles según selección
            w_trend = 50 if intervalo == "1d" else 20
            
            # Tendencia de Fondo
            df['Trend'] = (df['Close'] - df['Close'].shift(w_trend)) / df['Close'].shift(w_trend)
            
            # Física Simplificada
            df['Volatilidad'] = df['Ret'].rolling(20).std()
            
            # SEÑAL FLEXIBLE:
            # Si el usuario es "Quantum" o "Growth", permitimos más volatilidad
            umbral_trend = 0.0 if risk_profile != "Conservador" else 0.05
            
            # Lógica: Estar dentro si la tendencia es positiva, ignorando ruido diario
            df['Signal'] = np.where(df['Trend'] > umbral_trend, 1, 0)
            
            # Filtro de Caída Brusca (Solo salimos si cae muy fuerte rápido)
            # Esto evita salir en correcciones pequeñas (-5%) pero sale en crashes (-15%)
            crash_limit = -0.15 if risk_profile == "Quantum" else -0.10
            drawdown_fast = df['Close'].pct_change(10) 
            df['Signal'] = np.where(drawdown_fast < crash_limit, 0, df['Signal'])
            
            df['Signal'] = df['Signal'].shift(1)
            df['Strat'] = df['Ret'] * df['Signal']
            df['Cum_Faros'] = (1 + df['Strat']).cumprod()
            df['Cum_Hold'] = (1 + df['Ret']).cumprod()
            
            ret_f = (df['Cum_Faros'].iloc[-1]-1)*100
            ret_h = (df['Cum_Hold'].iloc[-1]-1)*100
            
            k1, k2 = st.columns(2)
            k1.metric("FAROS (Flexible)", f"{ret_f:,.1f}%", delta=f"{ret_f-ret_h:.1f}%")
            k2.metric("Buy & Hold", f"{ret_h:,.1f}%")
            
            st.line_chart(df[['Cum_Hold', 'Cum_Faros']])
        else: st.error("No hay datos para ese periodo.")

# --- ORÁCULO EXTENDIDO ---
elif app_mode == "🔮 ORÁCULO":
    st.title("Proyección Cuántica")
    
    c1, c2 = st.columns(2)
    t = c1.text_input("Activo:", "BTC-USD").upper()
    # Selector de horizonte más amplio
    horizonte_dias = c2.select_slider("Horizonte de Proyección:", options=[30, 90, 180, 365, 730], value=365)
    
    if st.button("Proyectar Futuro"):
        df = download_data(t, "2y") # Usamos 2 años para mejor estadística
        if not df.empty:
            last = df['Close'].iloc[-1]
            re, _, _, est = fisica.calcular_hidrodinamica(df, risk_profile, h_param)
            
            # Drift (Tendencia Anualizada)
            log_ret = np.log(df['Close']/df['Close'].shift(1))
            mu = log_ret.mean() * 252
            sigma = log_ret.std() * 252**0.5
            
            # Ajuste Físico: Si es PLASMA (Tendencia fuerte), proyectamos continuación
            if "PLASMA" in est or "LÍQUIDO" in est:
                mu = max(0.10, mu) # Asumimos piso de crecimiento positivo
            
            # Simulación
            T = horizonte_dias/365
            steps = horizonte_dias
            sims = 1000
            dt = 1/365
            
            paths = np.zeros((steps, sims))
            paths[0] = last
            
            for i in range(1, steps):
                # Movimiento Browniano Geométrico
                shock = np.random.normal(0, 1, sims)
                paths[i] = paths[i-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*shock)
                
            final = paths[-1]
            p_opt = np.percentile(final, 90)
            p_base = np.percentile(final, 50)
            p_pes = np.percentile(final, 10)
            
            st.subheader(f"Objetivos a {horizonte_dias} días")
            k1, k2, k3 = st.columns(3)
            k1.metric("🚀 Optimista", f"${p_opt:,.2f}", f"{((p_opt/last)-1)*100:.0f}%")
            k2.metric("⚖️ Base", f"${p_base:,.2f}", f"{((p_base/last)-1)*100:.0f}%")
            k3.metric("🛡️ Soporte", f"${p_pes:,.2f}", f"{((p_pes/last)-1)*100:.0f}%")
            
            # Gráfico de Cono
            fig = go.Figure()
            x_axis = [datetime.now() + timedelta(days=i) for i in range(steps)]
            
            fig.add_trace(go.Scatter(x=x_axis, y=np.percentile(paths, 90, axis=1), name='Techo', line=dict(color='green', dash='dash')))
            fig.add_trace(go.Scatter(x=x_axis, y=np.percentile(paths, 50, axis=1), name='Probable', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=x_axis, y=np.percentile(paths, 10, axis=1), name='Suelo', line=dict(color='red', dash='dash')))
            
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Basado en volatilidad histórica ({sigma*100:.1f}%) ajustada por estado actual: {est}")
