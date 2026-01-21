# ==============================================================================
# FAROS v5.0 - THE MATTER STATE EDITION
# Autor: Juan Arroyo | SG Consulting Group
# Características: Estados de la Materia, Perfiles de Riesgo, Oráculo Preciso
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
st.set_page_config(page_title="FAROS v5.0", page_icon="📡", layout="wide")
st.markdown("""
<style>
    .stApp { background-color: #FFFFFF; color: #111; }
    h1, h2, h3 { color: #0E1117 !important; } 
    .metric-card { background-color: #f8f9fa; border: 1px solid #ddd; padding: 15px; border-radius: 10px; text-align: center; }
    .info-text { font-size: 0.85rem; color: #555; background-color: #eef; padding: 10px; border-radius: 5px; margin-top: 10px;}
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
    selected = []
    for k in selection:
        if k in ASSET_DB: selected.append(ASSET_DB[k])
    if manual_input:
        manual_list = [x.strip().upper() for x in manual_input.split(',')]
        selected.extend(manual_list)
    return list(set(selected))

# ==============================================================================
# 1. GESTOR DE DATOS
# ==============================================================================
@st.cache_data(ttl=900)
def download_data(ticker, period="1y"):
    try:
        df = yf.Ticker(ticker).history(period=period).copy()
        if df.empty: return pd.DataFrame()
        return df
    except: return pd.DataFrame()

@st.cache_data(ttl=600)
def get_market_status(perfil):
    try:
        spy = download_data("SPY", "6mo")
        if spy.empty: return "UNKNOWN", 0, "Error", pd.DataFrame()
        re, h, psi, estado = fisica.calcular_hidrodinamica(spy, perfil)
        msg = f"{estado} (Re: {re:.0f})"
        color = "green" if "LÍQUIDO" in estado else ("red" if "GASEOSO" in estado else "orange")
        return color, msg, spy
    except: return "gray", "Offline", pd.DataFrame()

# ==============================================================================
# 2. INTERFAZ Y MÓDULOS
# ==============================================================================

with st.sidebar:
    st.title("📡 FAROS v5.0")
    st.caption("**Matter State Edition**")
    
    # --- PERFIL DE RIESGO ---
    st.markdown("### 🎚️ Configuración del Crítico")
    risk_profile = st.select_slider("Perfil de Riesgo", options=["Conservador", "Growth", "Quantum"], value="Growth")
    
    st.info(f"""
    **Modo: {risk_profile}**
    * Ajusta la tolerancia a la turbulencia.
    * 'Conservador' huye rápido del GAS.
    * 'Quantum' tolera PLASMA (Alta volatilidad).
    """)
    
    st.markdown("---")
    app_mode = st.radio("SISTEMA:", ["🤖 ANALISTA IA", "🌎 MACRO FRACTAL", "💼 GESTIÓN PORTAFOLIOS", "🔍 SCANNER FÍSICO", "⏳ BACKTEST LAB", "🔮 ORÁCULO"])
    
    # Estatus Global
    col_spy, msg_spy, _ = get_market_status(risk_profile)
    st.markdown("---")
    st.markdown(f"**Mercado Global (SPY):**")
    st.markdown(f"<div style='color:{col_spy}; font-weight:bold; border:1px solid {col_spy}; padding:5px; text-align:center; border-radius:5px;'>{msg_spy}</div>", unsafe_allow_html=True)

# --- MÓDULO 1: ANALISTA IA ---
if app_mode == "🤖 ANALISTA IA":
    st.title("Analista de Estados de la Materia")
    if "messages" not in st.session_state: st.session_state.messages = []
    
    for msg in st.session_state.messages: st.chat_message(msg["role"]).markdown(msg["content"])
    
    if prompt := st.chat_input("Consulta activo (ej: NVDA)"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Analizando Termodinámica..."):
                t = prompt.upper().replace("$","").split()[0]
                hist = download_data(t, "1y")
                
                if not hist.empty:
                    re, h, psi, estado = fisica.calcular_hidrodinamica(hist, risk_profile)
                    price = hist['Close'].iloc[-1]
                    
                    # Definición de colores
                    c_map = {"LÍQUIDO": "green", "PLASMA": "purple", "SÓLIDO": "gray", "GASEOSO": "red"}
                    c_estado = c_map.get(estado.split()[0], "black")
                    
                    res = f"""
                    ### 🌡️ Diagnóstico: {t}
                    **Precio:** ${price:.2f} | **Gobernanza Ψ:** {psi:.0f}%
                    
                    **Estado de la Materia:** <span style='color:{c_estado}; font-size:1.2em'>**{estado}**</span>
                    * **Reynolds:** {re:.0f} (Fricción Entrópica: {h:.2f})
                    
                    **¿Qué significa?**
                    """
                    if "LÍQUIDO" in estado: res += "El activo fluye libremente. Es zona de compra y acumulación."
                    elif "GASEOSO" in estado: res += "Alta entropía y caos. Las partículas (precios) chocan violentamente. **ALTO RIESGO.**"
                    elif "PLASMA" in estado: res += "Energía extrema. Subida vertical insostenible a largo plazo, pero muy rentable ahora."
                    elif "SÓLIDO" in estado: res += "El precio está congelado o cayendo ordenadamente. No hay energía cinética."
                    
                    st.markdown(res, unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": res})
                else:
                    st.error("Ticker no encontrado.")

# --- MÓDULO 2: PORTAFOLIO ---
elif app_mode == "💼 GESTIÓN PORTAFOLIOS":
    st.title("Gobernanza de Capital (Ψ)")
    st.markdown(f"Optimizado para perfil: **{risk_profile}**")
    
    with st.expander("📝 Configuración de Cartera", expanded=True):
        col1, col2 = st.columns(2)
        sel = col1.multiselect("Activos:", list(ASSET_DB.keys()), default=["PALANTIR (PLTR)", "NVIDIA (NVDA)"])
        manual = col2.text_input("Manual (Separado por comas):", "COIN")
        
        if st.button("⚖️ Calcular Pesos Termodinámicos"):
            tickers = get_tickers_from_selection(sel, manual)
            allocations = {}
            valid_tickers = []
            
            for t in tickers:
                hist = download_data(t, "6mo")
                if not hist.empty:
                    _, _, psi, estado = fisica.calcular_hidrodinamica(hist, risk_profile)
                    # Si es Gaseoso, Psi ya viene castigado por el motor físico
                    allocations[t] = psi
                    valid_tickers.append(t)
            
            total_psi = sum(allocations.values())
            if total_psi > 0:
                final_w = {k: v/total_psi for k, v in allocations.items() if v > 0}
                st.session_state['portfolio'] = (final_w, allocations)
            else:
                st.session_state['portfolio'] = ("CASH", {})

    if 'portfolio' in st.session_state:
        weights, raw_psi = st.session_state['portfolio']
        
        if weights == "CASH":
            st.error("⚠️ ALERTA DE SISTEMA: El mercado está GASEOSO. La recomendación es 100% CASH.")
        else:
            st.subheader("Distribución Sugerida")
            
            c1, c2 = st.columns([1, 1])
            with c1:
                df_show = pd.DataFrame(list(weights.items()), columns=['Activo', 'Peso'])
                df_show['Psi (Fuerza)'] = df_show['Activo'].map(raw_psi).map("{:.0f}%".format)
                df_show['Peso'] = df_show['Peso'].map("{:.1%}".format)
                st.dataframe(df_show, hide_index=True)
            with c2:
                fig = px.pie(names=weights.keys(), values=weights.values(), title="Allocation", hole=0.4)
                st.plotly_chart(fig, use_container_width=True)
                
            st.info("💡 **Nota:** Activos con Psi bajo (<20%) han sido excluidos o reducidos automáticamente para proteger el capital.")

# --- MÓDULO 3: SCANNER ---
elif app_mode == "🔍 SCANNER FÍSICO":
    st.title("Scanner de Estados de la Materia")
    defaults = ["NVIDIA (NVDA)", "TESLA (TSLA)", "BITCOIN (BTC-USD)", "GOOGLE (GOOGL)"]
    sel = st.multiselect("Universo:", list(ASSET_DB.keys()), default=defaults)
    
    if st.button("Escanear"):
        tickers = get_tickers_from_selection(sel, "")
        data = []
        
        for t in tickers:
            df = download_data(t, "6mo")
            if not df.empty:
                re, h, psi, estado = fisica.calcular_hidrodinamica(df, risk_profile)
                data.append({
                    "Ticker": t, "Precio": df['Close'].iloc[-1],
                    "Psi (Ψ)": psi, "Estado": estado, "Reynolds": re, "Entropía": h
                })
        
        if data:
            df_res = pd.DataFrame(data).sort_values("Psi (Ψ)", ascending=False)
            
            def color_states(val):
                if "LÍQUIDO" in val: return 'background-color: #d4edda; color: #155724' # Verde
                if "GASEOSO" in val: return 'background-color: #f8d7da; color: #721c24' # Rojo
                if "PLASMA" in val: return 'background-color: #e2e3e5; color: #383d41'  # Gris/Plasma
                return ''
            
            st.dataframe(df_res.style.applymap(color_states, subset=['Estado']))
            
            # Mapa Visual
            fig = px.scatter(df_res, x="Reynolds", y="Psi (Ψ)", color="Estado", text="Ticker", 
                             size="Psi (Ψ)", title="Mapa de Fases Termodinámico",
                             color_discrete_map={"LÍQUIDO":"green", "GASEOSO":"red", "PLASMA":"purple", "SÓLIDO":"gray"})
            fig.add_vline(x=4500, line_dash="dash", line_color="red", annotation_text="Límite Turbulencia")
            st.plotly_chart(fig)
            
            with st.expander("📚 Guía de Estados"):
                st.markdown("""
                * **🟢 LÍQUIDO:** Flujo laminar. Tendencia sana y predecible. **(Compra)**
                * **🟣 PLASMA:** Flujo super-laminar. Subida explosiva con alta energía. **(Mantener con Stop-Loss)**
                * **⚪ SÓLIDO:** Estancamiento o caída lenta. El dinero está congelado. **(Esperar)**
                * **🔴 GASEOSO:** Turbulencia caótica. El precio se mueve sin dirección clara y con violencia. **(Vender/Cash)**
                """)

# --- MÓDULO 4: ORÁCULO ---
elif app_mode == "🔮 ORÁCULO":
    st.title("Proyección Cuántica (Monte Carlo)")
    st.caption("Proyecta el cono de probabilidad basado en la física actual.")
    
    c1, c2 = st.columns(2)
    t = c1.text_input("Activo:", "BTC-USD").upper()
    days = c2.slider("Días a Futuro:", 30, 365, 90)
    
    if st.button("Consultar Oráculo"):
        hist = download_data(t, "1y")
        if not hist.empty:
            re, _, _, estado = fisica.calcular_hidrodinamica(hist, risk_profile)
            last_price = hist['Close'].iloc[-1]
            
            # Ajuste de Volatilidad por Física
            daily_vol = hist['Close'].pct_change().std()
            if "GASEOSO" in estado: daily_vol *= 1.8 # Mucha incertidumbre
            elif "PLASMA" in estado: daily_vol *= 1.2 # Volatilidad direccional
            
            # Simulación
            paths = np.zeros((days, 1000))
            paths[0] = last_price
            drift = hist['Close'].pct_change().mean()
            
            for d in range(1, days):
                shock = np.random.normal(0, daily_vol, 1000)
                paths[d] = paths[d-1] * np.exp(drift + shock)
            
            # Resultados Finales
            final_prices = paths[-1]
            p90 = np.percentile(final_prices, 90) # Optimista
            p50 = np.percentile(final_prices, 50) # Base
            p10 = np.percentile(final_prices, 10) # Pesimista
            
            # UI DE RESULTADOS
            st.markdown(f"### Proyección a {days} días para {t}")
            st.info(f"Estado Inicial: **{estado}**. La volatilidad se ha ajustado acorde al riesgo físico.")
            
            k1, k2, k3 = st.columns(3)
            k1.metric("🟢 Techo (Optimista)", f"${p90:,.2f}", f"{((p90/last_price)-1)*100:.1f}%")
            k2.metric("🔵 Escenario Base", f"${p50:,.2f}", f"{((p50/last_price)-1)*100:.1f}%")
            k3.metric("🔴 Suelo (Riesgo)", f"${p10:,.2f}", f"{((p10/last_price)-1)*100:.1f}%")
            
            # Gráfico de Cono
            fig = go.Figure()
            # Muestra solo 50 caminos para no saturar
            for i in range(50):
                fig.add_trace(go.Scatter(y=paths[:, i], line=dict(color='gray', width=0.5), opacity=0.1, showlegend=False))
            
            fig.add_trace(go.Scatter(y=np.percentile(paths, 90, axis=1), name='Optimista', line=dict(color='green', dash='dash')))
            fig.add_trace(go.Scatter(y=np.percentile(paths, 50, axis=1), name='Base', line=dict(color='blue', width=2)))
            fig.add_trace(go.Scatter(y=np.percentile(paths, 10, axis=1), name='Piso', line=dict(color='red', dash='dash')))
            
            st.plotly_chart(fig, use_container_width=True)

# --- MÓDULO 5: BACKTEST ---
elif app_mode == "⏳ BACKTEST LAB":
    st.title("Laboratorio de Validación")
    st.caption("Prueba cómo hubiera rendido la estrategia hidrodinámica en el pasado.")
    
    tck = st.text_input("Activo:", "NVDA").upper()
    
    if st.button("Ejecutar Simulación"):
        df = download_data(tck, "2y")
        if not df.empty:
            # Cálculo Vectorizado para Backtest (Simplificado)
            df['Ret'] = df['Close'].pct_change()
            df['Vol_SMA'] = df['Volume'].rolling(14).mean()
            df['Density'] = df['Volume'] / df['Vol_SMA']
            df['Velocity'] = df['Ret'].abs().rolling(3).mean()
            df['Spread'] = (df['High']-df['Low'])/df['Close']
            df['Viscosity'] = df['Spread'].rolling(3).mean().fillna(0.01)
            df['L'] = df['Close'].rolling(14).std() / df['Close']
            
            K = 150000
            df['Re'] = (df['Density'] * df['Velocity'] * df['L'] / df['Viscosity']) * K
            
            # Lógica de Compra/Venta según Perfil Growth
            # Comprar si es Laminar (<3000) o si es Super-Laminar (Re alto pero tendencia fuerte)
            sma_short = df['Close'].rolling(10).mean()
            sma_long = df['Close'].rolling(50).mean()
            uptrend = sma_short > sma_long
            
            # SEÑAL:
            # 1. Comprar si Re < 3000 (Laminar)
            # 2. Mantener si Re > 3000 PERO Uptrend es fuerte (Plasma)
            # 3. Vender si Re > 5000 y Uptrend se rompe (Gas)
            
            df['Signal'] = np.where((df['Re'] < 3000) | ((df['Re'] < 6000) & uptrend), 1, 0)
            df['Signal'] = df['Signal'].shift(1)
            
            df['Strat'] = df['Ret'] * df['Signal']
            df['Cum_Strat'] = (1 + df['Strat']).cumprod()
            df['Cum_BH'] = (1 + df['Ret']).cumprod()
            
            r_strat = (df['Cum_Strat'].iloc[-1]-1)*100
            r_bh = (df['Cum_BH'].iloc[-1]-1)*100
            
            k1, k2 = st.columns(2)
            k1.metric("Retorno FAROS", f"{r_strat:.1f}%", delta=f"{r_strat - r_bh:.1f}% vs Buy&Hold")
            k2.metric("Retorno Mercado", f"{r_bh:.1f}%")
            
            st.line_chart(df[['Cum_BH', 'Cum_Strat']])
            st.info("La línea azul es FAROS. Observa cómo se mantiene plana (Cash) durante las caídas turbulentas.")
