# ==============================================================================
# FAROS v10.0 - DATA LAB EDITION (VALIDATION SUITE)
# Autor: Juan Arroyo | SG Consulting Group
# Feature: Desglose detallado de cálculos (Raw Data) para validación teórica.
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf

st.set_page_config(page_title="FAROS | Data Lab", page_icon="📡", layout="wide")
st.markdown("""<style>.stApp { background-color: #FFFFFF; color: #111; } h1,h2,h3{color:#000!important;} 
.stExpander { border: 1px solid #eee; border-radius: 5px; background-color: #f9f9f9; }</style>""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# CORE LOGIC (Incluye extracción de DATOS CRUDOS para validación)
# ------------------------------------------------------------------------------

@st.cache_data(ttl=300)
def get_live_data(tickers_input, window_cfg, risk_tolerance):
    tickers_list = [x.strip().upper() for x in tickers_input.split(',')]
    data_list = []
    entropy_limit = risk_tolerance 

    for ticker in tickers_list:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=window_cfg['download'])
            if len(hist) > window_cfg['trend']:
                current_price = hist['Close'].iloc[-1]
                returns = hist['Close'].pct_change().dropna()
                
                # --- CÁLCULOS FÍSICOS (CRDOS) ---
                
                # 1. ENTROPÍA (Volatilidad)
                subset = returns.tail(window_cfg['volatility'])
                # Dato Crudo: Volatilidad Anualizada (%)
                raw_volatility = subset.std() * np.sqrt(252) * 100 if len(subset) > 1 else 0
                # Dato TAI: Z-Score
                z_entropy = (raw_volatility - 20) / 15 
                
                # 2. LIQUIDEZ (Volumen)
                # Dato Crudo: Volumen Promedio vs Actual
                vol_avg = hist['Volume'].rolling(window_cfg['volatility']).mean().iloc[-1]
                curr_vol = hist['Volume'].iloc[-1]
                # Dato TAI: Ratio relativo
                raw_vol_ratio = (curr_vol / vol_avg) if vol_avg > 0 else 0
                z_liq = (curr_vol - vol_avg) / vol_avg if vol_avg > 0 else 0
                
                # 3. GOBERNANZA (Tendencia)
                # Dato Crudo: Media Móvil
                sma_val = hist['Close'].rolling(window_cfg['trend']).mean().iloc[-1]
                # Dato TAI: Distancia %
                trend_pct = (current_price - sma_val) / sma_val
                
                # --- ÁRBOL DE DECISIÓN ---
                signal, category, narrative = "MANTENER", "neutral", "Consolidación."
                
                if z_entropy > entropy_limit:
                    if trend_pct > 0.15: 
                        signal = "GROWTH EXTREMO"
                        category = "warning"
                        narrative = f"⚡ Momentum supera volatilidad ({raw_volatility:.0f}% anual)."
                    else:
                        signal = "GAS / RIESGO"
                        category = "danger"
                        narrative = f"⚠️ Fase Gaseosa. Volatilidad {raw_volatility:.0f}% excede límite."
                elif trend_pct > 0.02:
                    if z_liq > 0.10:
                        signal = "COMPRA FUERTE"
                        category = "success"
                        narrative = "🚀 Fase Líquida Óptima."
                    else:
                        signal = "ACUMULAR"
                        category = "info"
                        narrative = "📈 Tendencia sana."
                elif trend_pct < -0.05:
                    signal = "SALIDA"
                    category = "warning"
                    narrative = "📉 Rotura de tendencia."

                data_list.append({
                    "Ticker": ticker, "Price": current_price, "Signal": signal, 
                    "Category": category, "Narrative": narrative,
                    # Datos TAI (Scores)
                    "Entropy": z_entropy, "Liquidity": z_liq, "Trend": trend_pct * 100,
                    # Datos LAB (Raw)
                    "Raw_Vol": raw_volatility, "Raw_Vol_Ratio": raw_vol_ratio, "SMA_Price": sma_val
                })
        except: pass
    return pd.DataFrame(data_list).sort_values('Category', ascending=False) if data_list else pd.DataFrame()

# BACKTESTING
def run_backtest(ticker, start, end, capital, risk_tolerance):
    try:
        df = yf.Ticker(ticker.strip().upper()).history(start=start, end=end)
        if df.empty: return None
        if df.index.tz: df.index = df.index.tz_localize(None)
        
        df['SMA'] = df['Close'].rolling(50).mean()
        df['Trend'] = (df['Close'] - df['SMA']) / df['SMA']
        df['Ret'] = df['Close'].pct_change()
        df['Vol_Ann'] = df['Ret'].rolling(20).std() * np.sqrt(252) * 100
        df['Z_Entropy'] = (df['Vol_Ann'] - 20) / 15
        
        entropy_limit = risk_tolerance
        conds = [
            (df['Z_Entropy'] > entropy_limit) & (df['Trend'] < 0.10), 
            (df['Trend'] > 0) & (df['Z_Entropy'] <= entropy_limit)
        ]
        df['Phase'] = np.select(conds, ['GAS', 'LIQUID'], default='SOLID')
        
        df['Signal'] = 0
        df.loc[(df['Phase'] == 'LIQUID') | ((df['Phase'] == 'SOLID') & (df['Trend'] > 0)), 'Signal'] = 1
        crash_cond = (df['Phase'] == 'GAS')
        trend_break = (df['Trend'] < -0.05)
        df.loc[crash_cond | trend_break, 'Signal'] = 0
        df['Signal'] = df['Signal'].ffill().fillna(0)
        
        df['Strat_Ret'] = df['Close'].pct_change() * df['Signal'].shift(1)
        df.dropna(inplace=True)
        df['Eq_Strat'] = capital * (1 + df['Strat_Ret']).cumprod()
        df['Eq_BH'] = capital * (1 + df['Close'].pct_change()).cumprod()
        return df
    except: return None

# ------------------------------------------------------------------------------
# UI
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("📡 FAROS")
    app_mode = st.radio("MÓDULO:", ["SCANNER PRO", "MÁQUINA DEL TIEMPO"])
    st.markdown("---")
    
    # CALIBRADOR DE RIESGO (VITAL PARA QBTS)
    st.subheader("🎛️ Calibración TAI")
    risk_profile = st.select_slider("Tolerancia al Caos", options=["Conservador", "Growth", "Quantum/Venture"], value="Growth")
    
    if "Conservador" in risk_profile: risk_sigma = 2.0
    elif "Growth" in risk_profile: risk_sigma = 3.0
    else: risk_sigma = 5.0 # Para QBTS
    
    st.caption(f"Límite de Entropía: **{risk_sigma:.1f}σ**")

if app_mode == "SCANNER PRO":
    time_h = st.selectbox("Horizonte", ["Corto", "Medio", "Largo"])
    if "Corto" in time_h: cfg = {'volatility': 10, 'trend': 20, 'download': '3mo'}
    elif "Medio" in time_h: cfg = {'volatility': 20, 'trend': 50, 'download': '6mo'}
    else: cfg = {'volatility': 60, 'trend': 200, 'download': '2y'}
    
    tickers = st.text_area("Cartera:", "QBTS, PLTR, NVDA, SPY", height=100)
    if st.button("Escanear Mercado"): st.cache_data.clear()
    
    df = get_live_data(tickers, cfg, risk_sigma)
    
    if not df.empty:
        c1, c2 = st.columns([2,1])
        with c2:
            st.markdown("#### 🧭 Radar")
            fig = px.scatter(df, x="Entropy", y="Liquidity", color="Category", text="Ticker", color_discrete_map={"success":"#28a745","warning":"#ffc107","danger":"#dc3545"})
            fig.add_vline(x=risk_sigma, line_dash="dash", line_color="red", annotation_text="Límite Riesgo")
            st.plotly_chart(fig, use_container_width=True)
            
        with c1:
            st.markdown("#### 📋 Resultados")
            for i, r in df.iterrows():
                with st.container(border=True):
                    # CABECERA
                    hc1, hc2 = st.columns([3,1])
                    hc1.markdown(f"### **{r['Ticker']}**")
                    hc2.markdown(f"### ${r['Price']:.2f}")
                    
                    # SEÑAL PRINCIPAL
                    msg = f"{r['Signal']}: {r['Narrative']}"
                    if r['Category']=='success': st.success(msg)
                    elif r['Category']=='warning': st.warning(msg)
                    elif r['Category']=='danger': st.error(msg)
                    else: st.info(msg)
                    
                    # --- AQUÍ ESTÁ LA MAGIA DE LA VALIDACIÓN ---
                    with st.expander(f"🔎 Ver Datos de Laboratorio ({r['Ticker']})"):
                        st.markdown(f"""
                        **Validación de Teoría TAI-ACF:**
                        
                        | Parámetro | Dato Crudo (Mercado) | Cálculo TAI (Algoritmo) | Interpretación |
                        | :--- | :--- | :--- | :--- |
                        | **Entropía ($H$)** | `{r['Raw_Vol']:.1f}%` (Anual) | **{r['Entropy']:.2f}σ** | {"🔴 Crítico" if r['Entropy'] > risk_sigma else "🟢 Estable"} bajo tu perfil. |
                        | **Liquidez ($L$)** | `{r['Raw_Vol_Ratio']:.2f}x` (vs Promedio) | **{r['Liquidity']:.2f}** | { "🌊 Flujo Entrando" if r['Liquidity'] > 0 else "🏜️ Secándose" } |
                        | **Gobernanza ($\Psi$)** | Precio vs SMA: `${r['SMA_Price']:.2f}` | **{r['Trend']:+.1f}%** | Distancia a la media móvil. |
                        """)

elif app_mode == "MÁQUINA DEL TIEMPO":
    st.title("Backtest Calibrado")
    tck = st.text_input("Activo:", "QBTS").upper()
    c1, c2 = st.columns(2)
    d_start = c1.date_input("Inicio", pd.to_datetime("2023-01-01"))
    d_end = c2.date_input("Fin", pd.to_datetime("2024-12-31"))
    
    if st.button("Ejecutar Prueba"):
        res = run_backtest(tck, d_start, d_end, 10000, risk_sigma)
        if res is not None:
            strat = res['Eq_Strat'].iloc[-1]
            bh = res['Eq_BH'].iloc[-1]
            st.metric("Resultado FAROS", f"${strat:,.0f}", delta=f"{(strat/10000 - 1)*100:.1f}%")
            st.metric("Buy & Hold", f"${bh:,.0f}", delta=f"{(bh/10000 - 1)*100:.1f}%")
            
            fig = go.Figure()
            gas = res[res['Phase']=='GAS']
            fig.add_trace(go.Scatter(x=gas.index, y=gas['Close'], mode='markers', marker=dict(color='red', size=5, opacity=0.3), name='Veto (Gas)'))
            fig.add_trace(go.Scatter(x=res.index, y=res['Eq_Strat'], name='FAROS', line=dict(color='blue', width=2)))
            fig.add_trace(go.Scatter(x=res.index, y=res['Eq_BH'], name='Hold', line=dict(color='gray', dash='dot')))
            st.plotly_chart(fig, use_container_width=True)
            st.info(f"Simulación ejecutada con tolerancia de **{risk_sigma}σ**.")
