# ==============================================================================
# FAROS v11.0 - FULL CYCLE EDITION
# Autor: Juan Arroyo | SG Consulting Group
# Mejoras: Salidas Dinámicas, Input de Capital y Visualización de 4 Fases
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf

st.set_page_config(page_title="FAROS | Full Cycle", page_icon="📡", layout="wide")
st.markdown("""<style>.stApp { background-color: #FFFFFF; color: #111; } h1,h2,h3{color:#000!important;} 
.stExpander { border: 1px solid #eee; background-color: #f8f9fa; }</style>""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# CORE LOGIC
# ------------------------------------------------------------------------------

@st.cache_data(ttl=300)
def get_live_data(tickers_input, window_cfg, risk_tolerance):
    tickers_list = [x.strip().upper() for x in tickers_input.split(',')]
    data_list = []
    
    # AJUSTE DE RIESGO
    entropy_limit = risk_tolerance 
    
    # AJUSTE DE SALIDA (STOP LOSS DINÁMICO)
    # Si eres Agresivo (5.0), aguantamos caídas de hasta -15% antes de salir.
    # Si eres Conservador (2.0), salimos al -5%.
    if risk_tolerance >= 5.0: exit_threshold = -0.15
    elif risk_tolerance >= 3.0: exit_threshold = -0.10
    else: exit_threshold = -0.05

    for ticker in tickers_list:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=window_cfg['download'])
            if len(hist) > window_cfg['trend']:
                current_price = hist['Close'].iloc[-1]
                returns = hist['Close'].pct_change().dropna()
                
                # Indicadores
                subset = returns.tail(window_cfg['volatility'])
                raw_vol = subset.std() * np.sqrt(252) * 100 if len(subset) > 1 else 0
                z_entropy = (raw_vol - 20) / 15 
                
                vol_avg = hist['Volume'].rolling(window_cfg['volatility']).mean().iloc[-1]
                curr_vol = hist['Volume'].iloc[-1]
                z_liq = (curr_vol - vol_avg) / vol_avg if vol_avg > 0 else 0
                raw_vol_ratio = (curr_vol / vol_avg) if vol_avg > 0 else 0
                
                sma_val = hist['Close'].rolling(window_cfg['trend']).mean().iloc[-1]
                trend_pct = (current_price - sma_val) / sma_val
                
                # --- ÁRBOL DE DECISIÓN v11 ---
                signal, category, narrative = "MANTENER", "neutral", "Consolidación (Sólido)."
                
                # 1. RIESGO (GAS)
                if z_entropy > entropy_limit:
                    if trend_pct > 0.15: 
                        signal = "GROWTH EXTREMO"
                        category = "warning"
                        narrative = f"⚡ Momentum (+{trend_pct*100:.1f}%) vence a la volatilidad."
                    else:
                        signal = "GAS / RIESGO"
                        category = "danger"
                        narrative = f"⚠️ Fase Gaseosa. Entropía excesiva ({z_entropy:.1f}σ)."
                
                # 2. SALIDA (Dinámica según perfil)
                elif trend_pct < exit_threshold:
                    signal = "SALIDA"
                    category = "danger" if risk_tolerance < 3 else "warning"
                    narrative = f"📉 Rotura de tendencia ({trend_pct*100:.1f}%). Límite: {exit_threshold*100:.0f}%"
                
                # 3. ENTRADA (LÍQUIDO)
                elif trend_pct > 0.02:
                    if z_liq > 0.10:
                        signal = "COMPRA FUERTE"
                        category = "success"
                        narrative = "🚀 Fase Líquida (Trend + Volumen)."
                    else:
                        signal = "ACUMULAR"
                        category = "info"
                        narrative = "📈 Tendencia sana (Sólido-Líquido)."
                
                # 4. TRAMPA (PLASMA)
                elif z_liq < -0.3 and trend_pct > -0.02 and trend_pct < 0.02:
                     signal = "PLASMA / LATERAL"
                     category = "neutral"
                     narrative = "🟡 Mercado seco (Iliquidez)."

                data_list.append({
                    "Ticker": ticker, "Price": current_price, "Signal": signal, 
                    "Category": category, "Narrative": narrative,
                    "Entropy": z_entropy, "Liquidity": z_liq, "Trend": trend_pct * 100,
                    "Raw_Vol": raw_vol, "Raw_Vol_Ratio": raw_vol_ratio, "SMA_Price": sma_val,
                    "Exit_Limit": exit_threshold * 100
                })
        except: pass
    return pd.DataFrame(data_list).sort_values('Category', ascending=False) if data_list else pd.DataFrame()

# BACKTESTING CON FASES COMPLETAS
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
        
        # Volumen relativo para detectar Plasma
        df['Vol_SMA'] = df['Volume'].rolling(20).mean()
        df['Z_Liq'] = (df['Volume'] - df['Vol_SMA']) / df['Vol_SMA']
        
        # --- CLASIFICACIÓN DE 4 FASES ---
        entropy_limit = risk_tolerance
        
        conditions = [
            (df['Z_Entropy'] > entropy_limit) & (df['Trend'] < 0.15), # GAS (Caos, salvo Growth extremo)
            (df['Trend'] > 0.02) & (df['Z_Entropy'] <= entropy_limit) & (df['Z_Liq'] > 0), # LÍQUIDO (Ideal)
            (df['Z_Liq'] < -0.3), # PLASMA (Sin volumen)
        ]
        choices = ['GAS', 'LIQUID', 'PLASMA']
        df['Phase'] = np.select(conditions, choices, default='SOLID') # SOLID es lo normal/estable
        
        # --- ESTRATEGIA ---
        df['Signal'] = 0
        
        # Comprar en Fases Favorables
        # Growth Mode: Compramos en LIQUID y SOLID si la tendencia es positiva
        buy_cond = (df['Phase'].isin(['LIQUID', 'SOLID'])) & (df['Trend'] > 0)
        
        # Vender en Fases Peligrosas
        # Ajuste dinámico de salida también en backtest
        exit_limit = -0.15 if risk_tolerance >= 5 else (-0.10 if risk_tolerance >= 3 else -0.05)
        
        sell_cond = (df['Phase'] == 'GAS') | (df['Trend'] < exit_limit)
        
        df.loc[buy_cond, 'Signal'] = 1
        df.loc[sell_cond, 'Signal'] = 0
        df['Signal'] = df['Signal'].ffill().fillna(0)
        
        # Retornos
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
    
    st.subheader("🎛️ Calibración")
    risk_profile = st.select_slider("Tolerancia", options=["Conservador", "Growth", "Quantum/Venture"], value="Growth")
    
    if "Conservador" in risk_profile: risk_sigma = 2.0
    elif "Growth" in risk_profile: risk_sigma = 3.0
    else: risk_sigma = 5.0 
    
    st.caption(f"Límite Entropía: {risk_sigma}σ")
    # Mostrar el stop loss dinámico calculado para referencia
    stop_display = "-15%" if risk_sigma==5 else ("-10%" if risk_sigma==3 else "-5%")
    st.caption(f"Stop Loss Táctico: {stop_display}")

if app_mode == "SCANNER PRO":
    tickers = st.text_area("Cartera:", "PLTR, QBTS, NVDA, SPY", height=100)
    if st.button("Escanear Mercado"): st.cache_data.clear()
    
    # Config dummy para el ejemplo
    cfg = {'volatility': 20, 'trend': 50, 'download': '1y'}
    df = get_live_data(tickers, cfg, risk_sigma)
    
    if not df.empty:
        c1, c2 = st.columns([2,1])
        with c2:
            st.markdown("#### 🧭 Radar")
            fig = px.scatter(df, x="Entropy", y="Liquidity", color="Category", text="Ticker", color_discrete_map={"success":"#28a745","warning":"#ffc107","danger":"#dc3545","neutral":"#6c757d"})
            fig.add_vline(x=risk_sigma, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
            
        with c1:
            st.markdown("#### 📋 Diagnóstico")
            for i, r in df.iterrows():
                with st.container(border=True):
                    hc1, hc2 = st.columns([3,1])
                    hc1.markdown(f"### **{r['Ticker']}**")
                    hc2.markdown(f"### ${r['Price']:.2f}")
                    
                    msg = f"{r['Signal']}: {r['Narrative']}"
                    if r['Category']=='success': st.success(msg)
                    elif r['Category']=='warning': st.warning(msg)
                    elif r['Category']=='danger': st.error(msg)
                    else: st.info(msg)
                    
                    with st.expander(f"🔎 Laboratorio ({r['Ticker']})"):
                        st.markdown(f"""
                        | Indicador | Valor TAI | Realidad Mercado |
                        | :--- | :--- | :--- |
                        | **Entropía** | `{r['Entropy']:.2f}σ` | Volatilidad {r['Raw_Vol']:.0f}% |
                        | **Tendencia** | `{r['Trend']:+.1f}%` | *Límite Salida: {r['Exit_Limit']}%* |
                        | **Liquidez** | `{r['Liquidity']:.2f}` | Ratio {r['Raw_Vol_Ratio']:.1f}x |
                        """)

elif app_mode == "MÁQUINA DEL TIEMPO":
    st.title("Backtest: Ciclo Termodinámico")
    
    # INPUTS RECUPERADOS
    c_tick, c_cap = st.columns([2, 1])
    tck = c_tick.text_input("Activo:", "PLTR").upper()
    cap_input = c_cap.number_input("Capital Inicial ($):", value=10000, step=1000)
    
    c1, c2 = st.columns(2)
    d_start = c1.date_input("Inicio", pd.to_datetime("2023-01-01"))
    d_end = c2.date_input("Fin", pd.to_datetime("2025-01-05"))
    
    if st.button("Ejecutar Análisis"):
        res = run_backtest(tck, d_start, d_end, cap_input, risk_sigma)
        
        if res is not None:
            # Métricas
            strat = res['Eq_Strat'].iloc[-1]
            bh = res['Eq_BH'].iloc[-1]
            st.metric("Resultado FAROS", f"${strat:,.0f}", delta=f"{(strat/cap_input - 1)*100:.1f}%")
            
            # --- VISUALIZACIÓN DE FASES (REGRESO TRIUNFAL) ---
            st.subheader("Análisis de Fases de Mercado")
            
            fig = go.Figure()
            
            # Dibujar el precio base
            fig.add_trace(go.Scatter(x=res.index, y=res['Close'], name='Precio', line=dict(color='black', width=1), opacity=0.3))
            
            # PINTAR LAS FASES (MARKERS)
            # Gas = Rojo, Líquido = Verde Neón, Plasma = Amarillo, Sólido = Gris
            phases = {'GAS': 'red', 'LIQUID': '#00FF41', 'PLASMA': '#FFD700', 'SOLID': 'gray'}
            
            for phase, color in phases.items():
                subset = res[res['Phase'] == phase]
                if not subset.empty:
                    fig.add_trace(go.Scatter(
                        x=subset.index, y=subset['Close'],
                        mode='markers', name=f'Fase {phase}',
                        marker=dict(color=color, size=5)
                    ))

            fig.update_layout(template="plotly_white", height=450, title=f"Ciclos TAI-ACF: {tck}", yaxis_title="Precio")
            st.plotly_chart(fig, use_container_width=True)
            
            # Curva Capital
            st.subheader("Curva de Capital")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=res.index, y=res['Eq_Strat'], name='FAROS Strategy', line=dict(color='blue', width=2)))
            fig2.add_trace(go.Scatter(x=res.index, y=res['Eq_BH'], name='Buy & Hold', line=dict(color='gray', dash='dot')))
            st.plotly_chart(fig2, use_container_width=True)
