# ==============================================================================
# FAROS v13.0 - MASTER SUITE (TAI-ACF FINAL)
# Autor: Juan Arroyo | SG Consulting Group
# Módulos: Scanner Multi-Frame, Backtest 4-Fases, Oráculo TAI
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf

st.set_page_config(page_title="FAROS | Master Suite", page_icon="📡", layout="wide")
st.markdown("""<style>.stApp { background-color: #FFFFFF; color: #111; } h1,h2,h3{color:#000!important;} 
.stExpander { border: 1px solid #eee; background-color: #f8f9fa; }</style>""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 1. MOTOR LÓGICO CENTRAL (CÁLCULOS TAI-ACF)
# ------------------------------------------------------------------------------

def calculate_psi(entropy, liquidity, trend, risk_sigma):
    """
    FÓRMULA MAESTRA (Gobernanza):
    Calcula un puntaje de 0 a 100 de la calidad del trade basado en la teoría.
    """
    # Base 50 puntos
    score = 50 
    
    # Penalización por Entropía (Riesgo)
    # Si la entropía supera el límite del usuario, penaliza fuerte.
    if entropy > risk_sigma: score -= 30
    else: score += (risk_sigma - entropy) * 10 # Premia baja entropía
    
    # Bonificación por Liquidez (Energía)
    if liquidity > 0: score += liquidity * 20
    elif liquidity < -0.2: score -= 20
    
    # Bonificación por Tendencia (Dirección)
    if trend > 0: score += trend * 100
    else: score -= 50
    
    # Limites 0-100
    return max(0, min(100, score))

@st.cache_data(ttl=300)
def get_live_data(tickers_input, window_cfg, risk_tolerance):
    tickers_list = [x.strip().upper() for x in tickers_input.split(',')]
    data_list = []
    
    entropy_limit = risk_tolerance 
    
    # Ajuste dinámico de salida
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
                
                # --- CÁLCULOS FÍSICOS ---
                subset = returns.tail(window_cfg['volatility'])
                raw_vol = subset.std() * np.sqrt(252) * 100 if len(subset) > 1 else 0
                z_entropy = (raw_vol - 20) / 15 
                
                vol_avg = hist['Volume'].rolling(window_cfg['volatility']).mean().iloc[-1]
                curr_vol = hist['Volume'].iloc[-1]
                z_liq = (curr_vol - vol_avg) / vol_avg if vol_avg > 0 else 0
                raw_vol_ratio = (curr_vol / vol_avg) if vol_avg > 0 else 0
                
                sma_val = hist['Close'].rolling(window_cfg['trend']).mean().iloc[-1]
                trend_pct = (current_price - sma_val) / sma_val
                
                # CÁLCULO DE GOBERNANZA (PSI)
                psi_score = calculate_psi(z_entropy, z_liq, trend_pct, risk_tolerance)
                
                # --- DIAGNÓSTICO DE ESTADO ---
                signal, category, narrative = "MANTENER", "neutral", "Sólido (Estable)."
                
                # 1. RIESGO (GAS)
                if z_entropy > entropy_limit:
                    if trend_pct > 0.15: 
                        signal = "GROWTH EXTREMO"
                        category = "warning"
                        narrative = f"⚡ Momentum (+{trend_pct*100:.1f}%) vence volatilidad."
                    else:
                        signal = "GAS / RIESGO"
                        category = "danger"
                        narrative = f"⚠️ Fase Gaseosa. Entropía excesiva ({z_entropy:.1f}σ)."
                
                # 2. SALIDA
                elif trend_pct < exit_threshold:
                    signal = "SALIDA"
                    category = "danger" if risk_tolerance < 3 else "warning"
                    narrative = f"📉 Rotura Estructural. ({trend_pct*100:.1f}% vs {exit_threshold*100:.0f}%)"
                
                # 3. ENTRADA (LÍQUIDO)
                elif trend_pct > 0.02:
                    if z_liq > 0.10:
                        signal = "COMPRA FUERTE"
                        category = "success"
                        narrative = "🚀 Fase Líquida Óptima."
                    else:
                        signal = "ACUMULAR"
                        category = "info"
                        narrative = "📈 Tendencia Sana."
                
                # 4. TRAMPA (PLASMA)
                elif z_liq < -0.3 and abs(trend_pct) < 0.02:
                     signal = "PLASMA"
                     category = "neutral"
                     narrative = "🟡 Iliquidez (Mercado Seco)."

                data_list.append({
                    "Ticker": ticker, "Price": current_price, "Signal": signal, 
                    "Category": category, "Narrative": narrative,
                    "Entropy": z_entropy, "Liquidity": z_liq, "Trend": trend_pct * 100,
                    "Psi": psi_score, # RESULTADO DE LA FÓRMULA MAESTRA
                    "Raw_Vol": raw_vol, "Raw_Vol_Ratio": raw_vol_ratio, "SMA_Price": sma_val
                })
        except: pass
    return pd.DataFrame(data_list).sort_values('Psi', ascending=False) if data_list else pd.DataFrame()

def run_backtest(ticker, start, end, capital, risk_tolerance):
    try:
        df = yf.Ticker(ticker.strip().upper()).history(start=start, end=end)
        if df.empty: return None
        if df.index.tz: df.index = df.index.tz_localize(None)
        
        # Indicadores
        df['SMA'] = df['Close'].rolling(50).mean()
        df['Trend'] = (df['Close'] - df['SMA']) / df['SMA']
        df['Ret'] = df['Close'].pct_change()
        df['Vol_Ann'] = df['Ret'].rolling(20).std() * np.sqrt(252) * 100
        df['Z_Entropy'] = (df['Vol_Ann'] - 20) / 15
        df['Vol_SMA'] = df['Volume'].rolling(20).mean()
        df['Z_Liq'] = (df['Volume'] - df['Vol_SMA']) / df['Vol_SMA']
        
        entropy_limit = risk_tolerance
        
        # --- DEFINICIÓN DE LOS 4 ESTADOS DE LA MATERIA ---
        conditions = [
            (df['Z_Entropy'] > entropy_limit) & (df['Trend'] < 0.15), # GAS (Rojo)
            (df['Z_Liq'] < -0.3),                                     # PLASMA (Amarillo)
            (df['Trend'] > 0.02) & (df['Z_Entropy'] <= entropy_limit) & (df['Z_Liq'] > 0), # LÍQUIDO (Verde)
        ]
        choices = ['GAS', 'PLASMA', 'LIQUID']
        df['Phase'] = np.select(conditions, choices, default='SOLID') # SOLID (Gris)
        
        # --- ESTRATEGIA ---
        df['Signal'] = 0
        
        # Comprar: LIQUID o SOLID (si tendencia positiva)
        buy_cond = (df['Phase'].isin(['LIQUID', 'SOLID'])) & (df['Trend'] > 0)
        
        # Vender: GAS o Rotura de Tendencia (Dinámica)
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

def run_oracle_sim(ticker, days, risk_tolerance):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        last_price = hist['Close'].iloc[-1]
        returns = hist['Close'].pct_change().dropna()
        daily_vol = returns.std()
        
        # Simulación Monte Carlo simple
        simulations = 200
        paths = np.zeros((days, simulations))
        paths[0] = last_price
        
        # Proyectamos volatilidad futura (Entropía)
        current_vol_ann = daily_vol * np.sqrt(252) * 100
        proj_entropy = (current_vol_ann - 20) / 15
        
        for t in range(1, days):
            shock = np.random.normal(0, daily_vol, simulations)
            paths[t] = paths[t-1] * (1 + shock)
            
        return paths, proj_entropy
    except: return None, 0

# ------------------------------------------------------------------------------
# 2. INTERFAZ DE USUARIO (UI)
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("📡 FAROS SUITE")
    app_mode = st.radio("MÓDULO:", ["🔍 SCANNER (Radar)", "⏳ BACKTEST (Time Machine)", "🔮 ORÁCULO (Escenarios)"])
    st.markdown("---")
    
    st.subheader("🎛️ Calibración Global")
    risk_profile = st.select_slider("Tolerancia al Riesgo", options=["Conservador", "Growth", "Quantum"], value="Growth")
    
    if "Conservador" in risk_profile: risk_sigma = 2.0
    elif "Growth" in risk_profile: risk_sigma = 3.0
    else: risk_sigma = 5.0 
    
    st.caption(f"Límite Entropía: **{risk_sigma}σ**")

# --- MÓDULO 1: SCANNER ---
if app_mode == "🔍 SCANNER (Radar)":
    # Selector de Temporalidad
    time_h = st.selectbox("⏱️ Horizonte Temporal", ["Corto Plazo (Días)", "Medio Plazo (Semanas)", "Largo Plazo (Meses)"])
    
    if "Corto" in time_h: cfg = {'volatility': 10, 'trend': 20, 'download': '3mo', 'desc': 'Trading'}
    elif "Medio" in time_h: cfg = {'volatility': 20, 'trend': 50, 'download': '6mo', 'desc': 'Swing'}
    else: cfg = {'volatility': 60, 'trend': 200, 'download': '2y', 'desc': 'Inversión'}
    
    tickers = st.text_area("Cartera:", "PLTR, QBTS, NVDA, SPY, BTC-USD", height=100)
    if st.button("Analizar Mercado"): st.cache_data.clear()
    
    df = get_live_data(tickers, cfg, risk_sigma)
    
    if not df.empty:
        c1, c2 = st.columns([2,1])
        with c2:
            st.markdown("#### 🧭 Radar Termodinámico")
            fig = px.scatter(df, x="Entropy", y="Liquidity", color="Category", text="Ticker", 
                             color_discrete_map={"success":"#28a745","warning":"#ffc107","danger":"#dc3545","neutral":"#6c757d"},
                             labels={"Entropy": "Caos (H)", "Liquidity": "Flujo (L)"})
            fig.add_vline(x=risk_sigma, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        
        with c1:
            st.markdown("#### 📋 Diagnóstico TAI")
            for i, r in df.iterrows():
                with st.container(border=True):
                    hc1, hc2 = st.columns([3,1])
                    hc1.markdown(f"### **{r['Ticker']}** ${r['Price']:.2f}")
                    
                    # Resultado Fórmula Maestra
                    psi_color = "green" if r['Psi'] > 70 else "orange" if r['Psi'] > 40 else "red"
                    hc2.markdown(f"**Ψ: :{psi_color}[{r['Psi']:.0f}/100]**")
                    
                    if r['Category']=='success': st.success(r['Narrative'])
                    elif r['Category']=='warning': st.warning(r['Narrative'])
                    elif r['Category']=='danger': st.error(r['Narrative'])
                    else: st.info(r['Narrative'])
                    
                    with st.expander(f"🔬 Laboratorio: {r['Ticker']}"):
                        st.markdown(f"""
                        **Datos Crudos vs Teoría:**
                        * **Entropía:** {r['Raw_Vol']:.0f}% Anual -> **{r['Entropy']:.2f}σ** (Límite: {risk_sigma}σ)
                        * **Liquidez:** {r['Raw_Vol_Ratio']:.1f}x Media -> **{r['Liquidity']:.2f}**
                        * **Tendencia:** {r['Trend']:+.1f}% vs Media Móvil
                        """)

# --- MÓDULO 2: BACKTEST ---
elif app_mode == "⏳ BACKTEST (Time Machine)":
    st.title("Validación de Ciclo Completo")
    
    c_tick, c_cap = st.columns([1, 1])
    tck = c_tick.text_input("Activo:", "PLTR").upper()
    cap_input = c_cap.number_input("Capital Inicial ($):", value=10000, step=1000)
    
    c1, c2 = st.columns(2)
    d_start = c1.date_input("Inicio", pd.to_datetime("2023-01-01"))
    d_end = c2.date_input("Fin", pd.to_datetime("2025-01-05"))
    
    if st.button("Ejecutar Simulación"):
        res = run_backtest(tck, d_start, d_end, cap_input, risk_sigma)
        
        if res is not None:
            fin_val = res['Eq_Strat'].iloc[-1]
            st.metric("Resultado Final (Estrategia)", f"${fin_val:,.0f}", delta=f"{(fin_val/cap_input - 1)*100:.1f}%")
            
            # 1. GRÁFICO DE LAS 4 FASES (COLORES)
            st.subheader("Estados de la Materia Detectados")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res.index, y=res['Close'], name='Precio', line=dict(color='black', width=1), opacity=0.3))
            
            # Definición de Colores de Teoría
            phases = {
                'GAS': {'color': 'red', 'name': '🔴 Gas (Caos/Venta)'},
                'LIQUID': {'color': '#00FF41', 'name': '🟢 Líquido (Growth)'},
                'PLASMA': {'color': '#FFD700', 'name': '🟡 Plasma (Trampa)'},
                'SOLID': {'color': 'gray', 'name': '⚪ Sólido (Estable)'}
            }
            
            for ph, attr in phases.items():
                subset = res[res['Phase'] == ph]
                if not subset.empty:
                    fig.add_trace(go.Scatter(x=subset.index, y=subset['Close'], mode='markers', name=attr['name'], marker=dict(color=attr['color'], size=5)))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 2. CURVA DE DINERO
            st.subheader("Evolución del Capital")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=res.index, y=res['Eq_Strat'], name='FAROS', line=dict(color='blue', width=2)))
            fig2.add_trace(go.Scatter(x=res.index, y=res['Eq_BH'], name='Buy & Hold', line=dict(color='gray', dash='dot')))
            st.plotly_chart(fig2, use_container_width=True)

# --- MÓDULO 3: ORÁCULO ---
elif app_mode == "🔮 ORÁCULO (Escenarios)":
    st.title("Proyección de Teoría TAI")
    st.caption("Simulación de escenarios futuros basada en la Entropía actual.")
    
    o_tick = st.text_input("Activo a Proyectar:", "QBTS").upper()
    o_days = st.slider("Días a Futuro:", 30, 365, 90)
    
    if st.button("Consultar Oráculo"):
        paths, proj_h = run_oracle_sim(o_tick, o_days, risk_sigma)
        
        if paths is not None:
            # Interpretación Teórica
            st.subheader("Diagnóstico de Probabilidad")
            
            # ¿Superará la entropía el límite de riesgo?
            if proj_h > risk_sigma:
                st.error(f"⚠️ **ALERTA DE FASE GASEOSA:** La volatilidad proyectada ({proj_h:.1f}σ) supera tu límite ({risk_sigma}σ).")
                st.write("La teoría predice que el activo entrará en una zona de caos incontrolable. Alta probabilidad de crashes.")
            elif proj_h > 1.5:
                st.warning(f"⚡ **ALERTA DE ALTA ENERGÍA:** Entropía proyectada ({proj_h:.1f}σ).")
                st.write("El activo será volátil. Si la tendencia acompaña, será 'Growth'. Si no, será 'Gas'.")
            else:
                st.success(f"✅ **ESTABILIDAD:** Entropía proyectada ({proj_h:.1f}σ).")
                st.write("Se espera comportamiento Sólido o Líquido estable.")

            # Gráfico de Abanico
            fig = go.Figure()
            for i in range(50): # 50 caminos aleatorios
                fig.add_trace(go.Scatter(y=paths[:, i], mode='lines', line=dict(color='gray', width=0.5), opacity=0.1, showlegend=False))
            
            # Media
            median_path = np.median(paths, axis=1)
            fig.add_trace(go.Scatter(y=median_path, mode='lines', name='Escenario Base', line=dict(color='blue', width=2)))
            
            st.plotly_chart(fig, use_container_width=True)
