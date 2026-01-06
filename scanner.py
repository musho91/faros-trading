# ==============================================================================
# FAROS v14.0 - GLOBAL AWARENESS EDITION
# Autor: Juan Arroyo | SG Consulting Group
# Novedad: Semáforo de Riesgo Sistémico (Ambient Temperature)
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf

st.set_page_config(page_title="FAROS | Global Awareness", page_icon="📡", layout="wide")
st.markdown("""<style>.stApp { background-color: #FFFFFF; color: #111; } h1,h2,h3{color:#000!important;} 
.stExpander { border: 1px solid #eee; background-color: #f8f9fa; }
/* Estilo para la barra de estado global */
.global-status { padding: 15px; border-radius: 8px; margin-bottom: 20px; font-weight: bold; text-align: center; border: 1px solid #ddd; }
</style>""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 1. MOTOR LÓGICO
# ------------------------------------------------------------------------------

def calculate_entropy(history, window=20):
    """Función auxiliar para calcular entropía rápido."""
    if len(history) < window: return 0, 0
    returns = history['Close'].pct_change().dropna()
    subset = returns.tail(window)
    raw_vol = subset.std() * np.sqrt(252) * 100 if len(subset) > 1 else 0
    z_entropy = (raw_vol - 20) / 15 
    return raw_vol, z_entropy

@st.cache_data(ttl=300)
def get_market_status():
    """
    ANALIZA EL ENTORNO (SPY) PRIMERO.
    Devuelve: Estado (GAS/LIQUID), Entropía Global, Mensaje.
    """
    try:
        spy = yf.Ticker("SPY").history(period="6mo")
        if spy.empty: return "UNKNOWN", 0, "Error conectando a mercado."
        
        raw_vol, z_entropy = calculate_entropy(spy)
        
        # Umbral de Pánico Global (Si el SPY tiene >3.0 sigma, es un Crash)
        if z_entropy > 3.0:
            return "GAS", z_entropy, "CRISIS DE MERCADO: Alta volatilidad sistémica."
        elif z_entropy > 2.0:
            return "WARNING", z_entropy, "PRECAUCIÓN: El mercado se está agitando."
        else:
            return "LIQUID", z_entropy, "MERCADO ESTABLE: Condiciones favorables."
    except:
        return "UNKNOWN", 0, "Sin datos de mercado."

def calculate_psi(entropy, liquidity, trend, risk_sigma, market_penalty=0):
    """
    FÓRMULA MAESTRA CON PENALIZACIÓN DE MERCADO
    """
    score = 50 
    if entropy > risk_sigma: score -= 30
    else: score += (risk_sigma - entropy) * 10 
    if liquidity > 0: score += liquidity * 20
    elif liquidity < -0.2: score -= 20
    if trend > 0: score += trend * 100
    else: score -= 50
    
    # NUEVO: Si el mercado está mal, bajamos el puntaje de todo
    score -= market_penalty
    
    return max(0, min(100, score))

@st.cache_data(ttl=300)
def get_live_data(tickers_input, window_cfg, risk_tolerance):
    # 1. OBTENER TEMPERATURA GLOBAL
    mkt_status, mkt_entropy, mkt_msg = get_market_status()
    
    # Definir penalización global para la fórmula PSI
    global_penalty = 0
    if mkt_status == "GAS": global_penalty = 30 # Resta 30 puntos a todo si hay crisis
    elif mkt_status == "WARNING": global_penalty = 10

    # 2. ANALIZAR ACTIVOS INDIVIDUALES
    tickers_list = [x.strip().upper() for x in tickers_input.split(',')]
    data_list = []
    entropy_limit = risk_tolerance 
    if risk_tolerance >= 5.0: exit_threshold = -0.15
    elif risk_tolerance >= 3.0: exit_threshold = -0.10
    else: exit_threshold = -0.05

    for ticker in tickers_list:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=window_cfg['download'])
            if len(hist) > window_cfg['trend']:
                current_price = hist['Close'].iloc[-1]
                
                # Cálculos
                raw_vol, z_entropy = calculate_entropy(hist, window_cfg['volatility'])
                
                vol_avg = hist['Volume'].rolling(window_cfg['volatility']).mean().iloc[-1]
                curr_vol = hist['Volume'].iloc[-1]
                z_liq = (curr_vol - vol_avg) / vol_avg if vol_avg > 0 else 0
                raw_vol_ratio = (curr_vol / vol_avg) if vol_avg > 0 else 0
                
                sma_val = hist['Close'].rolling(window_cfg['trend']).mean().iloc[-1]
                trend_pct = (current_price - sma_val) / sma_val
                
                # PSI con Factor Global
                psi_score = calculate_psi(z_entropy, z_liq, trend_pct, risk_tolerance, global_penalty)
                
                # --- LÓGICA DE DIAGNÓSTICO ---
                signal, category, narrative = "MANTENER", "neutral", "Sólido."
                
                # Filtro de Crisis Global
                systemic_warning = ""
                if mkt_status == "GAS":
                    systemic_warning = " ⚠️ [RIESGO SISTÉMICO]"
                    # Si el mercado es GAS, forzamos precaución incluso si el activo es bueno
                    category = "warning" 
                
                if z_entropy > entropy_limit:
                    if trend_pct > 0.15: 
                        signal = "GROWTH EXTREMO"
                        category = "warning"
                        narrative = f"⚡ Momentum vence volatilidad.{systemic_warning}"
                    else:
                        signal = "GAS / RIESGO"
                        category = "danger"
                        narrative = f"⚠️ Fase Gaseosa Local.{systemic_warning}"
                elif trend_pct < exit_threshold:
                    signal = "SALIDA"
                    category = "danger" if risk_tolerance < 3 else "warning"
                    narrative = f"📉 Rotura Estructural.{systemic_warning}"
                elif trend_pct > 0.02:
                    if z_liq > 0.10:
                        signal = "COMPRA FUERTE"
                        # Si hay crisis global, degradamos "Success" a "Warning"
                        category = "success" if mkt_status != "GAS" else "warning"
                        narrative = f"🚀 Fase Líquida.{systemic_warning}"
                    else:
                        signal = "ACUMULAR"
                        category = "info" if mkt_status != "GAS" else "neutral"
                        narrative = f"📈 Tendencia Sana.{systemic_warning}"
                elif z_liq < -0.3 and abs(trend_pct) < 0.02:
                     signal = "PLASMA"
                     category = "neutral"
                     narrative = "🟡 Iliquidez."

                data_list.append({
                    "Ticker": ticker, "Price": current_price, "Signal": signal, 
                    "Category": category, "Narrative": narrative,
                    "Entropy": z_entropy, "Liquidity": z_liq, "Trend": trend_pct * 100,
                    "Psi": psi_score, "Raw_Vol": raw_vol, "Raw_Vol_Ratio": raw_vol_ratio
                })
        except: pass
    
    return pd.DataFrame(data_list).sort_values('Psi', ascending=False) if data_list else pd.DataFrame(), mkt_status, mkt_entropy, mkt_msg

# (Backtest y Oráculo se mantienen igual, solo los incluimos para que el archivo esté completo)
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
        df['Vol_SMA'] = df['Volume'].rolling(20).mean()
        df['Z_Liq'] = (df['Volume'] - df['Vol_SMA']) / df['Vol_SMA']
        entropy_limit = risk_tolerance
        conditions = [(df['Z_Entropy'] > entropy_limit) & (df['Trend'] < 0.15), (df['Z_Liq'] < -0.3), (df['Trend'] > 0.02) & (df['Z_Entropy'] <= entropy_limit) & (df['Z_Liq'] > 0)]
        choices = ['GAS', 'PLASMA', 'LIQUID']
        df['Phase'] = np.select(conditions, choices, default='SOLID')
        df['Signal'] = 0
        buy_cond = (df['Phase'].isin(['LIQUID', 'SOLID'])) & (df['Trend'] > 0)
        exit_limit = -0.15 if risk_tolerance >= 5 else (-0.10 if risk_tolerance >= 3 else -0.05)
        sell_cond = (df['Phase'] == 'GAS') | (df['Trend'] < exit_limit)
        df.loc[buy_cond, 'Signal'] = 1; df.loc[sell_cond, 'Signal'] = 0
        df['Signal'] = df['Signal'].ffill().fillna(0)
        df['Strat_Ret'] = df['Close'].pct_change() * df['Signal'].shift(1)
        df.dropna(inplace=True)
        df['Eq_Strat'] = capital * (1 + df['Strat_Ret']).cumprod()
        df['Eq_BH'] = capital * (1 + df['Close'].pct_change()).cumprod()
        return df
    except: return None

def run_oracle_sim(ticker, days, risk_tolerance):
    try:
        stock = yf.Ticker(ticker); hist = stock.history(period="1y")
        last = hist['Close'].iloc[-1]; ret = hist['Close'].pct_change().dropna()
        vol = ret.std()
        sims = 200; paths = np.zeros((days, sims)); paths[0] = last
        proj_h = (vol * np.sqrt(252) * 100 - 20) / 15
        for t in range(1, days): paths[t] = paths[t-1] * (1 + np.random.normal(0, vol, sims))
        return paths, proj_h
    except: return None, 0

# ------------------------------------------------------------------------------
# 2. UI - AHORA CON BARRA DE ESTADO GLOBAL
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("📡 FAROS SUITE")
    app_mode = st.radio("MÓDULO:", ["SCANNER", "BACKTEST", "ORÁCULO"])
    st.markdown("---")
    st.subheader("🎛️ Calibración")
    risk_profile = st.select_slider("Tolerancia", options=["Conservador", "Growth", "Quantum"], value="Growth")
    if "Conservador" in risk_profile: risk_sigma = 2.0
    elif "Growth" in risk_profile: risk_sigma = 3.0
    else: risk_sigma = 5.0 

if app_mode == "SCANNER":
    time_h = st.selectbox("Horizonte", ["Corto Plazo", "Medio Plazo", "Largo Plazo"])
    if "Corto" in time_h: cfg = {'volatility': 10, 'trend': 20, 'download': '3mo', 'desc': 'Trading'}
    elif "Medio" in time_h: cfg = {'volatility': 20, 'trend': 50, 'download': '6mo', 'desc': 'Swing'}
    else: cfg = {'volatility': 60, 'trend': 200, 'download': '2y', 'desc': 'Inversión'}
    
    tickers = st.text_area("Cartera:", "PLTR, QBTS, NVDA, SPY, BTC-USD", height=100)
    if st.button("Analizar Mercado"): st.cache_data.clear()
    
    # LLAMADA PRINCIPAL
    df, m_status, m_entropy, m_msg = get_live_data(tickers, cfg, risk_sigma)
    
    # --- BARRA DE ESTADO GLOBAL (NUEVO) ---
    if m_status == "GAS":
        bg_color, txt_color = "#FFCDD2", "#B71C1C" # Rojo claro / Oscuro
        icon = "🔥"
    elif m_status == "WARNING":
        bg_color, txt_color = "#FFF9C4", "#F57F17" # Amarillo
        icon = "⚠️"
    else:
        bg_color, txt_color = "#C8E6C9", "#1B5E20" # Verde
        icon = "🌍"
        
    st.markdown(f"""
    <div class="global-status" style="background-color: {bg_color}; color: {txt_color};">
        {icon} TERMODINÁMICA GLOBAL (SPY): {m_msg} [Entropía: {m_entropy:.1f}σ]
    </div>
    """, unsafe_allow_html=True)
    # ---------------------------------------
    
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
                    hc1.markdown(f"### **{r['Ticker']}** ${r['Price']:.2f}")
                    psi_c = "green" if r['Psi']>70 else "orange" if r['Psi']>40 else "red"
                    hc2.markdown(f"**Ψ: :{psi_c}[{r['Psi']:.0f}]**")
                    
                    if r['Category']=='success': st.success(r['Narrative'])
                    elif r['Category']=='warning': st.warning(r['Narrative'])
                    elif r['Category']=='danger': st.error(r['Narrative'])
                    else: st.info(r['Narrative'])
                    
                    with st.expander("🔬 Lab Data"):
                        st.markdown(f"**Entropía:** {r['Raw_Vol']:.0f}% ({r['Entropy']:.2f}σ) | **Liq:** {r['Raw_Vol_Ratio']:.1f}x")

elif app_mode == "BACKTEST":
    # (Misma UI de Backtest v13)
    c_tick, c_cap = st.columns([1, 1])
    tck = c_tick.text_input("Activo:", "PLTR").upper()
    cap = c_cap.number_input("Capital ($):", value=10000)
    c1, c2 = st.columns(2)
    d1 = c1.date_input("Inicio", pd.to_datetime("2023-01-01")); d2 = c2.date_input("Fin", pd.to_datetime("2025-01-05"))
    if st.button("Simular"):
        res = run_backtest(tck, d1, d2, cap, risk_sigma)
        if res is not None:
            fin = res['Eq_Strat'].iloc[-1]
            st.metric("Resultado", f"${fin:,.0f}", delta=f"{(fin/cap-1)*100:.1f}%")
            st.subheader("Estados")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res.index, y=res['Close'], name='Precio', line=dict(color='black', width=1), opacity=0.3))
            phases = {'GAS': 'red', 'LIQUID': '#00FF41', 'PLASMA': '#FFD700', 'SOLID': 'gray'}
            for p, c in phases.items():
                s = res[res['Phase']==p]
                if not s.empty: fig.add_trace(go.Scatter(x=s.index, y=s['Close'], mode='markers', name=p, marker=dict(color=c, size=5)))
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Capital"); fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=res.index, y=res['Eq_Strat'], name='FAROS', line=dict(color='blue')))
            fig2.add_trace(go.Scatter(x=res.index, y=res['Eq_BH'], name='Hold', line=dict(color='gray', dash='dot')))
            st.plotly_chart(fig2, use_container_width=True)

# --- MÓDULO 3: ORÁCULO (CON FUTURE SCORE) ---
elif app_mode == "ORÁCULO":
    st.title("Proyección de Teoría TAI")
    st.caption("Simulación de escenarios futuros y Cálculo de Potencial (Phi).")
    
    # Inputs
    c_tick, c_days = st.columns([1, 1])
    o_tick = c_tick.text_input("Activo a Proyectar:", "PLTR").upper()
    o_days = c_days.slider("Horizonte de Proyección (Días):", 30, 365, 90)
    
    if st.button("Consultar Oráculo"):
        paths, proj_entropy = run_oracle_sim(o_tick, o_days, risk_sigma)
        
        if paths is not None:
            # 1. CÁLCULO DE ESCENARIOS (Matemática Pura)
            start_price = paths[0][0]
            final_prices = paths[-1]
            
            p95 = np.percentile(final_prices, 95) # Techo Optimista
            p50 = np.percentile(final_prices, 50) # Mediana
            p05 = np.percentile(final_prices, 5)  # Suelo Pesimista
            
            # 2. CÁLCULO DEL "FUTURE SCORE" (Phi - Φ)
            # Factor A: Probabilidad de Ganancia (% de caminos que terminan en verde)
            win_rate = np.mean(final_prices > start_price) # Ej: 0.65 (65%)
            
            # Factor B: Asimetría (Reward vs Risk)
            # ¿Cuánto puedo ganar vs cuánto puedo perder?
            upside = (p95 - start_price) / start_price
            downside = abs((p05 - start_price) / start_price)
            risk_reward_ratio = upside / downside if downside > 0 else 0
            
            # Factor C: Penalización por Entropía excesiva
            entropy_penalty = 0
            if proj_entropy > risk_sigma: entropy_penalty = 20
            
            # FÓRMULA DE POTENCIAL (0 a 100)
            # Base: Win Rate * 50 puntos
            # + Asimetría * 20 puntos (tope 40)
            # - Penalización
            phi_score = (win_rate * 60) + (min(risk_reward_ratio, 3) * 10) - entropy_penalty
            phi_score = max(0, min(100, phi_score)) # Normalizar 0-100
            
            # Color del Score
            phi_color = "green" if phi_score > 70 else "orange" if phi_score > 40 else "red"
            
            # --- INTERFAZ DE RESULTADOS ---
            
            # HEADER: El Score Phi
            st.markdown(f"""
            <div style="text-align:center; margin-bottom:20px; padding:10px; border:1px solid #ddd; border-radius:10px; background-color:#f9f9f9;">
                <h2 style="margin:0; color:#333;">POTENCIAL FUTURO (Φ)</h2>
                <h1 style="margin:0; font-size:3.5rem; color:{phi_color};">{phi_score:.0f}/100</h1>
                <p style="color:#666;">Probabilidad de Éxito: <b>{win_rate*100:.0f}%</b> | Ratio Riesgo/Beneficio: <b>{risk_reward_ratio:.1f}x</b></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Tarjetas de Precios
            k1, k2, k3 = st.columns(3)
            k1.metric("🟢 Techo (Optimista)", f"${p95:.2f}", f"+{((p95/start_price)-1)*100:.1f}%")
            k2.metric("🔵 Mediana (Base)", f"${p50:.2f}", f"{((p50/start_price)-1)*100:.1f}%")
            k3.metric("🔴 Suelo (Riesgo)", f"${p05:.2f}", f"{((p05/start_price)-1)*100:.1f}%")

            # Diagnóstico Escrito
            if phi_score > 75:
                st.success("💎 **POTENCIAL ALFA:** La simulación muestra una alta probabilidad de ganancia con un riesgo asimétrico a tu favor.")
            elif phi_score > 40:
                st.info("⚖️ **POTENCIAL NEUTRO:** El activo tiene posibilidades, pero el riesgo de caída es considerable.")
            else:
                st.error("💣 **POTENCIAL NEGATIVO:** Las probabilidades matemáticas están en tu contra. El riesgo supera al beneficio.")
            
            # Gráfico de Abanico
            fig = go.Figure()
            # 50 caminos aleatorios (Fondo)
            for i in range(50): 
                fig.add_trace(go.Scatter(y=paths[:, i], mode='lines', line=dict(color='gray', width=0.5), opacity=0.1, showlegend=False))
            
            # Líneas Clave
            fig.add_trace(go.Scatter(y=np.percentile(paths, 95, axis=1), mode='lines', name='Optimista', line=dict(color='green', width=2, dash='dash')))
            fig.add_trace(go.Scatter(y=np.percentile(paths, 50, axis=1), mode='lines', name='Mediana', line=dict(color='blue', width=3)))
            fig.add_trace(go.Scatter(y=np.percentile(paths, 5, axis=1), mode='lines', name='Pesimista', line=dict(color='red', width=2, dash='dash')))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Explicación Técnica (Desplegable)
            with st.expander("🧮 Ver desglose de la Fórmula Φ"):
                st.write(f"""
                **Cálculo de Φ (Phi):**
                1. **Probabilidad de Ganancia ({win_rate*100:.0f}%):** De los 200 futuros simulados, ¿cuántos terminaron arriba del precio actual? *(Aporta hasta 60 pts)*
                2. **Asimetría ({risk_reward_ratio:.1f}x):** Por cada dólar que arriesgas a la baja, ¿cuántos puedes ganar al alza? *(Aporta hasta 30 pts)*
                3. **Penalización de Entropía:** Si la volatilidad proyectada es peligrosa, restamos puntos. *(Descuento actual: -{entropy_penalty} pts)*
                """)
