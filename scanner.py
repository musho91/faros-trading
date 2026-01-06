# ==============================================================================
# FAROS v15.3 - MASTER SUITE (FINAL DEFINITIVE)
# Autor: Juan Arroyo | SG Consulting Group
# Módulos: 
#   1. Scanner Global (Con Señales Explícitas, PSI Score & Semáforo Sistémico)
#   2. Backtest Time Machine (4 Estados de la Materia & Capital Editable)
#   3. Oráculo Estocástico (Estable con Seed, Momentum Bias & PHI Score)
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf

# --- CONFIGURACIÓN VISUAL ---
st.set_page_config(page_title="FAROS | Master Suite", page_icon="📡", layout="wide")
st.markdown("""
<style>
    .stApp { background-color: #FFFFFF; color: #111; }
    h1, h2, h3 { color: #000 !important; } 
    .stExpander { border: 1px solid #eee; background-color: #f8f9fa; border-radius: 8px; }
    .global-status { padding: 15px; border-radius: 8px; margin-bottom: 20px; font-weight: bold; text-align: center; border: 1px solid #ddd; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. MOTOR LÓGICO & MATEMÁTICO
# ==============================================================================

def calculate_entropy(history, window=20):
    if len(history) < window: return 0, 0
    returns = history['Close'].pct_change().dropna()
    subset = returns.tail(window)
    raw_vol = subset.std() * np.sqrt(252) * 100 if len(subset) > 1 else 0
    z_entropy = (raw_vol - 20) / 15 
    return raw_vol, z_entropy

@st.cache_data(ttl=300)
def get_market_status():
    try:
        spy = yf.Ticker("SPY").history(period="6mo")
        if spy.empty: return "UNKNOWN", 0, "Error de Datos"
        raw, z = calculate_entropy(spy)
        if z > 3.0: return "GAS", z, "CRISIS SISTÉMICA (Crash Mode)"
        elif z > 2.0: return "WARNING", z, "ALTA TENSIÓN (Precaución)"
        else: return "LIQUID", z, "ESTABLE (Condiciones Favorables)"
    except: return "UNKNOWN", 0, "Desconectado"

def calculate_psi(entropy, liquidity, trend, risk_sigma, global_penalty=0):
    """Fórmula Maestra (PSI) - Calidad Presente"""
    score = 50 
    if entropy > risk_sigma: score -= 30
    else: score += (risk_sigma - entropy) * 10 
    if liquidity > 0: score += liquidity * 20
    elif liquidity < -0.2: score -= 20
    if trend > 0: score += trend * 100
    else: score -= 50
    score -= global_penalty
    return max(0, min(100, score))

@st.cache_data(ttl=300)
def get_live_data(tickers_input, window_cfg, risk_tolerance):
    m_status, m_entropy, m_msg = get_market_status()
    global_penalty = 30 if m_status == "GAS" else (10 if m_status == "WARNING" else 0)

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
                raw_vol, z_entropy = calculate_entropy(hist, window_cfg['volatility'])
                vol_avg = hist['Volume'].rolling(window_cfg['volatility']).mean().iloc[-1]
                curr_vol = hist['Volume'].iloc[-1]
                z_liq = (curr_vol - vol_avg) / vol_avg if vol_avg > 0 else 0
                raw_vol_ratio = (curr_vol / vol_avg) if vol_avg > 0 else 0
                sma_val = hist['Close'].rolling(window_cfg['trend']).mean().iloc[-1]
                trend_pct = (current_price - sma_val) / sma_val
                
                psi_score = calculate_psi(z_entropy, z_liq, trend_pct, risk_tolerance, global_penalty)
                
                signal, category, narrative = "MANTENER", "neutral", "Sólido (Estable)."
                sys_warn = " ⚠️ [RIESGO GLOBAL]" if m_status == "GAS" else ""
                
                if z_entropy > entropy_limit:
                    if trend_pct > 0.15: 
                        signal = "GROWTH EXTREMO"
                        category = "warning"
                        narrative = f"⚡ Momentum (+{trend_pct*100:.1f}%) vence volatilidad.{sys_warn}"
                    else:
                        signal = "GAS / RIESGO"
                        category = "danger"
                        narrative = f"⚠️ Fase Gaseosa Local ({z_entropy:.1f}σ).{sys_warn}"
                elif trend_pct < exit_threshold:
                    signal = "SALIDA"
                    category = "danger" if risk_tolerance < 3 else "warning"
                    narrative = f"📉 Rotura Estructural ({trend_pct*100:.1f}%).{sys_warn}"
                elif trend_pct > 0.02:
                    if z_liq > 0.10:
                        signal = "COMPRA FUERTE"
                        category = "success" if m_status != "GAS" else "warning"
                        narrative = f"🚀 Fase Líquida Óptima.{sys_warn}"
                    else:
                        signal = "ACUMULAR"
                        category = "info" if m_status != "GAS" else "neutral"
                        narrative = f"📈 Tendencia Sana.{sys_warn}"
                elif z_liq < -0.3 and abs(trend_pct) < 0.02:
                     signal = "PLASMA"
                     category = "neutral"
                     narrative = "🟡 Iliquidez (Mercado Seco)."

                data_list.append({
                    "Ticker": ticker, "Price": current_price, "Signal": signal, 
                    "Category": category, "Narrative": narrative,
                    "Entropy": z_entropy, "Liquidity": z_liq, "Trend": trend_pct * 100,
                    "Psi": psi_score, "Raw_Vol": raw_vol, "Raw_Vol_Ratio": raw_vol_ratio, "SMA_Price": sma_val,
                    "Exit_Limit": exit_threshold * 100
                })
        except: pass
    return pd.DataFrame(data_list).sort_values('Psi', ascending=False) if data_list else pd.DataFrame(), m_status, m_entropy, m_msg

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
        conditions = [
            (df['Z_Entropy'] > entropy_limit) & (df['Trend'] < 0.15),
            (df['Z_Liq'] < -0.3),
            (df['Trend'] > 0.02) & (df['Z_Entropy'] <= entropy_limit) & (df['Z_Liq'] > 0)
        ]
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
    """Oráculo Estabilizado con Semilla (Seed) y Momentum Bias"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        if len(hist) < 50: return None, 0, 0
        
        last_price = hist['Close'].iloc[-1]
        returns = hist['Close'].pct_change().dropna()
        daily_vol = returns.std()
        
        sma_50 = hist['Close'].rolling(50).mean().iloc[-1]
        trend_force = (last_price - sma_50) / sma_50
        
        hist_drift = returns.mean() - 0.5 * (daily_vol ** 2)
        trend_drift = trend_force / 252 * 2 
        final_drift = (hist_drift * 0.3) + (trend_drift * 0.7) if risk_tolerance >= 3 else hist_drift
        
        # SEED ESTABILIZADOR
        unique_seed = int(sum(ord(c) for c in ticker) + days)
        np.random.seed(unique_seed)

        sims = 1000 
        paths = np.zeros((days, sims)); paths[0] = last_price
        proj_h = (daily_vol * np.sqrt(252) * 100 - 20) / 15
        
        for t in range(1, days):
            shock = np.random.normal(0, daily_vol, sims)
            paths[t] = paths[t-1] * np.exp(final_drift + shock)
            
        return paths, proj_h, trend_force
    except: return None, 0, 0

# ==============================================================================
# 2. INTERFAZ DE USUARIO (FRONT-END)
# ==============================================================================

with st.sidebar:
    st.header("📡 FAROS SUITE")
    app_mode = st.radio("MÓDULO:", ["SCANNER", "BACKTEST", "ORÁCULO"])
    st.markdown("---")
    st.subheader("🎛️ Calibración")
    risk_profile = st.select_slider("Tolerancia", options=["Conservador", "Growth", "Quantum"], value="Growth")
    
    if "Conservador" in risk_profile: risk_sigma = 2.0
    elif "Growth" in risk_profile: risk_sigma = 3.0
    else: risk_sigma = 5.0 
    
    st.caption(f"Límite Entropía: **{risk_sigma}σ**")
    st.caption(f"Stop Táctico: **{'-15%' if risk_sigma==5 else '-10%' if risk_sigma==3 else '-5%'}**")

# --- MÓDULO 1: SCANNER ---
if app_mode == "SCANNER":
    time_h = st.selectbox("Horizonte", ["Corto Plazo", "Medio Plazo", "Largo Plazo"])
    if "Corto" in time_h: cfg = {'volatility': 10, 'trend': 20, 'download': '3mo', 'desc': 'Trading'}
    elif "Medio" in time_h: cfg = {'volatility': 20, 'trend': 50, 'download': '6mo', 'desc': 'Swing'}
    else: cfg = {'volatility': 60, 'trend': 200, 'download': '2y', 'desc': 'Inversión'}
    
    tickers = st.text_area("Cartera:", "PLTR, QBTS, NVDA, SPY, BTC-USD", height=100)
    if st.button("Analizar Mercado"): st.cache_data.clear()
    
    df, m_status, m_entropy, m_msg = get_live_data(tickers, cfg, risk_sigma)
    
    cols = {"GAS":("#FFCDD2","#B71C1C","🔥"), "WARNING":("#FFF9C4","#F57F17","⚠️"), "LIQUID":("#C8E6C9","#1B5E20","🌍")}
    bg, txt, ico = cols.get(m_status, ("#eee","#333","❓"))
    
    st.markdown(f"""
    <div class="global-status" style="background-color: {bg}; color: {txt};">
        {ico} TERMODINÁMICA GLOBAL (SPY): {m_msg} [{m_entropy:.1f}σ]
    </div>
    """, unsafe_allow_html=True)
    
    if not df.empty:
        c1, c2 = st.columns([2,1])
        with c2:
            st.markdown("#### 🧭 Radar")
            fig = px.scatter(df, x="Entropy", y="Liquidity", color="Category", text="Ticker", 
                             color_discrete_map={"success":"#28a745","warning":"#ffc107","danger":"#dc3545","neutral":"#6c757d"})
            fig.add_vline(x=risk_sigma, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        with c1:
            st.markdown("#### 📋 Diagnóstico")
            for i, r in df.iterrows():
                with st.container(border=True):
                    hc1, hc2 = st.columns([3,1])
                    hc1.markdown(f"### **{r['Ticker']}** ${r['Price']:.2f}")
                    # PSI Score Color
                    pc = "green" if r['Psi']>70 else "orange" if r['Psi']>40 else "red"
                    hc2.markdown(f"**Ψ: :{pc}[{r['Psi']:.0f}]**")
                    
                    # --- AQUÍ ESTÁ LA CORRECCIÓN: SEÑAL VISIBLE ---
                    msg = f"**{r['Signal']}**: {r['Narrative']}"
                    
                    if r['Category']=='success': st.success(msg)
                    elif r['Category']=='warning': st.warning(msg)
                    elif r['Category']=='danger': st.error(msg)
                    else: st.info(msg)
                    # ---------------------------------------------
                    
                    with st.expander("🔬 Lab Data"):
                        st.markdown(f"**Vol:** {r['Raw_Vol']:.0f}% | **Tendencia:** {r['Trend']:+.1f}% | **Liq:** {r['Raw_Vol_Ratio']:.1f}x")

# --- MÓDULO 2: BACKTEST ---
elif app_mode == "BACKTEST":
    st.title("Validación Histórica")
    c_tick, c_cap = st.columns([1, 1])
    tck = c_tick.text_input("Activo:", "PLTR").upper()
    cap = c_cap.number_input("Capital ($):", value=10000)
    c1, c2 = st.columns(2)
    d1 = c1.date_input("Inicio", pd.to_datetime("2023-01-01"))
    d2 = c2.date_input("Fin", pd.to_datetime("2025-01-05"))
    
    if st.button("Ejecutar"):
        res = run_backtest(tck, d1, d2, cap, risk_sigma)
        if res is not None:
            fin = res['Eq_Strat'].iloc[-1]
            st.metric("Resultado Final", f"${fin:,.0f}", delta=f"{(fin/cap-1)*100:.1f}%")
            
            st.subheader("Ciclos Detectados")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res.index, y=res['Close'], name='Precio', line=dict(color='black', width=1), opacity=0.3))
            
            colors = {'GAS': 'red', 'LIQUID': '#00FF41', 'PLASMA': '#FFD700', 'SOLID': 'gray'}
            for p, c in colors.items():
                s = res[res['Phase']==p]
                if not s.empty: fig.add_trace(go.Scatter(x=s.index, y=s['Close'], mode='markers', name=p, marker=dict(color=c, size=5)))
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Curva de Capital")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=res.index, y=res['Eq_Strat'], name='FAROS', line=dict(color='blue', width=2)))
            fig2.add_trace(go.Scatter(x=res.index, y=res['Eq_BH'], name='Hold', line=dict(color='gray', dash='dot')))
            st.plotly_chart(fig2, use_container_width=True)

# --- MÓDULO 3: ORÁCULO ---
elif app_mode == "ORÁCULO":
    st.title("Proyección TAI (Potencial Φ)")
    st.caption("Simulación Estocástica con Inercia de Tendencia.")
    
    c_tick, c_days = st.columns([1, 1])
    o_tick = c_tick.text_input("Activo:", "PLTR").upper()
    o_days = c_days.slider("Días:", 30, 365, 365)
    
    if st.button("Consultar"):
        paths, proj_h, trend_f = run_oracle_sim(o_tick, o_days, risk_sigma)
        
        if paths is not None:
            start = paths[0][0]; final = paths[-1]
            p95 = np.percentile(final, 95); p50 = np.percentile(final, 50); p05 = np.percentile(final, 5)
            
            win_rate = np.mean(final > start)
            upside = (p95 - start)/start; downside = abs((p05 - start)/start)
            rr = upside/downside if downside > 0 else 10
            
            bonus = 20 if trend_f > 0.10 else 0
            if trend_f > 0.10: proj_h = max(0, proj_h - 1.0)
            
            phi = (win_rate * 50) + (min(rr, 4) * 10) + bonus
            if proj_h > risk_sigma: phi -= 20
            phi = max(0, min(100, phi))
            
            p_col = "green" if phi > 70 else "orange" if phi > 40 else "red"
            
            # 1. SCORE
            st.markdown(f"""
            <div style="text-align:center; padding:15px; border:1px solid #ddd; border-radius:10px; background-color:#f9f9f9; margin-bottom:20px;">
                <h2 style="margin:0; color:#333;">POTENCIAL FUTURO (Φ)</h2>
                <h1 style="margin:0; font-size:4rem; color:{p_col};">{phi:.0f}/100</h1>
                <p style="color:#666;">Probabilidad: <b>{win_rate*100:.0f}%</b> | R/R: <b>{rr:.1f}x</b> | Momentum: <b>{trend_f*100:+.1f}%</b></p>
            </div>
            """, unsafe_allow_html=True)
            
            # 2. MÉTRICAS
            k1, k2, k3 = st.columns(3)
            k1.metric("🟢 Techo", f"${p95:.2f}", f"+{((p95/start)-1)*100:.1f}%")
            k2.metric("🔵 Base", f"${p50:.2f}", f"{((p50/start)-1)*100:.1f}%")
            k3.metric("🔴 Suelo", f"${p05:.2f}", f"{((p05/start)-1)*100:.1f}%")
            
            # 3. INTERPRETACIÓN TÁCTICA
            with st.expander("📖 Interpretación Táctica de Escenarios", expanded=True):
                st.markdown("""
                * **🟢 Techo (Optimista):** Rendimiento excepcional (Top 5% de probabilidad). Si llega aquí, es un 'Moonshot'.
                * **🔵 Base (Probable):** Mediana estadística. Si la inercia actual se mantiene, el precio orbitará esta zona.
                * **🔴 Suelo (Riesgo):** Escenario de Cisne Negro (Peor 5%). Este nivel marca tu **Riesgo Máximo Probable**.
                """)

            # 4. GRÁFICO
            fig = go.Figure()
            for i in range(50): fig.add_trace(go.Scatter(y=paths[:, i], line=dict(color='gray', width=0.5), opacity=0.1, showlegend=False))
            fig.add_trace(go.Scatter(y=np.percentile(paths, 95, axis=1), name='Optimista', line=dict(color='green', dash='dash')))
            fig.add_trace(go.Scatter(y=np.percentile(paths, 50, axis=1), name='Tendencia', line=dict(color='blue', width=3)))
            fig.add_trace(go.Scatter(y=np.percentile(paths, 5, axis=1), name='Pesimista', line=dict(color='red', dash='dash')))
            st.plotly_chart(fig, use_container_width=True)
