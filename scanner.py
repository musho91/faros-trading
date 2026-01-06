# ==============================================================================
# FAROS v9.0 - TIME MACHINE & THERMODYNAMIC PHASES
# Autor: Juan Arroyo | SG Consulting Group
# Feature: Backtesting con Rango de Fechas y Visualización de Estados (Gas/Líquido)
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta

# CONFIGURACIÓN
st.set_page_config(page_title="FAROS | Time Machine", page_icon="📡", layout="wide")
st.markdown("""
<style>
    .stApp { background-color: #FFFFFF; color: #111; }
    h1, h2, h3 { color: #000 !important; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 🧠 MÓDULO LÓGICO (CORE)
# ==============================================================================

@st.cache_data(ttl=300)
def get_live_data(tickers_input, window_cfg):
    # (Misma función del Scanner que ya funciona bien)
    tickers_list = [x.strip().upper() for x in tickers_input.split(',')]
    data_list = []
    w_calc = window_cfg['volatility']
    w_trend = window_cfg['trend']
    period_dl = window_cfg['download']

    for ticker in tickers_list:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period_dl)
            if len(hist) > w_trend:
                current_price = hist['Close'].iloc[-1]
                returns = hist['Close'].pct_change().dropna()
                subset_returns = returns.tail(w_calc)
                vol_annualized = subset_returns.std() * np.sqrt(252) * 100 if len(subset_returns) > 1 else 0
                z_entropy = (vol_annualized - 20) / 15 
                vol_avg = hist['Volume'].rolling(w_calc).mean().iloc[-1]
                curr_vol = hist['Volume'].iloc[-1]
                z_liquidity = (curr_vol - vol_avg) / vol_avg if vol_avg > 0 else 0
                sma_trend = hist['Close'].rolling(w_trend).mean().iloc[-1]
                trend_strength = (current_price - sma_trend) / sma_trend
                
                signal, category, narrative = "MANTENER", "neutral", "Consolidación."
                if z_entropy > 2.0:
                    if trend_strength > 0.10: 
                        signal = "GROWTH AGRESIVO"
                        category = "warning"
                        narrative = f"⚡ Tendencia fuerte (+{trend_strength*100:.1f}%) supera volatilidad."
                    else:
                        signal = "RIESGO / GAS"
                        category = "danger"
                        narrative = f"⚠️ Fase Gaseosa ({z_entropy:.1f}σ). Alta probabilidad de crash."
                elif trend_strength > 0.02:
                    if z_liquidity > 0.10:
                        signal = "FASE LÍQUIDA"
                        category = "success"
                        narrative = f"🚀 Estructura ordenada + Flujo (+{z_liquidity*100:.0f}%)."
                    else:
                        signal = "ACUMULAR"
                        category = "info"
                        narrative = "📈 Tendencia sana (Sólido)."
                elif trend_strength < -0.05:
                    signal = "SALIDA"
                    category = "warning"
                    narrative = "📉 Tendencia rota."

                data_list.append({
                    "Ticker": ticker, "Price": current_price, "Signal": signal, 
                    "Category": category, "Narrative": narrative,
                    "Entropy": z_entropy, "Liquidity": z_liquidity, "Trend": trend_strength * 100
                })
        except: pass
    df = pd.DataFrame(data_list)
    if not df.empty:
        prio = {"success": 0, "info": 1, "warning": 2, "neutral": 3, "danger": 4}
        df['P'] = df['Category'].map(prio)
        df = df.sort_values('P')
    return df

# --- FUNCIÓN DE BACKTESTING AVANZADA (CON FASES) ---
def run_backtest(ticker, start_date, end_date, initial_capital=10000):
    try:
        ticker = ticker.strip().upper()
        stock = yf.Ticker(ticker)
        # Descarga con fecha fin
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty or len(df) < 50:
            st.error("Datos insuficientes para el rango seleccionado.")
            return None
        
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        df = df[['Close', 'Volume']].copy()
        
        # --- CÁLCULO DE FÍSICA ---
        # 1. Tendencia
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['Trend_Str'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
        
        # 2. Entropía (Z_H)
        df['Returns'] = df['Close'].pct_change()
        df['Vol_Ann'] = df['Returns'].rolling(window=20).std() * np.sqrt(252) * 100
        df['Z_Entropy'] = (df['Vol_Ann'] - 20) / 15
        
        # --- LÓGICA DE FASES (Thermodynamic Cycle) ---
        # Definimos el estado para pintar el gráfico después
        conditions = [
            (df['Z_Entropy'] > 2.5) & (df['Trend_Str'] < 0.10), # GAS (Crash/Riesgo)
            (df['Trend_Str'] > 0.02) & (df['Z_Entropy'] < 2.0)  # LÍQUIDO (Growth)
        ]
        choices = ['GAS', 'LIQUID']
        df['Phase'] = np.select(conditions, choices, default='SOLID')
        
        # --- ESTRATEGIA ---
        df['Signal'] = 0
        
        # REGLA 1: PROTECCIÓN (COVID RULE)
        # Si entramos en GAS, Venta Inmediata (Cash)
        # REGLA 2: CRECIMIENTO
        # Si estamos en LIQUID o SOLID (con tendencia positiva), Compramos.
        
        buy_signal = (df['Phase'] == 'LIQUID') | ( (df['Phase'] == 'SOLID') & (df['Trend_Str'] > 0) )
        crash_signal = (df['Phase'] == 'GAS') | (df['Trend_Str'] < -0.05)
        
        df.loc[buy_signal, 'Signal'] = 1
        df.loc[crash_signal, 'Signal'] = 0
        
        # Mantener posición previa (ffill)
        df['Signal'] = df['Signal'].ffill().fillna(0)
        
        # --- RENDIMIENTO ---
        df['Market_Return'] = df['Close'].pct_change()
        df['Strategy_Return'] = df['Market_Return'] * df['Signal'].shift(1)
        df.dropna(inplace=True)
        
        df['Equity_BuyHold'] = initial_capital * (1 + df['Market_Return']).cumprod()
        df['Equity_Strategy'] = initial_capital * (1 + df['Strategy_Return']).cumprod()
        
        return df
        
    except Exception as e:
        st.error(f"Error Técnico: {str(e)}")
        return None

# ==============================================================================
# 📱 INTERFAZ DE USUARIO
# ==============================================================================

with st.sidebar:
    st.header("📡 FAROS")
    app_mode = st.radio("MÓDULO:", ["🔍 SCANNER MERCADO", "⏳ MÁQUINA DEL TIEMPO"])
    st.markdown("---")

if app_mode == "🔍 SCANNER MERCADO":
    # (Código del Scanner igual que antes...)
    with st.sidebar:
        time_horizon = st.selectbox("⏱️ Horizonte", ("Corto Plazo", "Medio Plazo", "Largo Plazo"))
        if "Corto" in time_horizon: window_config = {'volatility': 10, 'trend': 20, 'download': '3mo', 'desc': 'Días'}
        elif "Medio" in time_horizon: window_config = {'volatility': 20, 'trend': 50, 'download': '6mo', 'desc': 'Semanas'}
        else: window_config = {'volatility': 60, 'trend': 200, 'download': '2y', 'desc': 'Meses'}
        tickers = st.text_area("Cartera:", "PLTR, BTC-USD, CVX, SPY, TSLA", height=150)
        if st.button("Analizar"): st.cache_data.clear()

    st.title("Panel TAI-ACF en Vivo")
    df = get_live_data(tickers, window_config)
    if not df.empty:
        col_list, col_radar = st.columns([1.6, 1])
        with col_radar:
            fig = px.scatter(df, x="Entropy", y="Liquidity", color="Category", text="Ticker", color_discrete_map={"success":"#28a745", "info":"#17a2b8", "neutral":"#6c757d", "warning":"#ffc107", "danger":"#dc3545"})
            fig.update_layout(template="plotly_white", height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col_list:
            for i, row in df.iterrows():
                with st.container(border=True):
                    c1, c2 = st.columns([3, 1])
                    c1.markdown(f"**{row['Ticker']}** | {row['Signal']}")
                    c2.markdown(f"**${row['Price']:,.2f}**")
                    st.caption(f"{row['Narrative']} (H:{row['Entropy']:.1f}σ)")
    else: st.info("Cargando...")

elif app_mode == "⏳ MÁQUINA DEL TIEMPO":
    st.title("Validación Histórica TAI-ACF")
    st.markdown("Prueba la teoría en crisis reales (ej: COVID 2020).")
    
    with st.sidebar:
        st.subheader("Configuración de Prueba")
        bt_ticker = st.text_input("Activo:", "SPY").upper()
        
        # SELECTOR DE FECHAS (Range)
        c1, c2 = st.columns(2)
        bt_start = c1.date_input("Inicio", pd.to_datetime("2019-01-01"))
        bt_end = c2.date_input("Fin", pd.to_datetime("2021-01-01"))
        
        bt_capital = st.number_input("Capital Inicial ($)", value=10000)
        run_bt = st.button("REPRODUCIR HISTORIA ▶️", type="primary")

    if run_bt:
        with st.spinner(f"Simulando ciclo termodinámico en {bt_ticker}..."):
            bt_data = run_backtest(bt_ticker, bt_start, bt_end, bt_capital)
        
        if bt_data is not None:
            # MÉTRICAS
            final_strat = bt_data['Equity_Strategy'].iloc[-1]
            final_bh = bt_data['Equity_BuyHold'].iloc[-1]
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Resultado FAROS", f"${final_strat:,.0f}", delta=f"{((final_strat/bt_capital)-1)*100:.1f}%")
            m2.metric("Buy & Hold", f"${final_bh:,.0f}", delta=f"{((final_bh/bt_capital)-1)*100:.1f}%")
            
            alpha = final_strat - final_bh
            color_alpha = "normal" if alpha > 0 else "inverse"
            m3.metric("Protección/Alpha", f"${alpha:,.0f}", delta_color=color_alpha)
            
            # GRÁFICO AVANZADO CON FASES
            st.subheader("Análisis de Fases Termodinámicas")
            
            # Crear figura
            fig = go.Figure()
            
            # 1. Precio (Línea delgada negra)
            fig.add_trace(go.Scatter(x=bt_data.index, y=bt_data['Close'], name='Precio', line=dict(color='black', width=1), opacity=0.5))
            
            # 2. COLOREAR EL FONDO SEGÚN LA FASE (LA CLAVE DEL DIAGNÓSTICO)
            # FASE GAS (ROJO) = CAOS / CASH
            gas_zones = bt_data[bt_data['Phase'] == 'GAS']
            if not gas_zones.empty:
                fig.add_trace(go.Scatter(
                    x=gas_zones.index, y=gas_zones['Close'],
                    mode='markers', name='Fase Gaseosa (Venta/Cash)',
                    marker=dict(color='red', size=4, opacity=0.5)
                ))
            
            # FASE LÍQUIDA (VERDE) = CRECIMIENTO
            liquid_zones = bt_data[bt_data['Phase'] == 'LIQUID']
            if not liquid_zones.empty:
                 fig.add_trace(go.Scatter(
                    x=liquid_zones.index, y=liquid_zones['Close'],
                    mode='markers', name='Fase Líquida (Compra)',
                    marker=dict(color='#00FF41', size=4, opacity=0.5)
                ))

            fig.update_layout(template="plotly_white", height=400, title="Ciclos de Mercado Detectados", yaxis_title="Precio")
            st.plotly_chart(fig, use_container_width=True)
            
            # GRÁFICO DE CAPITAL
            st.subheader("Curva de Capital (Tu Dinero)")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=bt_data.index, y=bt_data['Equity_Strategy'], name='Estrategia FAROS', line=dict(color='#0055FF', width=3)))
            fig2.add_trace(go.Scatter(x=bt_data.index, y=bt_data['Equity_BuyHold'], name='Buy & Hold', line=dict(color='gray', dash='dot')))
            fig2.update_layout(template="plotly_white", height=400, yaxis_title="Capital ($)")
            st.plotly_chart(fig2, use_container_width=True)
            
            # DIAGNÓSTICO ESCRITO
            with st.expander("📝 Ver Diagnóstico del Período"):
                st.write(f"""
                **Análisis del Ciclo ({bt_start} a {bt_end}):**
                * En periodos de **Fase Gaseosa (Puntos Rojos)**, la entropía superó 2.5σ. El sistema debió pasar a Cash para evitar caídas.
                * En periodos de **Fase Líquida (Puntos Verdes)**, la entropía bajó y la tendencia subió. El sistema invirtió para capturar valor.
                """)
