# ==============================================================================
# FAROS v8.0 - INTELLIGENCE SUITE (SCANNER + BACKTESTER)
# Autor: Juan Arroyo | SG Consulting Group
# Módulos: 
#   1. Scanner en Tiempo Real (Live)
#   2. Motor de Backtesting Histórico (Simulation)
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta

# --- CONFIGURACIÓN GENERAL ---
st.set_page_config(page_title="FAROS | Intelligence", page_icon="📡", layout="wide")
st.markdown("""
<style>
    .stApp { background-color: #FFFFFF; color: #111; }
    h1, h2, h3 { color: #000 !important; }
    div[data-testid="stMetricValue"] { font-size: 1.1rem; }
    div[data-testid="stMetricLabel"] { font-size: 0.8rem; color: #666; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 🧠 MÓDULO 1: FUNCIONES DE CÁLCULO (CORE)
# ==============================================================================

@st.cache_data(ttl=300)
def get_live_data(tickers_input, window_cfg):
    # (Esta es la función del Scanner que ya funciona perfecta)
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
                
                # Entropía
                subset_returns = returns.tail(w_calc)
                vol_annualized = subset_returns.std() * np.sqrt(252) * 100 if len(subset_returns) > 1 else 0
                z_entropy = (vol_annualized - 20) / 15 
                
                # Liquidez
                vol_avg = hist['Volume'].rolling(w_calc).mean().iloc[-1]
                curr_vol = hist['Volume'].iloc[-1]
                z_liquidity = (curr_vol - vol_avg) / vol_avg if vol_avg > 0 else 0
                
                # Tendencia
                sma_trend = hist['Close'].rolling(w_trend).mean().iloc[-1]
                trend_strength = (current_price - sma_trend) / sma_trend
                
                # ÁRBOL DE DECISIÓN (Incluye la Excepción PLTR)
                signal, category, narrative = "MANTENER", "neutral", "Consolidación."
                
                if z_entropy > 2.0:
                    if trend_strength > 0.10: 
                        signal = "GROWTH AGRESIVO"
                        category = "warning"
                        narrative = f"⚡ Tendencia fuerte (+{trend_strength*100:.1f}%) supera alta volatilidad."
                    elif trend_strength > 0.02 and z_liquidity > -0.15:
                        signal = "GROWTH (VOLÁTIL)"
                        category = "warning"
                        narrative = f"⚡ Volatilidad alta sostenida por precio y volumen."
                    else:
                        signal = "RIESGO ALTO"
                        category = "danger"
                        narrative = f"⚠️ Estructura Inestable ({z_entropy:.1f}σ) sin tendencia."
                
                elif z_liquidity < -0.2 and trend_strength < -0.05:
                    signal = "VENTA / SALIDA"
                    category = "warning"
                    narrative = "📉 Debilidad confirmada por precio y volumen."
                
                elif trend_strength > 0.02:
                    if z_liquidity > 0.10:
                        signal = "COMPRA FUERTE"
                        category = "success"
                        narrative = f"🚀 Fase Líquida Óptima. Dinero entrando (+{z_liquidity*100:.0f}%)."
                    else:
                        signal = "ACUMULAR"
                        category = "info"
                        narrative = "📈 Tendencia alcista sana."

                data_list.append({
                    "Ticker": ticker, "Price": current_price, "Signal": signal, 
                    "Category": category, "Narrative": narrative,
                    "Entropy": z_entropy, "Liquidity": z_liquidity, "Trend": trend_strength * 100,
                    "Vol_Ann": vol_annualized
                })
        except: pass

    df = pd.DataFrame(data_list)
    if not df.empty:
        prio = {"success": 0, "info": 1, "warning": 2, "neutral": 3, "danger": 4}
        df['P'] = df['Category'].map(prio)
        df = df.sort_values('P')
    return df

# --- REEMPLAZA SOLO LA FUNCIÓN run_backtest CON ESTO ---

def run_backtest(ticker, start_date, initial_capital=10000):
    try:
        # 1. Limpieza del Ticker (evita errores por espacios)
        ticker = ticker.strip().upper()
        
        # 2. Descarga de datos
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date)
        
        # 3. Diagnóstico de errores comunes
        if df.empty:
            st.error(f"⚠️ Error: No se encontraron datos para '{ticker}'.")
            st.caption("Consejo: Verifica que el ticker sea correcto (ej: PLTR, BTC-USD, SPY) y que no tengas bloqueos de red.")
            return None
            
        if len(df) < 50:
            st.error(f"⚠️ Datos insuficientes: Se descargaron solo {len(df)} días.")
            st.caption("Solución: Selecciona una 'Fecha Inicio' más antigua (al menos 3 meses atrás).")
            return None
        
        # 4. Limpieza técnica (Eliminar zona horaria para evitar conflictos)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Seleccionar solo lo necesario
        df = df[['Close', 'Volume']].copy()
        
        # --- CÁLCULOS (MOTOR TAI-ACF) ---
        
        # Tendencia (SMA 50)
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['Trend_Str'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
        
        # Entropía (Volatilidad Anualizada)
        df['Returns'] = df['Close'].pct_change()
        df['Vol_Ann'] = df['Returns'].rolling(window=20).std() * np.sqrt(252) * 100
        df['Z_Entropy'] = (df['Vol_Ann'] - 20) / 15
        
        # Liquidez
        df['Vol_SMA'] = df['Volume'].rolling(window=20).mean()
        
        # ESTRATEGIA:
        # 1. Comprar si: Tendencia > 2% Y (Entropía Baja O Tendencia Muy Fuerte > 10%)
        # 2. Vender si: Tendencia rota (<-5%) O Riesgo Extremo (>3 sigma)
        
        # Inicializar vectores
        df['Signal'] = 0
        
        buy_cond = (df['Trend_Str'] > 0.02) & ( (df['Z_Entropy'] < 2.0) | (df['Trend_Str'] > 0.10) )
        sell_cond = (df['Trend_Str'] < -0.05) | ( (df['Z_Entropy'] > 3.0) & (df['Trend_Str'] < 0.05) )
        
        # Lógica vectorial rápida
        df.loc[buy_cond, 'Signal'] = 1
        df.loc[sell_cond, 'Signal'] = 0
        
        # Rellenar (Hold): Si no hay señal nueva, mantener la anterior
        df['Signal'] = df['Signal'].replace(to_replace=0, method='ffill') 
        # Nota: Si pandas es muy nuevo, replace/method puede fallar, usamos ffill directo:
        df['Signal'] = df['Signal'].ffill().fillna(0)
        
        # --- RESULTADOS ---
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
# 📱 MÓDULO 2: INTERFAZ DE USUARIO (NAVEGACIÓN)
# ==============================================================================

# BARRA LATERAL COMÚN
with st.sidebar:
    st.header("📡 FAROS SYSTEM")
    # MENÚ DE NAVEGACIÓN
    app_mode = st.radio("MÓDULO DE TRABAJO:", ["🔍 SCANNER EN VIVO", "🧪 BACKTESTING HISTÓRICO"])
    st.markdown("---")

# ------------------------------------------------------------------------------
# VISTA A: SCANNER EN VIVO (Lo que ya tenías)
# ------------------------------------------------------------------------------
if app_mode == "🔍 SCANNER EN VIVO":
    
    # Controles del Scanner
    with st.sidebar:
        time_horizon = st.selectbox("⏱️ Horizonte", ("Corto Plazo", "Medio Plazo", "Largo Plazo"))
        if "Corto" in time_horizon: window_config = {'volatility': 10, 'trend': 20, 'download': '3mo', 'desc': 'Días'}
        elif "Medio" in time_horizon: window_config = {'volatility': 20, 'trend': 50, 'download': '6mo', 'desc': 'Semanas'}
        else: window_config = {'volatility': 60, 'trend': 200, 'download': '2y', 'desc': 'Meses'}
        
        tickers = st.text_area("Cartera:", "PLTR, BTC-USD, CVX, SPY, TSLA, AMTB, NVDA", height=150)
        if st.button("Escanear Mercado", type="primary"): st.cache_data.clear()

    st.title("Panel de Inteligencia TAI-ACF")
    df = get_live_data(tickers, window_config)

    if not df.empty:
        col_list, col_radar = st.columns([1.6, 1])
        with col_radar:
            st.subheader("🧭 Radar de Mercado")
            fig = px.scatter(df, x="Entropy", y="Liquidity", color="Category", text="Ticker",
                             color_discrete_map={"success":"#28a745", "info":"#17a2b8", "neutral":"#6c757d", "warning":"#ffc107", "danger":"#dc3545"})
            fig.update_layout(template="plotly_white", height=450, showlegend=False)
            fig.add_vline(x=2.0, line_dash="dash", line_color="red", opacity=0.3)
            st.plotly_chart(fig, use_container_width=True)

        with col_list:
            st.subheader("📋 Matriz de Decisión")
            for i, row in df.iterrows():
                with st.container(border=True):
                    head_c1, head_c2 = st.columns([3, 1])
                    head_c1.markdown(f"### **{row['Ticker']}**") 
                    head_c2.markdown(f"### ${row['Price']:,.2f}")
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Entropía", f"{row['Entropy']:.1f}σ")
                    m2.metric("Liquidez", f"{row['Liquidity']*100:+.0f}%", delta_color="normal")
                    m3.metric("Tendencia", f"{row['Trend']:+.1f}%")
                    
                    msg = f"**{row['Signal']}:** {row['Narrative']}"
                    if row['Category'] == 'success': st.success(msg, icon="✅")
                    elif row['Category'] == 'info': st.info(msg, icon="ℹ️")
                    elif row['Category'] == 'warning': st.warning(msg, icon="⚠️")
                    elif row['Category'] == 'danger': st.error(msg, icon="⛔")
                    else: st.write(msg)
    else:
        st.info("Inicializando...")

# ------------------------------------------------------------------------------
# VISTA B: BACKTESTING (NUEVO MÓDULO)
# ------------------------------------------------------------------------------
elif app_mode == "🧪 BACKTESTING HISTÓRICO":
    
    st.title("Laboratorio de Estrategia TAI-ACF")
    st.markdown("Simulación de rendimiento histórico comparado contra 'Buy & Hold'.")
    
    # Controles del Backtest
    with st.sidebar:
        st.subheader("Parámetros de Prueba")
        bt_ticker = st.text_input("Activo a Probar:", "PLTR").upper()
        bt_start = st.date_input("Fecha Inicio:", pd.to_datetime("2023-01-01"))
        bt_capital = st.number_input("Capital Inicial ($)", value=10000)
        run_bt = st.button("EJECUTAR SIMULACIÓN ▶️", type="primary")

    if run_bt:
        with st.spinner(f"Simulando estrategia en {bt_ticker}..."):
            bt_data = run_backtest(bt_ticker, bt_start, bt_capital)
        
        if bt_data is not None:
            # RESULTADOS FINALES
            final_equity = bt_data['Equity_Strategy'].iloc[-1]
            bh_equity = bt_data['Equity_BuyHold'].iloc[-1]
            perf_pct = ((final_equity - bt_capital) / bt_capital) * 100
            bh_pct = ((bh_equity - bt_capital) / bt_capital) * 100
            
            # 1. TARJETAS DE RENDIMIENTO
            col1, col2, col3 = st.columns(3)
            col1.metric("Rendimiento Estrategia", f"{perf_pct:.2f}%", delta=f"${final_equity - bt_capital:,.0f}")
            col2.metric("Buy & Hold (Mercado)", f"{bh_pct:.2f}%", delta=f"${bh_equity - bt_capital:,.0f}")
            
            alpha = perf_pct - bh_pct
            col3.metric("Alpha (Exceso Retorno)", f"{alpha:.2f}%", delta_color="normal")
            
            st.markdown("---")
            
            # 2. GRÁFICO DE CURVA DE CAPITAL
            st.subheader("Curva de Crecimiento de Capital")
            
            fig_eq = go.Figure()
            # Línea de Estrategia
            fig_eq.add_trace(go.Scatter(
                x=bt_data.index, y=bt_data['Equity_Strategy'], 
                mode='lines', name='Estrategia FAROS',
                line=dict(color='#28a745', width=3)
            ))
            # Línea de Mercado
            fig_eq.add_trace(go.Scatter(
                x=bt_data.index, y=bt_data['Equity_BuyHold'], 
                mode='lines', name='Buy & Hold',
                line=dict(color='#6c757d', width=1, dash='dot')
            ))
            
            # Áreas de compra (Fondo verde cuando estamos comprados)
            # Truco visual: pintar el fondo cuando Signal == 1
            
            fig_eq.update_layout(
                template="plotly_white", 
                height=500,
                yaxis_title="Capital ($)",
                legend=dict(orientation="h", y=1.02, x=0)
            )
            st.plotly_chart(fig_eq, use_container_width=True)
            
            # 3. EXPLICACIÓN
            with st.expander("🔎 Ver Lógica de la Simulación"):
                st.write("""
                **Reglas de Entrada (Compra):**
                * Tendencia positiva (>2% sobre media de 50 días).
                * Entropía controlada (<2.0σ) O Tendencia muy fuerte (>10%).
                
                **Reglas de Salida (Venta):**
                * Tendencia rota (<-5%).
                * O Entropía extrema (>3.0σ) sin tendencia que la justifique.
                """)
                
        else:
            st.error("No se pudieron descargar datos o el rango de fechas es muy corto.")

