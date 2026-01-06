# ==============================================================================
# FAROS v7.0 - MULTI-TIMEFRAME & EDUCATION
# Autor: Juan Arroyo | SG Consulting Group
# Novedades: Selector de Temporalidad + Explicación Teórica
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf

# 1. CONFIGURACIÓN (Limpia y Blanca)
st.set_page_config(page_title="FAROS | TAI-ACF", page_icon="📡", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #FFFFFF; color: #111; }
    h1, h2, h3 { color: #000 !important; }
    .stExpander { border: 1px solid #ddd; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
</style>
""", unsafe_allow_html=True)

# 2. MOTOR LÓGICO DINÁMICO (Se adapta a la temporalidad)
@st.cache_data(ttl=300)
def get_quant_data(tickers_input, window_cfg):
    tickers_list = [x.strip().upper() for x in tickers_input.split(',')]
    data_list = []
    
    # Desempaquetar configuración de tiempo
    w_vol = window_cfg['volatility'] # Ventana para Entropía
    w_trend = window_cfg['trend']    # Ventana para Tendencia
    period_dl = window_cfg['download'] # Cuanta data descargar

    for ticker in tickers_list:
        try:
            stock = yf.Ticker(ticker)
            # Descargamos suficiente historia
            hist = stock.history(period=period_dl)
            
            if len(hist) > w_trend:
                current_price = hist['Close'].iloc[-1]
                
                # A. ENTROPÍA (H): Riesgo ajustado a la ventana
                returns = hist['Close'].pct_change().dropna()
                # Volatilidad en la ventana seleccionada
                volatility_window = returns.tail(w_vol).std() * np.sqrt(w_vol) * 100
                # Z-Score comparado con una base teórica del 5%
                z_entropy = (volatility_window - 5) / 2 
                
                # B. LIQUIDEZ (L): Flujo de energía
                # Comparamos volumen actual vs promedio de la ventana
                vol_avg = hist['Volume'].rolling(w_vol).mean().iloc[-1]
                curr_vol = hist['Volume'].iloc[-1]
                z_liquidity = (curr_vol - vol_avg) / vol_avg if vol_avg > 0 else 0
                
                # C. TENDENCIA (Gobernanza del Precio)
                sma_trend = hist['Close'].rolling(w_trend).mean().iloc[-1]
                trend_strength = (current_price - sma_trend) / sma_trend
                
                # Lógica de Señales (Gobernanza Psi implícita)
                signal = "MANTENER"
                category = "neutral" 
                narrative = "Equilibrio. Sin catalizadores claros en este horizonte."

                # Condiciones
                if z_entropy > 2.5:
                    signal = "NO OPERAR"
                    category = "danger"
                    narrative = f"⚠️ Entropía Crítica ({z_entropy:.1f}σ). El ruido supera a la señal en esta ventana de tiempo."
                
                elif z_liquidity < -0.2 and trend_strength < -0.03:
                    signal = "VENTA / SALIDA"
                    category = "warning"
                    narrative = "📉 Divergencia negativa. El precio cae y el volumen confirma la salida de capital."
                
                elif trend_strength > 0.02 and z_entropy < 1.5:
                    if z_liquidity > 0.15:
                        signal = "COMPRA FUERTE"
                        category = "success"
                        narrative = f"🚀 Fase Líquida. Estructura ordenada con inyección de capital (+{z_liquidity*100:.0f}% vol)."
                    else:
                        signal = "ACUMULAR"
                        category = "info"
                        narrative = "📈 Tendencia favorable. Acumulación silenciosa detectada."

                data_list.append({
                    "Ticker": ticker, "Price": current_price, "Signal": signal, 
                    "Category": category, "Narrative": narrative,
                    "Entropy": z_entropy, "Liquidity": z_liquidity, "Trend": trend_strength * 100
                })
        except: pass

    df = pd.DataFrame(data_list)
    if not df.empty:
        prio = {"success": 0, "info": 1, "neutral": 2, "warning": 3, "danger": 4}
        df['P'] = df['Category'].map(prio)
        df = df.sort_values('P')
    return df

# 3. BARRA LATERAL (CONTROLES)
with st.sidebar:
    st.header("📡 CONFIGURACIÓN")
    
    # SELECTOR DE TIEMPO (NUEVO)
    time_horizon = st.selectbox(
        "⏱️ Horizonte de Análisis",
        ("Corto Plazo (Trading)", "Medio Plazo (Swing)", "Largo Plazo (Inversión)")
    )
    
    # Lógica de configuración según selección
    if "Corto" in time_horizon:
        window_config = {'volatility': 10, 'trend': 20, 'download': '3mo', 'desc': 'Días a Semanas'}
    elif "Medio" in time_horizon:
        window_config = {'volatility': 20, 'trend': 50, 'download': '6mo', 'desc': 'Semanas a Meses'}
    else: # Largo
        window_config = {'volatility': 60, 'trend': 200, 'download': '2y', 'desc': 'Meses a Años'}
        
    st.info(f"Análisis ajustado para ventanas de: **{window_config['desc']}**")
    
    tickers = st.text_area("Cartera:", "PLTR, CVX, BTC-USD, SPY, TSLA, AMTB", height=150)
    if st.button("Ejecutar Análisis", type="primary"): st.cache_data.clear()

# 4. ÁREA PRINCIPAL
st.title("Panel de Inteligencia TAI-ACF")

# --- MÓDULO EDUCATIVO (NUEVO) ---
with st.expander("📘 Guía Teórica: ¿Cómo interpreta FAROS el mercado?"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 🎲 Entropía ($H$)")
        st.markdown("""
        Mide el **Caos y el Riesgo**. 
        * Una entropía alta significa que el precio es impredecible (Fase Gaseosa).
        * Buscamos entropía baja para operar con seguridad.
        """)
    with c2:
        st.markdown("### 🌊 Liquidez ($L$)")
        st.markdown("""
        Mide la **Energía y el Flujo**.
        * Es el combustible del movimiento. 
        * Si el precio sube sin liquidez, es una trampa (Plasma).
        * Necesitamos volumen creciente para confirmar tendencias.
        """)
    with c3:
        st.markdown("### 🧠 Gobernanza ($\Psi$)")
        st.markdown("""
        Es la **Señal de Decisión**.
        * El algoritmo combina $H$ y $L$.
        * Si el mercado está ordenado ($H$ baja) y hay energía ($L$ alta), la Gobernanza autoriza la **COMPRA**.
        """)

st.markdown("---")

# EJECUCIÓN DEL MODELO
df = get_quant_data(tickers, window_config)

if not df.empty:
    col_list, col_radar = st.columns([1.5, 1])
    
    with col_radar:
        st.subheader(f"🧭 Radar ({window_config['desc']})")
        fig = px.scatter(df, x="Entropy", y="Liquidity", color="Category", text="Ticker",
                         color_discrete_map={"success":"#28a745", "info":"#17a2b8", "neutral":"#6c757d", "warning":"#ffc107", "danger":"#dc3545"},
                         labels={"Entropy": "Caos (Riesgo)", "Liquidity": "Flujo (Energía)"})
        fig.update_layout(template="plotly_white", height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"*Mapa calculado con ventanas de volatilidad de {window_config['volatility']} días.*")

    with col_list:
        st.subheader("📋 Matriz de Decisión")
        
        for i, row in df.iterrows():
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                c1.markdown(f"### **{row['Ticker']}**") 
                c2.markdown(f"### ${row['Price']:.2f}")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Entropía", f"{row['Entropy']:.1f}σ")
                m2.metric("Liquidez", f"{row['Liquidity']*100:+.0f}%")
                m3.metric("Tendencia", f"{row['Trend']:+.1f}%")
                
                # Mensaje dinámico según temporalidad
                msg = f"**{row['Signal']} ({window_config['desc']}):** {row['Narrative']}"
                
                if row['Category'] == 'success': st.success(msg, icon="✅")
                elif row['Category'] == 'info': st.info(msg, icon="ℹ️")
                elif row['Category'] == 'warning': st.warning(msg, icon="⚠️")
                elif row['Category'] == 'danger': st.error(msg, icon="⛔")
                else: st.write(msg)

else:
    st.info("Cargando datos... un momento.")
