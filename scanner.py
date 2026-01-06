# ==============================================================================
# FAROS v6.0 - NATIVE DESIGN (CLEANEST VERSION)
# Autor: Juan Arroyo | SG Consulting Group
# Estilo: 100% Nativo, Blanco, Tipografía Standard
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf

# 1. CONFIGURACIÓN SIMPLE (Modo Wide)
st.set_page_config(page_title="FAROS", page_icon="📡", layout="wide")

# CSS MÍNIMO (Solo para asegurar fondo blanco puro)
st.markdown("""
<style>
    .stApp { background-color: #FFFFFF; }
    h1, h2, h3 { color: #000 !important; }
</style>
""", unsafe_allow_html=True)

# 2. MOTOR LÓGICO (El mismo cerebro potente)
@st.cache_data(ttl=300)
def get_quant_data(tickers_input):
    tickers_list = [x.strip().upper() for x in tickers_input.split(',')]
    data_list = []

    for ticker in tickers_list:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="6mo")
            if len(hist) > 20:
                current_price = hist['Close'].iloc[-1]
                returns = hist['Close'].pct_change().dropna()
                volatility_20d = returns.std() * np.sqrt(20) * 100
                z_entropy = (volatility_20d - 5) / 2 
                
                vol_sma_20 = hist['Volume'].rolling(20).mean().iloc[-1]
                curr_vol = hist['Volume'].iloc[-1]
                z_liquidity = (curr_vol - vol_sma_20) / vol_sma_20 
                
                sma_50 = hist['Close'].rolling(50).mean().iloc[-1]
                trend_strength = (current_price - sma_50) / sma_50
                
                # Lógica de Señales
                signal = "MANTENER"
                category = "neutral" # Para elegir el color de la alerta
                narrative = "Activo en rango lateral. Sin tendencia definida."

                if z_entropy > 2.0:
                    signal = "RIESGO ALTO"
                    category = "danger"
                    narrative = f"⚠️ Alta volatilidad ({z_entropy:.1f}σ). Estructura inestable. No comprar."
                
                elif z_liquidity < -0.3:
                    signal = "SALIDA / VENTA"
                    category = "warning"
                    narrative = "📉 El dinero está saliendo (Volumen bajo). Posible caída."
                
                elif trend_strength > 0.05 and z_entropy < 1.0:
                    if z_liquidity > 0.2:
                        signal = "COMPRA FUERTE"
                        category = "success"
                        narrative = f"🚀 Oportunidad Alfa. Baja volatilidad + Mucho volumen entrando."
                    else:
                        signal = "ACUMULAR"
                        category = "info"
                        narrative = "📈 Tendencia alcista sana. Bueno para comprar poco a poco."

                data_list.append({
                    "Ticker": ticker, "Price": current_price, "Signal": signal, 
                    "Category": category, "Narrative": narrative,
                    "Entropy": z_entropy, "Liquidity": z_liquidity, "Trend": trend_strength * 100
                })
        except: pass

    df = pd.DataFrame(data_list)
    if not df.empty:
        # Ordenar: Success (Compra) primero, Danger (Riesgo) último
        prio = {"success": 0, "info": 1, "neutral": 2, "warning": 3, "danger": 4}
        df['P'] = df['Category'].map(prio)
        df = df.sort_values('P')
    return df

# 3. INTERFAZ DE USUARIO (NATIVA)
with st.sidebar:
    st.header("📡 FAROS")
    tickers = st.text_area("Cartera:", "PLTR, CVX, BTC-USD, SPY, TSLA, AMTB", height=150)
    if st.button("Actualizar Análisis", type="primary"): st.cache_data.clear()

st.title("Panel de Inversión TAI-ACF")
st.markdown("---")

df = get_quant_data(tickers)

if not df.empty:
    col_list, col_radar = st.columns([1.5, 1])
    
    # SECCIÓN DERECHA: RADAR
    with col_radar:
        st.subheader("🧭 Radar de Mercado")
        fig = px.scatter(df, x="Entropy", y="Liquidity", color="Category", text="Ticker",
                         color_discrete_map={"success":"#28a745", "info":"#17a2b8", "neutral":"#6c757d", "warning":"#ffc107", "danger":"#dc3545"})
        fig.update_layout(template="plotly_white", height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # SECCIÓN IZQUIERDA: LISTA DE ACTIVOS (NATIVA)
    with col_list:
        st.subheader("📋 Recomendaciones")
        
        for i, row in df.iterrows():
            # USAMOS "st.container" CON BORDE (NATIVO, ELEGANTE)
            with st.container(border=True):
                
                # Encabezado: Ticker y Precio
                c1, c2 = st.columns([3, 1])
                c1.markdown(f"### **{row['Ticker']}**") 
                c2.markdown(f"### ${row['Price']:.2f}")
                
                # Métricas limpias
                m1, m2, m3 = st.columns(3)
                m1.metric("Riesgo", f"{row['Entropy']:.1f}σ")
                m2.metric("Flujo", f"{row['Liquidity']*100:+.0f}%", delta_color="normal")
                m3.metric("Tendencia", f"{row['Trend']:+.1f}%")
                
                # La "Descripción" usando alertas nativas (Se ve profesional, no código)
                msg = f"**{row['Signal']}:** {row['Narrative']}"
                
                if row['Category'] == 'success':
                    st.success(msg, icon="✅")
                elif row['Category'] == 'info':
                    st.info(msg, icon="ℹ️")
                elif row['Category'] == 'warning':
                    st.warning(msg, icon="⚠️")
                elif row['Category'] == 'danger':
                    st.error(msg, icon="⛔")
                else:
                    st.write(msg)

else:
    st.info("Cargando datos... un momento.")
