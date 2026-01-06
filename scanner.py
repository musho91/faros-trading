# ==============================================================================
# FAROS v5.1 - INSTITUTIONAL PRO (BETTER UX)
# Autor: Juan Arroyo | SG Consulting Group
# Mejora: Descripciones encapsuladas y diseño limpio
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf

# --- 1. ESTILO VISUAL (CLEAN & STRUCTURED) ---
st.set_page_config(page_title="FAROS | Quant Terminal", page_icon="📡", layout="wide")

st.markdown("""
<style>
    /* GENERAL */
    .stApp { background-color: #FFFFFF; color: #111; font-family: 'Helvetica Neue', sans-serif; }
    
    /* TARJETA PRINCIPAL */
    .quant-card {
        background-color: #F8F9FA; /* Gris muy pálido */
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    /* CAJA DE DESCRIPCIÓN (INSIGHT BOX) - AQUÍ ESTÁ EL CAMBIO */
    .insight-box {
        background-color: #FFFFFF; /* Blanco puro */
        border-left: 4px solid #CCC; /* Borde de color variable */
        padding: 12px 15px;
        margin-top: 15px;
        border-radius: 0 6px 6px 0;
        font-size: 0.95rem;
        color: #333;
        line-height: 1.5;
        display: flex;
        align-items: center;
        gap: 12px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    
    /* ELEMENTOS */
    .ticker-header { font-size: 1.6rem; font-weight: 800; color: #222; }
    .price-tag { font-family: 'Courier New', monospace; font-weight: 600; color: #555; font-size: 1.1rem; }
    
    .metric-container { display: flex; justify-content: space-between; margin-top: 15px; padding-bottom: 5px; border-bottom: 1px solid #EEE; }
    .metric-item { text-align: center; }
    .metric-lbl { font-size: 0.7rem; text-transform: uppercase; color: #888; letter-spacing: 1px; font-weight: 600; }
    .metric-val { font-size: 1rem; font-weight: 700; color: #000; }
    
    .badge { padding: 5px 12px; border-radius: 4px; font-weight: bold; font-size: 0.8rem; color: white; text-transform: uppercase; letter-spacing: 0.5px; }
</style>
""", unsafe_allow_html=True)

# --- 2. MOTOR LÓGICO ---
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
                
                # Cálculos
                returns = hist['Close'].pct_change().dropna()
                volatility_20d = returns.std() * np.sqrt(20) * 100
                z_entropy = (volatility_20d - 5) / 2 
                vol_sma_20 = hist['Volume'].rolling(20).mean().iloc[-1]
                curr_vol = hist['Volume'].iloc[-1]
                z_liquidity = (curr_vol - vol_sma_20) / vol_sma_20 
                sma_50 = hist['Close'].rolling(50).mean().iloc[-1]
                trend_strength = (current_price - sma_50) / sma_50
                
                # Lógica de Señales y Textos
                signal = "HOLD"
                color = "#6C757D" # Gris neutro
                icon = "⚖️"
                narrative = "El activo se mueve lateralmente. Esperar confirmación de ruptura."

                if z_entropy > 2.0:
                    signal = "ALTO RIESGO"
                    color = "#D32F2F" # Rojo oscuro
                    icon = "⚠️"
                    narrative = f"<b>Precaución:</b> Alta entropía ({z_entropy:.1f}σ). La estructura de precios es inestable. No operar."
                
                elif z_liquidity < -0.3:
                    signal = "SALIDA"
                    color = "#F57C00" # Naranja
                    icon = "📉"
                    narrative = "<b>Drenaje de capital:</b> El volumen está cayendo mientras el precio sube. Señal de debilidad."
                
                elif trend_strength > 0.05 and z_entropy < 1.0:
                    if z_liquidity > 0.2:
                        signal = "STRONG BUY"
                        color = "#2E7D32" # Verde bosque (serio)
                        icon = "🚀"
                        narrative = f"<b>Oportunidad Alfa:</b> Convergencia perfecta. Baja volatilidad + Entrada masiva de dinero (+{z_liquidity*100:.0f}%)."
                    else:
                        signal = "ACUMULAR"
                        color = "#1565C0" # Azul fuerte
                        icon = "📈"
                        narrative = "<b>Tendencia Sana:</b> Comportamiento alcista estable. Ideal para compras escalonadas."

                data_list.append({
                    "Ticker": ticker, "Price": current_price, "Signal": signal, 
                    "Color": color, "Narrative": narrative, "Icon": icon,
                    "Entropy": z_entropy, "Liquidity": z_liquidity, "Trend": trend_strength * 100
                })
        except: pass

    df = pd.DataFrame(data_list)
    if not df.empty:
        prio = {"STRONG BUY": 0, "ACUMULAR": 1, "HOLD": 2, "SALIDA": 3, "ALTO RIESGO": 4}
        df['P'] = df['Signal'].map(prio)
        df = df.sort_values('P')
    return df

# --- 3. INTERFAZ ---
with st.sidebar:
    st.header("📡 FAROS")
    tickers = st.text_area("Cartera:", "PLTR, CVX, BTC-USD, SPY, TSLA, AMTB", height=150)
    if st.button("Actualizar"): st.cache_data.clear()

st.title("Matriz de Decisión TAI-ACF")
df = get_quant_data(tickers)

if not df.empty:
    col_list, col_radar = st.columns([1.5, 1])
    
    with col_radar:
        st.markdown("#### 🧭 Radar")
        fig = px.scatter(df, x="Entropy", y="Liquidity", color="Signal", text="Ticker",
                         color_discrete_map={"STRONG BUY":"#2E7D32", "ACUMULAR":"#1565C0", "HOLD":"#6C757D", "SALIDA":"#F57C00", "ALTO RIESGO":"#D32F2F"})
        fig.update_layout(template="plotly_white", height=350, margin=dict(l=0,r=0,t=0,b=0), xaxis_title="Riesgo", yaxis_title="Flujo", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_list:
        st.markdown("#### 📋 Análisis")
        for i, row in df.iterrows():
            # NUEVO DISEÑO DE TARJETA
            st.markdown(f"""
            <div class="quant-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div><span class="ticker-header">{row['Ticker']}</span> <span class="price-tag">${row['Price']:.2f}</span></div>
                    <span class="badge" style="background-color:{row['Color']};">{row['Signal']}</span>
                </div>
                
                <div class="metric-container">
                    <div class="metric-item"><div class="metric-lbl">ENTROPÍA</div><div class="metric-val">{row['Entropy']:.1f}σ</div></div>
                    <div class="metric-item"><div class="metric-lbl">VOLUMEN</div><div class="metric-val" style="color:{row['Color']};">{row['Liquidity']*100:+.0f}%</div></div>
                    <div class="metric-item"><div class="metric-lbl">TENDENCIA</div><div class="metric-val">{row['Trend']:+.1f}%</div></div>
                </div>
                
                <div class="insight-box" style="border-left-color: {row['Color']};">
                    <div style="font-size:1.5rem;">{row['Icon']}</div>
                    <div>{row['Narrative']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("Cargando datos...")
