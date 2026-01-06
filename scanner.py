# ==============================================================================
# FAROS v3.5 - TAI-ACF COMPACT DESIGN
# Autor: Juan Arroyo | SG Consulting Group
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf

# --- 1. ESTILO VISUAL (CLEAN & COMPACT) ---
st.set_page_config(page_title="FAROS", page_icon="📡", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #E0E0E0; }
    /* Tarjetas de activos más pequeñas y elegantes */
    .asset-card {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .phase-tag { font-weight: bold; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; }
    /* Ajuste de textos */
    h1 { font-size: 1.8rem !important; }
    h3 { font-size: 1.2rem !important; }
    p { font-size: 0.9rem !important; }
</style>
""", unsafe_allow_html=True)

# --- 2. MOTOR DE CÁLCULO ---
@st.cache_data(ttl=600)
def get_live_data(tickers_input):
    tickers_list = [x.strip().upper() for x in tickers_input.split(',')]
    data_list = []

    for ticker in tickers_list:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="3mo")
            
            if len(hist) > 0:
                current_price = hist['Close'].iloc[-1]
                returns = hist['Close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(20) * 100 
                z_entropy = (volatility - 5) / 2
                
                avg_vol = hist['Volume'].mean()
                curr_vol = hist['Volume'].iloc[-1]
                z_liquidity = np.log(curr_vol / avg_vol) if avg_vol > 0 else 0
                
                # DEFINICIÓN DE ESTADOS Y DESCRIPCIONES HUMANAS
                if z_entropy > 2.5: 
                    fase = "GAS (CAOS)"
                    desc = "Alta volatilidad e incertidumbre. Riesgo de pérdidas rápidas. El precio no respeta soportes."
                    color = "#FF4B4B" # Rojo suave
                elif z_liquidity < -1.0: 
                    fase = "PLASMA (ILIQUIDO)"
                    desc = "Poco volumen. Difícil entrar o salir sin mover el precio. Posible trampa."
                    color = "#FFAA00" # Amarillo
                elif z_liquidity > 0 and z_entropy < 2.0: 
                    fase = "LÍQUIDO (ÓPTIMO)"
                    desc = "Flujo constante de dinero institucional. Tendencia saludable y predecible."
                    color = "#00CC96" # Verde menta
                else: 
                    fase = "SÓLIDO (ESTABLE)"
                    desc = "Baja volatilidad. El precio se mueve lento o lateral. Seguro, pero con poco retorno explosivo."
                    color = "#A6A6A6" # Gris

                data_list.append({
                    "Ticker": ticker,
                    "Precio": current_price,
                    "Fase": fase,
                    "Desc": desc,
                    "Color": color,
                    "Z_Entropia": z_entropy,
                    "Z_Liquidez": z_liquidity
                })
        except:
            continue

    return pd.DataFrame(data_list)

# --- 3. INTERFAZ DE USUARIO ---
with st.sidebar:
    st.header("📡 FAROS")
    user_tickers = st.text_area("Activos:", value="PLTR, CVX, BTC-USD, SPY, TSLA", height=80)
    st.caption("Escribe los tickers separados por coma.")
    if st.button("Actualizar"): st.cache_data.clear()

# Cargar datos
df = get_live_data(user_tickers)

if not df.empty:
    st.subheader("Mapas de Estado TAI-ACF")
    
    # --- DISEÑO DIVIDIDO (Izquierda: Radar Pequeño | Derecha: Lista Detallada) ---
    col_izq, col_der = st.columns([1, 1.5]) # Proporción 40% / 60%
    
    with col_izq:
        st.markdown("**Radar Termodinámico**")
        # Radar más pequeño (Height 350px) y limpio
        fig = px.scatter(df, x="Z_Entropia", y="Z_Liquidez", color="Fase", 
                         hover_name="Ticker", size_max=15,
                         color_discrete_map={
                             "LÍQUIDO (ÓPTIMO)": "#00CC96", 
                             "SÓLIDO (ESTABLE)": "#A6A6A6", 
                             "GAS (CAOS)": "#FF4B4B", 
                             "PLASMA (ILIQUIDO)": "#FFAA00"
                         })
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=350, # Altura reducida
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showgrid=False, title="Riesgo (Entropía)"),
            yaxis=dict(showgrid=False, title="Volumen (Liquidez)"),
            legend=dict(orientation="h", y=-0.2)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_der:
        st.markdown("**Diagnóstico por Activo**")
        # Lista de tarjetas compactas
        for index, row in df.iterrows():
            with st.container():
                # HTML personalizado para diseño compacto
                st.markdown(f"""
                <div class="asset-card" style="border-left: 4px solid {row['Color']};">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h3 style="margin:0; color:white;">{row['Ticker']} <span style="font-size:0.7em; color:#888;">${row['Precio']:.2f}</span></h3>
                        <span class="phase-tag" style="background-color:{row['Color']}20; color:{row['Color']}; border:1px solid {row['Color']};">{row['Fase']}</span>
                    </div>
                    <p style="margin-top:5px; margin-bottom:0; color:#BBB; font-size:0.85rem;">
                        <i>"{row['Desc']}"</i>
                    </p>
                </div>
                """, unsafe_allow_html=True)
else:
    st.info("Esperando datos... revisa los tickers en la barra lateral.")
