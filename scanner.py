# ==============================================================================
# FAROS v5.0 - INSTITUTIONAL LIGHT EDITION
# Autor: Juan Arroyo | SG Consulting Group
# Estilo: Clean, White, Professional (High Contrast)
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf

# --- 1. ESTILO VISUAL (INSTITUTIONAL CLEAN) ---
st.set_page_config(page_title="FAROS | Quant Terminal", page_icon="📡", layout="wide")

st.markdown("""
<style>
    /* FONDO BLANCO Y TEXTO OSCURO */
    .stApp { 
        background-color: #FFFFFF; 
        color: #111111; 
        font-family: 'Helvetica Neue', 'Arial', sans-serif;
    }
    
    /* TARJETAS DE ACTIVOS (Estilo "Paper") */
    .quant-card {
        background-color: #F8F9FA; /* Gris muy suave */
        border: 1px solid #E9ECEF;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05); /* Sombra sutil */
        transition: all 0.2s ease;
    }
    .quant-card:hover { 
        box-shadow: 0 4px 12px rgba(0,0,0,0.1); 
        transform: translateY(-2px);
        border-color: #CED4DA;
    }
    
    /* BADGES DE SEÑAL (Colores Sólidos) */
    .signal-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.8rem;
        color: white;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* TEXTOS */
    h1, h2, h3 { color: #212529; font-weight: 700; }
    .ticker-title { font-size: 1.5rem; font-weight: 800; color: #212529; }
    .price-tag { font-size: 1.1rem; color: #495057; font-family: 'Courier New', monospace; font-weight: bold; }
    .metric-label { font-size: 0.75rem; color: #6C757D; text-transform: uppercase; font-weight: 600; }
    .metric-value { font-size: 1rem; color: #000; font-weight: 700; }
    .analysis-text { color: #343A40; font-size: 0.95rem; margin-top: 10px; line-height: 1.5; border-top: 1px solid #E9ECEF; padding-top: 10px;}
    
</style>
""", unsafe_allow_html=True)

# --- 2. MOTOR LÓGICO (QUANT CORE) ---
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
                
                # --- LOGICA DE SEÑALES (Colores Institucionales) ---
                # Usamos colores sólidos, no neón.
                signal = "HOLD"
                bg_color = "#6C757D" # Gris Bootstrap
                narrative = "Activo en rango lateral. Sin catalizadores claros."
                
                if z_entropy > 2.0:
                    signal = "VETO / RIESGO"
                    bg_color = "#DC3545" # Rojo Profundo
                    narrative = f"⚠️ Alta Entropía ({z_entropy:.1f}σ). Estructura de precios rota. Evitar exposición hasta estabilización."
                
                elif z_liquidity < -0.3 and trend_strength < -0.05:
                    signal = "SELL / SALIDA"
                    bg_color = "#FD7E14" # Naranja
                    narrative = "📉 Salida de flujo institucional. El volumen decrece en las subidas. Distribución detectada."
                
                elif trend_strength > 0.05 and z_entropy < 1.0:
                    if z_liquidity > 0.2:
                        signal = "STRONG BUY"
                        bg_color = "#28A745" # Verde Éxito (Sólido)
                        narrative = f"🚀 **Fase Líquida Confirmada.** Combinación de baja volatilidad y alto volumen entrante (+{z_liquidity*100:.0f}%)."
                    else:
                        signal = "ACUMULAR"
                        bg_color = "#007BFF" # Azul Institucional
                        narrative = "Tendencia alcista estable. Zona segura para compras escalonadas."

                data_list.append({
                    "Ticker": ticker,
                    "Price": current_price,
                    "Signal": signal,
                    "BgColor": bg_color,
                    "Narrative": narrative,
                    "Entropy": z_entropy,
                    "Liquidity": z_liquidity,
                    "Trend": trend_strength * 100
                })
        except:
            pass

    df = pd.DataFrame(data_list)
    if not df.empty:
        # Priorizar compras
        priority_map = {"STRONG BUY": 0, "ACUMULAR": 1, "HOLD": 2, "SELL / SALIDA": 3, "VETO / RIESGO": 4}
        df['Priority'] = df['Signal'].map(priority_map)
        df = df.sort_values('Priority')
    return df

# --- 3. INTERFAZ DE USUARIO ---

# Sidebar Blanco
with st.sidebar:
    st.header("📡 FAROS")
    st.info("Institutional Access v5.0")
    tickers = st.text_area("Cartera de Análisis:", 
                           "PLTR, BTC-USD, CVX, TSLA, SPY, AMTB, NVDA", height=150)
    if st.button("Actualizar Análisis"):
        st.cache_data.clear()

# Main Area
st.title("Matriz de Decisión TAI-ACF")
st.markdown("### Reporte de Termodinámica Financiera")
st.markdown("---")

df = get_quant_data(tickers)

if not df.empty:
    
    col_list, col_radar = st.columns([1.8, 1])
    
    with col_radar:
        # RADAR EN MODO BLANCO (PLOTLY WHITE)
        st.markdown("#### 🧭 Radar de Fases")
        fig = px.scatter(df, x="Entropy", y="Liquidity", color="Signal", text="Ticker",
                         color_discrete_map={
                             "STRONG BUY": "#28A745", "ACUMULAR": "#007BFF",
                             "HOLD": "#6C757D", "SELL / SALIDA": "#FD7E14", "VETO / RIESGO": "#DC3545"
                         })
        fig.update_layout(
            template="plotly_white", # CLAVE: Fondo blanco para el gráfico
            height=400,
            margin=dict(l=0,r=0,t=0,b=0),
            xaxis_title="Caos / Riesgo (H)", 
            yaxis_title="Flujo de Capital (L)", 
            showlegend=False,
            font=dict(color="#000")
        )
        fig.update_traces(textposition='top center', marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
        st.plotly_chart(fig, use_container_width=True)
        
        with st.container():
            st.markdown("""
            <div style="background-color:#E9ECEF; padding:15px; border-radius:5px; font-size:0.85rem; color:#495057;">
                <b>Interpretación:</b><br>
                🟢 <b>Arriba-Izquierda:</b> Zona ideal (Compra).<br>
                🔴 <b>Abajo-Derecha:</b> Zona de peligro (Venta).
            </div>
            """, unsafe_allow_html=True)

    with col_list:
        st.markdown("#### 📋 Detalles de Ejecución")
        
        for index, row in df.iterrows():
            # CARD DESIGN ON WHITE
            st.markdown(f"""
            <div class="quant-card" style="border-left: 5px solid {row['BgColor']};">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span class="ticker-title">{row['Ticker']}</span>
                        <span class="price-tag" style="margin-left:10px;">${row['Price']:.2f}</span>
                    </div>
                    <span class="signal-badge" style="background-color:{row['BgColor']};">
                        {row['Signal']}
                    </span>
                </div>
                
                <div style="display:flex; justify-content:space-between; margin-top:15px; max-width:90%;">
                    <div><div class="metric-label">ENTROPÍA (σ)</div><div class="metric-value">{row['Entropy']:.2f}</div></div>
                    <div><div class="metric-label">VOLUMEN ($)</div><div class="metric-value" style="color:{'#28A745' if row['Liquidity']>0 else '#DC3545'};">{row['Liquidity']*100:+.1f}%</div></div>
                    <div><div class="metric-label">TENDENCIA</div><div class="metric-value">{row['Trend']:+.1f}%</div></div>
                </div>
                
                <div class="analysis-text">
                    {row['Narrative']}
                </div>
            </div>
            """, unsafe_allow_html=True)

else:
    st.info("Cargando datos de mercado... Por favor espere.")
