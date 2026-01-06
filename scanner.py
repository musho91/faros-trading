# ==============================================================================
# FAROS v4.0 - QUANTITATIVE PRIME EDITION
# Autor: Juan Arroyo | SG Consulting Group
# Enfoque: Señales Claras, Rigor Matemático, UI Limpia
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf

# --- 1. INGENIERÍA VISUAL (High-End Fintech) ---
st.set_page_config(page_title="FAROS | Quant Terminal", page_icon="📡", layout="wide")

st.markdown("""
<style>
    /* Fondo y Tipografía General */
    .stApp { background-color: #080808; color: #E0E0E0; font-family: 'Inter', sans-serif; }
    
    /* Títulos */
    h1, h2, h3 { font-family: 'Roboto Mono', monospace; font-weight: 600; letter-spacing: -0.5px; }
    
    /* TARJETA DE ACTIVO (Asset Card) - El núcleo del UX */
    .quant-card {
        background-color: #121212;
        border-left: 5px solid #333;
        border-radius: 4px;
        padding: 18px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: transform 0.2s;
    }
    .quant-card:hover { transform: translateX(5px); }
    
    /* Badges de Señal */
    .signal-badge {
        padding: 4px 10px;
        border-radius: 2px;
        font-weight: 800;
        font-size: 0.85rem;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    
    /* Métricas secundarias */
    .metric-label { color: #666; font-size: 0.75rem; text-transform: uppercase; }
    .metric-value { color: #FFF; font-size: 0.95rem; font-family: 'Roboto Mono', monospace; }
    .quant-desc { color: #AAA; font-size: 0.9rem; margin-top: 8px; line-height: 1.4; border-top: 1px solid #222; padding-top: 8px;}
</style>
""", unsafe_allow_html=True)

# --- 2. MOTOR LÓGICO (QUANT CORE) ---
@st.cache_data(ttl=300)
def get_quant_data(tickers_input):
    tickers_list = [x.strip().upper() for x in tickers_input.split(',')]
    data_list = []

    for ticker in tickers_list:
        try:
            # Descarga de datos (6 meses para robustez estadística)
            stock = yf.Ticker(ticker)
            hist = stock.history(period="6mo")
            
            if len(hist) > 20:
                # --- MATEMÁTICA TAI-ACF ---
                current_price = hist['Close'].iloc[-1]
                
                # 1. Entropía (H): Volatilidad normalizada (Z-Score)
                returns = hist['Close'].pct_change().dropna()
                volatility_20d = returns.std() * np.sqrt(20) * 100
                avg_volatility = returns.std() * np.sqrt(252) * 100 # Anualizada
                z_entropy = (volatility_20d - 5) / 2 # Baseline de mercado ~5%
                
                # 2. Liquidez (L): Momentum de Volumen
                vol_sma_20 = hist['Volume'].rolling(20).mean().iloc[-1]
                curr_vol = hist['Volume'].iloc[-1]
                z_liquidity = (curr_vol - vol_sma_20) / vol_sma_20 # % sobre la media
                
                # 3. Tendencia (Trend): Posición respecto a medias
                sma_50 = hist['Close'].rolling(50).mean().iloc[-1]
                trend_strength = (current_price - sma_50) / sma_50
                
                # --- GENERADOR DE SEÑALES ---
                signal = "HOLD"
                color = "#888888" # Gris
                narrative = "Activo en equilibrio termodinámico. Sin vectores de fuerza claros."
                
                # Lógica de Decisión Experta
                if z_entropy > 2.0:
                    signal = "VETO (RIESGO)"
                    color = "#FF3333" # Rojo
                    narrative = f"CRÍTICO: Ruptura de estructura. La entropía ({z_entropy:.1f}σ) indica comportamiento caótico (Fase Gaseosa). Alta probabilidad de drawdowns severos."
                
                elif z_liquidity < -0.3 and trend_strength < -0.05:
                    signal = "SELL / EXIT"
                    color = "#FF8800" # Naranja
                    narrative = "Drenaje de liquidez detectado. El capital institucional está rotando fuera del activo. Trampa de valor potencial."
                
                elif trend_strength > 0.05 and z_entropy < 1.0:
                    if z_liquidity > 0.2:
                        signal = "STRONG BUY"
                        color = "#00FF41" # Neon Green
                        narrative = f"ALFA PURO: Convergencia de baja entropía y alta inyección de capital (+{z_liquidity*100:.0f}% vol). El activo está en Fase Líquida Expansiva."
                    else:
                        signal = "ACCUMULATE"
                        color = "#00CCFF" # Cyan
                        narrative = "Tendencia sólida con volatilidad controlada. Ideal para construir posición escalonada (DCA)."

                data_list.append({
                    "Ticker": ticker,
                    "Price": current_price,
                    "Signal": signal,
                    "Color": color,
                    "Narrative": narrative,
                    "Entropy": z_entropy,
                    "Liquidity": z_liquidity,
                    "Trend": trend_strength * 100
                })
        except Exception:
            pass

    # Ordenar por "Signal strength" (Strong Buy primero)
    df = pd.DataFrame(data_list)
    if not df.empty:
        priority_map = {"STRONG BUY": 0, "ACCUMULATE": 1, "HOLD": 2, "SELL / EXIT": 3, "VETO (RIESGO)": 4}
        df['Priority'] = df['Signal'].map(priority_map)
        df = df.sort_values('Priority')
    return df

# --- 3. INTERFAZ DE USUARIO (UX) ---

# Sidebar Limpio
with st.sidebar:
    st.title("📡 FAROS Q-SYS")
    st.caption("Quantitative System for Asset Allocation")
    st.markdown("---")
    tickers = st.text_area("CARTERA DE VIGILANCIA:", 
                           "PLTR, BTC-USD, CVX, TSLA, NVDA, SPY, AMTB", height=150)
    if st.button("EJECUTAR ANÁLISIS ⚡", use_container_width=True):
        st.cache_data.clear()

# Main Area
st.title("Matriz de Decisión TAI-ACF")
st.markdown("*Análisis de Termodinámica Financiera en Tiempo Real*")

# Carga de Datos
df = get_quant_data(tickers)

if not df.empty:
    
    # 1. KPI RESUMEN (Metrics Row)
    kpi1, kpi2, kpi3 = st.columns(3)
    best_asset = df.iloc[0]
    kpi1.metric("Mejor Oportunidad", best_asset['Ticker'], best_asset['Signal'])
    kpi2.metric("Nivel de Entropía Global", "BAJO" if df['Entropy'].mean() < 1 else "ALTO", delta_color="inverse")
    kpi3.metric("Activos Analizados", len(df))
    
    st.markdown("---")

    # 2. LISTA DE ACCIÓN (Action List) - Aquí está el UX mejorado
    # Usamos columnas para Radar (Pequeño) vs Lista (Grande)
    
    col_list, col_radar = st.columns([2, 1])
    
    with col_radar:
        st.markdown("###### 🗺️ RADAR DE FASES")
        fig = px.scatter(df, x="Entropy", y="Liquidity", color="Signal", text="Ticker",
                         color_discrete_map={
                             "STRONG BUY": "#00FF41", "ACCUMULATE": "#00CCFF",
                             "HOLD": "#888888", "SELL / EXIT": "#FF8800", "VETO (RIESGO)": "#FF3333"
                         })
        fig.update_layout(template="plotly_dark", height=350, margin=dict(l=0,r=0,t=0,b=0),
                          xaxis_title="Caos (H)", yaxis_title="Flujo (L)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 **Guía Rápida:** Buscamos activos en la zona superior izquierda (Alto Flujo, Bajo Caos).")

    with col_list:
        st.markdown("###### 📋 REPORTE DE EJECUCIÓN")
        
        for index, row in df.iterrows():
            # Renderizado HTML de la tarjeta
            st.markdown(f"""
            <div class="quant-card" style="border-left-color: {row['Color']};">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span style="font-size:1.4rem; font-weight:700; color:#FFF;">{row['Ticker']}</span>
                        <span style="font-size:1rem; color:#888; margin-left:10px;">${row['Price']:.2f}</span>
                    </div>
                    <div class="signal-badge" style="background-color:{row['Color']}20; color:{row['Color']}; border:1px solid {row['Color']};">
                        {row['Signal']}
                    </div>
                </div>
                
                <div style="display:flex; gap: 20px; margin-top:10px;">
                    <div><span class="metric-label">ENTROPÍA (σ)</span><br><span class="metric-value">{row['Entropy']:.2f}</span></div>
                    <div><span class="metric-label">MOMENTUM ($)</span><br><span class="metric-value">{row['Liquidity']*100:+.1f}%</span></div>
                    <div><span class="metric-label">TENDENCIA</span><br><span class="metric-value">{row['Trend']:+.1f}%</span></div>
                </div>
                
                <div class="quant-desc">
                    ➤ <b>Análisis:</b> {row['Narrative']}
                </div>
            </div>
            """, unsafe_allow_html=True)

else:
    st.warning("Inicializando sistemas... Por favor espera o verifica los tickers.")
