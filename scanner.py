# ==============================================================================
# FAROS v7.2 - INTELLIGENCE PLATFORM (FINAL PRODUCTION)
# Autor: Juan Arroyo | SG Consulting Group
# Características: Diseño Nativo, Multi-Frame, Lógica "Growth Exception"
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf

# --- 1. CONFIGURACIÓN VISUAL (MODO LIMPIO) ---
st.set_page_config(page_title="FAROS | TAI-ACF", page_icon="📡", layout="wide")

# CSS Mínimo para forzar limpieza visual
st.markdown("""
<style>
    .stApp { background-color: #FFFFFF; color: #111; }
    h1, h2, h3 { color: #000 !important; }
    /* Ajuste sutil para las métricas */
    div[data-testid="stMetricValue"] { font-size: 1.1rem; }
    div[data-testid="stMetricLabel"] { font-size: 0.8rem; color: #666; }
</style>
""", unsafe_allow_html=True)

# --- 2. MOTOR LÓGICO TAI-ACF (CALIBRADO) ---
@st.cache_data(ttl=300)
def get_quant_data(tickers_input, window_cfg):
    tickers_list = [x.strip().upper() for x in tickers_input.split(',')]
    data_list = []
    
    # Desempaquetar configuración de ventanas (Timeframe)
    w_calc = window_cfg['volatility'] # Días para cálculo de volatilidad
    w_trend = window_cfg['trend']     # Días para tendencia (SMA)
    period_dl = window_cfg['download'] # Cuanta data bajar de Yahoo

    for ticker in tickers_list:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period_dl)
            
            if len(hist) > w_trend:
                current_price = hist['Close'].iloc[-1]
                
                # --- A. CÁLCULO DE ENTROPÍA (RIESGO) ---
                returns = hist['Close'].pct_change().dropna()
                
                # Tomamos la volatilidad de la ventana seleccionada (ej. 20 días)
                # IMPORTANTE: La anualizamos (* sqrt(252)) para que la escala sea universal
                subset_returns = returns.tail(w_calc)
                if len(subset_returns) > 1:
                    vol_annualized = subset_returns.std() * np.sqrt(252) * 100
                else:
                    vol_annualized = 0
                
                # Z-Score calibrado: (Volatilidad - Base 20%) / Desviación 15%
                z_entropy = (vol_annualized - 20) / 15 
                
                # --- B. CÁLCULO DE LIQUIDEZ (ENERGÍA) ---
                # Comparamos volumen actual vs media de la ventana
                vol_avg = hist['Volume'].rolling(w_calc).mean().iloc[-1]
                curr_vol = hist['Volume'].iloc[-1]
                z_liquidity = (curr_vol - vol_avg) / vol_avg if vol_avg > 0 else 0
                
                # --- C. CÁLCULO DE TENDENCIA (GOBERNANZA) ---
                sma_trend = hist['Close'].rolling(w_trend).mean().iloc[-1]
                trend_strength = (current_price - sma_trend) / sma_trend
                
                # --- D. ÁRBOL DE DECISIÓN (CALIBRADO PARA PLTR) ---
                signal = "MANTENER"
                category = "neutral" 
                narrative = "Consolidación. Esperando definición."

                # 1. FILTRO DE ENTROPÍA (ALTA VOLATILIDAD)
                if z_entropy > 2.0:
                    # AJUSTE CRÍTICO: Bajamos el umbral de 0.15 a 0.10
                    # Si la tendencia es > 10% (como PLTR con 14.9%), es Growth.
                    # Ignoramos la liquidez negativa temporalmente.
                    if trend_strength > 0.10: 
                        signal = "GROWTH AGRESIVO"
                        category = "warning"
                        narrative = f"⚡ Tendencia muy fuerte (+{trend_strength*100:.1f}%) supera a la alta volatilidad. Momentum puro."
                    
                    # Si la tendencia es suave (entre 2% y 10%) PERO el volumen es bueno:
                    elif trend_strength > 0.02 and z_liquidity > -0.15:
                        signal = "GROWTH (VOLÁTIL)"
                        category = "warning"
                        narrative = f"⚡ Volatilidad alta, pero el precio se sostiene con volumen aceptable."
                    
                    else:
                        # Solo si NO hay tendencia fuerte (<10%) Y la entropía es alta
                        signal = "RIESGO ALTO"
                        category = "danger"
                        narrative = f"⚠️ Estructura Inestable. Alta entropía ({z_entropy:.1f}σ) sin suficiente fuerza de tendencia."
                
                # 2. SEÑALES DE SALIDA (Para activos estables que se rompen)
                elif z_liquidity < -0.2 and trend_strength < -0.05:
                    signal = "VENTA / SALIDA"
                    category = "warning"
                    narrative = "📉 Debilidad confirmada. El precio cae y el volumen valida la salida."
                
                # 3. SEÑALES DE ENTRADA (Mercado Ideal)
                elif trend_strength > 0.02:
                    if z_liquidity > 0.10:
                        signal = "COMPRA FUERTE"
                        category = "success"
                        narrative = f"🚀 Fase Líquida Óptima. Baja volatilidad + Entrada de dinero (+{z_liquidity*100:.0f}%)."
                    else:
                        signal = "ACUMULAR"
                        category = "info"
                        narrative = "📈 Tendencia alcista sana. Zona de compra tranquila."
                data_list.append({
                    "Ticker": ticker, 
                    "Price": current_price, 
                    "Signal": signal, 
                    "Category": category, 
                    "Narrative": narrative,
                    "Entropy": z_entropy, 
                    "Liquidity": z_liquidity, 
                    "Trend": trend_strength * 100,
                    "Vol_Ann": vol_annualized
                })
        except Exception:
            pass # Si falla un ticker, continuamos con el siguiente

    df = pd.DataFrame(data_list)
    if not df.empty:
        # Ordenar por prioridad de acción: Success (0) -> Info (1) -> Warning (2) -> Danger (3) -> Neutral (4)
        # Ajustamos el mapa para que 'Warning' (Growth) salga antes que 'Danger'
        prio = {"success": 0, "info": 1, "warning": 2, "neutral": 3, "danger": 4}
        df['P'] = df['Category'].map(prio)
        df = df.sort_values('P')
    return df

# --- 3. BARRA LATERAL (CONTROLES) ---
with st.sidebar:
    st.header("📡 CONFIGURACIÓN")
    
    # Selector de Temporalidad
    time_horizon = st.selectbox(
        "⏱️ Horizonte de Inversión",
        ("Corto Plazo (Trading)", "Medio Plazo (Swing)", "Largo Plazo (Inversión)")
    )
    
    # Lógica de configuración de ventanas
    if "Corto" in time_horizon:
        # Muy reactivo: 10 días volatilidad, 20 días tendencia
        window_config = {'volatility': 10, 'trend': 20, 'download': '3mo', 'desc': 'Días a Semanas'}
    elif "Medio" in time_horizon:
        # Estándar: 20 días volatilidad (mensual), 50 días tendencia
        window_config = {'volatility': 20, 'trend': 50, 'download': '6mo', 'desc': 'Semanas a Meses'}
    else: 
        # Inversión: 60 días volatilidad (trimestral), 200 días tendencia
        window_config = {'volatility': 60, 'trend': 200, 'download': '2y', 'desc': 'Meses a Años'}
        
    st.caption(f"Ventanas ajustadas para: {window_config['desc']}")
    
    st.markdown("---")
    tickers = st.text_area("Cartera de Vigilancia:", 
                          "PLTR, BTC-USD, CVX, SPY, TSLA, AMTB, NVDA, MELI", 
                          height=150)
    
    if st.button("Ejecutar Análisis", type="primary"): 
        st.cache_data.clear()

# --- 4. ÁREA PRINCIPAL ---
st.title("Panel de Inteligencia TAI-ACF")

# Módulo Educativo (Desplegable)
with st.expander("📘 Guía Teórica: Entendiendo los Indicadores"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**🎲 Entropía (Riesgo)**")
        st.caption("Mide el caos. Valores altos (>2.0σ) indican inestabilidad, a menos que sea crecimiento explosivo.")
    with c2:
        st.markdown("**🌊 Liquidez (Energía)**")
        st.caption("Mide el flujo de dinero. Necesitamos valores positivos para confirmar que la subida es real.")
    with c3:
        st.markdown("**🧠 Gobernanza (Señal)**")
        st.caption("La decisión final del algoritmo basada en su horizonte temporal seleccionado.")

st.markdown("---")

# Ejecución
df = get_quant_data(tickers, window_config)

if not df.empty:
    # Diseño: Lista de Recomendaciones (Izquierda) + Radar (Derecha)
    col_list, col_radar = st.columns([1.6, 1])
    
    # --- SECCIÓN DERECHA: RADAR VISUAL ---
    with col_radar:
        st.subheader("🧭 Radar de Fases")
        fig = px.scatter(df, x="Entropy", y="Liquidity", color="Category", text="Ticker",
                         color_discrete_map={
                             "success":"#28a745", # Verde
                             "info":"#17a2b8",    # Azul
                             "neutral":"#6c757d", # Gris
                             "warning":"#ffc107", # Amarillo/Naranja
                             "danger":"#dc3545"   # Rojo
                         },
                         labels={"Entropy": "Caos / Riesgo (σ)", "Liquidity": "Flujo de Dinero"})
        
        fig.update_layout(template="plotly_white", height=450, showlegend=False)
        fig.update_traces(textposition='top center', marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
        
        # Zonas de referencia visual
        fig.add_vline(x=2.0, line_width=1, line_dash="dash", line_color="red", opacity=0.3)
        fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray", opacity=0.3)
        
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"Visualizando horizonte de **{time_horizon}**.")

    # --- SECCIÓN IZQUIERDA: TARJETAS DE ACCIÓN ---
    with col_list:
        st.subheader("📋 Matriz de Decisión")
        
        for i, row in df.iterrows():
            # Contenedor con borde (Diseño Nativo Limpio)
            with st.container(border=True):
                
                # Cabecera: Ticker y Precio
                head_c1, head_c2 = st.columns([3, 1])
                head_c1.markdown(f"### **{row['Ticker']}**") 
                head_c2.markdown(f"### ${row['Price']:,.2f}")
                
                # Métricas Clave
                m1, m2, m3 = st.columns(3)
                m1.metric("Entropía (Vol)", f"{row['Entropy']:.1f}σ", help=f"Volatilidad Anualizada: {row['Vol_Ann']:.0f}%")
                m2.metric("Liquidez", f"{row['Liquidity']*100:+.0f}%", delta_color="normal")
                m3.metric("Tendencia", f"{row['Trend']:+.1f}%")
                
                # Mensaje de IA (Alertas Nativas)
                msg = f"**{row['Signal']}:** {row['Narrative']}"
                
                if row['Category'] == 'success':
                    st.success(msg, icon="✅")
                elif row['Category'] == 'info':
                    st.info(msg, icon="ℹ️")
                elif row['Category'] == 'warning':
                    st.warning(msg, icon="⚠️") # Cubre tanto "Venta" como "Growth Agresivo"
                elif row['Category'] == 'danger':
                    st.error(msg, icon="⛔")
                else:
                    st.write(msg)

else:
    st.info("⏳ Inicializando sistemas... Por favor espere un momento.")


