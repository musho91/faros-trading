# ==============================================================================
# FAROS v2.0 - TAI-ACF EXECUTIVE DASHBOARD
# Autor: Juan Arroyo | SG Consulting Group
# Framework: Streamlit + Plotly
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# --- 1. CONFIGURACIÓN DE PÁGINA (ESTILO CYBERPUNK) ---
st.set_page_config(
    page_title="FAROS | TAI-ACF System",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para "Dark Mode" financiero
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #00FF00;
    }
    .metric-card {
        background-color: #1e2130;
        border: 1px solid #444;
        padding: 15px;
        border-radius: 5px;
        text-align: center;
    }
    h1, h2, h3 {
        font-family: 'Roboto Mono', monospace;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. MOTOR DE DATOS SIMULADO (MOCK DATA) ---
# En producción, esto se conecta a tu API de EvalIA o Yahoo Finance
def get_mock_data():
    data = {
        'Ticker': ['PLTR', 'CVX', 'VE-BONDS', 'SPY', 'BTC', 'TSLA', 'AMTB', 'LSEG', 'HAG'],
        'Precio': [167.86, 142.50, 32.91, 580.12, 102500, 240.50, 28.40, 115.20, 34.50],
        'Z_Liquidez': [1.2, 0.8, -2.5, 0.9, 1.1, -0.4, -0.2, 0.9, 0.4],  # Eje Y (Estructura)
        'Z_Entropia': [0.4, 0.3, 2.8, 0.5, 1.4, 1.8, 1.1, 0.2, 0.7],     # Eje X (Caos)
        'Flujo_M': [1, 1, 1, 0, 1, -1, 0, 1, 1] # 1=Entrada, -1=Salida
    }
    df = pd.DataFrame(data)
    
    # Clasificación de Fases (Lógica TAI-ACF)
    conditions = [
        (df['Z_Entropia'] > 1.5),              # GAS (Caos/Riesgo)
        (df['Z_Liquidez'] < -1.5),             # PLASMA (Iliquidez/Burbuja)
        (df['Z_Liquidez'] > 0) & (df['Z_Entropia'] < 1.0), # LÍQUIDO (Crecimiento Sano)
    ]
    choices = ['GAS 🔴', 'PLASMA 🟡', 'LÍQUIDO 🔵']
    df['Fase'] = np.select(conditions, choices, default='SÓLIDO ⚪')
    
    # Gobernanza Psi (Cálculo simplificado para demo)
    # Psi bajo = Veto/Cash. Psi alto = Full Investment.
    df['Psi'] = np.where(df['Fase'].str.contains('GAS'), 0.0, 
                np.where(df['Fase'].str.contains('LÍQUIDO'), 0.95, 0.5))
    
    return df

# --- 3. UI: SIDEBAR ---
with st.sidebar:
    st.title("📡 FAROS SYSTEM")
    st.markdown("`v2.0.1 | TAI-ACF CORE`")
    st.markdown("---")
    
    mode = st.radio("MODO DE OPERACIÓN:", ["Radar de Fases", "Análisis Profundo", "Señales IA"])
    
    st.markdown("---")
    st.info("⚡ Estado del Sistema: ONLINE")
    st.text("Conexión: EvalIA-Node-1")

# --- 4. UI: DASHBOARD PRINCIPAL (RADAR) ---
df = get_mock_data()

if mode == "Radar de Fases":
    st.title("🗺️ RADAR TERMODINÁMICO DE MERCADO")
    st.markdown("Visualización de transiciones de fase en tiempo real (No-Ergodicidad).")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # GRÁFICO DE DISPERSIÓN (Phase Diagram)
        fig = px.scatter(df, x="Z_Entropia", y="Z_Liquidez", 
                         color="Fase", 
                         size="Precio", 
                         text="Ticker",
                         hover_data=["Psi"],
                         color_discrete_map={
                             "LÍQUIDO 🔵": "#00CCFF", 
                             "SÓLIDO ⚪": "#AAAAAA", 
                             "GAS 🔴": "#FF0000", 
                             "PLASMA 🟡": "#FFFF00"
                         })
        
        # Zonas de Fondo (Cuadrantes TAI-ACF)
        fig.add_hrect(y0=-3, y1=-1.5, line_width=0, fillcolor="yellow", opacity=0.1) # Zona Plasma
        fig.add_vrect(x0=1.5, x1=4, line_width=0, fillcolor="red", opacity=0.1)     # Zona Gas
        
        fig.update_traces(textposition='top center', marker=dict(line=dict(width=2, color='DarkSlateGrey')))
        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Entropía (Caos H) →",
            yaxis_title="Liquidez (Estructura L) ↑",
            height=600,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🚨 Alertas de Fase")
        for index, row in df.iterrows():
            if "GAS" in row['Fase']:
                st.error(f"**{row['Ticker']}** ha entrado en FASE GASEOSA. Veto activo.")
            elif "BONDS" in row['Ticker']:
                st.success(f"**{row['Ticker']}** detecta inyección de energía ($\vec{{M}} > 0$).")
            elif "PLTR" in row['Ticker']:
                st.info(f"**{row['Ticker']}** mantiene estabilidad en FASE LÍQUIDA.")

# --- 5. UI: ANÁLISIS PROFUNDO (GAUGES) ---
elif mode == "Análisis Profundo":
    selected_ticker = st.selectbox("Seleccione Activo:", df['Ticker'])
    asset = df[df['Ticker'] == selected_ticker].iloc[0]
    
    st.title(f"🧬 ANÁLISIS TAI-ACF: {selected_ticker}")
    
    # Panel de Métricas
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precio Actual", f"${asset['Precio']}")
    m2.metric("Fase Actual", asset['Fase'])
    m3.metric("Gobernanza (Ψ)", f"{asset['Psi']:.2f}", delta_color="normal")
    m4.metric("Flujo Capital", "ENTRANDO" if asset['Flujo_M']>0 else "SALIENDO")
    
    st.markdown("---")
    
    # Medidores (Gauges)
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("### 🌊 LIQUIDEZ ($Z_L$)")
        fig_l = go.Figure(go.Indicator(
            mode = "gauge+number", value = asset['Z_Liquidez'],
            gauge = {'axis': {'range': [-3, 3]}, 'bar': {'color': "blue"}}
        ))
        fig_l.update_layout(height=300, margin=dict(t=10,b=10))
        st.plotly_chart(fig_l, use_container_width=True)
        
    with c2:
        st.markdown("### 🎲 ENTROPÍA ($Z_H$)")
        fig_h = go.Figure(go.Indicator(
            mode = "gauge+number", value = asset['Z_Entropia'],
            gauge = {'axis': {'range': [0, 3]}, 'bar': {'color': "red" if asset['Z_Entropia']>1.5 else "green"},
                     'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 1.5}}
        ))
        fig_h.update_layout(height=300, margin=dict(t=10,b=10))
        st.plotly_chart(fig_h, use_container_width=True)

    with c3:
        st.markdown("### 🧠 GOBERNANZA ($\Psi$)")
        fig_p = go.Figure(go.Indicator(
            mode = "gauge+number", value = asset['Psi'],
            gauge = {'axis': {'range': [0, 1]}, 'bar': {'color': "cyan"}}
        ))
        fig_p.update_layout(height=300, margin=dict(t=10,b=10))
        st.plotly_chart(fig_p, use_container_width=True)
