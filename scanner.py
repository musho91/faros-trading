# ==============================================================================
# FAROS v16.0 - INSTITUTIONAL PLATFORM (PORTFOLIO MANAGER)
# Autor: Juan Arroyo | SG Consulting Group
# Módulos: Scanner, Backtest, Oráculo, PORTFOLIO MANAGER, MACRO (Placeholder)
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import seaborn as sns # Para matriz de correlación estética
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN VISUAL ---
st.set_page_config(page_title="FAROS | Institutional", page_icon="🏛️", layout="wide")
st.markdown("""
<style>
    .stApp { background-color: #FFFFFF; color: #111; }
    h1, h2, h3 { color: #000 !important; } 
    .stExpander { border: 1px solid #ddd; background-color: #f8f9fa; border-radius: 8px; }
    .global-status { padding: 15px; border-radius: 8px; margin-bottom: 20px; font-weight: bold; text-align: center; border: 1px solid #ddd; }
    .metric-card { background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid #333; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. MOTOR LÓGICO (CORE)
# ==============================================================================

def calculate_entropy(history, window=20):
    if len(history) < window: return 0, 0
    returns = history['Close'].pct_change().dropna()
    subset = returns.tail(window)
    raw_vol = subset.std() * np.sqrt(252) * 100 if len(subset) > 1 else 0
    z_entropy = (raw_vol - 20) / 15 
    return raw_vol, z_entropy

def calculate_beta(ticker_hist, market_hist):
    """Calcula la sensibilidad (Beta) del activo respecto al SPY."""
    try:
        # Alinear fechas
        df = pd.DataFrame({'Asset': ticker_hist['Close'].pct_change(), 'Market': market_hist['Close'].pct_change()}).dropna()
        cov = df.cov().iloc[0, 1]
        var = df['Market'].var()
        return cov / var if var != 0 else 1.0
    except: return 1.0

def calculate_psi(entropy, liquidity, trend, risk_sigma, global_penalty=0):
    score = 50 
    if entropy > risk_sigma: score -= 30
    else: score += (risk_sigma - entropy) * 10 
    if liquidity > 0: score += liquidity * 20
    elif liquidity < -0.2: score -= 20
    if trend > 0: score += trend * 100
    else: score -= 50
    score -= global_penalty
    return max(0, min(100, score))

@st.cache_data(ttl=300)
def get_market_status():
    try:
        spy = yf.Ticker("SPY").history(period="6mo")
        if spy.empty: return "UNKNOWN", 0, "Error", pd.DataFrame()
        raw, z = calculate_entropy(spy)
        if z > 3.0: return "GAS", z, "CRISIS SISTÉMICA", spy
        elif z > 2.0: return "WARNING", z, "ALTA TENSIÓN", spy
        else: return "LIQUID", z, "ESTABLE", spy
    except: return "UNKNOWN", 0, "Desconectado", pd.DataFrame()

# ==============================================================================
# 2. FUNCIONES DE PORTAFOLIO
# ==============================================================================

def analyze_portfolio(holdings, risk_tolerance):
    """
    Analiza una lista de activos con pesos.
    holdings: dict {'TICKER': peso_float} (ej: {'PLTR': 0.3, 'NVDA': 0.7})
    """
    m_status, m_entropy, m_msg, spy_hist = get_market_status()
    global_penalty = 30 if m_status == "GAS" else 0
    
    results = []
    
    # Descarga masiva para correlaciones
    tickers = list(holdings.keys())
    if 'CASH' in tickers: tickers.remove('CASH') # Ignoramos cash para descarga
    
    if not tickers: return None, None, None
    
    # Análisis Individual + Beta
    for t in tickers:
        try:
            stock = yf.Ticker(t)
            hist = stock.history(period="1y")
            curr_price = hist['Close'].iloc[-1]
            
            # Física
            raw_vol, z_entropy = calculate_entropy(hist)
            sma = hist['Close'].rolling(50).mean().iloc[-1]
            trend = (curr_price - sma) / sma
            
            # Beta
            beta = calculate_beta(hist, spy_hist)
            
            # Psi
            psi = calculate_psi(z_entropy, 0, trend, risk_tolerance, global_penalty) # Liq simplificada
            
            # Diagnóstico
            status = "SÓLIDO"
            if z_entropy > risk_tolerance: status = "GAS (RIESGO)"
            elif trend > 0.02 and z_entropy < risk_tolerance: status = "LÍQUIDO (GROWTH)"
            
            # Acción sugerida
            action = "MANTENER"
            if status == "GAS (RIESGO)": action = "REDUCIR / VENDER"
            elif status == "LÍQUIDO (GROWTH)": action = "AUMENTAR"
            
            results.append({
                "Ticker": t,
                "Weight": holdings[t],
                "Price": curr_price,
                "Beta": beta,
                "Entropy": z_entropy,
                "Psi": psi,
                "Status": status,
                "Action": action
            })
        except: pass
        
    df_res = pd.DataFrame(results)
    
    # Matriz de Correlación
    data_corr = yf.download(tickers, period="6mo")['Close'].pct_change().dropna()
    corr_matrix = data_corr.corr()
    
    # Métricas Agregadas del Portafolio
    # Beta Ponderado = Sum(Peso * Beta)
    port_beta = (df_res['Beta'] * df_res['Weight']).sum()
    # Psi Ponderado
    port_psi = (df_res['Psi'] * df_res['Weight']).sum()
    
    return df_res, corr_matrix, {"Beta": port_beta, "Psi": port_psi}

# ==============================================================================
# 3. INTERFAZ DE USUARIO
# ==============================================================================

with st.sidebar:
    st.header("🏛️ SG CAPITAL | FAROS")
    app_mode = st.radio("SISTEMA:", [
        "💼 GESTIÓN PORTAFOLIOS", 
        "🔍 SCANNER MERCADO", 
        "⏳ BACKTEST LAB", 
        "🔮 ORÁCULO FUTURO"
    ])
    st.markdown("---")
    st.subheader("⚙️ Parámetros de Riesgo")
    risk_profile = st.select_slider("Perfil del Fondo", options=["Conservador", "Growth", "Quantum"], value="Growth")
    if "Conservador" in risk_profile: risk_sigma = 2.0
    elif "Growth" in risk_profile: risk_sigma = 3.0
    else: risk_sigma = 5.0 

# --- MÓDULO: GESTIÓN DE PORTAFOLIOS (NUEVO) ---
if app_mode == "💼 GESTIÓN PORTAFOLIOS":
    st.title("Gestión de Activos & Riesgo")
    st.caption("Análisis agregado de exposiciones, correlaciones y cumplimiento de teoría TAI.")
    
    # 1. INPUT DE PORTAFOLIO
    with st.expander("📝 Editar Composición del Portafolio", expanded=True):
        c1, c2 = st.columns(2)
        # Ejemplo de input texto simple para rapidez
        portfolio_txt = c1.text_area("Activos y Pesos (Formato: Ticker, Peso)", 
                                     "PLTR, 0.30\nNVDA, 0.25\nQBTS, 0.10\nSPY, 0.20\nBTC-USD, 0.15", 
                                     height=150)
        c2.info("💡 **Instrucciones:** Ingresa cada activo en una línea nueva. La suma de pesos debería ser 1.0 (100%).")
        
        if c2.button("Analizar Cartera"):
            # Parsear texto
            try:
                holdings = {}
                for line in portfolio_txt.split('\n'):
                    parts = line.split(',')
                    if len(parts) == 2:
                        holdings[parts[0].strip().upper()] = float(parts[1].strip())
                st.session_state['holdings'] = holdings
            except:
                st.error("Error de formato. Usa: TICKER, 0.XX")

    if 'holdings' in st.session_state:
        df_p, corr_m, metrics = analyze_portfolio(st.session_state['holdings'], risk_sigma)
        
        if df_p is not None:
            # 2. DASHBOARD DE RIESGO
            st.markdown("### 📊 Termodinámica del Portafolio")
            
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Activos", len(df_p))
            
            # Beta del Portafolio
            beta_delta = "Agresivo" if metrics['Beta'] > 1.2 else "Defensivo" if metrics['Beta'] < 0.8 else "Balanceado"
            k2.metric("Beta (Riesgo Mercado)", f"{metrics['Beta']:.2f}x", beta_delta)
            
            # Psi del Portafolio
            psi_val = metrics['Psi']
            psi_col = "normal" if psi_val > 60 else "inverse"
            k3.metric("Score TAI Agregado (Ψ)", f"{psi_val:.0f}/100", delta_color=psi_col)
            
            # Estado Predominante
            avg_entropy = (df_p['Entropy'] * df_p['Weight']).sum()
            state = "GASEOSO (PELIGRO)" if avg_entropy > risk_sigma else "LÍQUIDO (CRECIMIENTO)"
            k4.metric("Fase Predominante", state)

            st.markdown("---")

            # 3. ANÁLISIS DE CORRELACIÓN (Vital para Hedge Funds)
            c_left, c_right = st.columns([1, 1])
            
            with c_left:
                st.subheader("🔥 Mapa de Calor (Correlaciones)")
                st.caption("Evita tener todo en rojo oscuro (1.0). Busca diversificación (colores fríos).")
                # Plotly Heatmap
                fig_corr = px.imshow(corr_m, text_auto=True, color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
                st.plotly_chart(fig_corr, use_container_width=True)
            
            with c_right:
                st.subheader("⚖️ Distribución y Alertas")
                fig_pie = px.pie(df_p, values='Weight', names='Ticker', title='Asignación Actual')
                st.plotly_chart(fig_pie, use_container_width=True)

            # 4. GESTOR DE ALERTAS / ACCIONES
            st.subheader("🚨 Centro de Comando (Alertas TAI)")
            
            # Filtrar acciones críticas
            critical = df_p[df_p['Action'] != "MANTENER"]
            
            if not critical.empty:
                for i, row in critical.iterrows():
                    color = "red" if "VENDER" in row['Action'] else "green"
                    st.markdown(f"""
                    <div style="padding:10px; border:1px solid {color}; border-radius:5px; margin-bottom:5px; background-color:rgba(255,0,0,0.05);">
                        <b>{row['Ticker']}:</b> {row['Action']} (Ψ: {row['Psi']:.0f} | Estado: {row['Status']})
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ Todo el portafolio está en estado saludable. No se requieren acciones urgentes.")
            
            # 5. TABLA DETALLADA (Para Reguladores/Auditoría)
            with st.expander("📄 Ver Detalle Técnico (Para Reporte Regulatorio)"):
                st.dataframe(df_p.style.format({"Weight": "{:.0%}", "Price": "${:.2f}", "Beta": "{:.2f}", "Entropy": "{:.2f}σ", "Psi": "{:.0f}"}))
                
                st.download_button(
                    label="📥 Descargar Informe CSV",
                    data=df_p.to_csv().encode('utf-8'),
                    file_name='reporte_portafolio_faros.csv',
                    mime='text/csv',
                )

# --- (MANTENEMOS LOS OTROS MÓDULOS DEL CÓDIGO ANTERIOR AQUÍ) ---
# Copia aquí las secciones de SCANNER, BACKTEST y ORÁCULO del código v15.3
# Para que funcione, solo tienes que pegar los bloques `elif app_mode == "..."` debajo.
# ------------------------------------------------------------------

elif app_mode == "🔍 SCANNER MERCADO":
    # ... (Pegar lógica de Scanner v15.3) ...
    st.info("Módulo Scanner activo (usar código anterior)")

elif app_mode == "⏳ BACKTEST LAB":
    # ... (Pegar lógica de Backtest v15.3) ...
    st.info("Módulo Backtest activo (usar código anterior)")

elif app_mode == "🔮 ORÁCULO FUTURO":
    # ... (Pegar lógica de Oráculo v15.3) ...
    st.info("Módulo Oráculo activo (usar código anterior)")
