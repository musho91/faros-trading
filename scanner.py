# ==============================================================================
# FAROS v4.1 - PORTFOLIO RADAR & AI ANALYST
# Autor: Juan Arroyo | TAI-ACF Framework
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import warnings

warnings.filterwarnings("ignore")

# --- CONFIGURACIÓN VISUAL ---
st.set_page_config(page_title="FAROS AI Radar", page_icon="📡", layout="wide")

st.title("📡 FAROS: Portfolio Radar System")
st.markdown("### TAI-ACF Multi-Asset Scanner | v4.1 (AI Analyst Build)")

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎛️ Configuración")
    default_tickers = "ARM, NVDA, MSTR, AAPL, AMD, META, GOOG, GME, IONQ, PLTR, TSLA, AMZN"
    tickers_input = st.text_area("Watchlist (separados por coma):", value=default_tickers, height=150)
    lookback = st.slider("Ventana (Días):", 10, 60, 20)
    scan_button = st.button("🛰️ INICIAR ESCANEO", type="primary")
    st.markdown("---")
    st.caption("Juan Arroyo | CEO & Founder")

# --- MOTOR MATEMÁTICO ---
def analyze_asset(ticker, window=20):
    try:
        df = yf.download(ticker, period="1y", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if df.empty or len(df) < 60: return None

        # Física
        returns = df['Close'].pct_change()
        vol = returns.rolling(window).std()
        
        # Liquidez (L)
        l_raw = np.log(df['Volume'] + 1) / (vol + 1e-6)
        l_mean = l_raw.rolling(60).mean()
        l_std = l_raw.rolling(60).std()
        L_score = (l_raw.iloc[-1] - l_mean.iloc[-1]) / (l_std.iloc[-1] + 1e-6)

        # Entropía (H)
        net = df['Close'].diff(window).abs()
        path = df['Close'].diff().abs().rolling(window).sum()
        h_raw = 1 - (net / (path + 1e-6))
        h_mean = h_raw.rolling(60).mean()
        h_std = h_raw.rolling(60).std()
        H_score = (h_raw.iloc[-1] - h_mean.iloc[-1]) / (h_std.iloc[-1] + 1e-6)

        # Gobernanza Psi
        phase = "SÓLIDO (Hold)"
        action = "OBSERVAR"
        
        if L_score > -0.5 and H_score < 1.0: phase = "LÍQUIDO (Tendencia)"
        if H_score > 1.5: phase = "GAS (Crash/Caos)"; action = "VENTA (Cash)"
        if L_score < -1.5: phase = "PLASMA (Burbuja)"; action = "REDUCIR"

        raw_signal = L_score - H_score
        psi = 1 / (1 + np.exp(-raw_signal))
        
        if "GAS" in phase: psi = 0.0
        elif psi > 0.75 and "LÍQUIDO" in phase: action = "COMPRA FUERTE 🚀"
        elif psi > 0.6: action = "ACUMULAR ✅"

        return {
            "Ticker": ticker, "Precio": df['Close'].iloc[-1],
            "L": round(L_score, 2), "H": round(H_score, 2),
            "Ψ": round(psi, 2), "Fase": phase, "Estrategia": action
        }
    except: return None

# --- BOT NARRATIVO (La novedad) ---
def generate_bot_insight(row):
    ticker = row['Ticker']
    psi = row['Ψ']
    l = row['L']
    h = row['H']
    
    insight = f"**Análisis de Inteligencia para {ticker}:**\n\n"
    
    if psi > 0.75:
        insight += f"🔥 **¡Oportunidad Alpha Detectada!** {ticker} es el líder indiscutible del grupo.\n"
        insight += f"- **¿Por qué?** Su Liquidez ({l}σ) es extremadamente alta, lo que indica que las grandes instituciones están comprando fuerte y sosteniendo el precio. "
        insight += f"Al mismo tiempo, su Entropía ({h}σ) es baja, lo que significa que la subida es limpia, eficiente y ordenada.\n"
        insight += "- **Conclusión:** El sistema recomienda **ASIGNACIÓN MÁXIMA**."
    elif psi < 0.3:
        insight += f"⚠️ **Alerta de Riesgo Estructural.** El sistema ha vetado a {ticker}.\n"
        insight += f"- **El Problema:** Detectamos condiciones de 'Gas' o falta de soporte institucional. La probabilidad de caída es alta.\n"
        insight += "- **Conclusión:** Mantenerse en CASH o VENDER."
    else:
        insight += f"⚖️ **Condiciones Neutrales.** {ticker} es un activo seguro pero aburrido hoy.\n"
        insight += "- **Detalle:** Tiene liquidez decente, pero falta el 'momentum' explosivo que buscamos. "
        insight += "Es bueno para preservar capital, pero no para multiplicarlo agresivamente hoy."
        
    return insight

# --- INTERFAZ ---
if scan_button:
    ticker_list = [x.strip().upper() for x in tickers_input.split(',')]
    results = []
    bar = st.progress(0)
    
    for i, t in enumerate(ticker_list):
        res = analyze_asset(t, lookback)
        if res: results.append(res)
        bar.progress((i + 1) / len(ticker_list))
    
    if results:
        df = pd.DataFrame(results).sort_values(by="Ψ", ascending=False)
        top_pick = df.iloc[0] # El Ganador
        
        # --- SECCIÓN DEL BOT ANALISTA ---
        st.success("✅ Escaneo Finalizado.")
        
        with st.container():
            col_bot, col_kpi = st.columns([2, 1])
            
            with col_bot:
                st.markdown("### 🤖 El Analista Táctico Dice:")
                st.info(generate_bot_insight(top_pick))
            
            with col_kpi:
                st.markdown("### 🏆 Top Pick Metrics")
                st.metric(label=f"Ticker: {top_pick['Ticker']}", value=f"${top_pick['Precio']:.2f}")
                st.metric(
                    label="Gobernanza (Ψ)", 
                    value=top_pick['Ψ'], 
                    delta="Excelente" if top_pick['Ψ']>0.7 else "Normal",
                    help="Puntaje de 0 a 1 que combina Liquidez y Entropía. Más alto = Mejor compra."
                )

        st.markdown("---")

        # --- RADAR Y TABLA ---
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("🗺️ Mapa de Fases")
            fig = px.scatter(df, x="L", y="H", color="Fase", text="Ticker", size="Ψ",
                             color_discrete_map={"GAS (Crash/Caos)":"red", "LÍQUIDO (Tendencia)":"blue", "SÓLIDO (Hold)":"grey"},
                             title="Cuanto más abajo y a la derecha, MEJOR.")
            fig.add_hrect(y0=1.5, y1=4, line_width=0, fillcolor="red", opacity=0.1)
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.subheader("📋 Ranking Oficial")
            st.dataframe(df[["Ticker", "Ψ", "Estrategia"]], hide_index=True, use_container_width=True)