# ==============================================================================
# FAROS v3.0 - THE HYDRODYNAMIC SUITE
# Autor: Juan Arroyo | SG Consulting Group & Emporium
# Motor Físico: Navier-Stokes Financial Implementation
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta
from physics_engine import FarosPhysics  # IMPORTANTE: Tu nuevo cerebro

# Instancia Global del Motor Físico
fisica = FarosPhysics()

# --- CONFIGURACIÓN VISUAL ---
st.set_page_config(page_title="FAROS v3.0 | Hydrodynamic", page_icon="📡", layout="wide")
st.markdown("""
<style>
    .stApp { background-color: #FFFFFF; color: #111; }
    h1, h2, h3 { color: #0E1117 !important; } 
    .stExpander { border: 1px solid #ddd; background-color: #f8f9fa; border-radius: 8px; }
    .global-status { padding: 15px; border-radius: 8px; margin-bottom: 20px; font-weight: bold; text-align: center; border: 1px solid #ddd; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
    .macro-card { padding: 20px; background-color: #f8f9fa; border-radius: 10px; border: 1px solid #ddd; margin-bottom: 20px; text-align: center; }
    /* Estilo Chatbot */
    .stChatMessage { background-color: #f9f9f9; border-radius: 10px; padding: 10px; border: 1px solid #eee; }
    .stChatMessage[data-testid="user-message"] { background-color: #e3f2fd; }
    /* Alertas Físicas */
    .laminar { color: #1B5E20; font-weight: bold; }
    .turbulent { color: #B71C1C; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 0. BASE DE DATOS & UTILIDADES
# ==============================================================================
ASSET_DB = {
    "PALANTIR (PLTR)": "PLTR", "NVIDIA (NVDA)": "NVDA", "D-WAVE (QBTS)": "QBTS", 
    "TESLA (TSLA)": "TSLA", "APPLE (AAPL)": "AAPL", "MICROSOFT (MSFT)": "MSFT", 
    "AMAZON (AMZN)": "AMZN", "GOOGLE (GOOGL)": "GOOGL", "META (META)": "META",
    "BITCOIN (BTC-USD)": "BTC-USD", "ETHEREUM (ETH-USD)": "ETH-USD", 
    "S&P 500 (SPY)": "SPY", "NASDAQ 100 (QQQ)": "QQQ", "RUSSELL 2000 (IWM)": "IWM",
    "AMD (AMD)": "AMD", "INTEL (INTC)": "INTC", "TSMC (TSM)": "TSM",
    "COINBASE (COIN)": "COIN", "MICROSTRATEGY (MSTR)": "MSTR",
    "NETFLIX (NFLX)": "NFLX", "DISNEY (DIS)": "DIS",
    "VISA (V)": "V", "MASTERCARD (MA)": "MA", "JPMORGAN (JPM)": "JPM",
    "EXXON (XOM)": "XOM", "CHEVRON (CVX)": "CVX",
    "SUPER MICRO (SMCI)": "SMCI", "C3.AI (AI)": "AI", "IONQ (IONQ)": "IONQ"
}

def get_tickers_from_selection(selection, manual_input):
    selected = [ASSET_DB[k] for k in selection]
    if manual_input:
        manual_list = [x.strip().upper() for x in manual_input.split(',')]
        selected.extend(manual_list)
    return list(set(selected)) 

def get_ecuador_time():
    return (datetime.utcnow() - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S (Quito/EC)")

# ==============================================================================
# 1. CORE v3.0: INTEGRACIÓN FÍSICA
# ==============================================================================

@st.cache_data(ttl=300)
def get_market_status():
    """Analiza la Viscosidad Global usando SPY"""
    try:
        spy = yf.Ticker("SPY").history(period="6mo")
        if spy.empty: return "UNKNOWN", 0, "Error Data", pd.DataFrame()
        
        # --- CÁLCULO FÍSICO ---
        re, h, psi, estado = fisica.calcular_hidrodinamica(spy)
        
        msg = f"Re: {re:.0f} ({estado})"
        if estado == "TURBULENTO": return "GAS", h, "CRASH ALERT: " + msg, spy
        elif estado == "TRANSICION": return "WARNING", h, "VISCOSO: " + msg, spy
        else: return "LIQUID", h, "NOMINAL: " + msg, spy
    except: return "UNKNOWN", 0, "Desconectado", pd.DataFrame()

def calculate_beta(ticker_hist, market_hist):
    try:
        # Alinear fechas
        df = pd.DataFrame({'Asset': ticker_hist['Close'].pct_change(), 'Market': market_hist['Close'].pct_change()}).dropna()
        if df.empty: return 1.0
        cov = df.cov().iloc[0, 1]
        var = df['Market'].var()
        return cov / var if var != 0 else 1.0
    except: return 1.0

# ==============================================================================
# 2. FUNCIONES DE INTELIGENCIA (IA & TAI ENGINE)
# ==============================================================================

# --- CHATBOT HIDRODINÁMICO ---
def extract_ticker(user_input):
    words = user_input.upper().replace('?', '').replace('.', '').split()
    known_tickers = list(ASSET_DB.values()) + ["PLTR", "NVDA", "BTC", "ETH", "SPY", "QBTS"]
    for w in words:
        if w in known_tickers: return w
        if 2 <= len(w) <= 6 and w.isalpha(): return w 
    return None

def generate_faros_insight(ticker, risk_tolerance=3.0):
    try:
        ticker = ticker.upper()
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        if hist.empty: return f"⚠️ No encontré datos para **{ticker}**."
        
        # --- MOTOR FÍSICO ---
        re, h, psi, estado = fisica.calcular_hidrodinamica(hist)
        curr_price = hist['Close'].iloc[-1]
        
        # --- NARRATIVA AUTOMÁTICA ---
        if estado == "TURBULENTO":
            emoji = "⛔"; rec = "CASH / SALIDA"
            desc = f"Reynolds Crítico ({re:.0f}). El fluido se ha roto. Alta probabilidad de ruina."
        elif estado == "TRANSICION":
            emoji = "⚠️"; rec = "PRECAUCIÓN"
            desc = f"Viscosidad en aumento (Re {re:.0f}). Vórtices detectados."
        else: # LAMINAR
            if psi > 60:
                emoji = "🚀"; rec = "COMPRA FUERTE"
                desc = "Flujo Laminar puro. El capital fluye sin resistencia."
            elif psi > 20:
                emoji = "✅"; rec = "ACUMULAR"
                desc = "Estado Líquido estable. Tendencia saludable."
            else:
                emoji = "🧊"; rec = "ESPERAR"
                desc = "Estado Sólido (Congelado). Falta inercia."

        return f"""
        ### 📡 Hidrodinámica TAI: {ticker}
        **Precio:** ${curr_price:.2f} | **Gobernanza Ψ:** {psi:.0f}%
        
        **Estado:** {emoji} **{estado}** (Re: {re:.0f})
        > {desc}
        
        **Veredicto:** {rec}
        *(Entropía: {h:.2f} bits)*
        """
    except Exception as e: return f"Error analizando el activo: {str(e)}"

# --- ASIGNACIÓN DE PORTAFOLIO TAI ---
def calculate_tai_weights(tickers, risk_tolerance):
    scores = {}; valid_tickers = []
    
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(period="6mo")
            if len(hist) > 50:
                # Usamos Psi directamente del motor físico como peso
                _, _, psi, estado = fisica.calcular_hidrodinamica(hist)
                
                # Penalización extrema si es Turbulento
                if estado == "TURBULENTO": weight_score = 0
                else: weight_score = psi
                
                scores[t] = weight_score
                valid_tickers.append(t)
        except: pass
        
    total = sum(scores.values())
    w_str = ""
    if total > 0: 
        for t in valid_tickers: w_str += f"{t}, {scores[t]/total:.2f}\n"
    else: w_str = "SISTEMA EN CASH (PROTECCIÓN TURBULENCIA)"
    return w_str

# ==============================================================================
# 3. MÓDULOS DE ANÁLISIS
# ==============================================================================

def analyze_country(country_name):
    proxies = {
        "USA": {"ETF": "SPY", "FX": "DX-Y.NYB", "Name": "Estados Unidos"},
        "MEXICO": {"ETF": "EWW", "FX": "MXN=X", "Name": "México"},
        "BRASIL": {"ETF": "EWZ", "FX": "BRL=X", "Name": "Brasil"},
        "EUROPA": {"ETF": "VGK", "FX": "EURUSD=X", "Name": "Eurozona"},
        "CHINA": {"ETF": "MCHI", "FX": "CNY=X", "Name": "China"},
        "JAPON": {"ETF": "EWJ", "FX": "JPY=X", "Name": "Japón"},
        "ARGENTINA": {"ETF": "ARGT", "FX": "ARS=X", "Name": "Argentina"},
    }
    target = proxies.get(country_name)
    if not target: return None
    try:
        etf_h = yf.Ticker(target['ETF']).history(period="1y")
        fx_h = yf.Ticker(target['FX']).history(period="1y")
        
        # Análisis Fractal (Macro es igual a Micro)
        re, h, psi, estado = fisica.calcular_hidrodinamica(etf_h)
        
        return {
            "Name": target['Name'], "ETF_Ticker": target['ETF'], 
            "ETF_Price": etf_h['Close'].iloc[-1],
            "Status": estado, "Reynolds": re,
            "FX_Price": fx_h['Close'].iloc[-1],
            "Macro_Score": psi
        }
    except: return None

def analyze_portfolio(holdings, risk_tolerance):
    _, _, _, spy_hist = get_market_status()
    results = []
    tickers = [t for t in holdings.keys() if t != 'CASH']
    
    if not tickers: return None, None, None
    
    for t in tickers:
        try:
            stock = yf.Ticker(t)
            hist = stock.history(period="1y")
            if not hist.empty:
                curr_price = hist['Close'].iloc[-1]
                beta = calculate_beta(hist, spy_hist)
                
                # --- CÁLCULO FÍSICO ---
                re, h, psi, estado = fisica.calcular_hidrodinamica(hist)
                
                action = "MANTENER"
                if estado == "TURBULENTO": action = "VENDER (RIESGO)"
                elif estado == "LAMINAR" and psi > 60: action = "AUMENTAR"
                elif estado == "TRANSICION": action = "VIGILAR"
                
                results.append({
                    "Ticker": t, "Weight": holdings[t], "Price": curr_price, 
                    "Beta": beta, "Reynolds": re, "Psi": psi, 
                    "Status": estado, "Action": action
                })
        except: pass
    
    if not results: return None, pd.DataFrame(), {"Beta":0, "Psi":0}
    
    df_res = pd.DataFrame(results)
    try:
        data_corr = yf.download(tickers, period="6mo")['Close'].pct_change().dropna()
        corr_matrix = data_corr.corr()
    except: corr_matrix = pd.DataFrame()
    
    port_beta = (df_res['Beta'] * df_res['Weight']).sum()
    port_psi = (df_res['Psi'] * df_res['Weight']).sum()
    
    return df_res, corr_matrix, {"Beta": port_beta, "Psi": port_psi}

@st.cache_data(ttl=300)
def get_live_data(tickers_list, risk_tolerance):
    m_status, _, m_msg, _ = get_market_status()
    data_list = []
    
    for ticker in tickers_list:
        try:
            hist = yf.Ticker(ticker).history(period="6mo")
            if len(hist) > 20:
                re, h, psi, estado = fisica.calcular_hidrodinamica(hist)
                
                signal = "NEUTRAL"
                narrative = "Falta de definición."
                category = "neutral"
                
                if estado == "TURBULENTO":
                    signal = "SALIDA"; category = "danger"; narrative = f"Turbulencia (Re {re:.0f})"
                elif estado == "TRANSICION":
                    signal = "ALERTA"; category = "warning"; narrative = "Aumento de viscosidad"
                elif estado == "LAMINAR":
                    if psi > 60:
                        signal = "COMPRA"; category = "success"; narrative = "Flujo Laminar Óptimo"
                    else:
                        signal = "ACUMULAR"; category = "info"; narrative = "Laminar débil"

                data_list.append({
                    "Ticker": ticker, "Price": hist['Close'].iloc[-1], 
                    "Signal": signal, "Category": category, "Narrative": narrative,
                    "Entropy": h, "Reynolds": re, "Psi": psi, "Status": estado
                })
        except: pass
        
    return pd.DataFrame(data_list).sort_values('Psi', ascending=False) if data_list else pd.DataFrame(), m_status, 0, m_msg

# ==============================================================================
# 4. GENERADORES DE REPORTES HTML
# ==============================================================================

def generate_portfolio_report(df_portfolio, metrics, risk_profile):
    now_ec = get_ecuador_time()
    df_html = df_portfolio[['Ticker', 'Weight', 'Psi', 'Reynolds', 'Status', 'Action']].to_html(classes='table', index=False, float_format="%.0f")
    html = f"""
    <html><head><style>
        body {{ font-family: Helvetica, sans-serif; padding: 40px; color: #333; }}
        h1 {{ border-bottom: 2px solid #333; padding-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 12px; }}
        th {{ background-color: #333; color: white; padding: 8px; }}
        td {{ border-bottom: 1px solid #ddd; padding: 8px; text-align: center; }}
        .metric-box {{ display: inline-block; width: 30%; background: #f4f4f4; padding: 10px; margin-right: 10px; text-align: center; border-radius: 5px; }}
    </style></head><body>
        <h1>FAROS v3.0 | Auditoría Física</h1>
        <p><strong>Operador:</strong> SG Consulting Group | <strong>Fecha:</strong> {now_ec}</p>
        <h3>1. Métricas Termodinámicas</h3>
        <div>
            <div class="metric-box"><strong>Beta</strong><br><h2>{metrics['Beta']:.2f}x</h2></div>
            <div class="metric-box"><strong>Gobernanza (Ψ)</strong><br><h2>{metrics['Psi']:.0f}%</h2></div>
        </div>
        <h3>2. Diagnóstico Hidrodinámico</h3>
        {df_html}
    </body></html>
    """
    return html

# ==============================================================================
# 5. INTERFAZ DE USUARIO (FRONT-END)
# ==============================================================================

with st.sidebar:
    st.title("📡 FAROS v3.0")
    st.caption("**Hydrodynamic Governance Engine**")
    app_mode = st.radio("SISTEMA:", ["🤖 ANALISTA IA", "🌎 MACRO FRACTAL", "💼 GESTIÓN PORTAFOLIOS", "🔍 SCANNER FÍSICO", "⏳ BACKTEST LAB"])
    st.markdown("---")
    risk_profile = st.select_slider("Perfil del Crítico", options=["Conservador", "Growth", "Quantum"], value="Growth")

# --- MÓDULO 1: ANALISTA IA ---
if app_mode == "🤖 ANALISTA IA":
    st.title("Analista Hidrodinámico")
    st.caption("Consulta el Número de Reynolds y el estado físico de cualquier activo.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Sistema FAROS en línea. ¿Qué activo quieres escanear hoy? (Ej: NVDA)"}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("Escribe tu consulta..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Calculando física de fluidos..."):
                ticker = extract_ticker(prompt)
                response = generate_faros_insight(ticker) if ticker else "No detecté un ticker."
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

# --- MÓDULO 2: MACRO ---
elif app_mode == "🌎 MACRO FRACTAL":
    st.title("Sandbox de Política Soberana")
    st.info("Simulación Fractal: Las leyes del activo individual se aplican a la nación.")
    
    country_sel = st.selectbox("Jurisdicción:", ["USA", "MEXICO", "EUROPA", "CHINA", "BRASIL"])
    
    if st.button("Ejecutar Simulación"):
        with st.spinner(f"Calculando Reynolds Macro para {country_sel}..."):
            data = analyze_country(country_sel)
        
        if data:
            # Gauge Chart para Reynolds
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = data['Reynolds'],
                title = {'text': f"Turbulencia ({data['Name']})"},
                gauge = {
                    'axis': {'range': [None, 6000]},
                    'bar': {'color': "black"},
                    'steps': [
                        {'range': [0, 2300], 'color': "lightgreen"},
                        {'range': [2300, 4000], 'color': "yellow"},
                        {'range': [4000, 6000], 'color': "red"}
                    ],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 4000}
                }
            ))
            
            c1, c2 = st.columns([1, 2])
            with c1: st.plotly_chart(fig_gauge, use_container_width=True)
            with c2:
                st.subheader(f"Estado: {data['Status']}")
                st.metric("Score de Gobernanza (Ψ)", f"{data['Macro_Score']:.0f}/100")
                st.markdown(f"**Divisa:** ${data['FX_Price']:.2f}")

# --- MÓDULO 3: PORTAFOLIO ---
elif app_mode == "💼 GESTIÓN PORTAFOLIOS":
    st.title("Gobernanza de Capital (Ψ)")
    
    with st.expander("📝 Configuración de Activos", expanded=True):
        c1, c2 = st.columns(2)
        sel_assets = c1.multiselect("Activos:", list(ASSET_DB.keys()), default=["PALANTIR (PLTR)", "NVIDIA (NVDA)"])
        manual = c1.text_input("Manual (Tickers):", "")
        
        if c1.button("⚖️ Auto-Balanceo Hidrodinámico"):
            with st.spinner("Optimizando por Entropía..."):
                final_tickers = get_tickers_from_selection(sel_assets, manual)
                w_str = calculate_tai_weights(final_tickers, 3.0)
                st.session_state['weights_area'] = w_str
        
        weights_input = c2.text_area("Pesos:", value=st.session_state.get('weights_area', ""), height=150)
        
        if c2.button("Analizar"):
            try:
                holdings = {}; [holdings.update({l.split(',')[0].strip().upper(): float(l.split(',')[1].strip())}) for l in weights_input.split('\n') if ',' in l]
                st.session_state['holdings'] = holdings
            except: st.error("Error formato (Ticker, Peso)")

    if 'holdings' in st.session_state:
        df_p, corr_m, metrics = analyze_portfolio(st.session_state['holdings'], 3.0)
        if df_p is not None:
            k1, k2, k3 = st.columns(3)
            k1.metric("Gobernanza (Ψ)", f"{metrics['Psi']:.0f}%")
            k2.metric("Estado", "LÍQUIDO" if metrics['Psi']>50 else "VISCOSO")
            k3.metric("Beta", f"{metrics['Beta']:.2f}")
            
            st.dataframe(df_p.style.apply(lambda x: ['background-color: #ffcccc' if v == 'TURBULENTO' else '' for v in x], subset=['Status']))
            
            if st.button("🖨️ Informe Auditoría"):
                st.download_button("Descargar HTML", generate_portfolio_report(df_p, metrics, "Growth"), "faros_audit.html")

# --- MÓDULO 4: SCANNER ---
elif app_mode == "🔍 SCANNER FÍSICO":
    st.title("Scanner de Navier-Stokes")
    col_search, col_manual = st.columns([3, 1])
    selected_assets = col_search.multiselect("Universo:", list(ASSET_DB.keys()), default=["PALANTIR (PLTR)", "NVIDIA (NVDA)", "TESLA (TSLA)"])
    
    if st.button("Escanear Mercado"): 
        target_list = get_tickers_from_selection(selected_assets, col_manual.text_input("Otros:", ""))
        df, m_status, _, m_msg = get_live_data(target_list, 3.0)
        st.session_state['scan_data'] = df
        st.session_state['scan_meta'] = (m_status, m_msg)
    
    if 'scan_data' in st.session_state:
        df = st.session_state['scan_data']; m_status, m_msg = st.session_state['scan_meta']
        
        # Alerta Global
        bg = "#ffcdd2" if m_status == "GAS" else "#c8e6c9"
        st.markdown(f"<div class='global-status' style='background-color:{bg};'>CONTEXTO SPY: {m_msg}</div>", unsafe_allow_html=True)
        
        # Gráfico de Dispersión (Reynolds vs Psi)
        if not df.empty:
            fig = px.scatter(df, x="Reynolds", y="Psi", color="Status", text="Ticker", 
                             color_discrete_map={"TURBULENTO":"red", "LAMINAR":"green", "TRANSICION":"orange"})
            fig.add_vline(x=4000, line_dash="dash", line_color="red", annotation_text="Límite Ruina")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df)

# --- MÓDULO 5: BACKTEST ---
elif app_mode == "⏳ BACKTEST LAB":
    st.title("Validación Histórica")
    tck = st.text_input("Activo:", "BTC-USD").upper()
    
    if st.button("Simular Física"):
        try:
            df = yf.Ticker(tck).history(period="2y")
            
            # --- Lógica Vectorizada para Velocidad (Aprox de Physics Engine) ---
            df['Ret'] = df['Close'].pct_change()
            df['Velocity'] = df['Ret'].abs().rolling(3).mean() * 1000
            df['Vol_SMA'] = df['Volume'].rolling(14).mean()
            df['Density'] = df['Volume'] / df['Vol_SMA']
            df['High_Low'] = (df['High'] - df['Low']) / df['Close']
            df['Viscosity'] = df['High_Low'] / (df['Density'] + 0.01)
            df['L'] = df['Close'].rolling(14).std()
            
            # Reynolds Vectorizado
            df['Reynolds'] = (df['Density'] * df['Velocity'] * df['L'] * 100) / (df['Viscosity'] + 0.0001)
            
            # Señales
            df['Signal'] = np.where(df['Reynolds'] < 2300, 1, 0) # Comprar solo en Laminar
            df['Signal'] = np.where(df['Reynolds'] > 4000, 0, df['Signal']) # Vender en Turbulento
            
            # Resultados
            df['Strategy'] = df['Ret'] * df['Signal'].shift(1)
            df['Cum_Strat'] = (1 + df['Strategy']).cumprod()
            df['Cum_BH'] = (1 + df['Ret']).cumprod()
            
            st.line_chart(df[['Cum_Strat', 'Cum_BH']])
            st.metric("Retorno Estrategia", f"{(df['Cum_Strat'].iloc[-1]-1)*100:.1f}%")
            
        except Exception as e: st.error(f"Error: {e}")
