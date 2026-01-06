# ==============================================================================
# FAROS v21.0 - MASTER SUITE (STABLE RELEASE)
# Autor: Juan Arroyo | SG Consulting Group & Emporium
# Fixes: Persistencia de Reportes, Gráfico Oráculo Completo, Manejo de Nulls
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta

# --- CONFIGURACIÓN VISUAL ---
st.set_page_config(page_title="FAROS | Institutional", page_icon="📡", layout="wide")
st.markdown("""
<style>
    .stApp { background-color: #FFFFFF; color: #111; }
    h1, h2, h3 { color: #000 !important; } 
    .stExpander { border: 1px solid #ddd; background-color: #f8f9fa; border-radius: 8px; }
    .global-status { padding: 15px; border-radius: 8px; margin-bottom: 20px; font-weight: bold; text-align: center; border: 1px solid #ddd; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
    .macro-card { padding: 20px; background-color: #f8f9fa; border-radius: 10px; border: 1px solid #ddd; margin-bottom: 20px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 0. BASE DE DATOS (SMART SEARCH)
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
    try:
        df = pd.DataFrame({'Asset': ticker_hist['Close'].pct_change(), 'Market': market_hist['Close'].pct_change()}).dropna()
        if df.empty: return 1.0
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
# 2. MOTOR DE ASIGNACIÓN TAI
# ==============================================================================

def calculate_tai_weights(tickers, risk_tolerance):
    scores = {}
    valid_tickers = []
    m_status, _, _, _ = get_market_status()
    global_penalty = 30 if m_status == "GAS" else 0

    for t in tickers:
        try:
            hist = yf.Ticker(t).history(period="6mo")
            if len(hist) > 50:
                raw_vol, z_entropy = calculate_entropy(hist)
                sma = hist['Close'].rolling(50).mean().iloc[-1]
                trend = (hist['Close'].iloc[-1] - sma) / sma
                psi = calculate_psi(z_entropy, 0, trend, risk_tolerance, global_penalty)
                
                if z_entropy > risk_tolerance: weight_score = 0 
                else: weight_score = psi if psi > 0 else 0
                
                scores[t] = weight_score
                valid_tickers.append(t)
        except: pass
    
    total_score = sum(scores.values())
    weights_str = ""
    if total_score > 0:
        for t in valid_tickers:
            w = scores[t] / total_score
            weights_str += f"{t}, {w:.2f}\n"
    else:
        weights_str = "\n".join([f"{t}, {1/len(tickers):.2f}" for t in tickers])
    return weights_str

# ==============================================================================
# 3. GENERADORES DE REPORTES (FIXED)
# ==============================================================================

def get_ecuador_time():
    return (datetime.utcnow() - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S (Quito/EC)")

def generate_portfolio_report(df_portfolio, metrics, risk_profile):
    now_ec = get_ecuador_time()
    df_html = df_portfolio[['Ticker', 'Weight', 'Price', 'Beta', 'Psi', 'Status', 'Action']].to_html(classes='table', index=False, float_format="%.2f")
    html = f"""
    <html><head><style>
        body {{ font-family: Helvetica, sans-serif; padding: 40px; color: #333; }}
        h1 {{ border-bottom: 2px solid #333; padding-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 12px; }}
        th {{ background-color: #333; color: white; padding: 8px; }}
        td {{ border-bottom: 1px solid #ddd; padding: 8px; text-align: center; }}
        .metric-box {{ display: inline-block; width: 30%; background: #f4f4f4; padding: 10px; margin-right: 10px; text-align: center; border-radius: 5px; }}
    </style></head><body>
        <h1>FAROS | Auditoría de Portafolio</h1>
        <p><strong>Operador:</strong> SG Consulting Group & Emporium | <strong>Fecha:</strong> {now_ec}</p>
        <h3>1. Métricas Agregadas</h3>
        <div>
            <div class="metric-box"><strong>Beta Global</strong><br><h2>{metrics['Beta']:.2f}x</h2></div>
            <div class="metric-box"><strong>Calidad (Ψ)</strong><br><h2>{metrics['Psi']:.0f}/100</h2></div>
            <div class="metric-box"><strong>Perfil Riesgo</strong><br><h2>{risk_profile}</h2></div>
        </div>
        <h3>2. Posiciones y Diagnóstico Táctico</h3>
        {df_html}
        <p style="font-size: 10px; color: #777; margin-top: 50px;">Generado por Algoritmo TAI-ACF. Uso interno.</p>
    </body></html>
    """
    return html

def generate_scanner_report(df_scan, market_status, risk_profile):
    now_ec = get_ecuador_time()
    df_print = df_scan[['Ticker', 'Price', 'Signal', 'Psi', 'Entropy', 'Trend']].copy()
    df_print['Trend'] = (df_print['Trend']/100).map("{:.1%}".format)
    df_print['Entropy'] = df_print['Entropy'].map("{:.2f}σ".format)
    df_html = df_print.to_html(classes='table', index=False, float_format="%.2f")
    best_asset = df_scan.iloc[0] 
    html = f"""
    <html><head><style>
        body {{ font-family: Helvetica, sans-serif; padding: 40px; color: #333; }}
        h1 {{ border-bottom: 2px solid #004085; padding-bottom: 10px; color: #004085; }}
        .status-bar {{ padding: 10px; background-color: #e2e3e5; border-radius: 5px; margin-bottom: 20px; font-weight: bold; text-align: center; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 12px; }}
        th {{ background-color: #004085; color: white; padding: 8px; }}
        td {{ border-bottom: 1px solid #ddd; padding: 8px; text-align: center; }}
        .highlight {{ background-color: #d4edda; padding: 15px; border-left: 5px solid #28a745; margin: 20px 0; }}
    </style></head><body>
        <h1>FAROS | Informe de Inteligencia de Mercado</h1>
        <p><strong>Emisión:</strong> {now_ec} | <strong>Perfil:</strong> {risk_profile}</p>
        <div class="status-bar">CONTEXTO GLOBAL (SPY): {market_status}</div>
        <div class="highlight">
            <strong>🏆 Oportunidad Destacada:</strong> {best_asset['Ticker']} (${best_asset['Price']:.2f})<br>
            Score: <strong>{best_asset['Psi']:.0f}/100</strong> | Señal: {best_asset['Signal']}
        </div>
        <h3>Análisis Detallado de Activos Escaneados</h3>
        {df_html}
        <p style="font-size: 10px; color: #777; margin-top: 50px;">Metodología TAI-ACF propiedad de SG Consulting Group.</p>
    </body></html>
    """
    return html

# ==============================================================================
# 4. FUNCIONES DE MÓDULOS (CORE FUNCTIONS)
# ==============================================================================

# --- MACRO ---
def analyze_country(country_name):
    proxies = {
        "USA": {"ETF": "SPY", "FX": "DX-Y.NYB", "Name": "Estados Unidos", "FX_Inv": False},
        "MEXICO": {"ETF": "EWW", "FX": "MXN=X", "Name": "México", "FX_Inv": True},
        "BRASIL": {"ETF": "EWZ", "FX": "BRL=X", "Name": "Brasil", "FX_Inv": True},
        "EUROPA": {"ETF": "VGK", "FX": "EURUSD=X", "Name": "Eurozona", "FX_Inv": False},
        "CHINA": {"ETF": "MCHI", "FX": "CNY=X", "Name": "China", "FX_Inv": True},
        "JAPON": {"ETF": "EWJ", "FX": "JPY=X", "Name": "Japón", "FX_Inv": True},
        "ARGENTINA": {"ETF": "ARGT", "FX": "ARS=X", "Name": "Argentina", "FX_Inv": True},
    }
    target = proxies.get(country_name, None)
    if not target: return None
    try:
        etf_h = yf.Ticker(target['ETF']).history(period="1y")
        fx_h = yf.Ticker(target['FX']).history(period="1y")
        etf_vol, etf_z = calculate_entropy(etf_h)
        etf_sma = etf_h['Close'].rolling(50).mean().iloc[-1]
        etf_trend = (etf_h['Close'].iloc[-1] - etf_sma) / etf_sma
        fx_vol, fx_z = calculate_entropy(fx_h)
        fx_sma = fx_h['Close'].rolling(50).mean().iloc[-1]
        fx_change = (fx_h['Close'].iloc[-1] - fx_sma) / fx_sma
        local_currency_strength = -fx_change if target['FX_Inv'] else fx_change
        econ_score = calculate_psi(etf_z, 0, etf_trend, 3.0)
        return {
            "Name": target['Name'], "ETF_Ticker": target['ETF'], "ETF_Price": etf_h['Close'].iloc[-1],
            "ETF_Trend": etf_trend, "ETF_Vol": etf_vol, "FX_Ticker": target['FX'], "FX_Price": fx_h['Close'].iloc[-1],
            "Local_FX_Trend": local_currency_strength, "Macro_Score": econ_score
        }
    except: return None

# --- PORTAFOLIO ---
def analyze_portfolio(holdings, risk_tolerance):
    m_status, m_entropy, m_msg, spy_hist = get_market_status()
    global_penalty = 30 if m_status == "GAS" else 0
    results = []
    tickers = list(holdings.keys())
    if 'CASH' in tickers: tickers.remove('CASH')
    if not tickers: return None, None, None
    for t in tickers:
        try:
            stock = yf.Ticker(t)
            hist = stock.history(period="1y")
            if not hist.empty: # FIX: Check empty
                curr_price = hist['Close'].iloc[-1]
                raw_vol, z_entropy = calculate_entropy(hist)
                sma = hist['Close'].rolling(50).mean().iloc[-1]
                trend = (curr_price - sma) / sma
                beta = calculate_beta(hist, spy_hist)
                psi = calculate_psi(z_entropy, 0, trend, risk_tolerance, global_penalty)
                status = "SÓLIDO"
                if z_entropy > risk_tolerance: status = "GAS (RIESGO)"
                elif trend > 0.02 and z_entropy < risk_tolerance: status = "LÍQUIDO (GROWTH)"
                action = "MANTENER"
                if status == "GAS (RIESGO)": action = "REDUCIR"
                elif status == "LÍQUIDO (GROWTH)": action = "AUMENTAR"
                results.append({"Ticker": t, "Weight": holdings[t], "Price": curr_price, "Beta": beta, "Entropy": z_entropy, "Psi": psi, "Status": status, "Action": action})
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

# --- SCANNER ---
@st.cache_data(ttl=300)
def get_live_data(tickers_list, window_cfg, risk_tolerance):
    m_status, m_entropy, m_msg, _ = get_market_status()
    global_penalty = 30 if m_status == "GAS" else (10 if m_status == "WARNING" else 0)
    data_list = []
    entropy_limit = risk_tolerance 
    if risk_tolerance >= 5.0: exit_threshold = -0.15
    elif risk_tolerance >= 3.0: exit_threshold = -0.10
    else: exit_threshold = -0.05
    for ticker in tickers_list:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=window_cfg['download'])
            if len(hist) > window_cfg['trend']:
                current_price = hist['Close'].iloc[-1]
                raw_vol, z_entropy = calculate_entropy(hist, window_cfg['volatility'])
                vol_avg = hist['Volume'].rolling(window_cfg['volatility']).mean().iloc[-1]
                curr_vol = hist['Volume'].iloc[-1]
                z_liq = (curr_vol - vol_avg) / vol_avg if vol_avg > 0 else 0
                raw_vol_ratio = (curr_vol / vol_avg) if vol_avg > 0 else 0
                sma_val = hist['Close'].rolling(window_cfg['trend']).mean().iloc[-1]
                trend_pct = (current_price - sma_val) / sma_val
                psi_score = calculate_psi(z_entropy, z_liq, trend_pct, risk_tolerance, global_penalty)
                signal, category, narrative = "MANTENER", "neutral", "Sólido (Estable)."
                sys_warn = " ⚠️ [RIESGO GLOBAL]" if m_status == "GAS" else ""
                if z_entropy > entropy_limit:
                    if trend_pct > 0.15: signal = "GROWTH EXTREMO"; category = "warning"; narrative = f"⚡ Momentum (+{trend_pct*100:.1f}%) vence volatilidad.{sys_warn}"
                    else: signal = "GAS / RIESGO"; category = "danger"; narrative = f"⚠️ Fase Gaseosa Local ({z_entropy:.1f}σ).{sys_warn}"
                elif trend_pct < exit_threshold: signal = "SALIDA"; category = "danger" if risk_tolerance < 3 else "warning"; narrative = f"📉 Rotura Estructural ({trend_pct*100:.1f}%).{sys_warn}"
                elif trend_pct > 0.02:
                    if z_liq > 0.10: signal = "COMPRA FUERTE"; category = "success" if m_status != "GAS" else "warning"; narrative = f"🚀 Fase Líquida Óptima.{sys_warn}"
                    else: signal = "ACUMULAR"; category = "info" if m_status != "GAS" else "neutral"; narrative = f"📈 Tendencia Sana.{sys_warn}"
                elif z_liq < -0.3 and abs(trend_pct) < 0.02: signal = "PLASMA"; category = "neutral"; narrative = "🟡 Iliquidez (Mercado Seco)."
                data_list.append({"Ticker": ticker, "Price": current_price, "Signal": signal, "Category": category, "Narrative": narrative, "Entropy": z_entropy, "Liquidity": z_liq, "Trend": trend_pct * 100, "Psi": psi_score, "Raw_Vol": raw_vol, "Raw_Vol_Ratio": raw_vol_ratio, "SMA_Price": sma_val, "Exit_Limit": exit_threshold * 100})
        except: pass
    return pd.DataFrame(data_list).sort_values('Psi', ascending=False) if data_list else pd.DataFrame(), m_status, m_entropy, m_msg

# --- BACKTEST ---
def run_backtest(ticker, start, end, capital, risk_tolerance):
    try:
        df = yf.Ticker(ticker.strip().upper()).history(start=start, end=end)
        if df.empty: return None
        if df.index.tz: df.index = df.index.tz_localize(None)
        df['SMA'] = df['Close'].rolling(50).mean(); df['Trend'] = (df['Close'] - df['SMA']) / df['SMA']
        df['Ret'] = df['Close'].pct_change(); df['Vol_Ann'] = df['Ret'].rolling(20).std() * np.sqrt(252) * 100
        df['Z_Entropy'] = (df['Vol_Ann'] - 20) / 15
        df['Vol_SMA'] = df['Volume'].rolling(20).mean(); df['Z_Liq'] = (df['Volume'] - df['Vol_SMA']) / df['Vol_SMA']
        entropy_limit = risk_tolerance
        conditions = [(df['Z_Entropy'] > entropy_limit) & (df['Trend'] < 0.15), (df['Z_Liq'] < -0.3), (df['Trend'] > 0.02) & (df['Z_Entropy'] <= entropy_limit) & (df['Z_Liq'] > 0)]
        choices = ['GAS', 'PLASMA', 'LIQUID']; df['Phase'] = np.select(conditions, choices, default='SOLID')
        df['Signal'] = 0
        buy_cond = (df['Phase'].isin(['LIQUID', 'SOLID'])) & (df['Trend'] > 0)
        exit_limit = -0.15 if risk_tolerance >= 5 else (-0.10 if risk_tolerance >= 3 else -0.05)
        sell_cond = (df['Phase'] == 'GAS') | (df['Trend'] < exit_limit)
        df.loc[buy_cond, 'Signal'] = 1; df.loc[sell_cond, 'Signal'] = 0; df['Signal'] = df['Signal'].ffill().fillna(0)
        df['Strat_Ret'] = df['Close'].pct_change() * df['Signal'].shift(1); df.dropna(inplace=True)
        df['Eq_Strat'] = capital * (1 + df['Strat_Ret']).cumprod(); df['Eq_BH'] = capital * (1 + df['Close'].pct_change()).cumprod()
        return df
    except: return None

# --- ORÁCULO ---
def run_oracle_sim(ticker, days, risk_tolerance):
    try:
        stock = yf.Ticker(ticker); hist = stock.history(period="1y")
        if len(hist) < 50: return None, 0, 0
        last_price = hist['Close'].iloc[-1]; returns = hist['Close'].pct_change().dropna(); daily_vol = returns.std()
        sma_50 = hist['Close'].rolling(50).mean().iloc[-1]; trend_force = (last_price - sma_50) / sma_50
        hist_drift = returns.mean() - 0.5 * (daily_vol ** 2); trend_drift = trend_force / 252 * 2 
        final_drift = (hist_drift * 0.3) + (trend_drift * 0.7) if risk_tolerance >= 3 else hist_drift
        unique_seed = int(sum(ord(c) for c in ticker) + days); np.random.seed(unique_seed)
        sims = 1000; paths = np.zeros((days, sims)); paths[0] = last_price
        proj_h = (daily_vol * np.sqrt(252) * 100 - 20) / 15
        for t in range(1, days): shock = np.random.normal(0, daily_vol, sims); paths[t] = paths[t-1] * np.exp(final_drift + shock)
        return paths, proj_h, trend_force
    except: return None, 0, 0

# ==============================================================================
# 3. INTERFAZ DE USUARIO (FRONT-END)
# ==============================================================================

with st.sidebar:
    st.title("📡 FAROS")
    st.caption("**By SG Consulting Group y Emporium**")
    app_mode = st.radio("SISTEMA:", ["🌎 MACRO ECONOMÍA", "💼 GESTIÓN PORTAFOLIOS", "🔍 SCANNER MERCADO", "⏳ BACKTEST LAB", "🔮 ORÁCULO FUTURO"])
    st.markdown("---")
    risk_profile = st.select_slider("Perfil de Riesgo", options=["Conservador", "Growth", "Quantum"], value="Growth")
    if "Conservador" in risk_profile: risk_sigma = 2.0
    elif "Growth" in risk_profile: risk_sigma = 3.0
    else: risk_sigma = 5.0 
    st.caption(f"Límite Entropía: **{risk_sigma}σ**")

# --------------------------------------------------------------------------
# MÓDULO: MACRO
# --------------------------------------------------------------------------
if app_mode == "🌎 MACRO ECONOMÍA":
    st.title("Observatorio Macroeconómico")
    country_sel = st.selectbox("Seleccionar Jurisdicción:", ["USA", "MEXICO", "EUROPA", "CHINA", "BRASIL", "JAPON", "ARGENTINA"])
    if st.button("Escanear Economía"):
        with st.spinner(f"Analizando indicadores de {country_sel}..."):
            macro_data = analyze_country(country_sel)
            oil = yf.Ticker("CL=F").history(period="5d")['Close'].iloc[-1]; tnx = yf.Ticker("^TNX").history(period="5d")['Close'].iloc[-1]
        if macro_data:
            psi = macro_data['Macro_Score']; color_psi = "green" if psi > 60 else "orange" if psi > 40 else "red"
            st.markdown(f"<div class='macro-card'><h3 style='margin:0;'>ÍNDICE DE SALUD MACRO ({macro_data['Name']})</h3><h1 style='margin:0; font-size:4rem; color:{color_psi};'>{psi:.0f}/100</h1></div>", unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mercado (Acciones)", f"${macro_data['ETF_Price']:.2f}", f"{macro_data['ETF_Trend']*100:+.1f}%")
            c2.metric("Divisa vs USD", f"{macro_data['FX_Price']:.2f}", f"{macro_data['Local_FX_Trend']*100:+.1f}%", delta_color="normal")
            c3.metric("Riesgo (Bono 10Y)", f"{tnx:.2f}%"); c4.metric("Energía (WTI)", f"${oil:.2f}")
            st.subheader("Dinámica"); df_ch = pd.DataFrame()
            etf = yf.Ticker(macro_data['ETF_Ticker']).history(period="1y")['Close']; fx = yf.Ticker(macro_data['FX_Ticker']).history(period="1y")['Close']
            df_ch['Mercado'] = (etf/etf.iloc[0])*100; df_ch['Divisa'] = (fx/fx.iloc[0])*100
            st.plotly_chart(px.line(df_ch), use_container_width=True)

# --------------------------------------------------------------------------
# MÓDULO: PORTAFOLIO
# --------------------------------------------------------------------------
elif app_mode == "💼 GESTIÓN PORTAFOLIOS":
    st.title("Gestión de Activos & Riesgo")
    
    with st.expander("📝 Editar Composición del Portafolio", expanded=True):
        c1, c2 = st.columns(2)
        sel_assets = c1.multiselect("Seleccionar Activos:", options=list(ASSET_DB.keys()), default=["PALANTIR (PLTR)", "NVIDIA (NVDA)"])
        manual_assets = c1.text_input("Otros Tickers (separados por coma):", "")
        
        if c1.button("⚡ Auto-Balancear (Criterio TAI)"):
            with st.spinner("Calculando asignación óptima..."):
                final_tickers = get_tickers_from_selection(sel_assets, manual_assets)
                recommended_weights = calculate_tai_weights(final_tickers, risk_sigma)
                st.session_state['weights_area'] = recommended_weights
        
        default_val = st.session_state.get('weights_area', "")
        if not default_val:
            final_tickers = get_tickers_from_selection(sel_assets, manual_assets)
            if final_tickers: default_val = "\n".join([f"{t}, {1/len(final_tickers):.2f}" for t in final_tickers])

        weights_input = c2.text_area("Distribución de Capital (Ticker, Peso):", value=default_val, height=150)
        
        if c2.button("Analizar Cartera"):
            try:
                holdings = {}; [holdings.update({l.split(',')[0].strip().upper(): float(l.split(',')[1].strip())}) for l in weights_input.split('\n') if len(l.split(','))==2]
                st.session_state['holdings'] = holdings
            except: st.error("Error formato")

    if 'holdings' in st.session_state:
        df_p, corr_m, metrics = analyze_portfolio(st.session_state['holdings'], risk_sigma)
        if df_p is not None:
            st.markdown("### 📊 Termodinámica del Portafolio")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Activos", len(df_p)); k2.metric("Beta", f"{metrics['Beta']:.2f}x"); k3.metric("Score TAI", f"{metrics['Psi']:.0f}/100"); k4.metric("Estado", "LÍQUIDO" if metrics['Psi']>50 else "GASEOSO")
            if not corr_m.empty: st.plotly_chart(px.imshow(corr_m, text_auto=True, color_continuous_scale='RdBu_r'), use_container_width=True)
            
            st.subheader("🚨 Alertas Tácticas")
            critical = df_p[df_p['Action'] != "MANTENER"]
            if not critical.empty:
                for i, row in critical.iterrows():
                    color = "red" if "REDUCIR" in row['Action'] else "green"
                    st.markdown(f"<div style='padding:10px; border:1px solid {color}; border-radius:5px; margin-bottom:5px;'><b>{row['Ticker']}:</b> {row['Action']} (Ψ: {row['Psi']:.0f})</div>", unsafe_allow_html=True)
            else: st.success("✅ Portafolio saludable.")
            
            if st.button("🖨️ Generar Informe de Auditoría"):
                report_html = generate_portfolio_report(df_p, metrics, risk_profile)
                st.download_button("Descargar Informe (HTML)", report_html, "auditoria_faros.html", "text/html")
            
            with st.expander("📄 Detalle Técnico"): st.dataframe(df_p)

# --------------------------------------------------------------------------
# MÓDULO: SCANNER (PERSISTENCE FIX)
# --------------------------------------------------------------------------
elif app_mode == "🔍 SCANNER MERCADO":
    time_h = st.selectbox("Horizonte", ["Corto Plazo", "Medio Plazo", "Largo Plazo"])
    if "Corto" in time_h: cfg = {'volatility': 10, 'trend': 20, 'download': '3mo'}
    elif "Medio" in time_h: cfg = {'volatility': 20, 'trend': 50, 'download': '6mo'}
    else: cfg = {'volatility': 60, 'trend': 200, 'download': '2y'}
    
    col_search, col_manual = st.columns([3, 1])
    selected_assets = col_search.multiselect("Buscar Empresas:", options=list(ASSET_DB.keys()), default=["PALANTIR (PLTR)", "NVIDIA (NVDA)", "D-WAVE (QBTS)"])
    manual_tickers = col_manual.text_input("Otros (Tickers):", "")
    
    if st.button("Analizar Mercado"): 
        st.cache_data.clear()
        target_list = get_tickers_from_selection(selected_assets, manual_tickers)
        df, m_status, m_entropy, m_msg = get_live_data(target_list, cfg, risk_sigma)
        
        # SAVE TO SESSION
        st.session_state['scan_data'] = df
        st.session_state['scan_meta'] = (m_status, m_entropy, m_msg)
    
    # RENDER FROM SESSION
    if 'scan_data' in st.session_state:
        df = st.session_state['scan_data']
        m_status, m_entropy, m_msg = st.session_state['scan_meta']
        
        cols = {"GAS":("#FFCDD2","#B71C1C","🔥"), "WARNING":("#FFF9C4","#F57F17","⚠️"), "LIQUID":("#C8E6C9","#1B5E20","🌍")}
        bg, txt, ico = cols.get(m_status, ("#eee","#333","❓"))
        
        st.markdown(f"<div class='global-status' style='background-color:{bg}; color:{txt};'>{ico} SPY: {m_msg}</div>", unsafe_allow_html=True)
        
        if not df.empty:
            if st.button("🖨️ Generar Informe de Oportunidades"):
                scan_html = generate_scanner_report(df, m_status, risk_profile)
                st.download_button("Descargar Informe (HTML)", scan_html, "escaneo_faros.html", "text/html")
            
            c1, c2 = st.columns([2,1])
            with c2: st.plotly_chart(px.scatter(df, x="Entropy", y="Liquidity", color="Category", text="Ticker"), use_container_width=True)
            with c1:
                for i, r in df.iterrows():
                    with st.container(border=True):
                        hc1, hc2 = st.columns([3,1]); hc1.markdown(f"### **{r['Ticker']}** ${r['Price']:.2f}"); hc2.markdown(f"**Ψ: {r['Psi']:.0f}**")
                        msg = f"**{r['Signal']}**: {r['Narrative']}"
                        if r['Category']=='success': st.success(msg)
                        elif r['Category']=='warning': st.warning(msg)
                        elif r['Category']=='danger': st.error(msg)
                        else: st.info(msg)
                        with st.expander("🔬 Lab Data"):
                            st.markdown(f"**Vol:** {r['Raw_Vol']:.0f}% ({r['Entropy']:.2f}σ) | **Tendencia:** {r['Trend']:+.1f}% | **Liq:** {r['Raw_Vol_Ratio']:.1f}x")

# --------------------------------------------------------------------------
# MÓDULO: BACKTEST
# --------------------------------------------------------------------------
elif app_mode == "⏳ BACKTEST LAB":
    st.title("Validación Histórica")
    c_tick, c_cap = st.columns([1, 1]); tck = c_tick.text_input("Activo:", "PLTR").upper(); cap = c_cap.number_input("Capital", value=10000)
    c1, c2 = st.columns(2); d1 = c1.date_input("Inicio", pd.to_datetime("2023-01-01")); d2 = c2.date_input("Fin", pd.to_datetime("2025-01-05"))
    if st.button("Ejecutar"):
        res = run_backtest(tck, d1, d2, cap, risk_sigma)
        if res is not None:
            fin = res['Eq_Strat'].iloc[-1]; st.metric("Resultado Final", f"${fin:,.0f}", delta=f"{(fin/cap-1)*100:.1f}%")
            st.subheader("Ciclos Detectados")
            fig = go.Figure(); fig.add_trace(go.Scatter(x=res.index, y=res['Close'], name='Precio', line=dict(color='black', width=1), opacity=0.3))
            colors = {'GAS': 'red', 'LIQUID': '#00FF41', 'PLASMA': '#FFD700', 'SOLID': 'gray'}
            for p, c in colors.items(): 
                s = res[res['Phase']==p]; 
                if not s.empty: fig.add_trace(go.Scatter(x=s.index, y=s['Close'], mode='markers', name=p, marker=dict(color=c, size=5)))
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Curva de Capital (Tu Dinero)")
            fig2 = go.Figure(); fig2.add_trace(go.Scatter(x=res.index, y=res['Eq_Strat'], name='FAROS', line=dict(color='blue', width=2))); fig2.add_trace(go.Scatter(x=res.index, y=res['Eq_BH'], name='Buy & Hold', line=dict(color='gray', dash='dot')))
            st.plotly_chart(fig2, use_container_width=True)

# --------------------------------------------------------------------------
# MÓDULO: ORÁCULO (FIXED CHART)
# --------------------------------------------------------------------------
elif app_mode == "🔮 ORÁCULO FUTURO":
    st.title("Proyección TAI (Potencial Φ)")
    c_tick, c_days = st.columns([1, 1]); o_tick = c_tick.text_input("Activo:", "PLTR").upper(); o_days = c_days.slider("Días:", 30, 365, 365)
    if st.button("Consultar"):
        paths, proj_h, trend_f = run_oracle_sim(o_tick, o_days, risk_sigma)
        if paths is not None:
            start = paths[0][0]; final = paths[-1]
            p95 = np.percentile(final, 95); p50 = np.percentile(final, 50); p05 = np.percentile(final, 5)
            win_rate = np.mean(final > start); upside = (p95 - start)/start; downside = abs((p05 - start)/start); rr = upside/downside if downside > 0 else 10
            bonus = 20 if trend_f > 0.10 else 0
            phi = (win_rate * 50) + (min(rr, 4) * 10) + bonus; phi = max(0, min(100, phi))
            p_col = "green" if phi > 70 else "orange" if phi > 40 else "red"
            st.markdown(f"<div style='text-align:center; padding:15px; border:1px solid #ddd; border-radius:10px; background-color:#f9f9f9;'><h2 style='margin:0;'>POTENCIAL FUTURO (Φ)</h2><h1 style='margin:0; font-size:4rem; color:{p_col};'>{phi:.0f}/100</h1><p>Probabilidad: <b>{win_rate*100:.0f}%</b> | R/R: <b>{rr:.1f}x</b></p></div>", unsafe_allow_html=True)
            k1, k2, k3 = st.columns(3); k1.metric("🟢 Techo", f"${p95:.2f}"); k2.metric("🔵 Base", f"${p50:.2f}"); k3.metric("🔴 Suelo", f"${p05:.2f}")
            with st.expander("📖 Interpretación Táctica de Escenarios", expanded=True):
                st.markdown("""
                * **🟢 Techo (Optimista):** Rendimiento excepcional (Top 5% de probabilidad). Si llega aquí, es un 'Moonshot'.
                * **🔵 Base (Probable):** Mediana estadística. Si la inercia actual se mantiene, el precio orbitará esta zona.
                * **🔴 Suelo (Riesgo):** Escenario de Cisne Negro (Peor 5%). Este nivel marca tu **Riesgo Máximo Probable**.
                """)
            
            # --- FIXED PLOTLY CHART (Explicit Traces) ---
            fig = go.Figure()
            # Gray simulations
            for i in range(50): fig.add_trace(go.Scatter(y=paths[:, i], line=dict(color='gray', width=0.5), opacity=0.1, showlegend=False))
            # Green (Optimista)
            fig.add_trace(go.Scatter(y=np.percentile(paths, 95, axis=1), name='Optimista', line=dict(color='green', dash='dash', width=2)))
            # Blue (Trend)
            fig.add_trace(go.Scatter(y=np.percentile(paths, 50, axis=1), name='Tendencia', line=dict(color='blue', width=3)))
            # Red (Pesimista)
            fig.add_trace(go.Scatter(y=np.percentile(paths, 5, axis=1), name='Pesimista', line=dict(color='red', dash='dash', width=2)))
            
            st.plotly_chart(fig, use_container_width=True)
