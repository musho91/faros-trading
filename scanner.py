# ==============================================================================
# FAROS v7.0 - INSTITUTIONAL QUANT SUITE
# Autor: Juan Arroyo | SG Consulting Group
# Design: Option B — Clean Institutional
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from datetime import datetime
from physics_engine import FarosPhysics

fisica = FarosPhysics()

st.set_page_config(page_title="FAROS Institutional", page_icon="⬡", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif}
.stApp{background-color:#f8f9fc}
section[data-testid="stSidebar"]{background-color:#ffffff !important;border-right:1px solid #e5e7eb}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label p{color:#374151 !important;font-size:0.875rem !important}
section[data-testid="stSidebar"] label{color:#374151 !important}
#MainMenu,footer,header{visibility:hidden}
.stDeployButton{display:none}
.block-container{padding-top:1.5rem;padding-bottom:2rem;max-width:1200px}
.stButton>button[kind="primary"]{background:#1d4ed8 !important;color:white !important;border:none !important;border-radius:8px !important;font-weight:500 !important;padding:0.5rem 1.25rem !important}
.stButton>button[kind="primary"]:hover{background:#1e40af !important}
.stButton>button{border-radius:8px !important;font-weight:500 !important;border:1px solid #d1d5db !important;color:#374151 !important}
.stButton>button:hover{border-color:#1d4ed8 !important;color:#1d4ed8 !important}
.stTextInput>div>div>input{border-radius:8px !important;border:1px solid #d1d5db !important;background:white !important;color:#111827 !important}
.stTextInput>div>div>input::placeholder{color:#9ca3af !important}
.stSelectbox>div>div,.stMultiSelect>div>div{border-radius:8px !important;border:1px solid #d1d5db !important;background:white !important}
[data-testid="metric-container"]{background:white !important;border:1px solid #e5e7eb !important;border-radius:10px !important;padding:1rem 1.25rem !important}
[data-testid="metric-container"] label{font-size:0.72rem !important;font-weight:500 !important;letter-spacing:0.06em !important;text-transform:uppercase !important;color:#6b7280 !important}
[data-testid="metric-container"] [data-testid="stMetricValue"]{font-size:1.5rem !important;font-weight:600 !important;color:#0f172a !important}
[data-testid="stDataFrame"]{border:1px solid #e5e7eb !important;border-radius:10px !important;overflow:hidden !important}
[data-testid="stExpander"]{border:1px solid #e5e7eb !important;border-radius:10px !important;background:white !important}
.stProgress>div>div{background:#1d4ed8 !important;border-radius:4px !important}
.stProgress>div{background:#e5e7eb !important;border-radius:4px !important}
[data-testid="stChatMessage"]{border-radius:10px !important;border:1px solid #e5e7eb !important;background:white !important;margin-bottom:0.75rem !important}
[data-testid="stChatInputContainer"]{border-radius:10px !important;border:1px solid #d1d5db !important;background:white !important}
.stNumberInput>div>div>input{border-radius:8px !important;border:1px solid #d1d5db !important;color:#111827 !important}
.stAlert{border-radius:8px !important}
p,span,label{color:#374151}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# HELPERS
# ==============================================================================
def metric_card(label, value, sub="", sub_color="#6b7280"):
    s = f'<div style="font-size:0.75rem;color:{sub_color};margin-top:4px;">{sub}</div>' if sub else ""
    return f'''<div style="background:white;border:1px solid #e5e7eb;border-radius:10px;padding:1rem 1.25rem;">
        <div style="font-size:0.7rem;font-weight:500;letter-spacing:0.08em;text-transform:uppercase;color:#9ca3af;margin-bottom:6px;">{label}</div>
        <div style="font-size:1.5rem;font-weight:600;color:#0f172a;line-height:1.1;">{value}</div>{s}</div>'''

def section_header(title, subtitle=""):
    s = f'<p style="color:#6b7280;font-size:0.85rem;margin:4px 0 0 0;">{subtitle}</p>' if subtitle else ""
    return f'''<div style="margin-bottom:1.5rem;padding-bottom:1rem;border-bottom:1px solid #e5e7eb;">
        <h1 style="font-size:1.4rem;font-weight:600;color:#0f172a;margin:0;letter-spacing:-0.01em;">{title}</h1>{s}</div>'''

def signal_bar(ticker, regime, sig_text, sig_color, sig_bg, extra=""):
    return f'''<div style="display:flex;align-items:center;justify-content:space-between;
        background:{sig_bg};border:1px solid {sig_color}22;border-radius:10px;
        padding:0.875rem 1.25rem;margin:0.75rem 0;">
        <div style="display:flex;align-items:center;gap:12px;">
            <span style="font-size:1rem;font-weight:700;color:{sig_color};">{sig_text}</span>
            <span style="font-size:0.8rem;color:#374151;font-weight:500;">{ticker}</span>
            <span style="font-size:0.72rem;color:{sig_color};background:{sig_bg};
                border:1px solid {sig_color}44;border-radius:20px;padding:2px 10px;
                font-weight:600;letter-spacing:0.04em;">{regime}</span>
        </div>
        <div style="font-size:0.78rem;color:#6b7280;">{extra}</div>
    </div>'''

PLOT_LAYOUT = dict(
    paper_bgcolor='white', plot_bgcolor='#fafbfc',
    font=dict(family="Inter, sans-serif", color="#374151", size=12),
    margin=dict(l=12, r=12, t=40, b=12),
    xaxis=dict(showgrid=False, linecolor='#e5e7eb', tickfont=dict(size=11)),
    yaxis=dict(gridcolor='#f1f5f9', linecolor='#e5e7eb', tickfont=dict(size=11)),
    legend=dict(bgcolor='white', bordercolor='#e5e7eb', borderwidth=1,
                font=dict(size=11), orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
    hovermode='x unified'
)

# ==============================================================================
# ASSET DB
# ==============================================================================
ASSET_DB = {
    "NVIDIA Corp (NVDA)": "NVDA", "Palantir Tech (PLTR)": "PLTR",
    "Tesla Inc (TSLA)": "TSLA", "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD", "Apple Inc (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT", "Amazon (AMZN)": "AMZN",
    "Alphabet (GOOGL)": "GOOGL", "Meta Platforms (META)": "META",
    "S&P 500 ETF (SPY)": "SPY", "Nasdaq 100 (QQQ)": "QQQ",
    "Russell 2000 (IWM)": "IWM", "Coinbase (COIN)": "COIN",
    "MicroStrategy (MSTR)": "MSTR", "D-Wave Quantum (QBTS)": "QBTS",
    "IonQ Inc (IONQ)": "IONQ", "C3.ai (AI)": "AI",
}

def get_ticker_list(selection, manual_input):
    out = [ASSET_DB[i] for i in selection if i in ASSET_DB]
    if manual_input:
        out.extend([x.strip().upper() for x in manual_input.split(',') if x.strip()])
    return list(set(out))

# ==============================================================================
# DATA
# ==============================================================================
@st.cache_data(ttl=600)
def fetch_market_data(ticker, period="1y"):
    try:
        df = yf.Ticker(ticker).history(period=period)
        return df if not df.empty else pd.DataFrame()
    except:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_global_context(profile):
    spy = fetch_market_data("SPY", "1y")
    if spy.empty:
        return "offline", 0, "—"
    try:
        m = fisica.calcular_metricas_completas(spy, profile)
        return (m.regime, m.psi, f"{m.reynolds_pct:.0f}%ile") if m else ("neutral", 0, "—")
    except:
        return "neutral", 0, "—"

def signal_from_psi(psi, regime):
    if "ACCUMULATION" in regime and psi >= 50: return "✓ BUY",   "#15803d", "#f0fdf4"
    elif "MOMENTUM" in regime:                  return "▲ BUY+",  "#1d4ed8", "#eff6ff"
    elif "CONSOLIDATION" in regime:             return "◆ HOLD",  "#6b7280", "#f9fafb"
    elif "BREAK" in regime:                     return "⚠ CASH",  "#7c3aed", "#faf5ff"
    else:                                       return "✕ SELL",  "#b91c1c", "#fef2f2"

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown('''
    <div style="padding:0.5rem 0 1.25rem;">
      <div style="display:flex;align-items:center;gap:10px;">
        <div style="width:34px;height:34px;background:#1d4ed8;border-radius:8px;
          display:flex;align-items:center;justify-content:center;
          color:white;font-size:15px;font-weight:700;flex-shrink:0;">⬡</div>
        <div>
          <div style="font-size:1rem;font-weight:700;color:#0f172a;letter-spacing:-0.01em;">FAROS</div>
          <div style="font-size:0.68rem;color:#6b7280;letter-spacing:0.04em;">TAI-ACF Framework v3.0</div>
        </div>
      </div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('<div style="border-top:1px solid #e5e7eb;margin-bottom:1rem;"></div>', unsafe_allow_html=True)

    st.markdown('<div style="font-size:0.68rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:#9ca3af;margin-bottom:8px;">Perfil de riesgo</div>', unsafe_allow_html=True)
    risk_profile = st.select_slider("", options=["Conservador", "Growth", "Quantum"], value="Growth", label_visibility="collapsed")

    st.markdown('<div style="border-top:1px solid #e5e7eb;margin:1rem 0;"></div>', unsafe_allow_html=True)

    st.markdown('<div style="font-size:0.68rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:#9ca3af;margin-bottom:8px;">Módulos</div>', unsafe_allow_html=True)
    app_mode = st.radio("", [
        "🤖  Quant Analyst",
        "💼  Portfolio Builder",
        "🔍  Alpha Scanner",
        "⏳  Backtest Lab",
        "🔮  Oracle Projections",
    ], label_visibility="collapsed")

    st.markdown('<div style="border-top:1px solid #e5e7eb;margin:1rem 0;"></div>', unsafe_allow_html=True)

    g_regime, g_psi, g_re = get_global_context(risk_profile)
    RCFG = {
        "INSTITUTIONAL ACCUMULATION": ("#15803d", "#f0fdf4", "✓ ACCUMULATION"),
        "HIGH MOMENTUM":              ("#1d4ed8", "#eff6ff", "▲ HIGH MOMENTUM"),
        "CONSOLIDATION":              ("#6b7280", "#f9fafb", "◆ CONSOLIDATION"),
        "DISTRIBUTION/BEAR":          ("#b91c1c", "#fef2f2", "▼ BEAR"),
        "STRUCTURAL BREAK":           ("#7c3aed", "#faf5ff", "⚠ BREAK"),
    }
    rc, rbg, rl = RCFG.get(g_regime, ("#6b7280", "#f9fafb", g_regime or "LOADING"))

    st.markdown(f'''
    <div style="font-size:0.68rem;font-weight:600;letter-spacing:0.1em;
        text-transform:uppercase;color:#9ca3af;margin-bottom:8px;">S&P 500 · Contexto global</div>
    <div style="background:{rbg};border:1px solid {rc}33;border-radius:8px;padding:.75rem;">
        <div style="font-size:0.72rem;font-weight:700;color:{rc};letter-spacing:.04em;">{rl}</div>
        <div style="display:flex;justify-content:space-between;margin-top:8px;">
            <div style="font-size:0.75rem;color:#374151;">
                <span style="color:#6b7280;">Ψ</span>
                <strong style="margin-left:4px;color:#0f172a;">{g_psi:.0f}</strong>
            </div>
            <div style="font-size:0.75rem;color:#374151;">
                <span style="color:#6b7280;">Re</span>
                <strong style="margin-left:4px;color:#0f172a;">{g_re}</strong>
            </div>
        </div>
    </div>
    <div style="margin-top:1.25rem;font-size:0.68rem;color:#9ca3af;text-align:center;">
        {datetime.now().strftime("%d %b %Y · %H:%M")} UTC
    </div>
    ''', unsafe_allow_html=True)

# ==============================================================================
# MÓDULO 1 — QUANT ANALYST
# ==============================================================================
if "🤖" in app_mode:
    st.markdown(section_header("Quant Analyst", "Análisis estructural · TAI-ACF Navier-Stokes Financial Hydrodynamics"), unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"], unsafe_allow_html=True)

    if prompt := st.chat_input("Escribe un ticker — ej. NVDA, PLTR, BTC-USD..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Calculando métricas TAI-ACF..."):
                tokens = prompt.upper().replace("$", "").split()
                ticker = next(
                    (t for t in reversed(tokens) if t.replace("-", "").isalpha() and 1 <= len(t) <= 7),
                    tokens[-1]
                )
                df = fetch_market_data(ticker, "2y")

                if not df.empty:
                    metrics = fisica.calcular_metricas_completas(df, risk_profile)
                    if metrics:
                        lp = df['Close'].iloc[-1]
                        sig_text, sig_color, sig_bg = signal_from_psi(metrics.psi, metrics.regime)

                        c1, c2, c3, c4 = st.columns(4)
                        c1.markdown(metric_card("Último precio", f"${lp:,.2f}"), unsafe_allow_html=True)
                        c2.markdown(metric_card("Governance Ψ", f"{metrics.psi:.1f}", sub=sig_text, sub_color=sig_color), unsafe_allow_html=True)
                        re_lbl = "Laminar" if metrics.reynolds_pct < 50 else ("Transición" if metrics.reynolds_pct < 75 else "Turbulento")
                        re_col = "#15803d" if metrics.reynolds_pct < 50 else ("#d97706" if metrics.reynolds_pct < 75 else "#b91c1c")
                        c3.markdown(metric_card("Reynolds %ile", f"{metrics.reynolds_pct:.0f}", sub=re_lbl, sub_color=re_col), unsafe_allow_html=True)
                        c4.markdown(metric_card("Future Alpha", f"{metrics.future_score:.1f}", sub=f"α-flow {metrics.alpha_flow:.2f}", sub_color="#1d4ed8"), unsafe_allow_html=True)

                        st.markdown(signal_bar(
                            ticker, metrics.regime, sig_text, sig_color, sig_bg,
                            extra=f"Z-Score {metrics.z_score_price:.2f} · H {metrics.shannon_entropy:.2f} · µ {metrics.viscosity_mu:.4f}"
                        ), unsafe_allow_html=True)

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name=ticker,
                            line=dict(color='#1d4ed8', width=2),
                            fill='tozeroy', fillcolor='rgba(29,78,216,0.04)'))
                        fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(50).mean(),
                            name='SMA 50', line=dict(color='#f59e0b', width=1.5, dash='dot')))
                        fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(200).mean(),
                            name='SMA 200', line=dict(color='#ef4444', width=1.5, dash='dot')))
                        fig.update_layout(**PLOT_LAYOUT, height=280,
                            title=dict(text=f"{ticker} · Precio histórico con SMAs", font=dict(size=13), x=0))
                        st.plotly_chart(fig, use_container_width=True)

                        with st.expander("Métricas físicas completas (TAI-ACF)"):
                            d1, d2, d3, d4 = st.columns(4)
                            d1.metric("Shannon H", f"{metrics.shannon_entropy:.3f}")
                            d2.metric("α-flow", f"{metrics.alpha_flow:.3f}")
                            d3.metric("Viscosidad µ", f"{metrics.viscosity_mu:.5f}")
                            d4.metric("Densidad ρ", f"{metrics.density_rho:.3f}")

                        resp = f"**{ticker}** — `{metrics.regime}` · Ψ `{metrics.psi:.1f}` · {sig_text}"
                    else:
                        resp = f"⚠️ Datos insuficientes para **{ticker}**."
                else:
                    resp = f"❌ No se encontraron datos para **{ticker}**."

                st.markdown(resp)
                st.session_state.chat_history.append({"role": "assistant", "content": resp})

# ==============================================================================
# MÓDULO 2 — PORTFOLIO BUILDER
# ==============================================================================
elif "💼" in app_mode:
    st.markdown(section_header("Portfolio Builder", "Asignación óptima de capital ponderada por Governance Score Ψ"), unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])
    with c1:
        selection = st.multiselect("Seleccionar activos", list(ASSET_DB.keys()),
            default=["NVIDIA Corp (NVDA)", "Palantir Tech (PLTR)", "Apple Inc (AAPL)", "Bitcoin (BTC)"])
    with c2:
        manual = st.text_input("Tickers adicionales (separados por coma)", placeholder="ORCL, AMD...")

    capital = st.number_input("Capital del portafolio (USD)", min_value=1000, value=100000, step=5000)

    if st.button("Construir portafolio", type="primary"):
        tickers = get_ticker_list(selection, manual)
        if not tickers:
            st.warning("Selecciona al menos un activo.")
        else:
            scores, log_data = {}, []
            prog = st.progress(0)
            status = st.empty()
            for i, t in enumerate(tickers):
                status.markdown(f'<div style="font-size:0.8rem;color:#6b7280;">Analizando {t}...</div>', unsafe_allow_html=True)
                df = fetch_market_data(t, "1y")
                if not df.empty:
                    re, future, psi, regime = fisica.calcular_metricas_institucionales(df, risk_profile)
                    if psi > 20:
                        scores[t] = psi
                    log_data.append({
                        "Activo": t, "Precio": f"${df['Close'].iloc[-1]:,.2f}",
                        "Régimen": regime, "Future Alpha": round(future, 1),
                        "Governance Ψ": round(psi, 1),
                        "Estado": "✓ Incluido" if psi > 20 else "✕ Excluido"
                    })
                prog.progress((i + 1) / len(tickers))
            status.empty()
            prog.empty()

            if scores:
                total = sum(scores.values())
                weights = {t: s / total for t, s in scores.items()}

                k1, k2, k3 = st.columns(3)
                k1.metric("Activos invertibles", len(scores), f"de {len(tickers)} analizados")
                k2.metric("Capital asignado", f"${capital:,.0f}")
                k3.metric("Ψ promedio", f"{sum(scores.values()) / len(scores):.1f}")

                st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

                cc, tc = st.columns([1, 1])
                with cc:
                    colors = ['#1d4ed8', '#3b82f6', '#60a5fa', '#93c5fd', '#bfdbfe']
                    fig_pie = go.Figure(go.Pie(
                        labels=list(weights.keys()),
                        values=[round(w * 100, 1) for w in weights.values()],
                        hole=0.5, marker=dict(colors=colors[:len(weights)]),
                        textinfo='label+percent', textfont=dict(size=11)
                    ))
                    fig_pie.update_layout(
                        paper_bgcolor='white', font=dict(family="Inter"),
                        margin=dict(l=0, r=0, t=30, b=0), height=300,
                        showlegend=False,
                        title=dict(text="Distribución del capital", font=dict(size=13), x=0)
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                with tc:
                    st.dataframe(pd.DataFrame([{
                        "Ticker": t, "Peso": f"{w*100:.1f}%",
                        "USD": f"${w*capital:,.0f}", "Ψ": round(scores[t], 1)
                    } for t, w in weights.items()]),
                    use_container_width=True, hide_index=True, height=300)

                with st.expander("Ver análisis cuantitativo completo"):
                    st.dataframe(pd.DataFrame(log_data), use_container_width=True, hide_index=True)
            else:
                st.warning("Ningún activo supera Ψ > 20. El sistema recomienda mantener cash.")

# ==============================================================================
# MÓDULO 3 — ALPHA SCANNER
# ==============================================================================
elif "🔍" in app_mode:
    st.markdown(section_header("Alpha Scanner", "Universo institucional · Ranking por Governance Score Ψ"), unsafe_allow_html=True)

    sel_scan = st.multiselect("Universo de activos", list(ASSET_DB.keys()),
        default=["NVIDIA Corp (NVDA)", "Palantir Tech (PLTR)", "Tesla Inc (TSLA)", "Bitcoin (BTC)"])

    if st.button("Ejecutar scanner", type="primary"):
        tickers = get_ticker_list(sel_scan, "")
        results = []
        prog = st.progress(0)
        status = st.empty()
        for i, t in enumerate(tickers):
            status.markdown(f'<div style="font-size:0.8rem;color:#6b7280;">Escaneando {t}...</div>', unsafe_allow_html=True)
            df = fetch_market_data(t, "1y")
            if not df.empty:
                try:
                    m = fisica.calcular_metricas_completas(df, risk_profile)
                    if m:
                        results.append({
                            "Ticker": t, "Precio": df['Close'].iloc[-1],
                            "Régimen": m.regime,
                            "Reynolds %ile": round(m.reynolds_pct, 0),
                            "Shannon H": round(m.shannon_entropy, 3),
                            "α-flow": round(m.alpha_flow, 3),
                            "Future Alpha": round(m.future_score, 1),
                            "Ψ Score": round(m.psi, 1),
                        })
                except:
                    pass
            prog.progress((i + 1) / len(tickers))
        status.empty()
        prog.empty()

        if results:
            df_res = pd.DataFrame(results).sort_values("Ψ Score", ascending=False)
            inv = len(df_res[df_res["Ψ Score"] > 20])
            top = df_res.iloc[0]

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Activos escaneados", len(df_res))
            k2.metric("Invertibles (Ψ > 20)", inv)
            k3.metric("Top activo", top["Ticker"])
            k4.metric("Ψ más alto", f"{top['Ψ Score']:.1f}")

            st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

            def color_regime(val):
                m = {
                    "ACCUMULATION": "background-color:#f0fdf4;color:#15803d",
                    "MOMENTUM":     "background-color:#eff6ff;color:#1d4ed8",
                    "BREAK":        "background-color:#faf5ff;color:#7c3aed",
                    "BEAR":         "background-color:#fef2f2;color:#b91c1c",
                    "CONSOLIDATION":"background-color:#f9fafb;color:#6b7280",
                }
                for k, v in m.items():
                    if k in str(val): return v
                return ''

            styled = (df_res.style
                .map(color_regime, subset=['Régimen'])
                .format({"Precio": "${:.2f}", "Future Alpha": "{:.1f}", "Ψ Score": "{:.1f}", "Reynolds %ile": "{:.0f}"})
                .background_gradient(subset=['Ψ Score'], cmap='Blues'))
            st.dataframe(styled, use_container_width=True, hide_index=True)

            fig_sc = px.scatter(df_res, x="Future Alpha", y="Ψ Score",
                color="Régimen", size="Ψ Score", hover_name="Ticker",
                title="Alpha Map — Potencial futuro vs. Governance Score Ψ",
                color_discrete_map={
                    "INSTITUTIONAL ACCUMULATION": "#15803d",
                    "HIGH MOMENTUM": "#1d4ed8",
                    "STRUCTURAL BREAK": "#7c3aed",
                    "CONSOLIDATION": "#9ca3af",
                    "DISTRIBUTION/BEAR": "#b91c1c",
                })
            fig_sc.update_layout(**PLOT_LAYOUT, height=400)
            st.plotly_chart(fig_sc, use_container_width=True)
        else:
            st.warning("No se obtuvieron datos para los activos seleccionados.")

# ==============================================================================
# MÓDULO 4 — BACKTEST LAB
# ==============================================================================
elif "⏳" in app_mode:
    st.markdown(section_header("Backtest Lab", "Validación histórica · FAROS Strategy vs. Buy & Hold"), unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    tck = c1.text_input("Ticker", "NVDA").upper()
    years = c2.selectbox("Período", ["1y", "2y", "5y"], index=1)
    vol_thresh = c3.slider("Filtro de volatilidad (diaria)", 0.01, 0.06, 0.025, 0.005, format="%.3f")

    if st.button("Ejecutar simulación", type="primary"):
        df = fetch_market_data(tck, years)
        if not df.empty:
            df['Ret']    = df['Close'].pct_change()
            df['SMA50']  = df['Close'].rolling(50).mean()
            df['SMA200'] = df['Close'].rolling(200).mean()
            vol = df['Ret'].rolling(20).std()

            sig = np.where((df['SMA50'] > df['SMA200']) & (vol < vol_thresh), 1, 0)
            if risk_profile == "Quantum":
                sig = np.where(df['Close'] > df['SMA50'] * 1.1, 1, sig)

            df['Signal']   = pd.Series(sig, index=df.index).shift(1).fillna(0)
            df['Strategy'] = (1 + df['Ret'] * df['Signal']).cumprod()
            df['BuyHold']  = (1 + df['Ret']).cumprod()

            ps = (df['Strategy'].iloc[-1] - 1) * 100
            pb = (df['BuyHold'].iloc[-1]  - 1) * 100
            sr = df['Ret'] * df['Signal']
            sharpe = (sr.mean() / sr.std() * np.sqrt(252)) if sr.std() > 0 else 0
            mdd = ((df['Strategy'] / df['Strategy'].cummax()) - 1).min() * 100

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("FAROS Strategy", f"{ps:.1f}%", delta=f"{ps - pb:+.1f}% vs B&H")
            k2.metric("Buy & Hold",     f"{pb:.1f}%")
            k3.metric("Sharpe Ratio",   f"{sharpe:.2f}")
            k4.metric("Max Drawdown",   f"{mdd:.1f}%")

            st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=df.index, y=df['BuyHold'], name='Buy & Hold',
                line=dict(color='#d1d5db', width=2, dash='dash')))
            fig_eq.add_trace(go.Scatter(x=df.index, y=df['Strategy'], name='FAROS Strategy',
                line=dict(color='#1d4ed8', width=2.5),
                fill='tozeroy', fillcolor='rgba(29,78,216,0.04)'))
            fig_eq.update_layout(**PLOT_LAYOUT, height=360,
                title=dict(text=f"Equity Curve — {tck} ({years})", font=dict(size=13), x=0),
                yaxis_title="Crecimiento de $1")
            st.plotly_chart(fig_eq, use_container_width=True)

            dd = (df['Strategy'] / df['Strategy'].cummax() - 1) * 100
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=df.index, y=dd, name='Drawdown',
                fill='tozeroy', fillcolor='rgba(239,68,68,0.07)',
                line=dict(color='#ef4444', width=1)))
            fig_dd.update_layout(**PLOT_LAYOUT, height=180,
                title=dict(text="Drawdown", font=dict(size=12), x=0),
                yaxis_tickformat=".1f")
            st.plotly_chart(fig_dd, use_container_width=True)
        else:
            st.error(f"No hay datos disponibles para {tck}.")

# ==============================================================================
# MÓDULO 5 — ORACLE PROJECTIONS
# ==============================================================================
elif "🔮" in app_mode:
    st.markdown(section_header("Oracle Projections", "Monte Carlo · 1,000 simulaciones ajustadas por régimen estructural"), unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    t_input = c1.text_input("Ticker", "NVDA").upper()
    h_days  = c2.slider("Horizonte de proyección (días)", 30, 365, 252)

    if st.button("Generar proyección", type="primary"):
        df = fetch_market_data(t_input, "2y")
        if not df.empty:
            lr    = np.log(df['Close'] / df['Close'].shift(1)).dropna()
            mu    = lr.mean() * 252
            sigma = lr.std()  * np.sqrt(252)
            lp    = df['Close'].iloc[-1]

            re, future, psi, regime = fisica.calcular_metricas_institucionales(df, risk_profile)
            if "MOMENTUM" in regime or "ACCUMULATION" in regime:
                mu = max(0.15, mu)

            dt, N = 1 / 252, 1000
            paths    = np.zeros((h_days, N))
            paths[0] = lp
            for step in range(1, h_days):
                r = np.random.standard_normal(N)
                paths[step] = paths[step - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * r)

            p95, p50, p05 = [np.percentile(paths[-1], p) for p in [95, 50, 5]]

            k1, k2, k3 = st.columns(3)
            k1.metric("🟢 Bull Case P95", f"${p95:,.2f}", f"+{((p95/lp)-1)*100:.0f}%")
            k2.metric("🔵 Base Case P50", f"${p50:,.2f}", f"{((p50/lp)-1)*100:+.0f}%")
            k3.metric("🔴 Bear Case P5",  f"${p05:,.2f}", f"{((p05/lp)-1)*100:.0f}%")

            st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

            p95a = np.percentile(paths, 95, axis=1)
            p75a = np.percentile(paths, 75, axis=1)
            p50a = np.percentile(paths, 50, axis=1)
            p25a = np.percentile(paths, 25, axis=1)
            p05a = np.percentile(paths, 5,  axis=1)

            fig_mc = go.Figure()
            for i in range(min(60, N)):
                fig_mc.add_trace(go.Scatter(y=paths[:, i], mode='lines',
                    line=dict(color='rgba(29,78,216,0.04)', width=1), showlegend=False))
            fig_mc.add_trace(go.Scatter(y=p95a, mode='lines', name='P95 Bull',
                line=dict(color='#15803d', width=2, dash='dash')))
            fig_mc.add_trace(go.Scatter(y=p75a, fill='tonexty', fillcolor='rgba(29,78,216,0.05)',
                mode='lines', name='P75', line=dict(color='rgba(29,78,216,0.25)', width=1)))
            fig_mc.add_trace(go.Scatter(y=p50a, mode='lines', name='P50 Base',
                line=dict(color='#1d4ed8', width=2.5)))
            fig_mc.add_trace(go.Scatter(y=p25a, fill='tonexty', fillcolor='rgba(239,68,68,0.04)',
                mode='lines', name='P25', line=dict(color='rgba(239,68,68,0.25)', width=1)))
            fig_mc.add_trace(go.Scatter(y=p05a, mode='lines', name='P5 Bear',
                line=dict(color='#ef4444', width=2, dash='dash')))
            fig_mc.update_layout(**PLOT_LAYOUT, height=440,
                title=dict(text=f"Monte Carlo — {t_input} · {h_days} días · {regime}", font=dict(size=13), x=0),
                yaxis_title="Precio (USD)")
            st.plotly_chart(fig_mc, use_container_width=True)

            sig_text, sig_color, sig_bg = signal_from_psi(psi, regime)
            st.markdown(f'''
            <div style="display:flex;gap:2rem;flex-wrap:wrap;background:{sig_bg};
                border:1px solid {sig_color}22;border-radius:8px;
                padding:.875rem 1.25rem;font-size:0.8rem;color:#374151;margin-top:0.5rem;">
                <span>Régimen <strong style="color:{sig_color};">{regime}</strong></span>
                <span>Ψ Score <strong>{psi:.1f}</strong></span>
                <span>Volatilidad anual <strong>{sigma*100:.1f}%</strong></span>
                <span>Drift anual <strong>{mu*100:.1f}%</strong></span>
                <span>Simulaciones <strong>1,000</strong></span>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.error(f"No hay datos disponibles para {t_input}.")
