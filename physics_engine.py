# physics_engine.py
import numpy as np
from scipy.stats import entropy

class FarosPhysics:
    def __init__(self):
        # Umbrales ajustados para ser menos miedosos
        self.RE_CRITICO = 2500
        self.RE_TURBULENTO = 5000
        # Bajamos la K para que no salga siempre 10,000
        self.K_UNIVERSAL = 150000 

    def calcular_hidrodinamica(self, hist, window=14):
        """
        Calcula Navier-Stokes Financiero (Calibrado v4.0)
        """
        if len(hist) < window:
            return 0, 0, 0, "DATA_INSUFICIENTE"

        # --- 1. VISCOSIDAD (mu) ---
        # Spread relativo suavizado.
        spread = (hist['High'] - hist['Low']) / hist['Close']
        viscosity = spread.rolling(3).mean().iloc[-1]
        
        # --- 2. DENSIDAD (rho) ---
        # Volumen relativo.
        vol_sma = hist['Volume'].rolling(window).mean().iloc[-1]
        density = (hist['Volume'].iloc[-1] / vol_sma) if vol_sma > 0 else 1.0

        # --- 3. VELOCIDAD (v) ---
        # Momentum absoluto.
        velocity = hist['Close'].pct_change().abs().rolling(3).mean().iloc[-1]

        # --- 4. LONGITUD (L) ---
        # Volatilidad relativa (ATR % aprox).
        L = (hist['Close'].rolling(window).std() / hist['Close']).iloc[-1]

        # --- REYNOLDS (Re_f) ---
        denom = viscosity + 0.0001
        re_raw = (density * velocity * L) / denom
        reynolds = re_raw * self.K_UNIVERSAL
        
        # Cap visual pero no lógico
        reynolds_visual = min(reynolds, 10000)

        # --- ENTROPÍA (H) ---
        returns = hist['Close'].pct_change().tail(window).dropna()
        hist_counts, _ = np.histogram(returns, bins='auto')
        entropy_val = entropy(hist_counts, base=2)
        if np.isnan(entropy_val): entropy_val = 0
        
        # --- ESTADO Y EXCEPCIÓN "SUPER-LAMINAR" ---
        # Calculamos la tendencia pura
        trend = (hist['Close'].iloc[-1] - hist['Close'].iloc[-window]) / hist['Close'].iloc[-window]
        
        if reynolds < self.RE_CRITICO:
            estado = "LAMINAR"
            factor_re = 1.0
        elif reynolds < self.RE_TURBULENTO:
            # Si hay turbulencia pero la tendencia es MUY fuerte (>10%), es "Super-Laminar" (Cohete)
            if trend > 0.10: 
                estado = "SUPER-LAMINAR"
                factor_re = 0.9 # Penalización leve
            else:
                estado = "TRANSICION"
                factor_re = 0.5
        else:
            # Si es turbulento extremo
            if trend > 0.15: # Salvar activos hiper-growth (ej. NVDA en rally)
                estado = "SUPER-LAMINAR (RIESGO)"
                factor_re = 0.4
            else:
                estado = "TURBULENTO"
                factor_re = 0.0 # Veto

        # --- GOBERNANZA (Psi) ---
        # Psi base: Fuerza de tendencia * Calidad de Información
        trend_score = max(0, trend * 15) # Amplificamos señal de tendencia
        
        # Ecuación: Psi = tanh(Tendencia * Entropía) * FactorReynolds
        psi = np.tanh(trend_score * (entropy_val if entropy_val > 0.5 else 0.5)) * factor_re * 100
        
        return reynolds_visual, entropy_val, psi, estado
