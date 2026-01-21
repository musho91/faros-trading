# physics_engine.py
import numpy as np
import pandas as pd
from scipy.stats import entropy

class FarosPhysics:
    def __init__(self):
        self.RE_CRITICO = 2500
        self.RE_TURBULENTO = 5000
        self.K_UNIVERSAL = 150000 

    def calcular_hidrodinamica(self, hist, window=14):
        # 1. Validación de Datos Mínimos
        if hist is None or len(hist) < window:
            return 0, 0, 0, "DATA_INSUFICIENTE"

        try:
            # --- 1. VISCOSIDAD (mu) ---
            # Spread relativo. Rellenamos ceros para evitar errores.
            spread = (hist['High'] - hist['Low']) / hist['Close']
            viscosity = spread.rolling(3).mean().iloc[-1]
            if np.isnan(viscosity) or viscosity == 0: viscosity = 0.001
            
            # --- 2. DENSIDAD (rho) ---
            vol_sma = hist['Volume'].rolling(window).mean().iloc[-1]
            current_vol = hist['Volume'].iloc[-1]
            # Si vol_sma es 0 o NaN, asumimos densidad neutra (1.0)
            density = (current_vol / vol_sma) if (vol_sma > 0 and not np.isnan(vol_sma)) else 1.0

            # --- 3. VELOCIDAD (v) ---
            velocity = hist['Close'].pct_change().abs().rolling(3).mean().iloc[-1]
            if np.isnan(velocity): velocity = 0

            # --- 4. LONGITUD (L) ---
            L = (hist['Close'].rolling(window).std() / hist['Close']).iloc[-1]
            if np.isnan(L): L = 0.01

            # --- CÁLCULO DE REYNOLDS ---
            re_raw = (density * velocity * L) / viscosity
            reynolds = re_raw * self.K_UNIVERSAL
            
            # Limpieza de infinitos
            if np.isinf(reynolds) or np.isnan(reynolds): reynolds = 10000
            reynolds = min(reynolds, 10000)

            # --- ENTROPÍA ---
            returns = hist['Close'].pct_change().tail(window).dropna()
            if len(returns) > 1:
                hist_counts, _ = np.histogram(returns, bins='auto')
                entropy_val = entropy(hist_counts, base=2)
            else:
                entropy_val = 0
            
            if np.isnan(entropy_val): entropy_val = 0
            
            # --- ESTADO Y GOBERNANZA ---
            trend = (hist['Close'].iloc[-1] - hist['Close'].iloc[-window]) / hist['Close'].iloc[-window]
            if np.isnan(trend): trend = 0

            if reynolds < self.RE_CRITICO:
                estado = "LAMINAR"
                factor_re = 1.0
            elif reynolds < self.RE_TURBULENTO:
                # Lógica Super-Laminar: Si sube fuerte, ignoramos fricción
                if trend > 0.10: 
                    estado = "SUPER-LAMINAR"
                    factor_re = 0.9
                else:
                    estado = "TRANSICION"
                    factor_re = 0.5
            else:
                if trend > 0.15: 
                    estado = "SUPER-LAMINAR (RIESGO)"
                    factor_re = 0.4
                else:
                    estado = "TURBULENTO"
                    factor_re = 0.0 

            # Psi amplificado para reaccionar más rápido
            trend_score = max(0, trend * 20) 
            psi = np.tanh(trend_score * (entropy_val if entropy_val > 0.1 else 1.0)) * factor_re * 100
            
            return reynolds, entropy_val, psi, estado

        except Exception as e:
            # En caso de error matemático, retornamos estado seguro
            return 0, 0, 0, "ERROR_CALCULO"
