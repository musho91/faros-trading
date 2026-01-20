# physics_engine.py
import numpy as np
from scipy.stats import entropy

class FarosPhysics:
    def __init__(self):
        self.RE_CRITICO = 2300
        self.RE_TURBULENTO = 4000
        # Constante de calibración para escalar los ratios a valores de Reynolds (0-6000)
        self.K_UNIVERSAL = 250000 

    def calcular_hidrodinamica(self, hist, window=14):
        """
        Calcula las variables de Navier-Stokes Normalizadas.
        """
        if len(hist) < window:
            return 0, 0, 0, "DATA_INSUFICIENTE"

        # --- 1. VISCOSIDAD (mu): Fricción del mercado ---
        # Rango relativo (High-Low)/Close. 
        # Si es bajo, el mercado es eficiente (baja fricción). Si es alto, hay resistencia.
        # Añadimos un suavizado (rolling mean) para evitar ruido de 1 vela.
        spread_proxy = (hist['High'] - hist['Low']) / hist['Close']
        viscosity = spread_proxy.rolling(3).mean().iloc[-1]
        
        # --- 2. DENSIDAD (rho): Profundidad relativa ---
        # Volumen actual vs Media. >1.0 significa mercado denso (difícil de mover).
        vol_sma = hist['Volume'].rolling(window).mean().iloc[-1]
        if vol_sma == 0: vol_sma = 1
        density = hist['Volume'].iloc[-1] / vol_sma

        # --- 3. VELOCIDAD (v): Inercia del precio ---
        # Valor absoluto del cambio porcentual.
        velocity = hist['Close'].pct_change().abs().rolling(3).mean().iloc[-1]

        # --- 4. LONGITUD (L): Estructura del movimiento ---
        # Usamos la Volatilidad Relativa (StdDev / Precio) para que sea agnóstico al precio del activo.
        # Esto soluciona el error de que NVDA salga siempre turbulento.
        L = (hist['Close'].rolling(window).std() / hist['Close']).iloc[-1]

        # --- CÁLCULO DE REYNOLDS (Re_f) ---
        # Fórmula: Re = (rho * v * L) / mu
        # Multiplicamos por K_UNIVERSAL para llevarlo a la escala 0 - 6000
        # Evitamos división por cero en viscosidad
        denom = viscosity + 0.0001
        re_raw = (density * velocity * L) / denom
        reynolds = re_raw * self.K_UNIVERSAL
        
        # Cap de seguridad visual (para que no rompa la gráfica)
        reynolds = min(reynolds, 10000)

        # --- CÁLCULO DE ENTROPÍA (H) ---
        returns = hist['Close'].pct_change().tail(window).dropna()
        # Discretizamos los retornos en bins para calcular entropía
        hist_counts, _ = np.histogram(returns, bins='auto')
        entropy_val = entropy(hist_counts, base=2)
        if np.isnan(entropy_val): entropy_val = 0
        
        # --- GOBERNANZA (Psi) ---
        if reynolds < self.RE_CRITICO:
            estado = "LAMINAR"
            factor_re = 1.0
        elif reynolds < self.RE_TURBULENTO:
            estado = "TRANSICION"
            factor_re = 0.5
        else:
            estado = "TURBULENTO"
            factor_re = 0.0 # Veto del Crítico

        # Psi depende de la tendencia y la entropía, filtrado por Reynolds
        # Tendencia de corto plazo
        trend = (hist['Close'].iloc[-1] - hist['Close'].iloc[-window]) / hist['Close'].iloc[-window]
        
        # Normalizamos la tendencia (0.10 de subida = 100 puntos base)
        trend_score = max(0, trend * 10) 
        
        # Ecuación Psi
        psi = np.tanh(trend_score * (entropy_val / 2)) * factor_re * 100
        
        return reynolds, entropy_val, psi, estado
