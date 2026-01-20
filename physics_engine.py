# physics_engine.py
import numpy as np
from scipy.stats import entropy

class FarosPhysics:
    def __init__(self):
        self.RE_CRITICO = 2300
        self.RE_TURBULENTO = 4000

    def calcular_hidrodinamica(self, hist, window=14):
        """
        Calcula las variables de Navier-Stokes usando Proxies de Velas OHLCV.
        Retorna: Reynolds, Entropia, Psi, Estado
        """
        if len(hist) < window:
            return 0, 0, 0, "DATA_INSUFICIENTE"

        # 1. Velocidad (v): Momentum del precio (Derivada)
        # Cambio absoluto porcentual suavizado
        velocity = hist['Close'].pct_change().abs().rolling(3).mean().iloc[-1] * 1000 

        # 2. Densidad (rho): Profundidad relativa del mercado
        # Si hay mucho volumen, el fluido es denso y difícil de mover
        vol_sma = hist['Volume'].rolling(window).mean().iloc[-1]
        if vol_sma == 0: vol_sma = 1
        density = hist['Volume'].iloc[-1] / vol_sma

        # 3. Viscosidad (mu): Fricción del mercado (Spread estimado)
        # Eficiencia de Kawaller: (High - Low) / Volumen. 
        # Si es alto, hay mucha fricción/ineficiencia.
        high_low = (hist['High'].iloc[-1] - hist['Low'].iloc[-1]) / hist['Close'].iloc[-1]
        viscosity = high_low / (density + 0.01) # Evitar div/0
        
        # 4. Longitud Característica (L): Estructura del movimiento
        # Usamos la volatilidad ATR como proxy del tamaño del "vórtice"
        L = hist['Close'].rolling(window).std().iloc[-1]

        # --- CÁLCULO DE REYNOLDS (Re_f) ---
        # Re = (Densidad * Velocidad * Longitud) / Viscosidad
        # Ajustamos constantes para escalar a valores 0-6000 típicos
        reynolds = (density * velocity * L * 100) / (viscosity + 0.0001)
        reynolds = min(reynolds, 10000) # Cap de seguridad

        # --- CÁLCULO DE ENTROPÍA (H) ---
        # Analizamos la distribución de los retornos recientes
        returns = hist['Close'].pct_change().tail(window).dropna()
        hist_counts, _ = np.histogram(returns, bins='auto')
        entropy_val = entropy(hist_counts, base=2)
        
        # --- GOBERNANZA (Psi) ---
        # Definir estado
        if reynolds < self.RE_CRITICO:
            estado = "LAMINAR"
            factor_re = 1.0
        elif reynolds < self.RE_TURBULENTO:
            estado = "TRANSICION"
            factor_re = 0.5
        else:
            estado = "TURBULENTO"
            factor_re = 0.0 # Veto del Crítico

        # Ecuación Maestra Simplificada para Proxy
        # Psi = (Tendencia * Calidad Entrópica) * Filtro Reynolds
        trend = (hist['Close'].iloc[-1] - hist['Close'].iloc[-window]) / hist['Close'].iloc[-window]
        trend_score = max(0, trend * 10) # Normalizar tendencia positiva
        
        psi = np.tanh(trend_score * entropy_val) * factor_re * 100
        
        return reynolds, entropy_val, psi, estado
