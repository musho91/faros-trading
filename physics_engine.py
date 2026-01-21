import numpy as np
import pandas as pd
from scipy.stats import entropy

class FarosPhysics:
    def __init__(self):
        self.K_UNIVERSAL = 100000  # Bajamos K para reducir "miedo" general

    def calcular_hidrodinamica(self, hist, perfil_riesgo="Growth", horizonte="Mediano"):
        if hist is None or len(hist) < 50:
            return 0, 0, 0, "DATA_INSUFICIENTE"

        try:
            # --- 0. CONFIGURACIÓN DE HORIZONTE (Filtro de Ruido) ---
            # Si miramos más lejos, el ruido diario desaparece.
            if horizonte == "Largo":
                window = 50      # Mirar el trimestre
                smooth = 10      # Suavizado fuerte
            elif horizonte == "Mediano":
                window = 20      # Mirar el mes
                smooth = 5
            else: # Corto
                window = 10
                smooth = 3

            # --- 1. VARIABLES FÍSICAS SUAVIZADAS ---
            # Viscosidad: Usamos una media móvil más larga para no asustarnos por un día malo
            spread = (hist['High'] - hist['Low']) / hist['Close']
            viscosity = spread.rolling(smooth).mean().iloc[-1]
            if pd.isna(viscosity) or viscosity == 0: viscosity = 0.001
            
            # Densidad (Volumen Relativo)
            vol_sma = hist['Volume'].rolling(window*2).mean().iloc[-1] # Base más amplia
            current_vol = hist['Volume'].rolling(smooth).mean().iloc[-1]
            density = (current_vol / vol_sma) if (vol_sma > 0) else 1.0

            # Velocidad (Inercia)
            velocity = hist['Close'].pct_change().abs().rolling(smooth).mean().iloc[-1]
            
            # Longitud (Estructura de volatilidad)
            L = (hist['Close'].rolling(window).std() / hist['Close']).iloc[-1]
            if pd.isna(L): L = 0.01

            # --- CÁLCULO DE REYNOLDS (Re_f) ---
            re_raw = (density * velocity * L) / viscosity
            reynolds = min(re_raw * self.K_UNIVERSAL, 10000)

            # --- 2. EL "FACTOR COHETE" (Flexibilidad) ---
            # Aquí está la magia para NVDA/BTC.
            # Calculamos la tendencia de FONDO (200 periodos si es posible, o 50)
            long_period = 100 if len(hist) > 100 else 50
            sma_long = hist['Close'].rolling(long_period).mean().iloc[-1]
            trend_force = (hist['Close'].iloc[-1] - sma_long) / sma_long
            
            # Ajuste de Perfil
            limit_turb = 4000
            if perfil_riesgo == "Quantum": limit_turb = 6500
            elif perfil_riesgo == "Growth": limit_turb = 5000

            # LÓGICA DE ESTADOS v6.0 (Más Permisiva)
            if reynolds > limit_turb:
                # Si es turbulento PERO la tendencia es brutal (>15%), es PLASMA (Comprar)
                # Antes esto daba GASEOSO (Vender).
                if trend_force > 0.15: 
                    estado = "PLASMA (HIGH GROWTH)"
                    factor_psi = 0.8 # Invertimos fuerte, pero no el 100%
                elif trend_force > 0.05 and perfil_riesgo != "Conservador":
                    estado = "PLASMA (VOLÁTIL)"
                    factor_psi = 0.5 # Mitad de posición
                else:
                    estado = "GASEOSO (CRASH)"
                    factor_psi = 0.0 # Salir
            else:
                # Flujo estable
                if trend_force > 0.05:
                    estado = "LÍQUIDO (ALCISTA)"
                    factor_psi = 1.0 # All in
                elif trend_force < -0.05:
                    estado = "SÓLIDO (BAJISTA)"
                    factor_psi = 0.0
                else:
                    estado = "LÍQUIDO (LATERAL)"
                    factor_psi = 0.4

            # --- 3. GOBERNANZA (PSI) ---
            # Entropía como validador (Si sube sin volumen, bajamos Psi)
            # Simplificado para que no de 0 por error de datos
            psi = np.tanh(trend_force * 10) * factor_psi * 100
            psi = max(0, min(100, psi))

            return reynolds, 0, psi, estado # Entropía (0) placeholder por velocidad

        except Exception as e:
            return 0, 0, 0, "ERROR"
