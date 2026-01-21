# physics_engine.py
import numpy as np
import pandas as pd
from scipy.stats import entropy

class FarosPhysics:
    def __init__(self):
        self.K_UNIVERSAL = 150000 

    def calcular_hidrodinamica(self, hist, perfil_riesgo="Growth", window=14):
        if hist is None or len(hist) < 50:
            return 0, 0, 0, "DATA_INSUFICIENTE"

        try:
            # --- 0. AJUSTE POR PERFIL DE RIESGO ---
            # Definimos qué tanta turbulencia aguantamos antes de salir
            if perfil_riesgo == "Conservador":
                limite_turbulencia = 3000
                sensibilidad_psi = 0.8
            elif perfil_riesgo == "Quantum": # Muy agresivo
                limite_turbulencia = 6000
                sensibilidad_psi = 1.5
            else: # Growth (Default)
                limite_turbulencia = 4500
                sensibilidad_psi = 1.0

            # --- 1. VARIABLES FÍSICAS ---
            # Spread relativo (Viscosidad)
            spread = (hist['High'] - hist['Low']) / hist['Close']
            viscosity = spread.rolling(3).mean().iloc[-1]
            if np.isnan(viscosity) or viscosity == 0: viscosity = 0.001
            
            # Densidad
            vol_sma = hist['Volume'].rolling(window).mean().iloc[-1]
            density = (hist['Volume'].iloc[-1] / vol_sma) if vol_sma > 0 else 1.0

            # Velocidad y Longitud
            velocity = hist['Close'].pct_change().abs().rolling(3).mean().iloc[-1]
            L = (hist['Close'].rolling(window).std() / hist['Close']).iloc[-1]

            # REYNOLDS
            re_raw = (density * velocity * L) / viscosity
            reynolds = min(re_raw * self.K_UNIVERSAL, 10000)

            # --- 2. ESTADOS DE LA MATERIA (Mapeo) ---
            # Tendencia de largo plazo (SMA 50) para definir dirección
            sma_50 = hist['Close'].rolling(50).mean().iloc[-1]
            trend_force = (hist['Close'].iloc[-1] - sma_50) / sma_50
            
            # Entropía
            returns = hist['Close'].pct_change().tail(window).dropna()
            hist_counts, _ = np.histogram(returns, bins='auto')
            entropy_val = entropy(hist_counts, base=2)
            if np.isnan(entropy_val): entropy_val = 0

            # LÓGICA DE ESTADOS v5.0
            if reynolds > limite_turbulencia:
                # Si hay mucha turbulencia, es GAS o PLASMA
                if trend_force > 0.15 and perfil_riesgo != "Conservador":
                    estado = "PLASMA" # Alta energía, subida vertical (Riesgoso pero rentable)
                    factor_psi = 0.6  # Invertir con cuidado
                else:
                    estado = "GASEOSO" # Caos total
                    factor_psi = 0.0  # Salir
            else:
                # Si el flujo es calmado
                if trend_force > 0.02:
                    estado = "LÍQUIDO" # Flujo Laminar Alcista
                    factor_psi = 1.0
                elif trend_force < -0.05:
                    estado = "SÓLIDO (BAJISTA)" # Caída ordenada
                    factor_psi = 0.0
                else:
                    estado = "SÓLIDO (RANGO)" # Estancado
                    factor_psi = 0.2

            # --- 3. CÁLCULO DE GOBERNANZA (PSI) ---
            # Psi ahora mira la tendencia de fondo (SMA 50) no solo la de ayer.
            # Normalizamos: Una tendencia del 10% (0.10) debería dar un score alto.
            score_base = max(0, trend_force * 10) 
            
            # Ecuación Maestra v5:
            # Psi = tanh(Tendencia * Calidad) * Factor_Estado * Perfil
            psi = np.tanh(score_base * (entropy_val if entropy_val > 0.1 else 1)) * factor_psi * sensibilidad_psi * 100
            
            # Cap final entre 0 y 100
            psi = max(0, min(100, psi))

            return reynolds, entropy_val, psi, estado

        except:
            return 0, 0, 0, "ERROR"
