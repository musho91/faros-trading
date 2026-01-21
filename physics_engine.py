import numpy as np
import pandas as pd
from scipy.stats import entropy

class FarosPhysics:
    def __init__(self):
        self.K_UNIVERSAL = 150000 

    def calcular_hidrodinamica(self, hist, perfil_riesgo="Growth", window=14):
        # Validación de datos
        if hist is None or len(hist) < window:
            return 0, 0, 0, "DATA_INSUFICIENTE"

        try:
            # --- 0. CONFIGURACIÓN DEL CRÍTICO (PERFIL DE RIESGO) ---
            # Aquí traducimos el texto a parámetros matemáticos
            if perfil_riesgo == "Conservador":
                limite_turbulencia = 3000
                sensibilidad_psi = 0.8  # Más miedoso
            elif perfil_riesgo == "Quantum": 
                limite_turbulencia = 6000 # Tolera mucho caos
                sensibilidad_psi = 1.5  # Muy agresivo
            else: # Growth (Default)
                limite_turbulencia = 4500
                sensibilidad_psi = 1.0

            # --- 1. VARIABLES FÍSICAS ---
            # Viscosidad (Spread relativo)
            spread = (hist['High'] - hist['Low']) / hist['Close']
            viscosity = spread.rolling(3).mean().iloc[-1]
            if pd.isna(viscosity) or viscosity == 0: viscosity = 0.001
            
            # Densidad (Volumen relativo)
            vol_sma = hist['Volume'].rolling(window).mean().iloc[-1]
            current_vol = hist['Volume'].iloc[-1]
            density = (current_vol / vol_sma) if (vol_sma > 0 and not pd.isna(vol_sma)) else 1.0

            # Velocidad y Longitud
            velocity = hist['Close'].pct_change().abs().rolling(3).mean().iloc[-1]
            L = (hist['Close'].rolling(window).std() / hist['Close']).iloc[-1]
            
            if pd.isna(velocity): velocity = 0
            if pd.isna(L): L = 0.01

            # --- CÁLCULO DE REYNOLDS ---
            re_raw = (density * velocity * L) / viscosity
            reynolds = min(re_raw * self.K_UNIVERSAL, 10000)
            if pd.isna(reynolds): reynolds = 5000 # Fallback seguro

            # --- ENTROPÍA ---
            returns = hist['Close'].pct_change().tail(window).dropna()
            if len(returns) > 1:
                hist_counts, _ = np.histogram(returns, bins='auto')
                entropy_val = entropy(hist_counts, base=2)
            else:
                entropy_val = 0
            
            if pd.isna(entropy_val): entropy_val = 0

            # --- 2. DETERMINACIÓN DE ESTADO ---
            # Usamos SMA 50 para ver la tendencia de fondo
            if len(hist) > 50:
                sma_50 = hist['Close'].rolling(50).mean().iloc[-1]
                trend_force = (hist['Close'].iloc[-1] - sma_50) / sma_50
            else:
                trend_force = 0

            if reynolds > limite_turbulencia:
                # Si hay turbulencia excesiva
                if trend_force > 0.15 and perfil_riesgo != "Conservador":
                    estado = "PLASMA" # Subida vertical (Riesgo alto pero rentable)
                    factor_psi = 0.6  
                else:
                    estado = "GASEOSO" # Caos total (Salida)
                    factor_psi = 0.0  
            else:
                # Flujo estable
                if trend_force > 0.02:
                    estado = "LÍQUIDO" # Tendencia sana
                    factor_psi = 1.0
                elif trend_force < -0.05:
                    estado = "SÓLIDO (BAJISTA)" 
                    factor_psi = 0.0
                else:
                    estado = "SÓLIDO (RANGO)"
                    factor_psi = 0.2

            # --- 3. CÁLCULO DE PSI (GOBERNANZA) ---
            # Amplificamos la tendencia para que PSI reaccione
            trend_score = max(0, trend_force * 15) 
            
            # Ecuación: Psi = tanh(Tendencia * Calidad) * Factor_Estado * Perfil
            psi = np.tanh(trend_score * (entropy_val if entropy_val > 0.1 else 1)) * factor_psi * sensibilidad_psi * 100
            
            # Limites finales 0-100
            psi = max(0, min(100, psi))

            return reynolds, entropy_val, psi, estado

        except Exception as e:
            print(f"Error Física: {e}")
            return 0, 0, 0, "ERROR"
