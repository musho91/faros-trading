import numpy as np
import pandas as pd
from scipy.stats import entropy

class FarosPhysics:
    def __init__(self):
        self.K_UNIVERSAL = 150000 

    def calcular_metricas_institucionales(self, hist, perfil_riesgo="Growth"):
        """
        Calcula Métricas Cuantitativas (Navier-Stokes + Monte Carlo Drift).
        Retorna: Estabilidad, Alpha Score, Governance (Psi), Régimen
        """
        if hist is None or len(hist) < 50:
            return 0, 0, 0, "DATA_INSUFICIENTE"

        try:
            # --- 1. HIDRODINÁMICA (ESTABILIDAD ACTUAL) ---
            # Viscosidad (Spread Relativo)
            spread = (hist['High'] - hist['Low']) / hist['Close']
            viscosity = spread.rolling(5).mean().iloc[-1]
            if pd.isna(viscosity) or viscosity == 0: viscosity = 0.001
            
            # Densidad (Volumen Institucional)
            vol_sma = hist['Volume'].rolling(20).mean().iloc[-1]
            density = (hist['Volume'].iloc[-1] / vol_sma) if vol_sma > 0 else 1.0

            # Velocidad (Momentum)
            velocity = hist['Close'].pct_change().abs().rolling(5).mean().iloc[-1]
            
            # Longitud (Estructura de Volatilidad)
            L = (hist['Close'].rolling(20).std() / hist['Close']).iloc[-1]

            # Stability Index (Inverso de Reynolds para ser más intuitivo: Alto es bueno)
            # Re = (rho * v * L) / mu
            re_raw = (density * velocity * L) / viscosity
            reynolds = min(re_raw * self.K_UNIVERSAL, 10000)
            
            # --- 2. CÁLCULO DE ALPHA FUTURO (PROYECCIÓN) ---
            # Calculamos el CAGR implícito (Drift) de los últimos 6 meses
            drift_6m = hist['Close'].pct_change().mean() * 252
            future_score = np.tanh(drift_6m * 2) * 100 # Normalizado 0-100
            if future_score < 0: future_score = 0

            # --- 3. DETERMINACIÓN DE RÉGIMEN DE MERCADO ---
            # Tendencia Estructural (SMA 50 vs SMA 200 si hay datos, sino SMA 20 vs 50)
            sma_short = hist['Close'].rolling(20).mean().iloc[-1]
            sma_long = hist['Close'].rolling(50).mean().iloc[-1]
            trend_strength = (sma_short - sma_long) / sma_long

            # Umbrales ajustados por perfil
            limit_turb = 5000 if perfil_riesgo == "Quantum" else 3500

            if reynolds > limit_turb:
                # Alta Turbulencia
                if trend_strength > 0.10: 
                    regimen = "HIGH MOMENTUM" # Antes Plasma
                    factor_calidad = 0.7 
                else:
                    regimen = "STRUCTURAL BREAK" # Antes Gaseoso
                    factor_calidad = 0.0
            else:
                # Estabilidad
                if trend_strength > 0.02:
                    regimen = "INSTITUTIONAL ACCUMULATION" # Antes Líquido
                    factor_calidad = 1.0
                elif trend_strength < -0.05:
                    regimen = "DISTRIBUTION/BEAR" # Antes Sólido
                    factor_calidad = 0.0
                else:
                    regimen = "CONSOLIDATION" # Rango
                    factor_calidad = 0.4

            # --- 4. GOBERNANZA (PSI) INTEGRAL ---
            # Psi = (Calidad Actual * 60%) + (Potencial Futuro * 40%)
            # Esto soluciona que NVDA salga en 0. Si el futuro es brillante, compramos.
            
            current_score = factor_calidad * 100
            
            # Si el régimen es STRUCTURAL BREAK (Crash), el futuro no importa, se corta.
            if regimen == "STRUCTURAL BREAK":
                final_psi = 0
            else:
                # Ponderación
                final_psi = (current_score * 0.6) + (future_score * 0.4)
                
            # Ajuste final por Entropía (Calidad de señal)
            returns = hist['Close'].pct_change().tail(20).dropna()
            hist_counts, _ = np.histogram(returns, bins='auto')
            entropy_val = entropy(hist_counts, base=2)
            
            # Si la entropía es muy baja (manipulación), penalizamos levemente
            if entropy_val < 1.0: final_psi *= 0.9

            return reynolds, future_score, final_psi, regimen

        except Exception as e:
            return 0, 0, 0, "ERROR_CALCULO"
