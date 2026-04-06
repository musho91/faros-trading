"""
FAROS v7.0 — Physics Engine (Audited & Refactored)
Autor: Juan Arroyo | SG Consulting Group
Auditoría: Equipo Quant Elite (Física Matemática + Hedge Fund + FullStack)

Cambios Clave vs. v6.x:
  - Reynolds normalizado por Z-Score percentil en lugar de K_UNIVERSAL fijo.
  - Viscosidad extendida con estimación de Amihud Illiquidity (proxy de slippage institucional).
  - Entropía calculada sobre distribución de volumen relativo (proxy de flujo de órdenes).
  - FutureScore ponderado por estabilidad de Reynolds para evitar señales falsas en crash rallies.
  - Todos los métodos son stateless y vectorizados (compatibles con FastAPI).
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy, zscore
from scipy.signal import find_peaks
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class FarosMetrics:
    """Contenedor tipado para todos los outputs del motor físico."""
    reynolds: float
    reynolds_pct: float          # Percentil histórico del Reynolds (0-100)
    future_score: float
    psi: float                   # Governance Score final
    regime: str
    viscosity_mu: float          # Viscosidad dinámica estimada
    density_rho: float           # Densidad institucional
    amihud_illiquidity: float    # Proxy de slippage institucional
    shannon_entropy: float       # Entropía del flujo de volumen
    alpha_flow: float            # Coeficiente de autenticidad (0-1)
    z_score_price: float         # Estado termodinámico actual


class FarosPhysics:
    """
    Motor de Hidrodinámica Financiera — TAI-ACF Framework v3.0.

    Postulados físicos implementados:
      1. Navier-Stokes financiero: Re_f = (ρ · v · L) / µ
      2. Entropía de Shannon sobre distribución de volumen relativo
      3. Coeficiente de Autenticidad α_flow para filtrar liquidez sintética
      4. Ecuación Unificada de Gobernanza Ψ con veto topológico
    """

    # --- Umbrales de Régimen (basados en la taxonomía TAI-ACF) ---
    RE_LAMINAR_MAX = 2300
    RE_TRANSITION_MAX = 4000

    def __init__(self):
        pass  # K_UNIVERSAL eliminado — normalización ahora es relativa al histórico del activo

    # ===========================================================================
    # BLOQUE 1: COMPONENTES FÍSICOS INDIVIDUALES
    # ===========================================================================

    def _calc_viscosity(self, hist: pd.DataFrame) -> float:
        """
        Viscosidad Dinámica (µ) — Proxy de fricción del mercado.

        Combina:
          - Spread Bid-Ask relativo (High-Low)/Close — fricción retail
          - Ratio de Amihud Illiquidity — impacto de mercado institucional
            Amihud_i = |r_i| / Volume_i  (en USD si disponible, sino unidades)

        La combinación captura tanto la fricción superficial como la profunda.
        """
        # Componente 1: Spread promedio (ventana 5 días, suavizada)
        spread = (hist['High'] - hist['Low']) / hist['Close']
        spread_5d = spread.rolling(5).mean().iloc[-1]
        if pd.isna(spread_5d) or spread_5d <= 0:
            spread_5d = 0.01

        # Componente 2: Amihud Illiquidity (ventana 20 días)
        # Representa cuánto se mueve el precio por unidad de volumen — clave para slippage institucional
        daily_returns = hist['Close'].pct_change().abs()
        amihud_series = daily_returns / (hist['Volume'] * hist['Close'])
        amihud_20d = amihud_series.rolling(20).mean().iloc[-1]
        if pd.isna(amihud_20d) or amihud_20d <= 0:
            amihud_20d = 1e-10

        # Normalizar Amihud al rango [0, 0.1] usando percentil histórico
        amihud_hist = amihud_series.dropna()
        amihud_pct = amihud_hist.rank(pct=True).iloc[-1] * 0.1  # escalar a [0, 0.1]

        # Viscosidad compuesta: spread domina para activos líquidos, Amihud para ilíquidos
        mu = (spread_5d * 0.5) + (amihud_pct * 0.5)
        return max(mu, 1e-6), amihud_20d

    def _calc_density(self, hist: pd.DataFrame) -> float:
        """
        Densidad (ρ) — Proxy de profundidad de mercado institucional.

        Usa el ratio volumen actual vs. SMA20 de volumen, ponderado por
        la consistencia del volumen (bajo CV = mayor institucionalidad).
        """
        vol_sma = hist['Volume'].rolling(20).mean().iloc[-1]
        if vol_sma <= 0:
            return 1.0
        density = hist['Volume'].iloc[-1] / vol_sma

        # Penalizar si el volumen es errático (CV alto = menor confianza institucional)
        vol_cv = hist['Volume'].rolling(20).std().iloc[-1] / vol_sma
        consistency_factor = 1 / (1 + vol_cv) if not pd.isna(vol_cv) else 0.5

        return density * consistency_factor

    def _calc_velocity_and_length(self, hist: pd.DataFrame) -> Tuple[float, float]:
        """
        Velocidad (v) y Longitud Característica (L).

        v = momentum de precio (pct_change abs, rolling 5d)
        L = rango normalizado de volatilidad histórica (estructura de la tendencia)
          L más grande = tendencia de mayor escala = más inercia institucional
        """
        velocity = hist['Close'].pct_change().abs().rolling(5).mean().iloc[-1]
        if pd.isna(velocity):
            velocity = 0.001

        # L: volatilidad normalizada sobre 20 días
        L = (hist['Close'].rolling(20).std() / hist['Close']).iloc[-1]
        if pd.isna(L) or L <= 0:
            L = 0.01

        return velocity, L

    def _calc_reynolds(
        self,
        density: float,
        velocity: float,
        L: float,
        viscosity: float,
        hist: pd.DataFrame
    ) -> Tuple[float, float]:
        """
        Número de Reynolds Financiero (Re_f).

        CORRECCIÓN CRÍTICA vs. v6.x:
          Se elimina K_UNIVERSAL. En su lugar, se normaliza Re_raw usando su propio
          percentil histórico rodante (252 días). Esto garantiza invarianza de escala
          entre activos (BTC vs. SPY vs. small-cap).

        Retorna:
          reynolds_abs: valor absoluto (para diagnóstico)
          reynolds_pct: percentil 0-100 (para lógica de régimen — USAR ESTE)
        """
        re_raw = (density * velocity * L) / viscosity
        reynolds_abs = re_raw  # valor bruto sin escalar

        # Normalización por percentil histórico del propio activo (ventana 252 días)
        # Esto hace que Re_f = 80 siempre signifique "más turbulento que el 80% del historial"
        # independientemente de la capitalización o liquidez del activo.
        spread = (hist['High'] - hist['Low']) / hist['Close']
        vol_sma_series = hist['Volume'].rolling(20).mean()
        density_series = hist['Volume'] / vol_sma_series.clip(lower=1)
        velocity_series = hist['Close'].pct_change().abs().rolling(5).mean()
        L_series = hist['Close'].rolling(20).std() / hist['Close']
        spread_smooth = spread.rolling(5).mean().clip(lower=1e-6)

        re_series = (density_series * velocity_series * L_series) / spread_smooth
        re_series = re_series.dropna()

        if len(re_series) < 20:
            return reynolds_abs, 50.0  # default si no hay historial suficiente

        current_pct = (re_series < re_raw).mean() * 100
        return reynolds_abs, current_pct

    def _calc_shannon_entropy(self, hist: pd.DataFrame) -> float:
        """
        Entropía de Shannon sobre distribución de VOLUMEN RELATIVO.

        CORRECCIÓN CRÍTICA vs. v6.x:
          El paper TAI-ACF postula H(X) sobre tamaños de órdenes ejecutadas.
          Como no tenemos datos L2, usamos volumen relativo diario (Vol/Vol_mean)
          como proxy del tamaño relativo de participación institucional.

          Alta entropía = muchos actores distintos = demanda orgánica.
          Baja entropía = pocos actores dominantes = posible manipulación o recompra.

        Retorna entropía normalizada ∈ [0, 1].
        """
        vol_relative = (hist['Volume'] / hist['Volume'].rolling(20).mean()).tail(30).dropna()
        if len(vol_relative) < 10:
            return 0.5  # incertidumbre neutral por falta de datos

        # Discretizar en 10 bins para crear distribución de probabilidad
        counts, _ = np.histogram(vol_relative, bins=10, range=(0, vol_relative.quantile(0.99)))
        counts = counts + 1  # Laplace smoothing para evitar log(0)
        probs = counts / counts.sum()
        H = -np.sum(probs * np.log2(probs))
        H_max = np.log2(10)  # entropía máxima para 10 bins uniformes

        return min(H / H_max, 1.0)  # normalizado a [0, 1]

    def _calc_alpha_flow(self, shannon_entropy: float, reynolds_pct: float) -> float:
        """
        Coeficiente de Autenticidad α_flow ∈ [0, 1].

        Implementa la intuición del paper: penaliza cuando el precio sube
        con baja entropía (liquidez sintética / manipulación).

        En producción, debería incorporar ΔM2 (masa monetaria). Aquí usamos
        reynolds_pct como proxy de "presión artificial" cuando es >80 pero con
        baja entropía (el clásico patrón de burbuja).

        α_flow = H_norm × (1 - presión_artificial)
        """
        # Si Reynolds está muy alto (mercado turbulento) Y entropía es baja:
        # señal de manipulación / momentum artificial
        presion_artificial = max(0, (reynolds_pct - 70) / 30) * (1 - shannon_entropy)
        alpha = shannon_entropy * (1 - presion_artificial)
        return np.clip(alpha, 0.0, 1.0)

    def _calc_z_score(self, hist: pd.DataFrame, window: int = 50) -> float:
        """
        Z-Score termodinámico del precio actual.
        Define el Estado de la Materia Financiera según taxonomía TAI-ACF:
          Líquido: 0.5 < Z < 2.0  →  Zona operativa
          Sólido:  |Z| ≈ 0        →  Rango / Acumulación
          Gas:     |Z| > 3.0      →  Pánico / Caos
        """
        if len(hist) < window:
            return 0.0
        mu = hist['Close'].rolling(window).mean().iloc[-1]
        sigma = hist['Close'].rolling(window).std().iloc[-1]
        if pd.isna(mu) or pd.isna(sigma) or sigma == 0:
            return 0.0
        return (hist['Close'].iloc[-1] - mu) / sigma

    # ===========================================================================
    # BLOQUE 2: ECUACIÓN UNIFICADA DE GOBERNANZA Ψ
    # ===========================================================================

    def _calc_future_score(
        self, hist: pd.DataFrame, reynolds_pct: float, regime: str
    ) -> float:
        """
        FutureScore (Alpha Potencial).

        CORRECCIÓN vs. v6.x:
          El drift de 6 meses ahora se pondera por la estabilidad de Reynolds.
          En régimen turbulento (Re alto), el drift histórico es menos predictivo
          (el modelo reconoce su propia incertidumbre — principio de anti-fragilidad).

        Retorna valor normalizado [0, 100].
        """
        drift_6m = hist['Close'].pct_change().mean() * 252
        raw_score = np.tanh(drift_6m * 2) * 100
        raw_score = max(0.0, raw_score)

        # Penalización por turbulencia: si Re_pct > 70, el drift es menos confiable
        stability_weight = 1 - max(0, (reynolds_pct - 70) / 100)

        # En STRUCTURAL BREAK, el futuro no tiene valor predictivo inmediato
        if regime == "STRUCTURAL BREAK":
            return 0.0

        return raw_score * stability_weight

    def _determine_regime(
        self, hist: pd.DataFrame, reynolds_pct: float, perfil_riesgo: str
    ) -> Tuple[str, float]:
        """
        Clasificación de Régimen de Mercado.

        Usa Reynolds percentil (invariante de escala) + trend strength.
        Retorna (régimen, factor_calidad).
        """
        sma_short = hist['Close'].rolling(20).mean().iloc[-1]
        sma_long = hist['Close'].rolling(50).mean().iloc[-1]

        if pd.isna(sma_long) or sma_long == 0:
            return "DATA_INSUFICIENTE", 0.0

        trend_strength = (sma_short - sma_long) / sma_long

        # Umbral de turbulencia ajustado por perfil
        limit_turb = 85.0 if perfil_riesgo == "Quantum" else 75.0  # percentiles, no valores absolutos

        if reynolds_pct > limit_turb:
            if trend_strength > 0.10:
                return "HIGH MOMENTUM", 0.7
            else:
                return "STRUCTURAL BREAK", 0.0
        else:
            if trend_strength > 0.02:
                return "INSTITUTIONAL ACCUMULATION", 1.0
            elif trend_strength < -0.05:
                return "DISTRIBUTION/BEAR", 0.0
            else:
                return "CONSOLIDATION", 0.4

    def _governance_psi(
        self,
        factor_calidad: float,
        future_score: float,
        alpha_flow: float,
        z_score: float,
        regime: str
    ) -> float:
        """
        Ecuación Unificada de Gobernanza Ψ.

        Implementa la fórmula del paper TAI-ACF:
          Ψ = tanh(M⃗ · α_flow / Re_f^γ) · I(Z ∈ Líquido)

        Aquí simplificamos para el motor de scoring:
          - M⃗ (impulso) = factor_calidad + future_score
          - α_flow = coeficiente de autenticidad
          - Veto topológico: Si Z ∉ [0.3, 2.5] (estado Sólido o Gas) → Ψ *= penalización

        Retorna Ψ ∈ [0, 100].
        """
        if regime == "STRUCTURAL BREAK":
            return 0.0

        current_score = factor_calidad * 100

        # Ponderación: presente 60% + futuro 40%, modulada por autenticidad
        raw_psi = (current_score * 0.6 + future_score * 0.4) * alpha_flow

        # Veto topológico suave (en lugar de booleano duro, usamos penalización gradual)
        # Estado Líquido óptimo: 0.5 < Z < 2.0
        if 0.3 <= z_score <= 2.5:
            topo_factor = 1.0   # zona operativa
        elif z_score < 0:
            topo_factor = 0.5   # tendencia bajista confirmada
        elif z_score > 3.0:
            topo_factor = 0.2   # Gas/Pánico — reduce exposición fuertemente
        else:
            topo_factor = 0.8   # zona intermedia

        final_psi = raw_psi * topo_factor
        return np.clip(final_psi, 0.0, 100.0)

    # ===========================================================================
    # BLOQUE 3: INTERFAZ PÚBLICA
    # ===========================================================================

    def calcular_metricas_institucionales(
        self,
        hist: pd.DataFrame,
        perfil_riesgo: str = "Growth"
    ) -> Tuple[float, float, float, str]:
        """
        Punto de entrada principal — compatible con la interfaz v6.x del scanner.

        Retorna: (reynolds_percentil, future_score, psi, regime)
        """
        metrics = self.calcular_metricas_completas(hist, perfil_riesgo)
        if metrics is None:
            return 0.0, 0.0, 0.0, "DATA_INSUFICIENTE"
        return (
            metrics.reynolds_pct,
            metrics.future_score,
            metrics.psi,
            metrics.regime
        )

    def calcular_metricas_completas(
        self,
        hist: Optional[pd.DataFrame],
        perfil_riesgo: str = "Growth"
    ) -> Optional[FarosMetrics]:
        """
        Cálculo completo con todos los outputs físicos.
        Usar en FastAPI para respuestas ricas al frontend.
        """
        if hist is None or len(hist) < 50:
            return None

        try:
            # --- 1. Componentes físicos base ---
            viscosity, amihud = self._calc_viscosity(hist)
            density = self._calc_density(hist)
            velocity, L = self._calc_velocity_and_length(hist)

            # --- 2. Reynolds (normalizado por percentil) ---
            reynolds_abs, reynolds_pct = self._calc_reynolds(density, velocity, L, viscosity, hist)

            # --- 3. Régimen ---
            regime, factor_calidad = self._determine_regime(hist, reynolds_pct, perfil_riesgo)

            # --- 4. Entropía y autenticidad ---
            shannon_h = self._calc_shannon_entropy(hist)
            alpha_flow = self._calc_alpha_flow(shannon_h, reynolds_pct)

            # --- 5. Estado termodinámico (Z-Score) ---
            z_score = self._calc_z_score(hist)

            # --- 6. FutureScore ---
            future_score = self._calc_future_score(hist, reynolds_pct, regime)

            # --- 7. Gobernanza Ψ ---
            psi = self._governance_psi(factor_calidad, future_score, alpha_flow, z_score, regime)

            return FarosMetrics(
                reynolds=reynolds_abs,
                reynolds_pct=reynolds_pct,
                future_score=future_score,
                psi=psi,
                regime=regime,
                viscosity_mu=viscosity,
                density_rho=density,
                amihud_illiquidity=amihud,
                shannon_entropy=shannon_h,
                alpha_flow=alpha_flow,
                z_score_price=z_score,
            )

        except Exception as e:
            # Log estructurado (compatible con FastAPI logging)
            import logging
            logging.getLogger("faros.physics").error(f"Error en cálculo: {e}", exc_info=True)
            return None
