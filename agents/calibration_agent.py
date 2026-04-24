"""
agents/calibration_agent.py
----------------------------
The Calibration Agent:
  1. Receives raw probability from AnalystAgent
  2. Applies post-hoc calibration (Platt, isotonic, temperature, beta)
  3. Applies power-transform extremization (α-extremization)
  4. Computes Kelly criterion bet sizing
  5. Issues a trade recommendation (YES / NO / NO_TRADE)

This implements the "Multimodal Calibration" layer from the blueprint:
  - Post-hoc Platt scaling
  - Power-transform extremization with α ≈ 2.5
  - Quarter-Kelly position sizing for risk management
"""

import logging
import numpy as np
from agents.state import ForecastingState
from core.calibrator import ProbabilityCalibrator
from utils.kelly_criterion import KellyCriterion

logger = logging.getLogger(__name__)

# Minimum edge required to place a trade
MIN_EDGE_FOR_TRADE = 0.03    # 3 percentage points


class CalibrationAgent:
    """
    Probability calibration + bet sizing agent.

    Usage:
        agent = CalibrationAgent(calibrator, kelly, config)
        state = agent.run(state)
    """

    def __init__(self, calibrator: ProbabilityCalibrator, kelly: KellyCriterion, config):
        self.calibrator = calibrator
        self.kelly      = kelly
        self.config     = config

    # ------------------------------------------------------------------ #
    # LangGraph node entry point                                           #
    # ------------------------------------------------------------------ #

    def run(self, state: ForecastingState) -> ForecastingState:
        raw_prob   = state.get("raw_probability", 0.5)
        confidence = state.get("confidence", 0.6)
        iteration  = state.get("iteration", 0)

        logger.info(
            "[CalibrationAgent] Calibrating raw_prob=%.4f  confidence=%.4f  iter=%d",
            raw_prob, confidence, iteration,
        )

        try:
            # 1. Calibrate
            report = self.calibrator.calibrate(
                raw_probability=raw_prob,
                method=self.config.calibration.method,
            )

            # 2. Kelly sizing
            market_price = state.get("market_price", 0.5)   # default fair coin
            kelly_f      = self.kelly.compute(
                true_prob   = report.extremized_probability,
                market_price= market_price,
            )
            fractional_kelly = kelly_f * self.config.calibration.kelly_fraction

            # 3. Trade recommendation
            recommendation = self._recommend_trade(
                report.extremized_probability, market_price, fractional_kelly
            )

            calibration_report = {
                "raw_probability":        round(report.raw_probability, 4),
                "calibrated_probability": round(report.calibrated_probability, 4),
                "extremized_probability": round(report.extremized_probability, 4),
                "method":                 report.method,
                "extremize_alpha":        self.config.calibration.extremize_alpha,
                "kelly_full":             round(kelly_f, 4),
                "kelly_fractional":       round(fractional_kelly, 4),
                "kelly_fraction_used":    self.config.calibration.kelly_fraction,
                "market_price":           market_price,
                "edge":                   round(report.extremized_probability - market_price, 4),
                "recommendation":         recommendation,
            }

            logger.info(
                "[CalibrationAgent] cal_prob=%.4f  ext_prob=%.4f  kelly=%.4f  rec=%s",
                report.calibrated_probability,
                report.extremized_probability,
                fractional_kelly,
                recommendation,
            )

            return {
                **state,
                "calibrated_probability": report.calibrated_probability,
                "extremized_probability": report.extremized_probability,
                "calibration_report":     calibration_report,
                "kelly_fraction":         fractional_kelly,
                "recommended_position":   recommendation,
                "error":                  None,
            }

        except Exception as exc:
            logger.exception("[CalibrationAgent] Failed: %s", exc)
            return {**state, "error": f"CalibrationAgent: {exc}"}

    # ------------------------------------------------------------------ #
    # Trade decision                                                        #
    # ------------------------------------------------------------------ #

    def _recommend_trade(
        self, true_prob: float, market_price: float, kelly_f: float
    ) -> str:
        """
        Issue a directional recommendation.

        YES   → our probability is materially above the market price
        NO    → our probability is materially below the market price
        NO_TRADE → insufficient edge (|Δ| < MIN_EDGE_FOR_TRADE)
        """
        edge = true_prob - market_price

        if abs(edge) < MIN_EDGE_FOR_TRADE:
            return "NO_TRADE"
        elif edge > 0:
            return "YES"
        else:
            return "NO"
