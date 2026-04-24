"""
agents/evaluation_agent.py
---------------------------
The Evaluation Agent implements the "Reflection Mechanism":
  1. Splits GDELT data: uses 80% for fitting, 20% (held-out) for evaluation
  2. Scores the current probability using a Brier-like score on the held-out window
  3. Compares current score to previous iteration
  4. If evaluation fails threshold: generates updated reasoning notes for the next cycle
  5. Sets evaluation_passed → True/False to control the LangGraph loop

Bayesian reasoning is integrated here via a simple prior-update step:
  posterior = bayesian_update(prior=raw_probability, likelihood=tsfm_trend_signal)
"""

import logging
import numpy as np
import pandas as pd
from agents.state import ForecastingState
from utils.brier_score import BrierScoreCalculator

logger = logging.getLogger(__name__)


class EvaluationAgent:
    """
    Reflection + evaluation agent.

    Checks whether the current probability estimate is well-supported
    by the held-out GDELT signal window. Triggers re-analysis if not.
    """

    def __init__(self, brier_calculator: BrierScoreCalculator, config):
        self.brier = brier_calculator
        self.config = config

    # ------------------------------------------------------------------ #
    # LangGraph node entry point                                           #
    # ------------------------------------------------------------------ #

    def run(self, state: ForecastingState) -> ForecastingState:
        raw_prob     = state.get("raw_probability", 0.5)
        ext_prob     = state.get("extremized_probability", raw_prob)
        signals      = state.get("gdelt_signals")
        reasoning    = state.get("reasoning", "")
        iteration    = state.get("iteration", 0)
        max_iters    = state.get("max_iterations", self.config.agents.max_reflection_cycles)

        logger.info("[EvaluationAgent] Reflection cycle %d / %d", iteration + 1, max_iters)

        try:
            # 1. Compute held-out Brier score
            eval_score = self._compute_held_out_score(ext_prob, signals)

            # 2. Apply Bayesian update to refine probability
            updated_prob = self._bayesian_update(
                prior=ext_prob,
                signals=signals,
            )

            # 3. Decide: passed or needs another cycle?
            threshold = self.config.agents.reflection_threshold
            previous_score = state.get("evaluation_score", 1.0)
            improvement = previous_score - eval_score

            passed = self._evaluation_passed(eval_score, improvement, iteration, max_iters)

            # 4. Generate reflection notes
            notes = self._generate_reflection_notes(
                eval_score, improvement, passed, reasoning, signals
            )

            logger.info(
                "[EvaluationAgent] Brier=%.4f  improvement=%.4f  passed=%s",
                eval_score, improvement, passed,
            )

            new_state = {
                **state,
                "evaluation_score":   eval_score,
                "evaluation_passed":  passed,
                "reflection_notes":   notes,
                "updated_reasoning":  notes,
                "raw_probability":    updated_prob,
                "iteration":          iteration + 1,
                "error":              None,
            }

            return new_state

        except Exception as exc:
            logger.exception("[EvaluationAgent] Failed: %s", exc)
            # On failure, mark as passed to avoid infinite loops
            return {
                **state,
                "evaluation_passed": True,
                "iteration":         iteration + 1,
                "error":             f"EvaluationAgent (non-fatal): {exc}",
            }

    # ------------------------------------------------------------------ #
    # Held-out evaluation                                                  #
    # ------------------------------------------------------------------ #

    def _compute_held_out_score(self, prob: float, signals) -> float:
        """
        Use the last 20% of the GDELT series as a proxy ground truth.
        Convert the target signal into a binary outcome:
          outcome = 1 if final value > mean  else  0
        Brier score = (prob - outcome)^2
        """
        if signals is None:
            return 0.25  # baseline (random) Brier score

        target = signals.target_series
        if len(target) < 5:
            return 0.25

        split = int(len(target) * 0.8)
        held_out = target.iloc[split:]

        # Proxy outcome: did the trend improve relative to the training mean?
        train_mean = float(target.iloc[:split].mean())
        held_out_mean = float(held_out.mean())
        outcome = 1.0 if held_out_mean > train_mean else 0.0

        brier = self.brier.score(
            probabilities=np.array([prob]),
            outcomes=np.array([outcome]),
        )
        return float(brier)

    # ------------------------------------------------------------------ #
    # Bayesian update                                                       #
    # ------------------------------------------------------------------ #

    def _bayesian_update(self, prior: float, signals) -> float:
        """
        Simple Bayesian prior-to-posterior update using the TSFM trend
        as a likelihood signal.

        P(A | evidence) ∝ P(evidence | A) × P(A)

        Here:
          - P(A)  = prior probability
          - P(e|A)= sigmoid(trend_slope × 5)  [likelihood of evidence if A is true]
          - P(e|¬A) = 1 - P(e|A)
        """
        if signals is None:
            return prior

        summary = signals.summary
        trend = summary.get("trend_direction", "stable")

        # Convert trend direction into a likelihood ratio
        likelihood_yes = {
            "improving":     0.70,
            "stable":        0.50,
            "deteriorating": 0.30,
            "neutral":       0.50,
        }.get(trend, 0.50)

        likelihood_no = 1.0 - likelihood_yes

        # Bayesian update
        numerator   = likelihood_yes * prior
        denominator = likelihood_yes * prior + likelihood_no * (1 - prior)

        if denominator < 1e-9:
            return prior

        posterior = numerator / denominator
        return float(np.clip(posterior, 0.02, 0.98))

    # ------------------------------------------------------------------ #
    # Pass / fail decision                                                  #
    # ------------------------------------------------------------------ #

    def _evaluation_passed(
        self,
        score: float,
        improvement: float,
        iteration: int,
        max_iters: int,
    ) -> bool:
        if iteration >= max_iters - 1:
            return True   # force termination at max iterations

        # Pass if score is already good (below random Brier = 0.25)
        if score < 0.15:
            return True

        # Pass if improvement is marginal (converged)
        threshold = self.config.agents.reflection_threshold
        if iteration > 0 and improvement < threshold:
            return True

        return False

    # ------------------------------------------------------------------ #
    # Reflection notes generation                                          #
    # ------------------------------------------------------------------ #

    def _generate_reflection_notes(
        self,
        score: float,
        improvement: float,
        passed: bool,
        prior_reasoning: str,
        signals,
    ) -> str:
        if passed and score < 0.15:
            return (
                f"Evaluation PASSED (Brier={score:.4f}). "
                f"Probability estimate is well-calibrated. No further refinement needed."
            )

        if passed and improvement < self.config.agents.reflection_threshold:
            return (
                f"Evaluation CONVERGED (Brier={score:.4f}, improvement={improvement:.4f}). "
                f"Further iterations unlikely to improve estimate."
            )

        if passed:
            return f"Maximum iterations reached. Final Brier={score:.4f}."

        # Build specific feedback for the next cycle
        inflections = getattr(signals, "inflections", [])
        missed_signals = [
            i.description for i in inflections
            if i.magnitude > 3.0
        ][:3]

        notes = (
            f"Evaluation NEEDS REFINEMENT (Brier={score:.4f}, improvement={improvement:.4f}). "
            f"Previous reasoning: '{prior_reasoning[:100]}...'. "
        )
        if missed_signals:
            notes += (
                f"High-magnitude inflections to incorporate: "
                + "; ".join(missed_signals)
                + ". Consider reweighting these signals."
            )
        else:
            notes += (
                "Consider adjusting the relevance filter to capture "
                "slower-moving structural signals."
            )
        return notes
