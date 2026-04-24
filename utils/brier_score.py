"""
utils/brier_score.py
--------------------
Brier score computation with 3-component decomposition:

  BS = Reliability - Resolution + Uncertainty

  Reliability : how close forecast probabilities are to observed frequencies
  Resolution  : how much forecasts differ from the base rate
  Uncertainty : inherent difficulty of the event (base rate entropy)

A perfect system minimises Reliability while maximising Resolution.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class BrierDecomposition:
    """3-component Brier score decomposition."""
    brier_score:  float
    reliability:  float     # lower is better  (0 = perfect calibration)
    resolution:   float     # higher is better (0 = useless)
    uncertainty:  float     # base rate entropy (fixed, independent of model)
    base_rate:    float
    n_samples:    int
    skill_score:  float     # 1 - BS/BS_reference  (>0 = better than climatology)


class BrierScoreCalculator:
    """
    Computes and decomposes the Brier score for probabilistic forecasts.

    Usage:
        brier = BrierScoreCalculator()

        # Simple score
        score = brier.score(probabilities, outcomes)

        # Full decomposition
        decomp = brier.decompose(probabilities, outcomes, n_bins=10)
    """

    def score(self, probabilities: np.ndarray, outcomes: np.ndarray) -> float:
        """
        Mean squared Brier score.
        BS = (1/N) Σ (p_i - o_i)²
        Range: [0, 1].  0 = perfect, 1 = worst possible.
        """
        p = np.asarray(probabilities, dtype=float)
        o = np.asarray(outcomes, dtype=float)
        self._validate(p, o)
        return float(np.mean((p - o) ** 2))

    def log_score(self, probabilities: np.ndarray, outcomes: np.ndarray) -> float:
        """
        Logarithmic (Brier) scoring rule.
        LS = (1/N) Σ o_i * log(p_i) + (1-o_i) * log(1-p_i)
        """
        p = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1 - 1e-7)
        o = np.asarray(outcomes, dtype=float)
        return float(np.mean(o * np.log(p) + (1 - o) * np.log(1 - p)))

    def decompose(
        self,
        probabilities: np.ndarray,
        outcomes: np.ndarray,
        n_bins: int = 10,
    ) -> BrierDecomposition:
        """
        Murphy decomposition of the Brier score.

        Ref: Murphy (1973) – "A New Vector Partition of the Probability Score"

        BS = REL - RES + UNC
        """
        p = np.asarray(probabilities, dtype=float)
        o = np.asarray(outcomes, dtype=float)
        self._validate(p, o)

        bs = float(np.mean((p - o) ** 2))
        base_rate = float(o.mean())
        n = len(o)

        # Bin forecasts
        bins = np.linspace(0, 1, n_bins + 1)
        rel = res = 0.0

        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            mask = (p >= lo) & (p < hi)
            if mask.sum() == 0:
                continue
            n_k    = mask.sum()
            p_k    = p[mask].mean()          # mean forecast in bin
            o_k    = o[mask].mean()          # observed frequency in bin
            rel   += (n_k / n) * (p_k - o_k) ** 2
            res   += (n_k / n) * (o_k - base_rate) ** 2

        unc = base_rate * (1 - base_rate)

        # Skill score: 1 - BS/BS_ref   (BS_ref = predicting base rate every time)
        bs_ref = unc
        skill = 1.0 - (bs / bs_ref) if bs_ref > 0 else 0.0

        return BrierDecomposition(
            brier_score=round(bs, 6),
            reliability=round(float(rel), 6),
            resolution=round(float(res), 6),
            uncertainty=round(float(unc), 6),
            base_rate=round(base_rate, 4),
            n_samples=n,
            skill_score=round(float(skill), 4),
        )

    def expected_calibration_error(
        self,
        probabilities: np.ndarray,
        outcomes: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        ECE = Σ_b (|B_b|/N) * |conf(B_b) - acc(B_b)|
        """
        p = np.asarray(probabilities, dtype=float)
        o = np.asarray(outcomes, dtype=float)
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n = len(p)

        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            mask = (p >= lo) & (p < hi)
            if mask.sum() == 0:
                continue
            ece += (mask.sum() / n) * abs(p[mask].mean() - o[mask].mean())

        return float(ece)

    def compare(
        self,
        model_probs: np.ndarray,
        baseline_probs: np.ndarray,
        outcomes: np.ndarray,
    ) -> dict:
        """Compare two forecasting systems on the same outcomes."""
        model_bs    = self.score(model_probs, outcomes)
        baseline_bs = self.score(baseline_probs, outcomes)
        return {
            "model_brier":    round(model_bs, 6),
            "baseline_brier": round(baseline_bs, 6),
            "improvement":    round(baseline_bs - model_bs, 6),
            "model_wins":     model_bs < baseline_bs,
        }

    @staticmethod
    def _validate(p: np.ndarray, o: np.ndarray):
        if len(p) != len(o):
            raise ValueError("probabilities and outcomes must have the same length")
        if not np.all((p >= 0) & (p <= 1)):
            raise ValueError("probabilities must be in [0, 1]")
        if not np.all((o == 0) | (o == 1)):
            raise ValueError("outcomes must be binary (0 or 1)")
