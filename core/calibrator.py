"""
core/calibrator.py
------------------
Probability calibration pipeline.

Implements:
  - Platt scaling (logistic regression on raw scores)
  - Isotonic regression
  - Beta calibration
  - Temperature scaling
  - Power-transform extremization  (α-extremization)
  - Expected Calibration Error (ECE) computation
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CalibrationReport:
    """Metrics from a calibration run."""
    raw_probability: float
    calibrated_probability: float
    extremized_probability: float
    method: str
    ece: Optional[float] = None
    reliability_score: Optional[float] = None


class ProbabilityCalibrator:
    """
    Applies post-hoc calibration to raw LLM/model probability outputs.

    Usage:
        cal = ProbabilityCalibrator(config)
        report = cal.calibrate(raw_prob=0.55, method="platt")
        print(report.extremized_probability)
    """

    def __init__(self, config):
        self.config = config
        self._platt_model = None
        self._isotonic_model = None

    # ------------------------------------------------------------------ #
    # Primary calibration entry point                                      #
    # ------------------------------------------------------------------ #

    def calibrate(
        self,
        raw_probability: float,
        method: Optional[str] = None,
        calibration_data: Optional[tuple] = None,  # (probs, outcomes)
    ) -> CalibrationReport:
        """
        Calibrate a single raw probability estimate.

        Args:
            raw_probability:  uncalibrated probability in [0, 1]
            method:           override config method
            calibration_data: (probs_array, outcomes_array) for fitting

        Returns:
            CalibrationReport
        """
        method = method or self.config.method
        p = float(np.clip(raw_probability, 1e-6, 1 - 1e-6))

        # Fit calibrators if data provided
        if calibration_data is not None:
            probs, outcomes = calibration_data
            self._fit_calibrators(probs, outcomes)

        # Apply calibration
        calibrated = self._apply_method(p, method)
        calibrated = float(np.clip(calibrated, 1e-6, 1 - 1e-6))

        # Apply extremization
        extremized = self._extremize(calibrated) if self.config.extremize else calibrated

        return CalibrationReport(
            raw_probability=raw_probability,
            calibrated_probability=calibrated,
            extremized_probability=extremized,
            method=method,
        )

    def calibrate_batch(
        self,
        raw_probs: np.ndarray,
        outcomes: Optional[np.ndarray] = None,
        method: Optional[str] = None,
    ) -> np.ndarray:
        """
        Calibrate a batch of probabilities.
        If outcomes are provided, fits a calibrator first.
        """
        method = method or self.config.method
        if outcomes is not None:
            self._fit_calibrators(raw_probs, outcomes)

        calibrated = np.array([self._apply_method(p, method) for p in raw_probs])
        calibrated = np.clip(calibrated, 1e-6, 1 - 1e-6)
        if self.config.extremize:
            calibrated = self._extremize_array(calibrated)
        return calibrated

    # ------------------------------------------------------------------ #
    # Calibration methods                                                  #
    # ------------------------------------------------------------------ #

    def _apply_method(self, p: float, method: str) -> float:
        if method == "platt" and self._platt_model is not None:
            return self._platt_transform(p)
        elif method == "isotonic" and self._isotonic_model is not None:
            return self._isotonic_transform(p)
        elif method == "temperature":
            return self._temperature_scaling(p)
        elif method == "beta":
            return self._beta_calibration(p)
        else:
            # No-op if no fitted model
            return p

    def _fit_calibrators(self, probs: np.ndarray, outcomes: np.ndarray):
        """Fit Platt and isotonic calibrators from historical data."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.isotonic import IsotonicRegression

        probs = np.clip(probs, 1e-6, 1 - 1e-6)
        log_odds = np.log(probs / (1 - probs)).reshape(-1, 1)

        # Platt scaling: LR on log-odds
        self._platt_model = LogisticRegression()
        self._platt_model.fit(log_odds, outcomes)
        logger.debug("Platt scaling fitted on %d samples", len(probs))

        # Isotonic regression
        self._isotonic_model = IsotonicRegression(out_of_bounds="clip")
        self._isotonic_model.fit(probs, outcomes)
        logger.debug("Isotonic regression fitted")

    def _platt_transform(self, p: float) -> float:
        log_odds = np.log(p / (1 - p)).reshape(1, 1)
        return float(self._platt_model.predict_proba(log_odds)[0, 1])

    def _isotonic_transform(self, p: float) -> float:
        return float(self._isotonic_model.predict([p])[0])

    def _temperature_scaling(self, p: float, temperature: float = 1.3) -> float:
        """
        Scale the log-odds by 1/T.
        T > 1 → softer (moves toward 0.5).
        T < 1 → sharper.
        LLMs tend to over-hedge so T = 1.3 softens further before extremization.
        """
        log_odds = np.log(p / (1 - p)) / temperature
        return float(1 / (1 + np.exp(-log_odds)))

    def _beta_calibration(self, p: float, a: float = 1.0, b: float = 1.0, c: float = 0.5) -> float:
        """
        Beta calibration: a broader family than Platt.
        f(p) = sigmoid(a * log(p) - b * log(1-p) + c)
        Default params are identity; tune after fitting.
        """
        eps = 1e-7
        p = np.clip(p, eps, 1 - eps)
        logit = a * np.log(p) - b * np.log(1 - p) + c
        return float(1 / (1 + np.exp(-logit)))

    # ------------------------------------------------------------------ #
    # Extremization                                                        #
    # ------------------------------------------------------------------ #

    def _extremize(self, p: float) -> float:
        """
        Power-transform extremization:
            p_ext = p^α / (p^α + (1-p)^α)
        α > 1 pushes predictions away from 0.5 toward 0 or 1.
        """
        alpha = self.config.extremize_alpha
        eps = 1e-7
        p = np.clip(p, eps, 1 - eps)
        p_alpha = p ** alpha
        q_alpha = (1 - p) ** alpha
        return float(p_alpha / (p_alpha + q_alpha))

    def _extremize_array(self, probs: np.ndarray) -> np.ndarray:
        alpha = self.config.extremize_alpha
        eps = 1e-7
        probs = np.clip(probs, eps, 1 - eps)
        p_alpha = probs ** alpha
        q_alpha = (1 - probs) ** alpha
        return p_alpha / (p_alpha + q_alpha)

    # ------------------------------------------------------------------ #
    # Diagnostics                                                          #
    # ------------------------------------------------------------------ #

    def expected_calibration_error(
        self,
        probs: np.ndarray,
        outcomes: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).
        ECE = Σ (|B_i| / n) * |avg_confidence(B_i) - accuracy(B_i)|
        """
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n = len(probs)

        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            mask = (probs >= lo) & (probs < hi)
            if mask.sum() == 0:
                continue
            avg_conf = probs[mask].mean()
            avg_acc  = outcomes[mask].mean()
            ece += (mask.sum() / n) * abs(avg_conf - avg_acc)

        return float(ece)

    def reliability_diagram_data(
        self, probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10
    ) -> dict:
        """Return data for plotting a reliability diagram."""
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers, bin_accs, bin_counts = [], [], []

        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            mask = (probs >= lo) & (probs < hi)
            if mask.sum() == 0:
                continue
            bin_centers.append((lo + hi) / 2)
            bin_accs.append(outcomes[mask].mean())
            bin_counts.append(mask.sum())

        return {
            "bin_centers": bin_centers,
            "bin_accuracies": bin_accs,
            "bin_counts": bin_counts,
        }
