"""
core/tsfm_forecaster.py
-----------------------
Time Series forecasting with ARIMA / naive fallback.
Fixed: always converts input to clean numpy array before ARIMA.
"""

import logging
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    point_forecast: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    quantiles: dict
    horizon: int
    model_name: str
    uncertainty: float
    trend_slope: float
    trend_direction: str        # "up" | "down" | "flat"
    trend_strength: float       # 0-1 normalised slope magnitude
    momentum: float             # recent acceleration
    metadata: dict = field(default_factory=dict)

    @property
    def median_forecast(self) -> np.ndarray:
        return self.quantiles.get(0.5, self.point_forecast)


class TSFMForecaster:
    def __init__(self, config):
        self.config = config
        self._backend = self._detect_backend()
        logger.info("TSFM backend: %s", self._backend)

    def forecast(self, series, horizon: Optional[int] = None,
                 covariates=None) -> ForecastResult:
        h = horizon or self.config.prediction_length

        # ── Always produce a clean numpy float array ──────────────────
        if hasattr(series, "values"):
            arr = series.values.astype(float)
        else:
            arr = np.asarray(series, dtype=float)

        # Replace NaN/inf
        finite_mask = np.isfinite(arr)
        if finite_mask.sum() == 0:
            arr = np.zeros(len(arr))
        else:
            mean_val = arr[finite_mask].mean()
            arr = np.where(np.isfinite(arr), arr, mean_val)

        if self._backend == "timesfm":
            return self._forecast_timesfm(arr, h, covariates)
        elif self._backend == "chronos":
            return self._forecast_chronos(arr, h)
        else:
            return self._forecast_fallback(arr, h)

    # ------------------------------------------------------------------ #
    # ARIMA fallback                                                       #
    # ------------------------------------------------------------------ #

    def _forecast_fallback(self, arr: np.ndarray, horizon: int) -> ForecastResult:
        """ARIMA(2,1,2) on clean numpy array — no Series, no iloc."""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            train = arr[-min(200, len(arr)):]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = ARIMA(train, order=(2, 1, 2)).fit()
            fc     = fitted.get_forecast(steps=horizon)
            pf     = np.asarray(fc.predicted_mean, dtype=float)
            ci     = fc.conf_int(alpha=0.2)
            lo     = np.asarray(ci.iloc[:, 0], dtype=float)
            hi     = np.asarray(ci.iloc[:, 1], dtype=float)
            qd = {0.1: lo, 0.25: lo*0.5+pf*0.5, 0.5: pf,
                  0.75: hi*0.5+pf*0.5, 0.9: hi}
            return self._build_result(arr, pf, qd, horizon, "ARIMA(2,1,2)")
        except Exception as exc:
            logger.warning("ARIMA failed (%s); using naive extrapolation", exc)
            return self._naive_extrapolation(arr, horizon)

    def _naive_extrapolation(self, arr: np.ndarray, horizon: int) -> ForecastResult:
        """Polynomial trend extrapolation with seasonal decomposition."""
        tail = arr[-min(60, len(arr)):]
        x    = np.arange(len(tail))

        # Fit degree-2 polynomial for curved trends
        coeffs   = np.polyfit(x, tail, 2)
        future_x = np.arange(len(tail), len(tail) + horizon)
        pf       = np.polyval(coeffs, future_x)

        noise = tail.std() if tail.std() > 0 else 0.1
        qd = {
            0.1:  pf - 1.645 * noise,
            0.25: pf - 0.674 * noise,
            0.5:  pf,
            0.75: pf + 0.674 * noise,
            0.9:  pf + 1.645 * noise,
        }
        return self._build_result(arr, pf, qd, horizon, "PolyTrend")

    # ------------------------------------------------------------------ #
    # Optional backends                                                    #
    # ------------------------------------------------------------------ #

    def _forecast_timesfm(self, arr, horizon, covariates) -> ForecastResult:
        try:
            import timesfm
            tfm = timesfm.TimesFm(
                context_len=min(self.config.context_length, len(arr)),
                horizon_len=horizon, backend="cpu",
            )
            point, quant = tfm.forecast([arr[-self.config.context_length:]], freq=[0])
            pf = np.array(point[0])
            qd = {q: quant[0][:, i] if quant[0].ndim > 1 else pf
                  for i, q in enumerate(self.config.quantiles)}
            return self._build_result(arr, pf, qd, horizon, "TimesFM-2.5")
        except Exception as exc:
            logger.warning("TimesFM failed: %s", exc)
            return self._forecast_fallback(arr, horizon)

    def _forecast_chronos(self, arr, horizon) -> ForecastResult:
        try:
            import torch
            from chronos import ChronosPipeline
            pipeline = ChronosPipeline.from_pretrained(
                "amazon/chronos-t5-small", device_map="cpu",
                torch_dtype=torch.float32)
            ctx = torch.tensor(arr[-self.config.context_length:]).unsqueeze(0)
            fc  = pipeline.predict(ctx, horizon)
            samples = fc[0].numpy()
            pf = samples.mean(axis=0)
            qd = {0.1: np.percentile(samples,10,axis=0),
                  0.25: np.percentile(samples,25,axis=0),
                  0.5:  np.percentile(samples,50,axis=0),
                  0.75: np.percentile(samples,75,axis=0),
                  0.9:  np.percentile(samples,90,axis=0)}
            return self._build_result(arr, pf, qd, horizon, "Chronos-T5")
        except Exception as exc:
            logger.warning("Chronos failed: %s", exc)
            return self._forecast_fallback(arr, horizon)

    # ------------------------------------------------------------------ #
    # Result builder — computes rich trend metrics                         #
    # ------------------------------------------------------------------ #

    def _build_result(self, historical: np.ndarray, pf: np.ndarray,
                      quantiles: dict, horizon: int, model_name: str) -> ForecastResult:
        pf = np.asarray(pf, dtype=float)
        lo = np.asarray(quantiles.get(0.1, pf - pf.std()), dtype=float)
        hi = np.asarray(quantiles.get(0.9, pf + pf.std()), dtype=float)

        uncertainty = float(np.mean(hi - lo))
        slope       = float(np.polyfit(np.arange(horizon), pf, 1)[0])

        # Trend direction with threshold
        if slope > 0.02:
            direction = "up"
        elif slope < -0.02:
            direction = "down"
        else:
            direction = "flat"

        # Normalised trend strength (0-1)
        hist_std = float(historical.std()) if historical.std() > 0 else 1.0
        strength = float(min(abs(slope) / (hist_std + 1e-9), 1.0))

        # Momentum: compare last-quarter forecast slope vs overall slope
        if len(pf) >= 4:
            q_slope = float(np.polyfit(np.arange(len(pf)//4), pf[:len(pf)//4], 1)[0])
            momentum = q_slope - slope
        else:
            momentum = 0.0

        return ForecastResult(
            point_forecast=pf,
            lower_bound=lo,
            upper_bound=hi,
            quantiles=quantiles,
            horizon=horizon,
            model_name=model_name,
            uncertainty=uncertainty,
            trend_slope=slope,
            trend_direction=direction,
            trend_strength=strength,
            momentum=momentum,
        )

    def _detect_backend(self) -> str:
        backend = self.config.backend
        if backend == "timesfm":
            try:
                import timesfm; return "timesfm"
            except ImportError:
                pass
        if backend in ("chronos", "timesfm"):
            try:
                import chronos; return "chronos"
            except ImportError:
                pass
        return "fallback"
