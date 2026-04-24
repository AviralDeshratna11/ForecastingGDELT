"""
core/signal_processor.py
------------------------
Transforms raw GDELT DataFrames into normalised signals:
  - Z-score computation using rolling windows
  - Trend inflection detection
  - Pearson cross-correlation for historical analogues
  - PCA-based dimensionality reduction on GKG dimensions
  - Fourier noise filtering
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy import signal as scipy_signal
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class InflectionPoint:
    """A detected trend inflection in the signal."""
    date: pd.Timestamp
    column: str
    z_score: float
    direction: str          # "up" | "down"
    magnitude: float        # absolute z-score
    description: str = ""


@dataclass
class ProcessedSignals:
    """Container for all derived signals ready for TSFM input."""
    raw: pd.DataFrame
    normalised: pd.DataFrame
    z_scores: pd.DataFrame
    inflections: list[InflectionPoint]
    target_series: pd.Series      # primary time series for forecasting
    covariate_series: pd.DataFrame
    feature_names: list[str] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


class SignalProcessor:
    """
    Converts raw GDELT DataFrames into predictive signals.

    Usage:
        processor = SignalProcessor(config)
        signals   = processor.process(combined_df, target_col="avg_tone")
    """

    def __init__(self, config):
        self.config = config
        self._scaler = StandardScaler()

    def process(
        self,
        df: pd.DataFrame,
        target_col: str = "avg_tone",
        extra_covariates: Optional[list[str]] = None,
    ) -> ProcessedSignals:
        """Full processing pipeline."""
        if df.empty:
            raise ValueError("Cannot process an empty DataFrame")

        df = self._ensure_datetime_index(df)
        df = self._fill_gaps(df)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col not in numeric_cols:
            target_col = numeric_cols[0]
            logger.warning("Target column not found; using '%s'", target_col)

        normalised = self._normalise(df[numeric_cols])
        z_scores   = self._compute_z_scores(df[numeric_cols])
        inflections = self._detect_inflections(z_scores)

        # Build covariate matrix
        covariate_cols = [c for c in numeric_cols if c != target_col]
        if extra_covariates:
            covariate_cols = list(set(covariate_cols + extra_covariates))
        covariate_cols = [c for c in covariate_cols if c in df.columns]

        # Optional: Fourier denoise the target
        denoised_target = self._fourier_denoise(df[target_col])

        summary = {
            "n_days": len(df),
            "n_inflections": len(inflections),
            "target_col": target_col,
            "target_mean": float(df[target_col].mean()),
            "target_std": float(df[target_col].std()),
            "latest_z_score": float(z_scores[target_col].iloc[-1]) if target_col in z_scores else 0.0,
            "trend_direction": self._overall_trend(df[target_col]),
            "significant_inflections": [i for i in inflections if abs(i.z_score) > self.config.z_score_threshold],
        }

        return ProcessedSignals(
            raw=df,
            normalised=normalised,
            z_scores=z_scores,
            inflections=inflections,
            target_series=denoised_target,
            covariate_series=df[covariate_cols] if covariate_cols else pd.DataFrame(),
            feature_names=numeric_cols,
            summary=summary,
        )

    # ------------------------------------------------------------------ #
    # Signal engineering methods                                           #
    # ------------------------------------------------------------------ #

    def _ensure_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df

    def _fill_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample to daily frequency and forward-fill missing values."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df = df[numeric_cols].resample("D").mean()
        df = df.ffill().bfill()
        return df

    def _normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standard scaling (z-normalise) the entire feature matrix."""
        arr = self._scaler.fit_transform(df.values)
        return pd.DataFrame(arr, index=df.index, columns=df.columns)

    def _compute_z_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rolling z-scores for each column.
        z = (x - rolling_mean) / rolling_std
        Uses a 60-day long window and a 7-day short window.
        """
        long_w  = self.config.rolling_window_days
        short_w = self.config.short_window_days
        z_df = pd.DataFrame(index=df.index)

        for col in df.columns:
            series = df[col]
            roll_mean = series.rolling(window=long_w, min_periods=max(1, long_w // 3)).mean()
            roll_std  = series.rolling(window=long_w, min_periods=max(1, long_w // 3)).std()
            roll_std  = roll_std.replace(0, np.nan).ffill().bfill().fillna(1e-6)
            z_df[col] = (series - roll_mean) / roll_std

        return z_df

    def _detect_inflections(self, z_df: pd.DataFrame) -> list[InflectionPoint]:
        """Find dates where |z| exceeds threshold and constitutes a directional shift."""
        threshold = self.config.z_score_threshold
        inflections = []

        for col in z_df.columns:
            series = z_df[col].dropna()
            for i in range(1, len(series)):
                z_now  = series.iloc[i]
                z_prev = series.iloc[i - 1]
                if abs(z_now) >= threshold and abs(z_prev) < threshold:
                    direction = "up" if z_now > 0 else "down"
                    inflections.append(InflectionPoint(
                        date=series.index[i],
                        column=col,
                        z_score=z_now,
                        direction=direction,
                        magnitude=abs(z_now),
                        description=(
                            f"{col} crossed z={z_now:.2f} "
                            f"({'above' if direction == 'up' else 'below'} "
                            f"{threshold}-sigma threshold)"
                        ),
                    ))

        inflections.sort(key=lambda x: x.date)
        return inflections

    def _fourier_denoise(self, series: pd.Series, cutoff_fraction: float = 0.2) -> pd.Series:
        """
        Project series into frequency domain and suppress high-frequency noise.
        Keeps the lowest `cutoff_fraction` of frequency components.
        """
        if len(series) < 10:
            return series

        arr = series.fillna(series.mean()).values
        fft_vals = np.fft.rfft(arr)
        freqs = np.fft.rfftfreq(len(arr))
        cutoff = cutoff_fraction * freqs.max()
        fft_vals[freqs > cutoff] = 0
        denoised = np.fft.irfft(fft_vals, n=len(arr))
        return pd.Series(denoised, index=series.index, name=series.name)

    def _overall_trend(self, series: pd.Series) -> str:
        """Simple linear regression slope to characterise trend direction."""
        if len(series) < 5:
            return "neutral"
        x = np.arange(len(series))
        y = series.fillna(series.mean()).values
        slope = np.polyfit(x, y, 1)[0]
        if slope > 0.05:
            return "improving"
        elif slope < -0.05:
            return "deteriorating"
        return "stable"

    def compute_cross_correlation(
        self, series_a: pd.Series, series_b: pd.Series
    ) -> float:
        """Pearson correlation between two aligned series."""
        aligned = pd.concat([series_a, series_b], axis=1).dropna()
        if len(aligned) < 5:
            return 0.0
        return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))

    def pca_reduce(self, df: pd.DataFrame, n_components: int = 5) -> pd.DataFrame:
        """Reduce high-dimensional GKG features to top-N principal components."""
        if df.shape[1] <= n_components:
            return df
        pca = PCA(n_components=n_components)
        arr = pca.fit_transform(df.fillna(0).values)
        cols = [f"pc_{i+1}" for i in range(n_components)]
        logger.debug(
            "PCA retained %.1f%% variance",
            pca.explained_variance_ratio_.sum() * 100,
        )
        return pd.DataFrame(arr, index=df.index, columns=cols)
