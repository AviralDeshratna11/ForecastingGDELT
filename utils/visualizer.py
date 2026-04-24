"""
utils/visualizer.py
-------------------
Generates diagnostic visualisations for the forecasting pipeline.
Produces matplotlib figures saved to the outputs/ directory.
"""

import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
from typing import Optional, Any

logger = logging.getLogger(__name__)


class ForecastVisualizer:
    """
    Creates diagnostic charts for the GDELT forecasting pipeline.

    Charts produced:
      1. Signal dashboard   – raw signal + z-score + inflection markers
      2. TSFM forecast      – historical + predicted series with quantile bands
      3. Calibration chart  – reliability diagram + probability comparison
      4. Summary report     – final probability + Kelly sizing
    """

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        plt.rcParams.update({
            "figure.facecolor": "#0f0f1a",
            "axes.facecolor":   "#1a1a2e",
            "axes.edgecolor":   "#4a4a6a",
            "axes.labelcolor":  "#e0e0ff",
            "text.color":       "#e0e0ff",
            "xtick.color":      "#a0a0c0",
            "ytick.color":      "#a0a0c0",
            "grid.color":       "#2a2a4a",
            "grid.linestyle":   "--",
            "grid.alpha":       0.5,
        })

    def plot_all(
        self,
        signals: Any,
        forecast: Any,
        final_output: dict,
        query: str,
    ) -> list[str]:
        """Generate all diagnostic charts. Returns list of saved file paths."""
        paths = []
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        slug = query[:30].replace(" ", "_").lower()

        try:
            p = self._plot_signal_dashboard(signals, query, f"{ts}_{slug}_signals.png")
            paths.append(p)
        except Exception as exc:
            logger.warning("Signal dashboard failed: %s", exc)

        try:
            p = self._plot_forecast(signals, forecast, query, f"{ts}_{slug}_forecast.png")
            paths.append(p)
        except Exception as exc:
            logger.warning("Forecast plot failed: %s", exc)

        try:
            p = self._plot_summary(final_output, f"{ts}_{slug}_summary.png")
            paths.append(p)
        except Exception as exc:
            logger.warning("Summary plot failed: %s", exc)

        return paths

    # ------------------------------------------------------------------ #
    # Individual chart generators                                          #
    # ------------------------------------------------------------------ #

    def _plot_signal_dashboard(self, signals: Any, query: str, filename: str) -> str:
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(f"GDELT Signal Dashboard — {query[:60]}", fontsize=14, color="#c8d8ff", y=0.98)
        gs = gridspec.GridSpec(3, 1, hspace=0.35)

        target = signals.target_series
        z_scores = signals.z_scores
        inflections = signals.inflections

        # Panel 1: Raw target signal
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(target.index, target.values, color="#4fc3f7", linewidth=1.5, label="Target signal")
        ax1.fill_between(target.index, target.values, alpha=0.15, color="#4fc3f7")
        ax1.axhline(0, color="#666688", linewidth=0.8, linestyle=":")
        ax1.set_ylabel("Signal Value", fontsize=10)
        ax1.set_title("Raw GDELT Signal", fontsize=11, loc="left")
        ax1.grid(True)
        ax1.legend(fontsize=9)

        # Panel 2: Z-score
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        target_col = signals.summary.get("target_col", target.name or "signal")
        if target_col in z_scores.columns:
            z_vals = z_scores[target_col]
            threshold = 2.0
            ax2.plot(z_vals.index, z_vals.values, color="#ce93d8", linewidth=1.2, label="Z-score")
            ax2.fill_between(z_vals.index, threshold, z_vals.values, where=z_vals.values > threshold,
                             alpha=0.3, color="#ef5350", label=f"|z|>{threshold}")
            ax2.fill_between(z_vals.index, -threshold, z_vals.values, where=z_vals.values < -threshold,
                             alpha=0.3, color="#42a5f5")
            ax2.axhline(threshold,  color="#ef5350", linewidth=1, linestyle="--", alpha=0.7)
            ax2.axhline(-threshold, color="#42a5f5", linewidth=1, linestyle="--", alpha=0.7)
            ax2.axhline(0, color="#666688", linewidth=0.8)
        ax2.set_ylabel("Z-score", fontsize=10)
        ax2.set_title("Rolling Z-score (Trend Inflection Detector)", fontsize=11, loc="left")
        ax2.grid(True)
        ax2.legend(fontsize=9)

        # Panel 3: Article volume / activity density
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        raw = signals.raw
        vol_col = next((c for c in ["num_articles", "activity_density", "num_mentions"] if c in raw.columns), None)
        if vol_col:
            ax3.bar(raw.index, raw[vol_col].values, color="#80cbc4", alpha=0.6, width=0.8, label=vol_col)

        # Mark inflections
        sig_inflections = [i for i in inflections if abs(i.z_score) > 2.0]
        for inf in sig_inflections[:10]:
            for ax in [ax1, ax2, ax3]:
                ax.axvline(inf.date, color="#ffd54f", linewidth=1.0, alpha=0.6, linestyle=":")

        ax3.set_ylabel("Volume", fontsize=10)
        ax3.set_xlabel("Date", fontsize=10)
        ax3.set_title("Media Volume + Inflection Markers (yellow)", fontsize=11, loc="left")
        ax3.grid(True)
        ax3.legend(fontsize=9)

        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        logger.info("Saved signal dashboard: %s", path)
        return path

    def _plot_forecast(self, signals: Any, forecast: Any, query: str, filename: str) -> str:
        fig, ax = plt.subplots(figsize=(14, 7))
        fig.suptitle(f"TSFM Forecast — {query[:60]}", fontsize=13, color="#c8d8ff")

        target = signals.target_series
        horizon = forecast.horizon
        last_date = target.index[-1]
        future_dates = pd.date_range(last_date, periods=horizon + 1, freq="D")[1:]

        # Historical
        ax.plot(target.index, target.values, color="#4fc3f7", linewidth=1.8, label="Historical signal", zorder=3)

        # Forecast
        pf = forecast.point_forecast
        lo = forecast.lower_bound
        hi = forecast.upper_bound

        ax.plot(future_dates, pf, color="#ffb74d", linewidth=2.0, linestyle="--", label=f"Forecast ({forecast.model_name})", zorder=4)
        ax.fill_between(future_dates, lo, hi, alpha=0.2, color="#ffb74d", label="80% CI")

        # Quantile bands
        if 0.25 in forecast.quantiles and 0.75 in forecast.quantiles:
            ax.fill_between(
                future_dates,
                forecast.quantiles[0.25],
                forecast.quantiles[0.75],
                alpha=0.3, color="#ff8f00", label="50% CI"
            )

        # Boundary line
        ax.axvline(last_date, color="#888899", linewidth=1.2, linestyle=":", alpha=0.8, label="Forecast start")
        ax.axhline(0, color="#555566", linewidth=0.8, linestyle=":")

        ax.set_ylabel("Signal Value", fontsize=11)
        ax.set_xlabel("Date", fontsize=11)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True)

        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        logger.info("Saved forecast chart: %s", path)
        return path

    def _plot_summary(self, final_output: dict, filename: str) -> str:
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        fig.suptitle("Forecasting System — Final Summary", fontsize=14, color="#c8d8ff", y=1.02)

        # Panel 1: Probability gauge
        ax = axes[0]
        prob = final_output.get("final_probability", 0.5)
        colors = ["#ef5350" if prob < 0.3 else "#ffd54f" if prob < 0.6 else "#66bb6a"]
        ax.barh(["Probability"], [prob], color=colors, height=0.4)
        ax.barh(["Probability"], [1 - prob], left=[prob], color="#2a2a3e", height=0.4)
        ax.set_xlim(0, 1)
        ax.axvline(0.5, color="#888899", linewidth=1.2, linestyle="--")
        ax.text(prob / 2, 0, f"{prob:.1%}", ha="center", va="center", fontsize=14, color="white", fontweight="bold")
        ax.set_title("Final Probability (YES)", fontsize=11)
        ax.set_xlabel("Probability")
        ax.grid(False)

        # Panel 2: Calibration pipeline
        ax2 = axes[1]
        cal_report = final_output.get("calibration_report", {})
        stages = ["Raw", "Calibrated", "Extremized"]
        vals = [
            final_output.get("raw_probability", 0.5),
            final_output.get("calibrated_probability", 0.5),
            final_output.get("extremized_probability", 0.5),
        ]
        bar_colors = ["#7986cb", "#42a5f5", "#26c6da"]
        bars = ax2.bar(stages, vals, color=bar_colors, width=0.5)
        ax2.set_ylim(0, 1)
        ax2.axhline(0.5, color="#888899", linewidth=1, linestyle="--", alpha=0.7)
        for bar, val in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=11, color="white")
        ax2.set_title("Calibration Pipeline", fontsize=11)
        ax2.set_ylabel("Probability")
        ax2.grid(axis="y")

        # Panel 3: Key metrics text
        ax3 = axes[2]
        ax3.axis("off")
        metrics = [
            ("Recommendation",   final_output.get("recommendation", "—")),
            ("Kelly Fraction",   f"{final_output.get('kelly_fraction', 0):.2%}"),
            ("Brier Score",      f"{final_output.get('evaluation_score', 0):.4f}"),
            ("Iterations",       str(final_output.get("iterations", 1))),
            ("Inflections",      str(final_output.get("n_inflections", 0))),
            ("Trend",            final_output.get("trend_analysis", {}).get("overall_trend", "—")),
        ]
        y_pos = 0.90
        for label, value in metrics:
            color = "#ffd54f" if label == "Recommendation" else "#c8d8ff"
            ax3.text(0.05, y_pos, f"{label}:", fontsize=11, color="#a0a0c0", transform=ax3.transAxes)
            ax3.text(0.55, y_pos, value, fontsize=11, color=color, fontweight="bold", transform=ax3.transAxes)
            y_pos -= 0.13
        ax3.set_title("Key Metrics", fontsize=11)

        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0f0f1a")
        plt.close()
        logger.info("Saved summary chart: %s", path)
        return path
