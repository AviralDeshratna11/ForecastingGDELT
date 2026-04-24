"""
agents/librarian_agent.py
--------------------------
Librarian Agent — fetches, processes, and builds a rich numerical
summary that the Analyst LLM can actually reason from.
Includes full time-series stats, Goldstein scale breakdown,
volume spikes, and dated inflection points.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any

from core.gdelt_client import GDELTClient
from core.signal_processor import SignalProcessor
from agents.state import ForecastingState

logger = logging.getLogger(__name__)

QUAD_CLASS_LABELS = {
    1: "Verbal Cooperation",
    2: "Material Cooperation",
    3: "Verbal Conflict",
    4: "Material Conflict",
}


class LibrarianAgent:
    def __init__(self, gdelt_client: GDELTClient, signal_processor: SignalProcessor):
        self.client    = gdelt_client
        self.processor = signal_processor

    def run(self, state: ForecastingState) -> ForecastingState:
        query     = state.get("query", "")
        lookback  = state.get("lookback_days", 90)
        iteration = state.get("iteration", 0)

        logger.info("[LibrarianAgent] Fetching GDELT data for: %s  (iter %d)", query, iteration)

        try:
            combined_df = self.client.fetch_combined(query, days=lookback)

            if combined_df.empty:
                from datetime import timedelta
                end = datetime.utcnow()
                start = end - timedelta(days=lookback)
                combined_df = self.client._generate_synthetic_events(query, start, end)

            signals      = self.processor.process(combined_df, target_col="avg_tone")
            news_summary = self._build_rich_summary(query, signals, combined_df)
            themes       = self._extract_themes(combined_df)

            logger.info("[LibrarianAgent] Done. %d rows, %d inflections, summary=%d chars",
                        len(combined_df), len(signals.inflections), len(news_summary))

            return {
                **state,
                "gdelt_raw":         combined_df,
                "gdelt_signals":     signals,
                "inflections":       signals.inflections,
                "news_summary":      news_summary,
                "thematic_clusters": themes,
                "error":             None,
            }

        except Exception as exc:
            logger.exception("[LibrarianAgent] Failed: %s", exc)
            return {**state, "error": f"LibrarianAgent: {exc}"}

    # ------------------------------------------------------------------ #
    # Rich numerical summary builder                                       #
    # ------------------------------------------------------------------ #

    def _build_rich_summary(self, query: str, signals: Any, df: Any) -> str:
        s           = signals.summary
        inflections = signals.inflections
        raw         = signals.raw

        # ── Tone statistics ──────────────────────────────────────────────
        tone_col = "avg_tone" if "avg_tone" in df.columns else "tone"
        tone_series = raw[tone_col].dropna() if tone_col in raw.columns else pd.Series([0])

        tone_7d  = tone_series.iloc[-7:].mean()  if len(tone_series) >= 7  else tone_series.mean()
        tone_30d = tone_series.iloc[-30:].mean() if len(tone_series) >= 30 else tone_series.mean()
        tone_all = tone_series.mean()
        tone_min = tone_series.min()
        tone_max = tone_series.max()
        tone_std = tone_series.std()

        # Recent trend: compare last 7d vs prior 7d
        if len(tone_series) >= 14:
            recent_delta = tone_series.iloc[-7:].mean() - tone_series.iloc[-14:-7].mean()
            trend_text = f"{recent_delta:+.3f} vs prior week ({'improving' if recent_delta > 0 else 'worsening'})"
        else:
            trend_text = "insufficient data"

        # ── Goldstein scale ──────────────────────────────────────────────
        gs_text = "N/A"
        if "goldstein_scale" in raw.columns:
            gs = raw["goldstein_scale"].dropna()
            if len(gs) > 0:
                gs_text = (
                    f"mean={gs.mean():.3f}, "
                    f"recent={gs.iloc[-7:].mean():.3f}, "
                    f"min={gs.min():.2f}, max={gs.max():.2f}"
                )

        # ── Volume metrics ───────────────────────────────────────────────
        vol_col = next((c for c in ["num_articles", "num_mentions", "num_articles_evt",
                                     "num_articles_gkg"] if c in raw.columns), None)
        vol_text = "N/A"
        if vol_col:
            vol = raw[vol_col].dropna()
            if len(vol) > 0:
                vol_text = (
                    f"mean={vol.mean():.0f}/day, "
                    f"peak={vol.max():.0f} on {vol.idxmax().strftime('%Y-%m-%d') if hasattr(vol.idxmax(), 'strftime') else 'N/A'}, "
                    f"recent={vol.iloc[-7:].mean():.0f}/day"
                )

        # ── Sentiment spread ─────────────────────────────────────────────
        sent_text = "N/A"
        if "positive_score" in raw.columns and "negative_score" in raw.columns:
            pos = raw["positive_score"].dropna().mean()
            neg = raw["negative_score"].dropna().mean()
            pol = raw["polarity"].dropna().mean() if "polarity" in raw.columns else neg - pos
            sent_text = f"positive={pos:.2f}, negative={neg:.2f}, polarity={pol:.2f}"

        # ── QuadClass distribution ───────────────────────────────────────
        quad_text = "N/A"
        if "quad_class" in df.columns:
            counts = df["quad_class"].value_counts().to_dict()
            total  = sum(counts.values())
            quad_text = ", ".join(
                f"{QUAD_CLASS_LABELS.get(int(k), f'Q{k}')}: {v} ({100*v/total:.0f}%)"
                for k, v in sorted(counts.items())
            )

        # ── Inflection points (top 8 by magnitude) ──────────────────────
        top_inf = sorted(inflections, key=lambda x: -x.magnitude)[:8]
        if top_inf:
            inf_lines = "\n".join(
                f"  [{i.date.strftime('%Y-%m-%d')}] z={i.z_score:+.2f} | {i.direction.upper()} | {i.column} | {i.description}"
                for i in top_inf
            )
        else:
            inf_lines = "  No significant inflections detected"

        # ── Raw tone time series (last 30 days, weekly sampled) ──────────
        if len(tone_series) >= 14:
            sample = tone_series.iloc[-30::3]  # every 3rd point of last 30 days
            ts_lines = "  " + "  ".join(
                f"{d.strftime('%m/%d')}:{v:.2f}"
                for d, v in zip(sample.index, sample.values)
            )
        else:
            ts_lines = "  Insufficient history"

        summary = f"""=== GDELT QUANTITATIVE SIGNAL REPORT ===
Query          : {query}
Analysis Window: {s.get('n_days', '?')} days  |  Data rows: {len(raw)}

── TONE METRICS (AvgTone: negative=conflict, positive=cooperation) ──
Overall mean   : {tone_all:.3f}  (range {tone_min:.2f} to {tone_max:.2f}, std={tone_std:.3f})
30-day mean    : {tone_30d:.3f}
7-day mean     : {tone_7d:.3f}  |  Weekly change: {trend_text}
Current Z-score: {s.get('latest_z_score', 0.0):.3f}  (>2.0 = statistically significant shift)

── GOLDSTEIN SCALE (conflict potential: -10=max conflict, +10=max cooperation) ──
{gs_text}

── MEDIA VOLUME (coverage intensity) ──
{vol_text}

── GKG SENTIMENT BREAKDOWN ──
{sent_text}

── EVENT CLASSIFICATION (CAMEO QuadClass) ──
{quad_text}

── TOP TREND INFLECTIONS (statistically significant signal shifts) ──
{inf_lines}

── RECENT TONE TIME SERIES (last 30 days) ──
{ts_lines}

── INTERPRETATION GUIDE ──
AvgTone < -3    : High conflict/negativity environment
AvgTone -3 to 0 : Mild conflict/tension
AvgTone 0 to +3 : Mild cooperation/positive coverage
AvgTone > +3    : Strong cooperation/positive environment
Goldstein < 0   : Net conflict-generating events dominate
Z-score > 2.0   : Signal is a statistically significant outlier vs 60-day baseline
Material Conflict (Q4) high % → physical/military actions occurring
Verbal Conflict (Q3) high %  → diplomatic tensions, threats, accusations
"""
        return summary

    # ------------------------------------------------------------------ #
    # Theme extraction                                                     #
    # ------------------------------------------------------------------ #

    def _extract_themes(self, df: Any) -> list[str]:
        themes = []
        if "quad_class" in df.columns:
            dominant = df["quad_class"].mode()
            if not dominant.empty:
                themes.append(f"Dominant: {QUAD_CLASS_LABELS.get(int(dominant.iloc[0]), 'Unknown')}")
        if "avg_tone" in df.columns:
            mean_tone = df["avg_tone"].mean()
            if mean_tone < -3:   themes.append("HIGH_NEGATIVITY")
            elif mean_tone < -1: themes.append("MODERATE_NEGATIVITY")
            elif mean_tone > 1:  themes.append("POSITIVE_COVERAGE")
            else:                themes.append("NEUTRAL_COVERAGE")
        if "goldstein_scale" in df.columns:
            gs = df["goldstein_scale"].mean()
            if gs < -5:   themes.append("INTENSE_CONFLICT")
            elif gs < 0:  themes.append("MILD_CONFLICT")
            elif gs > 3:  themes.append("STRONG_COOPERATION")
        return themes
