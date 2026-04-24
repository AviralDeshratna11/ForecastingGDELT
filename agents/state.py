"""
agents/state.py
---------------
Defines the shared state TypedDict passed between all LangGraph agents.
Every agent reads from and writes to this state dict.
"""

from typing import TypedDict, Optional, Any
import pandas as pd
import numpy as np


class ForecastingState(TypedDict, total=False):
    """
    Shared state for the multi-agent forecasting graph.

    Lifecycle:
        query          → set by the caller
        gdelt_data     → populated by LibrarianAgent
        signals        → populated by LibrarianAgent
        inflections    → populated by LibrarianAgent
        news_summary   → populated by LibrarianAgent
        trend_analysis → populated by AnalystAgent
        forecast       → populated by AnalystAgent
        raw_probability→ populated by AnalystAgent
        calibrated_prob→ populated by CalibrationAgent
        kelly_fraction → populated by CalibrationAgent
        evaluation     → populated by EvaluationAgent
        final_output   → populated by OrchestratorAgent
        error          → set on failure
        iteration      → reflection loop counter
    """

    # Input
    query: str
    event_description: str
    lookback_days: int

    # Librarian output
    gdelt_raw: Any                    # pd.DataFrame
    gdelt_signals: Any                # ProcessedSignals
    inflections: list
    news_summary: str
    thematic_clusters: list[str]

    # Analyst output
    trend_analysis: dict
    tsfm_forecast: Any                # ForecastResult
    raw_probability: float
    reasoning: str
    confidence: float

    # Calibration output
    calibrated_probability: float
    extremized_probability: float
    calibration_report: dict
    kelly_fraction: float
    recommended_position: str        # "YES" | "NO" | "NO_TRADE"

    # Evaluation / Reflection output
    evaluation_score: float          # Brier-like score on held-out window
    evaluation_passed: bool
    reflection_notes: str
    updated_reasoning: str

    # System
    iteration: int
    max_iterations: int
    error: Optional[str]
    final_output: dict
