"""
agents/orchestrator.py
----------------------
The Orchestrator Agent builds and runs a LangGraph state machine
that coordinates the full multi-agent forecasting pipeline:

    START
      │
      ▼
  [LibrarianAgent]  ── fetch & process GDELT data
      │
      ▼
  [AnalystAgent]    ── TSFM forecast + probability estimation
      │
      ▼
  [CalibrationAgent] ── calibrate + extremize + Kelly sizing
      │
      ▼
  [EvaluationAgent] ── reflection loop (Brier score check)
      │
    ┌─┴──────────────────────────────┐
    │  evaluation_passed?            │
    │  YES → [FinaliseNode] → END    │
    │  NO  → [AnalystAgent]  (loop)  │
    └────────────────────────────────┘

Falls back to sequential execution if LangGraph is not installed.
"""

import json
import logging
from datetime import datetime
from typing import Optional

from agents.state import ForecastingState
from agents.librarian_agent import LibrarianAgent
from agents.analyst_agent import AnalystAgent
from agents.calibration_agent import CalibrationAgent
from agents.evaluation_agent import EvaluationAgent

logger = logging.getLogger(__name__)


class ForecastingOrchestrator:
    """
    Multi-agent orchestrator for event outcome forecasting.

    Usage:
        orchestrator = ForecastingOrchestrator.build(config)
        result = orchestrator.run("Will X happen within 30 days?", lookback_days=90)
    """

    def __init__(
        self,
        librarian: LibrarianAgent,
        analyst: AnalystAgent,
        calibration: CalibrationAgent,
        evaluation: EvaluationAgent,
        config,
    ):
        self.librarian  = librarian
        self.analyst    = analyst
        self.calibration= calibration
        self.evaluation = evaluation
        self.config     = config
        self._graph     = self._build_graph()

    # ------------------------------------------------------------------ #
    # Factory                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def build(cls, config) -> "ForecastingOrchestrator":
        """Construct and wire all agents from config."""
        from core.gdelt_client import GDELTClient
        from core.signal_processor import SignalProcessor
        from core.tsfm_forecaster import TSFMForecaster
        from core.calibrator import ProbabilityCalibrator
        from utils.brier_score import BrierScoreCalculator
        from utils.kelly_criterion import KellyCriterion

        gdelt_client  = GDELTClient(config.gdelt)
        processor     = SignalProcessor(config.gdelt)
        forecaster    = TSFMForecaster(config.tsfm)
        calibrator    = ProbabilityCalibrator(config.calibration)
        brier         = BrierScoreCalculator()
        kelly         = KellyCriterion()

        return cls(
            librarian   = LibrarianAgent(gdelt_client, processor),
            analyst     = AnalystAgent(forecaster, config),
            calibration = CalibrationAgent(calibrator, kelly, config),
            evaluation  = EvaluationAgent(brier, config),
            config      = config,
        )

    # ------------------------------------------------------------------ #
    # Public run                                                           #
    # ------------------------------------------------------------------ #

    def run(
        self,
        event_query: str,
        event_description: Optional[str] = None,
        lookback_days: int = 90,
        market_price: float = 0.5,
    ) -> dict:
        """
        Execute the full forecasting pipeline.

        Args:
            event_query:       Short query string for GDELT (e.g. "US-China trade")
            event_description: Full question for the LLM ("Will X happen by Y?")
            lookback_days:     GDELT history window
            market_price:      Current market implied probability (for Kelly calc)

        Returns:
            Final state dict with all outputs
        """
        initial_state: ForecastingState = {
            "query":             event_query,
            "event_description": event_description or event_query,
            "lookback_days":     lookback_days,
            "market_price":      market_price,
            "iteration":         0,
            "max_iterations":    self.config.agents.max_reflection_cycles,
        }

        logger.info("[Orchestrator] Starting pipeline for: %s", event_query)
        start_time = datetime.utcnow()

        if self._graph is not None:
            final_state = self._run_langgraph(initial_state)
        else:
            final_state = self._run_sequential(initial_state)

        elapsed = (datetime.utcnow() - start_time).total_seconds()
        final_state = self._finalise(final_state, elapsed)

        logger.info("[Orchestrator] Completed in %.2fs", elapsed)
        return final_state

    # ------------------------------------------------------------------ #
    # LangGraph execution                                                  #
    # ------------------------------------------------------------------ #

    def _build_graph(self):
        try:
            from langgraph.graph import StateGraph, END

            graph = StateGraph(ForecastingState)

            # Register nodes
            graph.add_node("librarian",   self.librarian.run)
            graph.add_node("analyst",     self.analyst.run)
            graph.add_node("calibration", self.calibration.run)
            graph.add_node("evaluation",  self.evaluation.run)
            graph.add_node("finalise",    self._finalise_node)

            # Edges
            graph.set_entry_point("librarian")
            graph.add_edge("librarian",   "analyst")
            graph.add_edge("analyst",     "calibration")
            graph.add_edge("calibration", "evaluation")

            # Conditional: reflection loop or finalise
            graph.add_conditional_edges(
                "evaluation",
                self._should_continue,
                {
                    "continue": "analyst",
                    "done":     "finalise",
                }
            )
            graph.add_edge("finalise", END)

            compiled = graph.compile()
            logger.info("[Orchestrator] LangGraph compiled successfully")
            return compiled

        except ImportError:
            logger.warning("LangGraph not installed – using sequential execution")
            return None
        except Exception as exc:
            logger.warning("LangGraph build failed (%s) – using sequential execution", exc)
            return None

    def _should_continue(self, state: ForecastingState) -> str:
        if state.get("evaluation_passed", True):
            return "done"
        if state.get("error"):
            return "done"
        return "continue"

    def _run_langgraph(self, initial_state: ForecastingState) -> ForecastingState:
        # NOTE: LangGraph's AddableValuesDict MERGES list fields (consequences,
        # scenarios) across iterations instead of replacing them, causing duplicates
        # or loss. We bypass LangGraph and always use sequential mode which gives
        # us full control over state replacement with **state spreading.
        logger.info("[Orchestrator] Using sequential mode (LangGraph list-merge bypass)")
        return self._run_sequential(initial_state)

    # ------------------------------------------------------------------ #
    # Sequential fallback execution                                        #
    # ------------------------------------------------------------------ #

    def _run_sequential(self, state: ForecastingState) -> ForecastingState:
        """Run agents in sequence without LangGraph."""
        logger.info("[Orchestrator] Running in sequential mode")

        # Step 1: Librarian
        state = self.librarian.run(state)
        if state.get("error"):
            return state

        # Step 2+: Analyst → Calibration → Evaluation loop
        max_iters = state.get("max_iterations", self.config.agents.max_reflection_cycles)

        for i in range(max_iters):
            state = self.analyst.run(state)
            if state.get("error"):
                break
            state = self.calibration.run(state)
            if state.get("error"):
                break
            state = self.evaluation.run(state)
            if state.get("evaluation_passed", True) or state.get("error"):
                break

        return state

    # ------------------------------------------------------------------ #
    # Finalisation                                                         #
    # ------------------------------------------------------------------ #

    def _finalise_node(self, state: ForecastingState) -> ForecastingState:
        return self._finalise(state, elapsed_seconds=None)

    def _finalise(self, state: ForecastingState, elapsed_seconds=None) -> ForecastingState:
        """Package the final output for the caller."""
        prob = state.get("extremized_probability",
               state.get("calibrated_probability",
               state.get("raw_probability", 0.5)))

        final_output = {
            "query":               state.get("query", ""),
            "event_description":   state.get("event_description", ""),
            "final_probability":   round(float(prob), 4),
            "raw_probability":     round(float(state.get("raw_probability", 0.5)), 4),
            "calibrated_probability": round(float(state.get("calibrated_probability", prob)), 4),
            "extremized_probability": round(float(state.get("extremized_probability", prob)), 4),
            "recommendation":      state.get("recommended_position", "NO_TRADE"),
            "kelly_fraction":      round(float(state.get("kelly_fraction", 0.0)), 4),
            "reasoning":           state.get("reasoning", ""),
            "reflection_notes":    state.get("reflection_notes", ""),
            "evaluation_score":    round(float(state.get("evaluation_score", 0.0)), 4),
            "iterations":          state.get("iteration", 1),
            "trend_analysis":      state.get("trend_analysis", {}),
            "calibration_report":  state.get("calibration_report", {}),
            "thematic_clusters":   state.get("thematic_clusters", []),
            "n_inflections":       len(state.get("inflections", [])),
            "consequences":        state.get("consequences", []),
            "key_indicators":      state.get("key_indicators", []),
            "scenarios":           state.get("scenarios", []),
            "elapsed_seconds":     round(elapsed_seconds, 2) if elapsed_seconds else None,
            "timestamp":           datetime.utcnow().isoformat() + "Z",
            "error":               state.get("error"),
        }

        return {**state, "final_output": final_output, "consequences": state.get("consequences", []), "scenarios": state.get("scenarios", []), "key_indicators": state.get("key_indicators", [])}
