"""
demo.py
-------
Standalone demo — no API keys, no BigQuery, no TSFM installation required.
Uses synthetic GDELT-like data and ARIMA fallback forecasting.

Run:
    python demo.py
"""

import sys
import json
import logging
from pathlib import Path

# ── project root on path ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.WARNING,   # suppress info noise in demo
    format="%(levelname)-8s  %(name)s — %(message)s",
)

from config.settings import CONFIG
from agents.orchestrator import ForecastingOrchestrator
from utils.visualizer import ForecastVisualizer
from utils.brier_score import BrierScoreCalculator
from utils.kelly_criterion import KellyCriterion

import numpy as np


SEPARATOR = "=" * 65


def print_section(title: str):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def demo_full_pipeline():
    """Run a complete multi-agent forecasting cycle with synthetic data."""
    print_section("GDELT Multi-Agent Forecasting System — DEMO")
    print("\nThis demo uses synthetic GDELT-like data (no API keys required).")
    print("Install optional dependencies for full functionality.")

    DEMO_EVENTS = [
        {
            "query":       "US China trade tariff negotiations",
            "description": "Will the US and China reach a new trade agreement within 90 days?",
            "market_price": 0.38,
            "lookback_days": 90,
        },
        {
            "query":       "central bank interest rate hike",
            "description": "Will the central bank raise interest rates at the next meeting?",
            "market_price": 0.62,
            "lookback_days": 60,
        },
        {
            "query":       "election protest unrest stability",
            "description": "Will the current political unrest escalate to a major crisis?",
            "market_price": 0.25,
            "lookback_days": 120,
        },
    ]

    print_section("Building Orchestrator")
    orchestrator = ForecastingOrchestrator.build(CONFIG)
    print("✓ All agents initialised")
    print("  • LibrarianAgent  (GDELT retrieval + signal processing)")
    print("  • AnalystAgent    (TSFM forecasting + probability estimation)")
    print("  • CalibrationAgent (Platt scaling + α-extremization + Kelly)")
    print("  • EvaluationAgent  (Reflection loop + Bayesian update)")
    print("  • Orchestrator     (LangGraph / sequential coordination)")

    viz = ForecastVisualizer(output_dir=CONFIG.output_dir)
    results = []

    for i, event in enumerate(DEMO_EVENTS, 1):
        print_section(f"Event {i}/{len(DEMO_EVENTS)}: {event['query'][:50]}")
        print(f"Question : {event['description']}")
        print(f"Market   : {event['market_price']:.0%} (current implied probability)")
        print(f"Lookback : {event['lookback_days']} days")

        result = orchestrator.run(
            event_query=event["query"],
            event_description=event["description"],
            lookback_days=event["lookback_days"],
            market_price=event["market_price"],
        )

        final = result.get("final_output", {})
        results.append(final)

        # Print result
        prob  = final.get("final_probability", 0.0)
        rec   = final.get("recommended_position", final.get("recommendation", "NO_TRADE"))
        kelly = final.get("kelly_fraction", 0.0)
        brier = final.get("evaluation_score", 0.0)
        iters = final.get("iterations", 1)
        trend = final.get("trend_analysis", {}).get("overall_trend", "—")
        n_inf = final.get("n_inflections", 0)

        print(f"\n  ┌─ RESULT ──────────────────────────────────────────────┐")
        print(f"  │  Raw probability      : {final.get('raw_probability', 0):.4f}")
        print(f"  │  Calibrated           : {final.get('calibrated_probability', 0):.4f}")
        print(f"  │  Extremized (final)   : {prob:.4f}  ({prob:.1%})")
        print(f"  │  Market price         : {event['market_price']:.4f}")
        edge = prob - event['market_price']
        print(f"  │  Edge                 : {edge:+.4f}  ({'favourable' if abs(edge) > 0.03 else 'minimal'})")
        print(f"  │  Recommendation       : {rec}")
        print(f"  │  Kelly (quarter)      : {kelly:.4f}  ({kelly:.1%} of bankroll)")
        print(f"  │  Brier score          : {brier:.4f}")
        print(f"  │  Trend direction      : {trend}")
        print(f"  │  GDELT inflections    : {n_inf}")
        print(f"  │  Reflection cycles    : {iters}")
        print(f"  └──────────────────────────────────────────────────────────┘")

        # Generate charts
        signals  = result.get("gdelt_signals")
        forecast = result.get("tsfm_forecast")
        if signals and forecast:
            paths = viz.plot_all(signals, forecast, final, event["query"])
            print(f"\n  Charts saved:")
            for p in paths:
                print(f"    → {p}")

    # ── Brier Score Decomposition Demo ─────────────────────────────────────
    print_section("Brier Score Decomposition Demo")
    brier_calc = BrierScoreCalculator()

    np.random.seed(42)
    n_test = 200
    true_probs  = np.random.beta(2, 2, n_test)
    outcomes    = (np.random.rand(n_test) < true_probs).astype(float)

    # Simulate miscalibrated model (compressed probabilities)
    model_probs = 0.5 + 0.4 * (true_probs - 0.5)  # hedge toward 0.5

    decomp = brier_calc.decompose(model_probs, outcomes)
    print(f"\n  Sample forecasting system (n={n_test}):")
    print(f"  Brier Score   : {decomp.brier_score:.5f}  (0=perfect, 0.25=random)")
    print(f"  Reliability   : {decomp.reliability:.5f}  (lower = better calibrated)")
    print(f"  Resolution    : {decomp.resolution:.5f}  (higher = more informative)")
    print(f"  Uncertainty   : {decomp.uncertainty:.5f}  (base-rate entropy, fixed)")
    print(f"  Skill Score   : {decomp.skill_score:.4f}   (>0 = beats climatology)")
    print(f"  Base Rate     : {decomp.base_rate:.4f}")
    print(f"\n  ✓  BS = REL - RES + UNC: {decomp.reliability:.5f} - {decomp.resolution:.5f} + {decomp.uncertainty:.5f} ≈ {decomp.reliability - decomp.resolution + decomp.uncertainty:.5f}")

    # ── Kelly Criterion Demo ───────────────────────────────────────────────
    print_section("Kelly Criterion Demo")
    kelly_calc = KellyCriterion()

    scenarios = [
        (0.65, 0.50, "Strong YES edge"),
        (0.35, 0.50, "Strong NO edge"),
        (0.52, 0.50, "Marginal edge"),
        (0.50, 0.50, "No edge (fair coin)"),
        (0.80, 0.60, "High conviction vs expensive market"),
    ]

    print(f"\n  {'True P':>8}  {'Market':>8}  {'Edge':>8}  {'Full Kelly':>11}  {'Qtr-Kelly':>10}  {'Description'}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*11}  {'-'*10}  {'-'*30}")
    for tp, mp, desc in scenarios:
        full_k = kelly_calc.compute(tp, mp)
        frac_k = full_k * 0.25
        edge = tp - mp
        print(f"  {tp:>8.3f}  {mp:>8.3f}  {edge:>+8.3f}  {full_k:>+11.4f}  {frac_k:>+10.4f}  {desc}")

    # ── Summary ────────────────────────────────────────────────────────────
    print_section("Multi-Event Summary")
    print(f"\n  {'Event Query':45}  {'Prob':>7}  {'Rec':>10}  {'Kelly':>7}")
    print(f"  {'-'*45}  {'-'*7}  {'-'*10}  {'-'*7}")
    for event, final in zip(DEMO_EVENTS, results):
        q    = event["query"][:43]
        prob = final.get("final_probability", 0.0)
        rec  = final.get("recommended_position", final.get("recommendation", "—"))
        kelly = final.get("kelly_fraction", 0.0)
        print(f"  {q:45}  {prob:>7.4f}  {rec:>10}  {kelly:>7.4f}")

    print(f"\n✓  Demo complete. Charts and JSON outputs saved to: {CONFIG.output_dir}/")
    print(f"\nNext steps:")
    print(f"  1. Set ANTHROPIC_API_KEY for LLM-powered CoT reasoning")
    print(f"  2. Set BQ_PROJECT for live GDELT BigQuery access")
    print(f"  3. pip install timesfm  →  for Google's TSFM backend")
    print(f"  4. python main.py --event 'your event' --days 90")
    print()


if __name__ == "__main__":
    demo_full_pipeline()
