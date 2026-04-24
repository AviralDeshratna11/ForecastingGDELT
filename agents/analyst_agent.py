"""
agents/analyst_agent.py
------------------------
Analyst Agent: TSFM forecasting + LLM future outcome prediction.
Uses string Template (not .format()) to avoid conflicts with
curly braces in the glossary and JSON template.
"""

import json
import logging
import numpy as np
from string import Template
from typing import Any
from datetime import datetime, timedelta

from core.tsfm_forecaster import TSFMForecaster, ForecastResult
from agents.state import ForecastingState
from config.glossary import GDELT_GLOSSARY

logger = logging.getLogger(__name__)


# Uses $var instead of {var} — safe against curly braces in glossary/JSON
ANALYST_PROMPT = Template("""You are a senior geopolitical and financial forecaster.

$glossary

Today's date: $today
Forecast horizon: $horizon_end ($horizon days from today)

=== GDELT SIGNAL SUMMARY (past $lookback days) ===
$news_summary

=== TIME-SERIES FORECAST ===
Model         : $model_name
Trend         : $trend_direction (slope=$trend_slope, strength=$trend_strength/1.0)
Momentum      : $momentum
Forecast range: [$lower, $upper]
Uncertainty   : $uncertainty

=== YOUR TASK ===
Event: "$event_description"

IMPORTANT: Do NOT quote probability values. Your job is to give FUTURE PREDICTIONS.

Step 1 — PROBABILITY
What is the numeric probability this event occurs within $horizon days?
Do NOT say 50% unless signals are genuinely neutral.
- Trend UP + positive momentum = 60-85%
- Trend DOWN + negative momentum = 15-40%
- Stable with mixed signals = 40-60%
Justify with specific numbers from the GDELT summary above.

Step 2 — FUTURE CONSEQUENCES (what happens AFTER the event occurs)
Give 3 specific consequences. Each MUST have:
- A specific date (e.g. "By August 2025")
- A quantity (e.g. "oil prices rise 15-20%", "GDP drops 0.8%")
- Named actors (countries, institutions, companies)
Do NOT describe the event itself — describe what comes NEXT.

Step 3 — SCENARIOS (next $horizon days)
3 mutually exclusive futures. Probabilities must sum to 1.0.
Each must say what WILL happen, not what might.

Step 4 — KEY WATCHLIST
3 specific indicators to monitor. Each must be a concrete measurable signal.

Respond ONLY with this exact JSON structure, no text before or after:
{
  "probability": 0.65,
  "probability_rationale": "cite specific z-score, tone value, and trend slope",
  "consequences": [
    {
      "consequence": "By [Month Year], [Actor] will [specific action], causing [quantified impact] in [region]",
      "timeframe": "By [Month Year]",
      "severity": "high",
      "regions": ["Region1", "Region2"],
      "probability_if_event_occurs": 0.80
    },
    {
      "consequence": "By [Month Year], [Actor] will [specific action], causing [quantified impact]",
      "timeframe": "By [Month Year]",
      "severity": "medium",
      "regions": ["Region1"],
      "probability_if_event_occurs": 0.65
    },
    {
      "consequence": "By [Month Year], [Actor] will [specific action] in [region]",
      "timeframe": "By [Month Year]",
      "severity": "low",
      "regions": ["Global"],
      "probability_if_event_occurs": 0.50
    }
  ],
  "scenarios": [
    {
      "label": "Accelerated",
      "description": "What specifically happens if signals continue strengthening",
      "probability": 0.25,
      "key_trigger": "Specific event that causes this",
      "expected_date": "[Month Year]"
    },
    {
      "label": "Base Case",
      "description": "Most likely specific outcome given current trajectory",
      "probability": 0.50,
      "key_trigger": "Current trend continues",
      "expected_date": "[Month Year]"
    },
    {
      "label": "Reversal",
      "description": "What specifically happens if trend reverses",
      "probability": 0.25,
      "key_trigger": "Specific event that causes reversal",
      "expected_date": "[Month Year]"
    }
  ],
  "key_indicators_to_watch": [
    "Specific measurable indicator 1",
    "Specific measurable indicator 2",
    "Specific measurable indicator 3"
  ],
  "confidence": 0.70,
  "reasoning": "2-3 sentences explaining the probability. Must cite: trend direction, slope value, z-score, tone value, and momentum. Do NOT use hedging language like 'it is unclear' — make a definitive statement."
}""")


class AnalystAgent:
    def __init__(self, tsfm_forecaster: TSFMForecaster, config):
        self.forecaster = tsfm_forecaster
        self.config     = config
        self._llm       = self._init_llm()

    # ------------------------------------------------------------------ #
    # LangGraph node                                                       #
    # ------------------------------------------------------------------ #

    def run(self, state: ForecastingState) -> ForecastingState:
        query        = state.get("query", "")
        event_desc   = state.get("event_description", query)
        signals      = state.get("gdelt_signals")
        news_summary = state.get("news_summary", "")
        reflection   = state.get("updated_reasoning", "")
        iteration    = state.get("iteration", 0)
        lookback     = state.get("lookback_days", 90)

        logger.info("[AnalystAgent] Running analysis (iter %d)", iteration)

        if signals is None:
            return {**state, "error": "AnalystAgent: No signals from LibrarianAgent"}

        try:
            forecast = self.forecaster.forecast(
                signals.target_series,
                horizon=self.config.tsfm.prediction_length,
            )

            prob, reasoning, confidence, consequences, scenarios, indicators = \
                self._estimate(event_desc, news_summary, forecast, lookback, reflection)

            trend_analysis = self._build_trend_analysis(signals, forecast)

            logger.info("[AnalystAgent] model=%s  raw_prob=%.3f  consequences=%d  scenarios=%d",
                        forecast.model_name, prob, len(consequences), len(scenarios))

            return {
                **state,
                "tsfm_forecast":    forecast,
                "raw_probability":  prob,
                "reasoning":        reasoning,
                "confidence":       confidence,
                "consequences":     consequences,
                "scenarios":        scenarios,
                "key_indicators":   indicators,
                "trend_analysis":   trend_analysis,
                "error":            None,
            }

        except Exception as exc:
            logger.exception("[AnalystAgent] Failed: %s", exc)
            return {**state, "error": f"AnalystAgent: {exc}"}

    # ------------------------------------------------------------------ #
    # Estimation                                                           #
    # ------------------------------------------------------------------ #

    def _estimate(self, event_desc, news_summary, forecast, lookback, reflection=""):
        if self._llm is not None:
            return self._llm_estimate(event_desc, news_summary, forecast, lookback, reflection)
        return self._rule_based(forecast)

    def _llm_estimate(self, event_desc, news_summary, forecast, lookback, reflection=""):
        today       = datetime.utcnow()
        horizon_end = (today + timedelta(days=forecast.horizon)).strftime("%B %d, %Y")

        # Build prompt using Template — safe against curly braces in glossary
        prompt = ANALYST_PROMPT.substitute(
            glossary        = GDELT_GLOSSARY,
            today           = today.strftime("%B %d, %Y"),
            horizon_end     = horizon_end,
            horizon         = forecast.horizon,
            lookback        = lookback,
            news_summary    = news_summary,
            model_name      = forecast.model_name,
            trend_direction = forecast.trend_direction,
            trend_slope     = f"{forecast.trend_slope:+.4f}",
            trend_strength  = f"{forecast.trend_strength:.2f}",
            momentum        = f"{forecast.momentum:+.4f}",
            lower           = f"{float(forecast.lower_bound[-1]):.3f}",
            upper           = f"{float(forecast.upper_bound[-1]):.3f}",
            uncertainty     = f"{forecast.uncertainty:.3f}",
            event_description = event_desc,
        )

        if reflection:
            prompt += f"\n\nPREVIOUS CYCLE FEEDBACK:\n{reflection}\nIncorporate this feedback into your revised analysis."

        try:
            response = self._llm.invoke(prompt)
            text = response.content if hasattr(response, "content") else str(response)
            text = text.strip()

            # Strip markdown code fences e.g. ```json ... ```
            if "```" in text:
                # Split on ``` and find the chunk that contains JSON
                chunks = text.split("```")
                for chunk in chunks:
                    chunk = chunk.strip()
                    if chunk.lower().startswith("json"):
                        chunk = chunk[4:].strip()
                    if chunk.startswith("{"):
                        text = chunk
                        break

            # Extract JSON from first { to last } (handles any leading/trailing text)
            start = text.find("{")
            end   = text.rfind("}") + 1
            if start >= 0 and end > start:
                text = text[start:end]

            logger.info("[AnalystAgent] Parsing JSON length=%d", len(text))
            data = json.loads(text)

            prob         = float(np.clip(data.get("probability", 0.5), 0.05, 0.95))
            confidence   = float(np.clip(data.get("confidence", 0.6), 0.1, 0.95))
            rationale    = data.get("probability_rationale", "")
            reasoning    = data.get("reasoning", "")
            if rationale:
                reasoning = reasoning + "\n\nRationale: " + rationale
            consequences = data.get("consequences", [])
            scenarios    = data.get("scenarios", [])
            indicators   = data.get("key_indicators_to_watch", [])

            # Validate we got real content
            if not consequences:
                logger.warning("[AnalystAgent] LLM returned empty consequences")
            if not scenarios:
                logger.warning("[AnalystAgent] LLM returned empty scenarios")

            logger.info("[AnalystAgent] LLM prob=%.3f  consequences=%d  scenarios=%d  indicators=%d",
                        prob, len(consequences), len(scenarios), len(indicators))

            return prob, reasoning, confidence, consequences, scenarios, indicators

        except json.JSONDecodeError as exc:
            logger.warning("[AnalystAgent] JSON parse failed (%s) — raw text: %s", exc, text[:200])
            return self._rule_based(forecast)
        except Exception as exc:
            logger.warning("[AnalystAgent] LLM call failed (%s)", exc)
            return self._rule_based(forecast)

    def _rule_based(self, forecast):
        """Fallback when no LLM — derives probability from trend metrics."""
        raw = 0.5 + (forecast.trend_slope * 10 * forecast.trend_strength)
        raw += forecast.momentum * 2
        prob = float(np.clip(raw, 0.05, 0.95))

        direction_map = {
            "up":   "Upward trend suggests increasing likelihood",
            "down": "Downward trend suggests decreasing likelihood",
            "flat": "Flat trend — uncertain outcome",
        }
        reasoning = (
            f"{direction_map.get(forecast.trend_direction, '')}. "
            f"Slope={forecast.trend_slope:+.4f}, strength={forecast.trend_strength:.2f}, "
            f"momentum={forecast.momentum:+.4f}. "
            f"Set OPENROUTER_API_KEY in .env for full LLM consequence analysis."
        )
        consequences = [
            {"consequence": "LLM not configured — add OPENROUTER_API_KEY to .env for detailed consequence predictions",
             "timeframe": "N/A", "severity": "unknown", "regions": ["Global"],
             "probability_if_event_occurs": 0.5}
        ]
        scenarios = [
            {"label": "Accelerated",  "description": "Add OPENROUTER_API_KEY for scenario analysis",
             "probability": 0.30, "key_trigger": "N/A", "expected_date": "N/A"},
            {"label": "Base Case",    "description": "Add OPENROUTER_API_KEY for scenario analysis",
             "probability": 0.50, "key_trigger": "N/A", "expected_date": "N/A"},
            {"label": "Reversal",     "description": "Add OPENROUTER_API_KEY for scenario analysis",
             "probability": 0.20, "key_trigger": "N/A", "expected_date": "N/A"},
        ]
        return prob, reasoning, 0.4, consequences, scenarios, []

    # ------------------------------------------------------------------ #
    # Trend analysis                                                       #
    # ------------------------------------------------------------------ #

    def _build_trend_analysis(self, signals: Any, forecast: ForecastResult) -> dict:
        s = signals.summary
        return {
            "overall_trend":        s.get("trend_direction", "unknown"),
            "latest_z_score":       round(s.get("latest_z_score", 0.0), 3),
            "target_mean":          round(s.get("target_mean", 0.0), 3),
            "target_std":           round(s.get("target_std", 0.0), 3),
            "n_inflections":        s.get("n_inflections", 0),
            "n_significant":        len(s.get("significant_inflections", [])),
            "forecast_model":       forecast.model_name,
            "forecast_horizon":     forecast.horizon,
            "forecast_slope":       round(forecast.trend_slope, 5),
            "forecast_direction":   forecast.trend_direction,
            "forecast_strength":    round(forecast.trend_strength, 3),
            "forecast_momentum":    round(forecast.momentum, 4),
            "forecast_uncertainty": round(forecast.uncertainty, 4),
            "top_inflections": [
                {"date": str(i.date), "z_score": round(i.z_score, 3),
                 "direction": i.direction, "description": i.description}
                for i in sorted(signals.inflections, key=lambda x: -x.magnitude)[:5]
            ],
        }

    # ------------------------------------------------------------------ #
    # LLM init                                                             #
    # ------------------------------------------------------------------ #

    def _init_llm(self):
        try:
            provider = self.config.llm.provider
            key      = self.config.llm.api_key
            if provider == "openrouter" and key:
                from langchain_openai import ChatOpenAI
                logger.info("LLM: OpenRouter - model=%s", self.config.llm.model)
                return ChatOpenAI(
                    model=self.config.llm.model,
                    api_key=key,
                    base_url=self.config.llm.openrouter_base_url,
                    temperature=self.config.llm.temperature,
                    max_tokens=self.config.llm.max_tokens,
                    default_headers={
                        "HTTP-Referer": "https://github.com/gdelt-forecaster",
                        "X-Title": "GDELT Multi-Agent Forecaster",
                    },
                )
            elif provider == "anthropic" and key:
                from langchain_anthropic import ChatAnthropic
                logger.info("LLM: Anthropic - model=%s", self.config.llm.model)
                return ChatAnthropic(model=self.config.llm.model, api_key=key,
                                     temperature=self.config.llm.temperature,
                                     max_tokens=self.config.llm.max_tokens)
            elif provider == "openai" and key:
                from langchain_openai import ChatOpenAI
                logger.info("LLM: OpenAI - model=%s", self.config.llm.model)
                return ChatOpenAI(model=self.config.llm.model, api_key=key,
                                  temperature=self.config.llm.temperature,
                                  max_tokens=self.config.llm.max_tokens)
            else:
                logger.warning("No API key for '%s' — rule-based mode. Set OPENROUTER_API_KEY in .env", provider)
        except Exception as exc:
            logger.warning("LLM init failed (%s)", exc)
        return None
