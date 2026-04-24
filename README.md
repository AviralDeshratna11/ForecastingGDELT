# GDELT Multi-Agent Predictive Forecasting System

A real-time event outcome prediction system integrating GDELT global news data with Time Series Foundation Models and a multi-agent orchestration layer.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR AGENT                           │
│              (Coordinates all agents via LangGraph)             │
└────────┬──────────┬────────────────┬───────────────┬───────────┘
         │          │                │               │
    ┌────▼────┐ ┌───▼────┐   ┌──────▼─────┐  ┌──────▼──────┐
    │LIBRARIAN│ │ANALYST │   │ EVALUATION │  │ CALIBRATION │
    │  AGENT  │ │ AGENT  │   │   AGENT    │  │    AGENT    │
    │(GDELT   │ │(Trend  │   │(Reflection │  │(Brier/Platt │
    │Retrieval│ │Analysis│   │  Loop)     │  │  Scaling)   │
    └────┬────┘ └───┬────┘   └──────┬─────┘  └──────┬──────┘
         │          │               │               │
    ┌────▼──────────▼───────────────▼───────────────▼────────┐
    │              SHARED STATE (LangGraph)                    │
    │     GDELT signals + TSFM predictions + probabilities    │
    └──────────────────────────────────────────────────────────┘
```

## Key Components

| Component | Description |
|---|---|
| `core/gdelt_client.py` | Fetches GDELT Event DB + GKG data via BigQuery / HTTP |
| `core/signal_processor.py` | Z-score normalization, rolling windows, trend inflection detection |
| `core/tsfm_forecaster.py` | TimesFM / Chronos wrapper for zero-shot time-series predictions |
| `core/calibrator.py` | Platt scaling, isotonic regression, temperature scaling, extremization |
| `agents/librarian_agent.py` | News retrieval & GDELT summarization agent |
| `agents/analyst_agent.py` | Trend analysis, CAMEO scoring, inflection detection |
| `agents/evaluation_agent.py` | Reflection loop – evaluates predictions vs. ground truth |
| `agents/calibration_agent.py` | Probability calibration & Kelly criterion sizing |
| `agents/orchestrator.py` | LangGraph state-machine that coordinates all agents |
| `utils/brier_score.py` | Brier score + 3-component decomposition |
| `utils/kelly_criterion.py` | Fractional Kelly bet sizing |
| `main.py` | Entry point – run a full prediction cycle |
| `demo.py` | Demo mode (no API keys required) |

## Setup

```bash
pip install -r requirements.txt
```

### Optional: Google BigQuery (for live GDELT)
```bash
pip install google-cloud-bigquery
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service_account.json"
```

### Optional: TimesFM
```bash
pip install timesfm
```

## Quick Start

```bash
# Demo mode (uses synthetic data, no API keys)
python demo.py

# Full prediction cycle with your event query
python main.py --event "US election results" --days 90

# Specify output format
python main.py --event "oil price spike" --days 60 --output-format json
```

## Configuration

Edit `config/settings.py` to configure:
- BigQuery project & dataset
- LLM model (Claude / GPT-4 / local)
- TSFM backend (TimesFM / Chronos / fallback ARIMA)
- Calibration method
- Kelly fraction
