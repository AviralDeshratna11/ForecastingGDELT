"""
config/settings.py
------------------
Central configuration for the GDELT Forecasting System.
Override any value via environment variables or .env file.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv
from pathlib import Path

# Load .env from the project root (parent of config/) — works regardless of CWD
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / '.env')
# Also try current directory as fallback
load_dotenv()


@dataclass
class GDELTConfig:
    """GDELT data source configuration."""
    bigquery_project: Optional[str] = os.getenv("BQ_PROJECT", None)
    bigquery_dataset: str = "gdelt-bq.gdeltv2"
    gdelt_api_base: str = "http://api.gdeltproject.org/api/v2"
    gdelt_event_csv_base: str = "http://data.gdeltproject.org/gdeltv2"
    default_lookback_days: int = 90
    update_interval_minutes: int = 15
    z_score_threshold: float = 2.0
    rolling_window_days: int = 60
    short_window_days: int = 7


@dataclass
class LLMConfig:
    """LLM backend configuration."""
    provider: str = os.getenv("LLM_PROVIDER", "openrouter")   # openrouter | anthropic | openai | local
    model: str = os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4-5")

    # OpenRouter (primary)
    openrouter_api_key: Optional[str] = os.getenv("OPENROUTER_API_KEY", None)
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Anthropic direct (fallback)
    anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY", None)

    # OpenAI direct (fallback)
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY", None)

    temperature: float = 0.2
    max_tokens: int = 4096

    @property
    def api_key(self) -> Optional[str]:
        """Return whichever key matches the active provider."""
        if self.provider == "openrouter":
            return self.openrouter_api_key
        elif self.provider == "anthropic":
            return self.anthropic_api_key
        elif self.provider == "openai":
            return self.openai_api_key
        return None


@dataclass
class TSFMConfig:
    """Time Series Foundation Model configuration."""
    backend: str = os.getenv("TSFM_BACKEND", "fallback")
    context_length: int = 512
    prediction_length: int = 30
    num_samples: int = 20
    quantiles: list = field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9])


@dataclass
class CalibrationConfig:
    """Probability calibration settings."""
    method: str = "platt"
    extremize: bool = True
    extremize_alpha: float = 2.5
    kelly_fraction: float = 0.25


@dataclass
class AgentConfig:
    """Multi-agent orchestration settings."""
    max_reflection_cycles: int = 1   # 1 = single LLM call, no reflection loop
    reflection_threshold: float = 0.15
    enable_bayesian_update: bool = True
    langgraph_recursion_limit: int = 50


@dataclass
class SystemConfig:
    """Top-level system config."""
    gdelt: GDELTConfig = field(default_factory=GDELTConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tsfm: TSFMConfig = field(default_factory=TSFMConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)

    output_dir: str = "outputs"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    demo_mode: bool = os.getenv("DEMO_MODE", "false").lower() == "true"


# Singleton
CONFIG = SystemConfig()
