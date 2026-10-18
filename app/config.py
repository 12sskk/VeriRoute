"""Centralized configuration for the VeriRoute LLM Gateway."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


@dataclass
class ModelArmSpec:
    """Specification for an upstream LLM candidate arm in the bandit router."""

    arm_id: str
    display_name: str
    provider: str
    tier: str  # "fast", "balanced", "frontier"
    cost_per_1k_input: float  # USD
    cost_per_1k_output: float  # USD
    base_ttft_ms: float  # Baseline Time-To-First-Token (ms)
    capability_profile: dict[str, float] = field(default_factory=dict)


DEFAULT_MODEL_ARMS: list[ModelArmSpec] = [
    ModelArmSpec(
        arm_id="llama-3.1-8b-instant",
        display_name="Llama-3.1-8B-Instant (Groq/Local)",
        provider="groq",
        tier="fast",
        cost_per_1k_input=0.00005,
        cost_per_1k_output=0.00008,
        base_ttft_ms=42.0,
        capability_profile={
            "simple_qa": 0.94,
            "summarization": 0.91,
            "code_simple": 0.84,
            "complex_reasoning": 0.42,
            "system_design": 0.48,
            "math_proof": 0.38,
        },
    ),
    ModelArmSpec(
        arm_id="gpt-4o-mini",
        display_name="GPT-4o-Mini (Balanced)",
        provider="openai",
        tier="balanced",
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.00060,
        base_ttft_ms=115.0,
        capability_profile={
            "simple_qa": 0.97,
            "summarization": 0.95,
            "code_simple": 0.92,
            "complex_reasoning": 0.74,
            "system_design": 0.78,
            "math_proof": 0.71,
        },
    ),
    ModelArmSpec(
        arm_id="claude-3-5-sonnet",
        display_name="Claude-3.5-Sonnet / GPT-4o (Frontier)",
        provider="frontier",
        tier="frontier",
        cost_per_1k_input=0.00300,
        cost_per_1k_output=0.01500,
        base_ttft_ms=260.0,
        capability_profile={
            "simple_qa": 0.99,
            "summarization": 0.98,
            "code_simple": 0.98,
            "complex_reasoning": 0.97,
            "system_design": 0.96,
            "math_proof": 0.95,
        },
    ),
]


@dataclass
class GatewayConfig:
    """Runtime settings loaded from environment variables with sensible defaults."""

    provider_mode: str = os.getenv("VERIROUTE_PROVIDER_MODE", "simulator")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

    # Two-stage Semantic Cache thresholds
    cache_cosine_threshold: float = float(os.getenv("CACHE_COSINE_THRESHOLD", "0.82"))
    cache_entailment_threshold: float = float(os.getenv("CACHE_ENTAILMENT_THRESHOLD", "0.70"))
    cache_contradiction_ceiling: float = float(os.getenv("CACHE_CONTRADICTION_CEILING", "0.25"))
    cache_max_entries: int = int(os.getenv("CACHE_MAX_ENTRIES", "2000"))

    # Contextual Bandit (LinUCB) settings
    bandit_alpha: float = float(os.getenv("BANDIT_ALPHA", "0.85"))
    bandit_latency_penalty: float = float(os.getenv("BANDIT_LATENCY_PENALTY", "0.15"))
    shadow_sample_rate: float = float(os.getenv("SHADOW_SAMPLE_RATE", "0.15"))
    sqlite_db_path: str = str(DATA_DIR / "veriroute_state.db")

    # Rate Limiting
    rate_limit_capacity: float = float(os.getenv("RATE_LIMIT_CAPACITY", "60.0"))
    rate_limit_refill_per_sec: float = float(os.getenv("RATE_LIMIT_REFILL_PER_SEC", "2.0"))

    # Agent Loop Circuit Breaker
    loop_window_size: int = 6
    loop_similarity_threshold: float = 0.88
    loop_repeat_trigger: int = 3

    onnx_nli_model_path: str = os.getenv("ONNX_NLI_MODEL_PATH", "")


settings = GatewayConfig()
