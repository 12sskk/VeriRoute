"""Pydantic data models for OpenAI-compatible API and VeriRoute diagnostics."""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


class FunctionCallSpec(BaseModel):
    name: str
    arguments: str


class ToolCallSpec(BaseModel):
    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:10]}")
    type: Literal["function"] = "function"
    function: FunctionCallSpec


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = ""
    name: Optional[str] = None
    tool_calls: Optional[list[ToolCallSpec]] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = Field(
        default="auto",
        description="Use 'auto' to let the LinUCB Bandit Router select the optimal model, or specify a model ID.",
    )
    messages: list[ChatMessage]
    temperature: float = 0.2
    max_tokens: int = 512
    stream: bool = False
    session_id: Optional[str] = Field(
        default=None,
        description="Optional conversation/agent trace ID for stateful loop detection.",
    )
    bypass_cache: bool = False
    force_vanilla_cosine_cache: bool = Field(
        default=False,
        description="For benchmarking/demo: if True, skips Stage-2 NLI verification and uses raw cosine similarity.",
    )


class Stage1ConstraintReport(BaseModel):
    numbers_query: list[str] = Field(default_factory=list)
    numbers_candidate: list[str] = Field(default_factory=list)
    numbers_match: bool = True
    entities_query: list[str] = Field(default_factory=list)
    entities_candidate: list[str] = Field(default_factory=list)
    entities_match: bool = True
    negation_polarity_match: bool = True
    antonym_conflict: Optional[tuple[str, str]] = None
    passed_hard_constraints: bool = True
    rejection_reason: Optional[str] = None


class NLIVerificationResult(BaseModel):
    entailment_prob: float
    neutral_prob: float
    contradiction_prob: float
    label: Literal["ENTAILMENT", "NEUTRAL", "CONTRADICTION"]
    inference_time_ms: float
    engine_type: str
    constraint_report: Stage1ConstraintReport


class CacheDecisionMetadata(BaseModel):
    cache_hit: bool
    vanilla_cosine_would_hit: bool
    false_positive_prevented: bool = False
    top_cosine_similarity: float = 0.0
    matched_query: Optional[str] = None
    matched_entry_id: Optional[str] = None
    stage1_latency_ms: float = 0.0
    stage2_nli: Optional[NLIVerificationResult] = None
    decision_reason: str = "no_candidates"


class BanditArmScore(BaseModel):
    arm_id: str
    display_name: str
    tier: str
    expected_reward: float
    uncertainty_bonus: float
    latency_penalty: float
    ucb_score: float
    ema_ttft_ms: float
    estimated_cost_usd: float


class RouterDecisionMetadata(BaseModel):
    selected_model: str
    selected_tier: str
    routing_policy: Literal["linucb_bandit", "explicit_override", "loop_breaker_escalation", "cache_hit"]
    complexity_score: float
    detected_domain: str
    context_features: dict[str, float]
    arm_scores: list[BanditArmScore] = Field(default_factory=list)
    shadow_candidate_triggered: Optional[str] = None
    router_latency_ms: float = 0.0


class LoopCircuitBreakerMetadata(BaseModel):
    loop_detected: bool = False
    session_id: Optional[str] = None
    repeated_action_signature: Optional[str] = None
    repeat_count: int = 0
    action_taken: Literal["none", "injected_system_directive_and_escalated"] = "none"


class VeriRouteTelemetry(BaseModel):
    request_id: str = Field(default_factory=lambda: f"req_{uuid.uuid4().hex[:12]}")
    gateway_overhead_ms: float
    total_latency_ms: float
    cost_usd: float
    baseline_frontier_cost_usd: float
    cost_saved_usd: float
    cache: CacheDecisionMetadata
    router: RouterDecisionMetadata
    loop_guard: LoopCircuitBreakerMetadata


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Literal["stop", "length", "tool_calls"] = "stop"


class UsageStats(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageStats
    veriroute: VeriRouteTelemetry


class FeedbackRequest(BaseModel):
    request_id: Optional[str] = None
    arm_id: str
    prompt: str
    quality_score: float = Field(ge=0.0, le=1.0, description="Observed quality reward in [0, 1]")
    latency_ms: Optional[float] = None
    cost_usd: Optional[float] = None
    signal_source: Literal["explicit_user", "json_schema_validator", "shadow_judge", "implicit_retry"] = "explicit_user"


class CacheCompareRequest(BaseModel):
    seed_query: str
    seed_response: str = "Cached reference response for seed query."
    test_query: str
