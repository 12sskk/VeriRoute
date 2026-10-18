"""FastAPI Application Entrypoint for VeriRoute LLM Gateway & Bandit Router."""

from __future__ import annotations

import asyncio
from pathlib import Path
import time
import uuid

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse

from app.cache.semantic_cache import EntailmentSemanticCache
from app.config import DEFAULT_MODEL_ARMS, settings
from app.eval.shadow import ShadowEvaluator
from app.guardrails.loop_detector import AgentLoopCircuitBreaker
from app.infra.metrics import GatewayMetricsCollector
from app.infra.rate_limiter import TokenBucketRateLimiter
from app.models import (
    CacheCompareRequest,
    CacheDecisionMetadata,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    FeedbackRequest,
    RouterDecisionMetadata,
    UsageStats,
    VeriRouteTelemetry,
)
from app.providers.engine import UpstreamProviderEngine
from app.router.bandit import LinUCBBanditRouter
from app.router.features import extract_prompt_context
from benchmarks.dataset import BANDIT_WORKLOAD_PROMPTS
from benchmarks.run_benchmarks import run_all_benchmarks

app = FastAPI(
    title="VeriRoute: Entailment-Verified Semantic LLM Gateway & LinUCB Bandit Router",
    version="1.0.0",
    description=(
        "High-performance AI Gateway featuring Two-Stage NLI Entailment Semantic Caching, "
        "Online Disjoint LinUCB Contextual Bandit Routing, Shadow-Mode Auto-Evaluation, "
        "and Stateful Agent Loop Circuit Breaking."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Core Engine Components
semantic_cache = EntailmentSemanticCache(
    cosine_threshold=settings.cache_cosine_threshold,
    entailment_threshold=settings.cache_entailment_threshold,
    contradiction_ceiling=settings.cache_contradiction_ceiling,
    max_entries=settings.cache_max_entries,
    onnx_nli_model_path=settings.onnx_nli_model_path or None,
)

bandit_router = LinUCBBanditRouter(
    arms=DEFAULT_MODEL_ARMS,
    alpha=settings.bandit_alpha,
    latency_penalty_weight=settings.bandit_latency_penalty,
    db_path=settings.sqlite_db_path,
    warm_start=True,
)

provider_engine = UpstreamProviderEngine(arms=DEFAULT_MODEL_ARMS)
shadow_evaluator = ShadowEvaluator(
    provider_engine=provider_engine,
    bandit_router=bandit_router,
    sample_rate=settings.shadow_sample_rate,
)
loop_breaker = AgentLoopCircuitBreaker(
    repeat_trigger=settings.loop_repeat_trigger,
    similarity_threshold=settings.loop_similarity_threshold,
)
rate_limiter = TokenBucketRateLimiter(
    capacity=settings.rate_limit_capacity,
    refill_per_sec=settings.rate_limit_refill_per_sec,
)
metrics_collector = GatewayMetricsCollector()

STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/", include_in_schema=False)
async def serve_dashboard():
    """Serve the interactive single-page Interview Demo Dashboard."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health_check() -> dict:
    return {
        "status": "healthy",
        "service": "VeriRoute LLM Gateway",
        "provider_mode": settings.provider_mode,
        "cache_entries": semantic_cache.size(),
        "total_requests": metrics_collector.total_requests,
    }


@app.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics() -> str:
    """Prometheus scrape endpoint."""
    return metrics_collector.format_prometheus()


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    x_tenant_id: str = Header(default="default"),
):
    """OpenAI-compatible `/v1/chat/completions` endpoint with NLI Cache, LinUCB Router, and Loop Guard."""
    t_gateway_start = time.perf_counter()

    # 1. Per-Tenant Token Bucket Rate Limiting
    allowed, remaining_tokens, retry_after = rate_limiter.check_and_consume(x_tenant_id)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded for tenant",
                "tenant_id": x_tenant_id,
                "remaining_tokens": remaining_tokens,
                "retry_after_sec": retry_after,
            },
        )

    if not request.messages:
        raise HTTPException(status_code=400, detail="messages array cannot be empty")

    latest_user_prompt = next(
        (m.content or "" for m in reversed(request.messages) if m.role == "user"),
        request.messages[-1].content or "",
    ).strip()

    # 2. Stateful Agent Loop Circuit Breaker
    processed_messages, loop_meta = loop_breaker.inspect_and_intervene(
        request.messages,
        session_id=request.session_id,
    )

    # 3. Two-Stage Entailment-Verified Semantic Cache Lookup
    # Only cache/lookup single-turn or non-looping user queries
    if not request.bypass_cache and not loop_meta.loop_detected and latest_user_prompt:
        cached_entry, cache_meta = semantic_cache.lookup(
            query=latest_user_prompt,
            force_vanilla_cosine=request.force_vanilla_cosine_cache,
        )
    else:
        cached_entry = None
        cache_meta = CacheDecisionMetadata(
            cache_hit=False,
            vanilla_cosine_would_hit=False,
            decision_reason="cache_bypassed" if request.bypass_cache else "agent_loop_escalation",
        )

    # 4A. Handle Verified Cache Hit (0 LLM token cost!)
    if cached_entry is not None:
        ctx = extract_prompt_context(processed_messages)
        gateway_overhead_ms = (time.perf_counter() - t_gateway_start) * 1000.0
        comp_tokens = max(20, int(len(cached_entry.response_text.split()) * 1.3))
        frontier_spec = provider_engine.frontier_spec
        baseline_cost = (
            (ctx.estimated_prompt_tokens / 1000.0) * frontier_spec.cost_per_1k_input
            + (comp_tokens / 1000.0) * frontier_spec.cost_per_1k_output
        )

        router_meta = RouterDecisionMetadata(
            selected_model=cached_entry.model_used,
            selected_tier="cache",
            routing_policy="cache_hit",
            complexity_score=ctx.complexity_score,
            detected_domain=ctx.detected_domain,
            context_features=ctx.feature_dict,
            arm_scores=[],
            router_latency_ms=0.0,
        )
        telemetry = VeriRouteTelemetry(
            gateway_overhead_ms=round(gateway_overhead_ms, 2),
            total_latency_ms=round(gateway_overhead_ms, 2),
            cost_usd=0.0,
            baseline_frontier_cost_usd=round(baseline_cost, 6),
            cost_saved_usd=round(baseline_cost, 6),
            cache=cache_meta,
            router=router_meta,
            loop_guard=loop_meta,
        )
        metrics_collector.record(latest_user_prompt, telemetry)

        if request.stream:
            return StreamingResponse(
                provider_engine.stream_sse_chunks(
                    request_id=telemetry.request_id,
                    model_used=cached_entry.model_used,
                    response_text=cached_entry.response_text,
                    veriroute_meta=telemetry.model_dump(),
                ),
                media_type="text/event-stream",
            )

        return ChatCompletionResponse(
            model=cached_entry.model_used,
            choices=[
                ChatCompletionChoice(
                    message=ChatMessage(role="assistant", content=cached_entry.response_text),
                )
            ],
            usage=UsageStats(
                prompt_tokens=ctx.estimated_prompt_tokens,
                completion_tokens=comp_tokens,
                total_tokens=ctx.estimated_prompt_tokens + comp_tokens,
            ),
            veriroute=telemetry,
        )

    # 4B. Cache Miss -> Route via Online Disjoint LinUCB Contextual Bandit
    selected_spec, ctx, router_meta = bandit_router.select_arm(
        messages=processed_messages,
        requested_model=request.model,
        escalate_to_frontier=loop_meta.loop_detected,
    )
    gateway_overhead_ms = (time.perf_counter() - t_gateway_start) * 1000.0

    # 5. Execute Upstream LLM Completion
    exec_result = await provider_engine.complete(
        spec=selected_spec,
        messages=processed_messages,
        ctx=ctx,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        loop_escalated=loop_meta.loop_detected,
    )

    # 6. Online Bandit Weight Update & Semantic Cache Store
    bandit_router.update_arm(
        arm_id=selected_spec.arm_id,
        context_vector=ctx.vector,
        quality_score=exec_result.observed_quality_score,
        latency_ms=exec_result.ttft_ms,
        cost_usd=exec_result.cost_usd,
        signal_source="online_eval",
    )

    if not request.bypass_cache and not loop_meta.loop_detected and latest_user_prompt:
        semantic_cache.store(
            query=latest_user_prompt,
            response_text=exec_result.response_text,
            model_used=selected_spec.arm_id,
        )

    # 7. Optional Background Shadow Mode Evaluation (non-blocking async task)
    shadow_spec = shadow_evaluator.pick_shadow_candidate(selected_spec.arm_id)
    if shadow_spec is not None:
        router_meta.shadow_candidate_triggered = shadow_spec.arm_id
        asyncio.create_task(
            shadow_evaluator.run_shadow_evaluation(
                shadow_spec=shadow_spec,
                primary_arm_id=selected_spec.arm_id,
                primary_quality=exec_result.observed_quality_score,
                messages=processed_messages,
                ctx=ctx,
            )
        )

    total_latency_ms = gateway_overhead_ms + exec_result.total_latency_ms
    cost_saved = max(0.0, exec_result.baseline_frontier_cost_usd - exec_result.cost_usd)

    telemetry = VeriRouteTelemetry(
        gateway_overhead_ms=round(gateway_overhead_ms, 2),
        total_latency_ms=round(total_latency_ms, 2),
        cost_usd=exec_result.cost_usd,
        baseline_frontier_cost_usd=exec_result.baseline_frontier_cost_usd,
        cost_saved_usd=round(cost_saved, 6),
        cache=cache_meta,
        router=router_meta,
        loop_guard=loop_meta,
    )
    metrics_collector.record(latest_user_prompt, telemetry)

    if request.stream:
        return StreamingResponse(
            provider_engine.stream_sse_chunks(
                request_id=telemetry.request_id,
                model_used=exec_result.model_used,
                response_text=exec_result.response_text,
                veriroute_meta=telemetry.model_dump(),
            ),
            media_type="text/event-stream",
        )

    return ChatCompletionResponse(
        model=exec_result.model_used,
        choices=[
            ChatCompletionChoice(
                message=ChatMessage(role="assistant", content=exec_result.response_text),
            )
        ],
        usage=UsageStats(
            prompt_tokens=exec_result.prompt_tokens,
            completion_tokens=exec_result.completion_tokens,
            total_tokens=exec_result.prompt_tokens + exec_result.completion_tokens,
        ),
        veriroute=telemetry,
    )


@app.post("/api/v1/cache/compare")
async def compare_cache_behavior(req: CacheCompareRequest) -> dict:
    """Side-by-side comparison of Vanilla Cosine Cache vs VeriRoute 2-Stage NLI Cache on any pair."""
    temp_cache = EntailmentSemanticCache(
        cosine_threshold=settings.cache_cosine_threshold,
        entailment_threshold=settings.cache_entailment_threshold,
        contradiction_ceiling=settings.cache_contradiction_ceiling,
    )
    temp_cache.store(req.seed_query, req.seed_response, model_used="claude-3-5-sonnet")
    _, decision = temp_cache.lookup(req.test_query)

    # Also record in live gateway cache if it blocked a false positive so dashboard stats update
    semantic_cache.store(req.seed_query, req.seed_response, model_used="claude-3-5-sonnet")
    _, live_decision = semantic_cache.lookup(req.test_query)

    return {
        "seed_query": req.seed_query,
        "test_query": req.test_query,
        "cosine_similarity": decision.top_cosine_similarity,
        "cosine_threshold": settings.cache_cosine_threshold,
        "vanilla_cosine_cache": {
            "would_hit": decision.vanilla_cosine_would_hit,
            "verdict": "CACHE HIT" if decision.vanilla_cosine_would_hit else "CACHE MISS",
        },
        "veriroute_two_stage_cache": {
            "cache_hit": decision.cache_hit,
            "false_positive_prevented": decision.false_positive_prevented,
            "verdict": (
                "VERIFIED HIT (Safe Paraphrase)"
                if decision.cache_hit
                else (
                    "BLOCKED FALSE POSITIVE (Contradiction/Mismatch Caught!)"
                    if decision.false_positive_prevented
                    else "CACHE MISS (Below Cosine Threshold)"
                )
            ),
            "decision_reason": decision.decision_reason,
            "stage1_latency_ms": decision.stage1_latency_ms,
            "stage2_nli": decision.stage2_nli.model_dump() if decision.stage2_nli else None,
        },
    }


@app.post("/api/v1/feedback")
async def submit_feedback(req: FeedbackRequest) -> dict:
    """Ingest explicit user feedback or downstream validator signals into the LinUCB Bandit."""
    ctx = extract_prompt_context([ChatMessage(role="user", content=req.prompt)])
    reward = bandit_router.update_arm(
        arm_id=req.arm_id,
        context_vector=ctx.vector,
        quality_score=req.quality_score,
        latency_ms=req.latency_ms,
        cost_usd=req.cost_usd or 0.0,
        signal_source=req.signal_source,
    )
    return {
        "status": "updated",
        "arm_id": req.arm_id,
        "composite_reward": round(reward, 4),
        "bandit_diagnostics": bandit_router.get_diagnostics(),
    }


@app.post("/api/v1/bandit/simulate")
async def simulate_bandit_workload(rounds: int = 32) -> dict:
    """Run N mixed-complexity requests through the gateway to demonstrate online bandit learning."""
    rounds = max(1, min(120, rounds))
    results_log = []
    for i in range(rounds):
        prompt = BANDIT_WORKLOAD_PROMPTS[i % len(BANDIT_WORKLOAD_PROMPTS)]
        messages = [ChatMessage(role="user", content=prompt)]
        spec, ctx, router_meta = bandit_router.select_arm(messages)
        exec_res = await provider_engine.complete(spec, messages, ctx, fast_benchmark_mode=True)
        reward = bandit_router.update_arm(
            arm_id=spec.arm_id,
            context_vector=ctx.vector,
            quality_score=exec_res.observed_quality_score,
            latency_ms=exec_res.ttft_ms,
            cost_usd=exec_res.cost_usd,
            signal_source="simulation_batch",
        )
        cost_saved = max(0.0, exec_res.baseline_frontier_cost_usd - exec_res.cost_usd)
        telemetry = VeriRouteTelemetry(
            gateway_overhead_ms=router_meta.router_latency_ms,
            total_latency_ms=exec_res.total_latency_ms,
            cost_usd=exec_res.cost_usd,
            baseline_frontier_cost_usd=exec_res.baseline_frontier_cost_usd,
            cost_saved_usd=round(cost_saved, 6),
            cache=CacheDecisionMetadata(
                cache_hit=False,
                vanilla_cosine_would_hit=False,
                decision_reason="bandit_simulation",
            ),
            router=router_meta,
            loop_guard=loop_breaker.inspect_and_intervene(messages)[1],
        )
        metrics_collector.record(prompt, telemetry)
        results_log.append(
            {
                "round": i + 1,
                "prompt": prompt[:70],
                "domain": ctx.detected_domain,
                "complexity": ctx.complexity_score,
                "selected_model": spec.arm_id,
                "quality": exec_res.observed_quality_score,
                "reward": round(reward, 4),
                "cost_saved_usd": round(cost_saved, 6),
            }
        )

    return {
        "rounds_executed": rounds,
        "metrics_summary": metrics_collector.summary(),
        "bandit_diagnostics": bandit_router.get_diagnostics(),
        "sample_decisions": results_log[-12:],
    }


@app.get("/api/v1/stats")
async def get_gateway_stats() -> dict:
    """Return full real-time gateway telemetry for the interactive dashboard."""
    return {
        "metrics": metrics_collector.summary(),
        "cache": {
            "entries_count": semantic_cache.size(),
            "total_lookups": semantic_cache.total_lookups,
            "verified_hits": semantic_cache.verified_hits,
            "false_positives_blocked": semantic_cache.false_positives_blocked,
            "recent_blocked_events": semantic_cache.recent_blocked_events[:10],
        },
        "bandit": bandit_router.get_diagnostics(),
        "shadow_evals": shadow_evaluator.shadow_runs[:10],
        "loop_breaker": {
            "total_loops_broken": loop_breaker.total_loops_broken,
        },
    }


@app.get("/api/v1/benchmarks/run")
async def trigger_benchmark_suite() -> dict:
    """Execute the full Adversarial Cache & Bandit Pareto Benchmark Suite."""
    return await run_all_benchmarks()


@app.post("/api/v1/reset")
async def reset_gateway_state(warm_start: bool = True) -> dict:
    """Reset cache, bandit matrices, and telemetry counters."""
    semantic_cache.clear()
    bandit_router.reset(warm_start=warm_start)
    loop_breaker.reset()
    metrics_collector.reset()
    shadow_evaluator.shadow_runs.clear()
    return {"status": "reset_complete", "warm_start": warm_start}
