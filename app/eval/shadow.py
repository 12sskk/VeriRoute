"""Async Shadow Mode & Automated Judge Evaluator.

Mirrors a configurable fraction (e.g., 15%) of live requests in the background
to an alternate candidate model arm without adding any latency to the user's request.
Scores quality parity and feeds the reward directly into the LinUCB Bandit Router.
"""

from __future__ import annotations

import random
import time
from typing import Optional

from app.config import DEFAULT_MODEL_ARMS, ModelArmSpec
from app.models import ChatMessage
from app.providers.engine import UpstreamProviderEngine
from app.router.bandit import LinUCBBanditRouter
from app.router.features import ExtractedPromptContext


class ShadowEvaluator:
    """Background shadow traffic runner and LLM-as-a-Judge quality feedback loop."""

    def __init__(
        self,
        provider_engine: UpstreamProviderEngine,
        bandit_router: LinUCBBanditRouter,
        sample_rate: float = 0.15,
    ) -> None:
        self.provider_engine = provider_engine
        self.bandit_router = bandit_router
        self.sample_rate = sample_rate
        self.shadow_runs: list[dict] = []

    def pick_shadow_candidate(self, primary_arm_id: str) -> Optional[ModelArmSpec]:
        """Select a cheaper or alternative model arm to shadow-test against primary_arm_id."""
        if random.random() > self.sample_rate:
            return None
        candidates = [s for s in DEFAULT_MODEL_ARMS if s.arm_id != primary_arm_id]
        if not candidates:
            return None
        # Prefer testing cheaper tiers to discover cost-saving opportunities
        cheaper = [c for c in candidates if c.tier in ("fast", "balanced")]
        return random.choice(cheaper if cheaper else candidates)

    async def run_shadow_evaluation(
        self,
        shadow_spec: ModelArmSpec,
        primary_arm_id: str,
        primary_quality: float,
        messages: list[ChatMessage],
        ctx: ExtractedPromptContext,
    ) -> dict:
        """Execute shadow completion in background, evaluate parity, and update LinUCB."""
        res = await self.provider_engine.complete(
            spec=shadow_spec,
            messages=messages,
            ctx=ctx,
            fast_benchmark_mode=True,
        )
        reward = self.bandit_router.update_arm(
            arm_id=shadow_spec.arm_id,
            context_vector=ctx.vector,
            quality_score=res.observed_quality_score,
            latency_ms=res.ttft_ms,
            cost_usd=res.cost_usd,
            signal_source="shadow_judge",
        )
        record = {
            "timestamp": time.time(),
            "primary_model": primary_arm_id,
            "shadow_model": shadow_spec.arm_id,
            "domain": ctx.detected_domain,
            "complexity": ctx.complexity_score,
            "primary_quality": round(primary_quality, 3),
            "shadow_quality": round(res.observed_quality_score, 3),
            "quality_delta": round(res.observed_quality_score - primary_quality, 3),
            "shadow_cost_usd": res.cost_usd,
            "bandit_reward_updated": round(reward, 4),
        }
        self.shadow_runs.insert(0, record)
        if len(self.shadow_runs) > 30:
            self.shadow_runs.pop()
        return record
