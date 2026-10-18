"""Gateway Telemetry Collector & Prometheus Exposition Formatter."""

from __future__ import annotations

from collections import Counter
import time
import numpy as np

from app.models import VeriRouteTelemetry


class GatewayMetricsCollector:
    """Tracks real-time latency percentiles, cost savings, cache accuracy, and model distribution."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.start_time = time.time()
        self.total_requests: int = 0
        self.cache_hits: int = 0
        self.false_positives_prevented: int = 0
        self.loops_broken: int = 0
        self.total_cost_usd: float = 0.0
        self.baseline_frontier_cost_usd: float = 0.0
        self.model_counts: Counter = Counter()
        self.gateway_overheads_ms: list[float] = []
        self.recent_requests: list[dict] = []

    def record(self, prompt_preview: str, telemetry: VeriRouteTelemetry) -> None:
        self.total_requests += 1
        if telemetry.cache.cache_hit:
            self.cache_hits += 1
        if telemetry.cache.false_positive_prevented:
            self.false_positives_prevented += 1
        if telemetry.loop_guard.loop_detected:
            self.loops_broken += 1

        self.total_cost_usd += telemetry.cost_usd
        self.baseline_frontier_cost_usd += telemetry.baseline_frontier_cost_usd
        self.model_counts[telemetry.router.selected_model] += 1
        self.gateway_overheads_ms.append(telemetry.gateway_overhead_ms)
        if len(self.gateway_overheads_ms) > 500:
            self.gateway_overheads_ms.pop(0)

        self.recent_requests.insert(
            0,
            {
                "request_id": telemetry.request_id,
                "prompt": prompt_preview[:95],
                "selected_model": telemetry.router.selected_model,
                "policy": telemetry.router.routing_policy,
                "cache_hit": telemetry.cache.cache_hit,
                "fp_prevented": telemetry.cache.false_positive_prevented,
                "cosine_sim": telemetry.cache.top_cosine_similarity,
                "nli_label": telemetry.cache.stage2_nli.label if telemetry.cache.stage2_nli else None,
                "complexity": telemetry.router.complexity_score,
                "domain": telemetry.router.detected_domain,
                "gateway_overhead_ms": telemetry.gateway_overhead_ms,
                "cost_usd": telemetry.cost_usd,
                "cost_saved_usd": telemetry.cost_saved_usd,
                "loop_detected": telemetry.loop_guard.loop_detected,
            },
        )
        if len(self.recent_requests) > 40:
            self.recent_requests.pop()

    def summary(self) -> dict:
        overheads = self.gateway_overheads_ms or [0.0]
        p50 = float(np.percentile(overheads, 50))
        p95 = float(np.percentile(overheads, 95))
        saved = max(0.0, self.baseline_frontier_cost_usd - self.total_cost_usd)
        savings_pct = (
            (saved / self.baseline_frontier_cost_usd) * 100.0
            if self.baseline_frontier_cost_usd > 1e-9
            else 0.0
        )
        return {
            "uptime_sec": round(time.time() - self.start_time, 1),
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_hit_rate_pct": round((self.cache_hits / max(1, self.total_requests)) * 100.0, 1),
            "false_positives_prevented": self.false_positives_prevented,
            "agent_loops_broken": self.loops_broken,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "baseline_frontier_cost_usd": round(self.baseline_frontier_cost_usd, 6),
            "total_cost_saved_usd": round(saved, 6),
            "cost_savings_pct": round(savings_pct, 1),
            "gateway_overhead_p50_ms": round(p50, 2),
            "gateway_overhead_p95_ms": round(p95, 2),
            "model_distribution": dict(self.model_counts),
            "recent_requests": self.recent_requests[:20],
        }

    def format_prometheus(self) -> str:
        """Render Prometheus text-based exposition format."""
        s = self.summary()
        lines = [
            "# HELP veriroute_requests_total Total LLM requests processed by VeriRoute gateway",
            "# TYPE veriroute_requests_total counter",
            f"veriroute_requests_total {s['total_requests']}",
            "# HELP veriroute_cache_hits_total Total entailment-verified semantic cache hits",
            "# TYPE veriroute_cache_hits_total counter",
            f"veriroute_cache_hits_total {s['cache_hits']}",
            "# HELP veriroute_cache_false_positives_prevented_total Adversarial cache hits blocked by Stage-2 NLI",
            "# TYPE veriroute_cache_false_positives_prevented_total counter",
            f"veriroute_cache_false_positives_prevented_total {s['false_positives_prevented']}",
            "# HELP veriroute_agent_loops_broken_total Agent infinite tool loops broken by circuit breaker",
            "# TYPE veriroute_agent_loops_broken_total counter",
            f"veriroute_agent_loops_broken_total {s['agent_loops_broken']}",
            "# HELP veriroute_cost_saved_usd_total Cumulative USD saved vs frontier baseline",
            "# TYPE veriroute_cost_saved_usd_total gauge",
            f"veriroute_cost_saved_usd_total {s['total_cost_saved_usd']}",
            "# HELP veriroute_gateway_overhead_p95_ms 95th percentile gateway overhead in ms",
            "# TYPE veriroute_gateway_overhead_p95_ms gauge",
            f"veriroute_gateway_overhead_p95_ms {s['gateway_overhead_p95_ms']}",
        ]
        for model_name, count in s["model_distribution"].items():
            lines.append(f'veriroute_routed_model_total{{model="{model_name}"}} {count}')
        return "\n".join(lines) + "\n"
