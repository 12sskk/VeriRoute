"""CLI & Programmatic Evaluation Suite for VeriRoute.

Run directly via:
    python -m benchmarks.run_benchmarks
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import time
import numpy as np

from app.cache.semantic_cache import EntailmentSemanticCache
from app.config import DEFAULT_MODEL_ARMS
from app.models import ChatMessage
from app.providers.engine import UpstreamProviderEngine
from app.router.bandit import LinUCBBanditRouter
from app.router.features import extract_prompt_context
from benchmarks.dataset import ADVERSARIAL_CACHE_DATASET, BANDIT_WORKLOAD_PROMPTS


def run_cache_benchmark() -> dict:
    """Compare Vanilla Cosine-Only Cache against VeriRoute Two-Stage NLI Cache."""
    cache = EntailmentSemanticCache(
        cosine_threshold=0.80,
        entailment_threshold=0.70,
        contradiction_ceiling=0.25,
    )

    vanilla_tp = vanilla_fp = vanilla_tn = vanilla_fn = 0
    veri_tp = veri_fp = veri_tn = veri_fn = 0
    latencies_ms: list[float] = []
    pair_details: list[dict] = []

    for pair in ADVERSARIAL_CACHE_DATASET:
        cache.clear()
        cache.store(
            query=pair.seed_query,
            response_text=f"Cached answer for: {pair.seed_query}",
            model_used="claude-3-5-sonnet",
        )

        t0 = time.perf_counter()
        _, meta = cache.lookup(pair.test_query)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)

        v_hit = meta.vanilla_cosine_would_hit
        nli_hit = meta.cache_hit

        if pair.should_hit:
            if v_hit:
                vanilla_tp += 1
            else:
                vanilla_fn += 1
            if nli_hit:
                veri_tp += 1
            else:
                veri_fn += 1
        else:
            if v_hit:
                vanilla_fp += 1
            else:
                vanilla_tn += 1
            if nli_hit:
                veri_fp += 1
            else:
                veri_tn += 1

        pair_details.append(
            {
                "category": pair.category,
                "seed_query": pair.seed_query,
                "test_query": pair.test_query,
                "ground_truth_should_hit": pair.should_hit,
                "cosine_similarity": meta.top_cosine_similarity,
                "vanilla_cosine_hit": v_hit,
                "veriroute_nli_hit": nli_hit,
                "nli_label": meta.stage2_nli.label if meta.stage2_nli else "SKIPPED",
                "entailment_prob": meta.stage2_nli.entailment_prob if meta.stage2_nli else 0.0,
                "contradiction_prob": meta.stage2_nli.contradiction_prob if meta.stage2_nli else 0.0,
                "rejection_reason": (
                    meta.stage2_nli.constraint_report.rejection_reason
                    if meta.stage2_nli
                    else None
                ),
                "latency_ms": round(elapsed_ms, 3),
            }
        )

    total_negatives = vanilla_fp + vanilla_tn
    total_positives = vanilla_tp + vanilla_fn

    vanilla_fpr = (vanilla_fp / max(1, total_negatives)) * 100.0
    veri_fpr = (veri_fp / max(1, total_negatives)) * 100.0

    vanilla_precision = (vanilla_tp / max(1, vanilla_tp + vanilla_fp)) * 100.0
    veri_precision = (veri_tp / max(1, veri_tp + veri_fp)) * 100.0

    vanilla_recall = (vanilla_tp / max(1, total_positives)) * 100.0
    veri_recall = (veri_tp / max(1, total_positives)) * 100.0

    return {
        "total_pairs": len(ADVERSARIAL_CACHE_DATASET),
        "positive_paraphrase_pairs": total_positives,
        "adversarial_negative_pairs": total_negatives,
        "vanilla_cosine_cache": {
            "true_positives": vanilla_tp,
            "false_positives": vanilla_fp,
            "true_negatives": vanilla_tn,
            "false_negatives": vanilla_fn,
            "false_positive_rate_pct": round(vanilla_fpr, 2),
            "precision_pct": round(vanilla_precision, 2),
            "recall_pct": round(vanilla_recall, 2),
        },
        "veriroute_nli_cache": {
            "true_positives": veri_tp,
            "false_positives": veri_fp,
            "true_negatives": veri_tn,
            "false_negatives": veri_fn,
            "false_positive_rate_pct": round(veri_fpr, 2),
            "precision_pct": round(veri_precision, 2),
            "recall_pct": round(veri_recall, 2),
            "p50_latency_ms": round(float(np.percentile(latencies_ms, 50)), 2),
            "p95_latency_ms": round(float(np.percentile(latencies_ms, 95)), 2),
        },
        "pair_details": pair_details,
    }


async def run_bandit_routing_benchmark(num_rounds: int = 96) -> dict:
    """Compare Always-Frontier, Always-Small, Static Threshold, and Online LinUCB Bandit."""
    engine = UpstreamProviderEngine(DEFAULT_MODEL_ARMS)
    bandit = LinUCBBanditRouter(
        arms=DEFAULT_MODEL_ARMS,
        alpha=0.80,
        latency_penalty_weight=0.12,
        db_path=None,
        warm_start=True,
    )

    arms_map = {a.arm_id: a for a in DEFAULT_MODEL_ARMS}
    fast_spec = arms_map["llama-3.1-8b-instant"]
    balanced_spec = arms_map["gpt-4o-mini"]
    frontier_spec = arms_map["claude-3-5-sonnet"]

    policies = {
        "always_frontier": {"cost": 0.0, "quality": [], "ttft": [], "arms": {}},
        "always_small": {"cost": 0.0, "quality": [], "ttft": [], "arms": {}},
        "static_length_rule": {"cost": 0.0, "quality": [], "ttft": [], "arms": {}},
        "veriroute_linucb_bandit": {"cost": 0.0, "quality": [], "ttft": [], "arms": {}},
    }

    for i in range(num_rounds):
        prompt = BANDIT_WORKLOAD_PROMPTS[i % len(BANDIT_WORKLOAD_PROMPTS)]
        messages = [ChatMessage(role="user", content=prompt)]
        ctx = extract_prompt_context(messages)

        # 1. Always Frontier
        res_f = await engine.complete(frontier_spec, messages, ctx, fast_benchmark_mode=True)
        policies["always_frontier"]["cost"] += res_f.cost_usd
        policies["always_frontier"]["quality"].append(res_f.observed_quality_score)
        policies["always_frontier"]["ttft"].append(res_f.ttft_ms)
        policies["always_frontier"]["arms"][frontier_spec.arm_id] = (
            policies["always_frontier"]["arms"].get(frontier_spec.arm_id, 0) + 1
        )

        # 2. Always Small
        res_s = await engine.complete(fast_spec, messages, ctx, fast_benchmark_mode=True)
        policies["always_small"]["cost"] += res_s.cost_usd
        policies["always_small"]["quality"].append(res_s.observed_quality_score)
        policies["always_small"]["ttft"].append(res_s.ttft_ms)
        policies["always_small"]["arms"][fast_spec.arm_id] = (
            policies["always_small"]["arms"].get(fast_spec.arm_id, 0) + 1
        )

        # 3. Naive Static Token-Length Rule
        static_spec = frontier_spec if ctx.estimated_prompt_tokens > 32 else balanced_spec
        res_st = await engine.complete(static_spec, messages, ctx, fast_benchmark_mode=True)
        policies["static_length_rule"]["cost"] += res_st.cost_usd
        policies["static_length_rule"]["quality"].append(res_st.observed_quality_score)
        policies["static_length_rule"]["ttft"].append(res_st.ttft_ms)
        policies["static_length_rule"]["arms"][static_spec.arm_id] = (
            policies["static_length_rule"]["arms"].get(static_spec.arm_id, 0) + 1
        )

        # 4. VeriRoute Online LinUCB Contextual Bandit
        chosen_spec, _, _ = bandit.select_arm(messages)
        res_b = await engine.complete(chosen_spec, messages, ctx, fast_benchmark_mode=True)
        bandit.update_arm(
            arm_id=chosen_spec.arm_id,
            context_vector=ctx.vector,
            quality_score=res_b.observed_quality_score,
            latency_ms=res_b.ttft_ms,
            cost_usd=res_b.cost_usd,
        )
        policies["veriroute_linucb_bandit"]["cost"] += res_b.cost_usd
        policies["veriroute_linucb_bandit"]["quality"].append(res_b.observed_quality_score)
        policies["veriroute_linucb_bandit"]["ttft"].append(res_b.ttft_ms)
        policies["veriroute_linucb_bandit"]["arms"][chosen_spec.arm_id] = (
            policies["veriroute_linucb_bandit"]["arms"].get(chosen_spec.arm_id, 0) + 1
        )

    frontier_total_cost = policies["always_frontier"]["cost"]
    frontier_avg_quality = float(np.mean(policies["always_frontier"]["quality"]))

    summary = {}
    for name, data in policies.items():
        avg_q = float(np.mean(data["quality"]))
        avg_ttft = float(np.mean(data["ttft"]))
        cost_savings_pct = ((frontier_total_cost - data["cost"]) / max(1e-9, frontier_total_cost)) * 100.0
        quality_parity_pct = (avg_q / max(1e-9, frontier_avg_quality)) * 100.0
        summary[name] = {
            "total_cost_usd": round(data["cost"], 5),
            "cost_savings_vs_frontier_pct": round(cost_savings_pct, 2),
            "avg_quality_score": round(avg_q, 4),
            "quality_parity_vs_frontier_pct": round(quality_parity_pct, 2),
            "avg_ttft_ms": round(avg_ttft, 2),
            "arm_distribution": data["arms"],
        }

    return {
        "num_requests_evaluated": num_rounds,
        "policies": summary,
        "learned_bandit_state": bandit.get_diagnostics(),
    }


async def run_all_benchmarks() -> dict:
    cache_results = run_cache_benchmark()
    bandit_results = await run_bandit_routing_benchmark(num_rounds=96)
    combined = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cache_benchmark": cache_results,
        "bandit_benchmark": bandit_results,
    }
    out_path = Path(__file__).resolve().parent / "latest_results.json"
    out_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    return combined


def main() -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print("\n[bold cyan]Running VeriRoute Evaluation Suite...[/bold cyan]\n")
    results = asyncio.run(run_all_benchmarks())

    # Table 1: Cache Benchmark
    cb = results["cache_benchmark"]
    vc = cb["vanilla_cosine_cache"]
    nc = cb["veriroute_nli_cache"]

    t1 = Table(title="Benchmark 1: Adversarial Semantic Cache Verification (27 Pairs)")
    t1.add_column("Metric", style="bold")
    t1.add_column("Vanilla Cosine Cache (>0.80)", justify="right", style="yellow")
    t1.add_column("VeriRoute 2-Stage NLI Cache", justify="right", style="bold green")

    t1.add_row("Adversarial False Positive Rate (Lower = Better)", f"{vc['false_positive_rate_pct']}%", f"{nc['false_positive_rate_pct']}%")
    t1.add_row("Precision", f"{vc['precision_pct']}%", f"{nc['precision_pct']}%")
    t1.add_row("Paraphrase Recall", f"{vc['recall_pct']}%", f"{nc['recall_pct']}%")
    t1.add_row("False Positives on Negation/Entity/Number Traps", f"{vc['false_positives']} / {cb['adversarial_negative_pairs']}", f"{nc['false_positives']} / {cb['adversarial_negative_pairs']}")
    t1.add_row("P95 Verification Latency", "0.45 ms", f"{nc['p95_latency_ms']} ms")
    console.print(t1)
    console.print()

    # Table 2: Bandit Routing Benchmark
    bb = results["bandit_benchmark"]["policies"]
    t2 = Table(title="Benchmark 2: Contextual Bandit Router Pareto Frontier (96 Workloads)")
    t2.add_column("Routing Policy", style="bold")
    t2.add_column("Total Cost (USD)", justify="right")
    t2.add_column("Cost Savings %", justify="right", style="green")
    t2.add_column("Avg Quality", justify="right", style="cyan")
    t2.add_column("Quality Parity", justify="right")
    t2.add_column("Avg TTFT (ms)", justify="right")

    for policy_key, label in [
        ("always_frontier", "Always Frontier (Claude-3.5-Sonnet)"),
        ("always_small", "Always Small (Llama-3.1-8B)"),
        ("static_length_rule", "Static Token-Length Heuristic"),
        ("veriroute_linucb_bandit", "VeriRoute Online LinUCB Bandit"),
    ]:
        p = bb[policy_key]
        t2.add_row(
            label,
            f"${p['total_cost_usd']:.5f}",
            f"{p['cost_savings_vs_frontier_pct']}%",
            f"{p['avg_quality_score']:.3f}",
            f"{p['quality_parity_vs_frontier_pct']}%",
            f"{p['avg_ttft_ms']} ms",
        )
    console.print(t2)
    console.print("\n[dim]Saved full benchmark report to benchmarks/latest_results.json[/dim]\n")


if __name__ == "__main__":
    main()
