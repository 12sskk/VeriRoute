"""Two-Stage Entailment-Verified Semantic Cache Manager.

Stage 1: Vector similarity search over normalized Bi-Encoder embeddings (Top-K retrieval).
Stage 2: Hard Constraint Verification + Cross-Encoder NLI Entailment Verification.

Also tracks whether a conventional Cosine-Only cache (like vanilla GPTCache) would have
triggered a False Positive on negated, number-swapped, or entity-swapped queries.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from app.cache.embeddings import BiEncoderEmbedder
from app.cache.nli_verifier import NLICrossEncoderVerifier
from app.models import CacheDecisionMetadata


@dataclass
class CacheEntry:
    entry_id: str
    query: str
    response_text: str
    model_used: str
    embedding: np.ndarray
    created_at: float = field(default_factory=time.time)
    hit_count: int = 0


class EntailmentSemanticCache:
    """Two-Stage Semantic Cache with Bi-Encoder Retrieval + Cross-Encoder NLI Guard."""

    def __init__(
        self,
        cosine_threshold: float = 0.82,
        entailment_threshold: float = 0.70,
        contradiction_ceiling: float = 0.25,
        max_entries: int = 2000,
        onnx_nli_model_path: Optional[str] = None,
    ) -> None:
        self.cosine_threshold = cosine_threshold
        self.entailment_threshold = entailment_threshold
        self.contradiction_ceiling = contradiction_ceiling
        self.max_entries = max_entries

        self.embedder = BiEncoderEmbedder(dim=128)
        self.nli_verifier = NLICrossEncoderVerifier(onnx_model_path=onnx_nli_model_path)
        self._entries: list[CacheEntry] = []

        # Telemetry counters
        self.total_lookups: int = 0
        self.verified_hits: int = 0
        self.false_positives_blocked: int = 0
        self.recent_blocked_events: list[dict] = []

    def clear(self) -> None:
        self._entries.clear()
        self.total_lookups = 0
        self.verified_hits = 0
        self.false_positives_blocked = 0
        self.recent_blocked_events.clear()

    def size(self) -> int:
        return len(self._entries)

    def store(self, query: str, response_text: str, model_used: str) -> CacheEntry:
        """Insert or update a verified query-response pair in the cache."""
        vec = self.embedder.encode(query)
        entry = CacheEntry(
            entry_id=f"cache_{uuid.uuid4().hex[:8]}",
            query=query,
            response_text=response_text,
            model_used=model_used,
            embedding=vec,
        )
        if len(self._entries) >= self.max_entries:
            self._entries.pop(0)
        self._entries.append(entry)
        return entry

    def lookup(
        self,
        query: str,
        top_k: int = 3,
        force_vanilla_cosine: bool = False,
    ) -> tuple[Optional[CacheEntry], CacheDecisionMetadata]:
        """Execute Stage-1 Vector Search followed by Stage-2 NLI Entailment Verification."""
        self.total_lookups += 1
        t0 = time.perf_counter()

        if not self._entries:
            return None, CacheDecisionMetadata(
                cache_hit=False,
                vanilla_cosine_would_hit=False,
                decision_reason="cache_empty",
            )

        q_vec = self.embedder.encode(query)
        matrix = np.stack([e.embedding for e in self._entries], axis=0)
        sims = np.dot(matrix, q_vec)

        ranked_indices = np.argsort(sims)[::-1][:top_k]
        best_idx = int(ranked_indices[0])
        best_sim = float(sims[best_idx])
        best_candidate = self._entries[best_idx]
        stage1_ms = (time.perf_counter() - t0) * 1000.0

        vanilla_would_hit = best_sim >= self.cosine_threshold

        # If Stage-1 cosine similarity is below threshold, fast reject without running Stage-2
        if best_sim < self.cosine_threshold:
            return None, CacheDecisionMetadata(
                cache_hit=False,
                vanilla_cosine_would_hit=False,
                top_cosine_similarity=round(best_sim, 4),
                matched_query=best_candidate.query,
                matched_entry_id=best_candidate.entry_id,
                stage1_latency_ms=round(stage1_ms, 3),
                decision_reason=f"stage1_cosine_below_threshold ({best_sim:.3f} < {self.cosine_threshold})",
            )

        # Evaluate Stage-2 NLI across Top-K candidates that passed Stage-1 threshold
        best_nli = None
        for idx in ranked_indices:
            sim_val = float(sims[int(idx)])
            if sim_val < self.cosine_threshold:
                break
            cand = self._entries[int(idx)]
            nli_res = self.nli_verifier.verify(query, cand.query, sim_val)
            if best_nli is None:
                best_nli = nli_res

            if force_vanilla_cosine:
                cand.hit_count += 1
                self.verified_hits += 1
                return cand, CacheDecisionMetadata(
                    cache_hit=True,
                    vanilla_cosine_would_hit=True,
                    false_positive_prevented=False,
                    top_cosine_similarity=round(sim_val, 4),
                    matched_query=cand.query,
                    matched_entry_id=cand.entry_id,
                    stage1_latency_ms=round(stage1_ms, 3),
                    stage2_nli=nli_res,
                    decision_reason="vanilla_cosine_forced_hit",
                )

            if (
                nli_res.constraint_report.passed_hard_constraints
                and nli_res.label == "ENTAILMENT"
                and nli_res.entailment_prob >= self.entailment_threshold
                and nli_res.contradiction_prob <= self.contradiction_ceiling
            ):
                cand.hit_count += 1
                self.verified_hits += 1
                return cand, CacheDecisionMetadata(
                    cache_hit=True,
                    vanilla_cosine_would_hit=True,
                    false_positive_prevented=False,
                    top_cosine_similarity=round(sim_val, 4),
                    matched_query=cand.query,
                    matched_entry_id=cand.entry_id,
                    stage1_latency_ms=round(stage1_ms, 3),
                    stage2_nli=nli_res,
                    decision_reason="stage2_nli_entailment_verified",
                )

        # Stage-1 Cosine wanted to hit, but Stage-2 NLI blocked a False Positive!
        self.false_positives_blocked += 1
        rejection_detail = (
            best_nli.constraint_report.rejection_reason
            if best_nli and best_nli.constraint_report.rejection_reason
            else f"nli_label={best_nli.label if best_nli else 'UNKNOWN'}"
        )
        blocked_event = {
            "timestamp": time.time(),
            "incoming_query": query,
            "cached_query": best_candidate.query,
            "cosine_similarity": round(best_sim, 4),
            "nli_label": best_nli.label if best_nli else "CONTRADICTION",
            "contradiction_prob": best_nli.contradiction_prob if best_nli else 1.0,
            "rejection_reason": rejection_detail,
        }
        self.recent_blocked_events.insert(0, blocked_event)
        if len(self.recent_blocked_events) > 25:
            self.recent_blocked_events.pop()

        return None, CacheDecisionMetadata(
            cache_hit=False,
            vanilla_cosine_would_hit=vanilla_would_hit,
            false_positive_prevented=True,
            top_cosine_similarity=round(best_sim, 4),
            matched_query=best_candidate.query,
            matched_entry_id=best_candidate.entry_id,
            stage1_latency_ms=round(stage1_ms, 3),
            stage2_nli=best_nli,
            decision_reason=f"stage2_blocked_false_positive: {rejection_detail}",
        )
