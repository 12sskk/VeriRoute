"""Stateful Agent Tool-Call & Semantic Loop Circuit Breaker.

Detects when multi-step AI agents enter repetitive tool-calling cycles or oscillating
states (either within the `messages` history of a single request or across consecutive
requests sharing a `session_id`).

When triggered:
1. Injects a deterministic loop-breaking system directive telling the agent which repeated
   action failed and instructing it to pivot strategies.
2. Signals the Bandit Router to escalate this stuck turn to the Frontier Reasoning model.
"""

from __future__ import annotations

from collections import defaultdict
import json
import re
from typing import Optional

from app.cache.embeddings import BiEncoderEmbedder
from app.models import ChatMessage, LoopCircuitBreakerMetadata


def _normalize_action_signature(msg: ChatMessage) -> str:
    """Create a canonical signature for an assistant tool call or message."""
    if msg.tool_calls:
        parts = []
        for tc in msg.tool_calls:
            try:
                parsed_args = json.loads(tc.function.arguments)
                canon_args = json.dumps(parsed_args, sort_keys=True)
            except Exception:
                canon_args = re.sub(r"\s+", " ", (tc.function.arguments or "").strip().lower())
            parts.append(f"tool:{tc.function.name}({canon_args})")
        return "|".join(parts)

    content = re.sub(r"\s+", " ", (msg.content or "").strip().lower())
    return content[:240]


class AgentLoopCircuitBreaker:
    """Monitors agent trajectories and breaks infinite tool/reasoning loops."""

    def __init__(
        self,
        repeat_trigger: int = 3,
        similarity_threshold: float = 0.90,
    ) -> None:
        self.repeat_trigger = repeat_trigger
        self.similarity_threshold = similarity_threshold
        self.embedder = BiEncoderEmbedder(dim=128)
        self._session_signatures: dict[str, list[str]] = defaultdict(list)
        self.total_loops_broken: int = 0

    def reset(self) -> None:
        self._session_signatures.clear()
        self.total_loops_broken = 0

    def inspect_and_intervene(
        self,
        messages: list[ChatMessage],
        session_id: Optional[str] = None,
    ) -> tuple[list[ChatMessage], LoopCircuitBreakerMetadata]:
        """Check message trajectory and session history for repetitive agent loops."""
        # Collect assistant actions or repeated user/tool error cycles in the message history
        signatures: list[str] = []
        for m in messages:
            if m.role == "assistant" and (m.tool_calls or m.content):
                sig = _normalize_action_signature(m)
                if sig:
                    signatures.append(sig)
            elif m.role == "tool" and m.content:
                cleaned_tool = re.sub(r"\s+", " ", m.content.strip().lower())[:160]
                sig = f"tool_output:{cleaned_tool}"
                signatures.append(sig)

        # Also include tracked session history if session_id is provided
        if session_id:
            latest_user = next((m.content for m in reversed(messages) if m.role == "user" and m.content), "")
            if latest_user:
                cleaned_u = re.sub(r"\s+", " ", latest_user.strip().lower())[:200]
                norm_u = f"user_turn:{cleaned_u}"
                self._session_signatures[session_id].append(norm_u)
                if len(self._session_signatures[session_id]) > 12:
                    self._session_signatures[session_id].pop(0)
                signatures.extend(self._session_signatures[session_id])

        if len(signatures) < self.repeat_trigger:
            return messages, LoopCircuitBreakerMetadata(
                loop_detected=False,
                session_id=session_id,
            )

        # 1. Exact signature repetition check
        counts: dict[str, int] = defaultdict(int)
        for sig in signatures:
            counts[sig] += 1
            if counts[sig] >= self.repeat_trigger:
                return self._trigger_intervention(messages, session_id, sig, counts[sig])

        # 2. Semantic near-duplicate check on recent assistant/user signatures
        recent = signatures[-6:]
        vecs = [self.embedder.encode(s) for s in recent]
        for i in range(len(recent)):
            similar_count = 1
            for j in range(i + 1, len(recent)):
                sim = float(self.embedder.cosine_similarity(vecs[i], vecs[j]))
                if sim >= self.similarity_threshold:
                    similar_count += 1
            if similar_count >= self.repeat_trigger:
                return self._trigger_intervention(messages, session_id, recent[i], similar_count)

        return messages, LoopCircuitBreakerMetadata(
            loop_detected=False,
            session_id=session_id,
        )

    def _trigger_intervention(
        self,
        messages: list[ChatMessage],
        session_id: Optional[str],
        repeated_sig: str,
        repeat_count: int,
    ) -> tuple[list[ChatMessage], LoopCircuitBreakerMetadata]:
        self.total_loops_broken += 1
        directive = ChatMessage(
            role="system",
            content=(
                "[VERIROUTE CIRCUIT BREAKER ACTIVATED]: Detected an infinite agent loop "
                f"repeating `{repeated_sig[:120]}` ({repeat_count} times). "
                "Do NOT repeat the same tool call or identical query. Diagnose the root cause, "
                "change your parameters or strategy, or return a clear fallback explanation."
            ),
        )
        updated_messages = list(messages) + [directive]
        return updated_messages, LoopCircuitBreakerMetadata(
            loop_detected=True,
            session_id=session_id,
            repeated_action_signature=repeated_sig[:120],
            repeat_count=repeat_count,
            action_taken="injected_system_directive_and_escalated",
        )
