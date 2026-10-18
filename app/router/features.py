"""Prompt Context & Complexity Feature Extractor for LinUCB Contextual Bandit.

Extracts an 8-dimensional context vector x_t in R^8 alongside human-interpretable
domain classification and complexity scores.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
import numpy as np
from app.models import ChatMessage

FEATURE_NAMES: list[str] = [
    "bias",
    "simplicity_signal",
    "length_log_norm",
    "code_density",
    "reasoning_math_depth",
    "systems_arch_depth",
    "strict_constraint_signal",
    "conversation_depth",
]

REASONING_KEYWORDS = {
    "prove", "proof", "derive", "theorem", "induction", "contradiction", "optimal",
    "dynamic programming", "np-hard", "time complexity", "space complexity", "amortized",
    "eigenvalue", "hessian", "jacobian", "bayesian", "posterior", "martingale",
    "formal verification", "invariant", "recurrence", "edge cases", "step-by-step",
    "counterexample", "deadlock", "race condition", "linearizability",
}

SYSTEMS_ARCH_KEYWORDS = {
    "distributed", "microservices", "consensus", "raft", "paxos", "byzantine",
    "sharding", "partitioning", "replication", "eventual consistency", "cap theorem",
    "idempotency", "backpressure", "circuit breaker", "kafka", "zero-downtime",
    "multi-region", "high availability", "load shedding", "lock-free", "mvcc",
    "write-ahead log", "b-tree", "lsm-tree", "vector clock", "architecture",
}

CODE_KEYWORDS = {
    "def ", "class ", "import ", "async ", "await ", "function", "return ",
    "select ", "from ", "where ", "join ", "group by", "```", "interface ",
    "struct ", "impl ", "public static", "try:", "except:", "const ", "=>",
}

STRICT_CONSTRAINT_KEYWORDS = {
    "json", "schema", "strictly", "exact format", "valid xml", "yaml only",
    "no markdown", "regex", "rfc", "must contain", "do not include",
}

SIMPLE_QA_KEYWORDS = {
    "what is the capital", "who invented", "translate", "summarize in one",
    "tl;dr", "define ", "hello", "hi ", "convert ", "syntax for", "how to print",
    "what does http stand for", "list 3 ", "give me a short",
}


@dataclass
class ExtractedPromptContext:
    vector: np.ndarray  # Shape: (8,)
    feature_dict: dict[str, float]
    complexity_score: float  # In [0.0, 1.0]
    detected_domain: str  # "simple_qa", "summarization", "code_simple", "complex_reasoning", "system_design", "math_proof"
    estimated_prompt_tokens: int


def extract_prompt_context(messages: list[ChatMessage]) -> ExtractedPromptContext:
    """Compute context vector x_t in R^8 from a chat message sequence."""
    user_texts = [m.content or "" for m in messages if m.role == "user"]
    full_text = "\n".join(m.content or "" for m in messages).strip()
    latest_user = (user_texts[-1] if user_texts else full_text).strip()
    lower = latest_user.lower()

    words = re.findall(r"\S+", full_text)
    est_tokens = max(1, int(len(words) * 1.3))

    # 1. Length log norm in [0, 1]
    length_norm = min(1.0, math.log1p(est_tokens) / math.log1p(600))

    # 2. Code syntax density
    code_hits = sum(1 for kw in CODE_KEYWORDS if kw in lower)
    symbol_count = len(re.findall(r"[{}\[\]();=<>_]", latest_user))
    code_density = min(1.0, (code_hits * 0.22) + (symbol_count / max(30.0, len(latest_user) * 0.15)))

    # 3. Reasoning & math depth
    reasoning_hits = sum(1 for kw in REASONING_KEYWORDS if kw in lower)
    math_symbols = len(re.findall(r"[\^\\\$\∑\∫\√]|O\([^)]+\)", latest_user))
    reasoning_depth = min(1.0, reasoning_hits * 0.32 + math_symbols * 0.20)

    # 4. Systems & architecture depth
    arch_hits = sum(1 for kw in SYSTEMS_ARCH_KEYWORDS if kw in lower)
    arch_depth = min(1.0, arch_hits * 0.32)

    # 5. Strict output constraint signal
    constraint_hits = sum(1 for kw in STRICT_CONSTRAINT_KEYWORDS if kw in lower)
    constraint_signal = min(1.0, constraint_hits * 0.4)

    # 6. Conversation depth
    conv_depth = min(1.0, max(0, len(messages) - 1) / 8.0)

    # 7. Simplicity signal (high when short and no deep reasoning/arch markers)
    simple_hits = sum(1 for kw in SIMPLE_QA_KEYWORDS if kw in lower)
    raw_technical = max(reasoning_depth, arch_depth, code_density * 0.8)
    simplicity_signal = max(0.0, min(1.0, (1.0 - raw_technical) * (0.7 + 0.3 * min(1, simple_hits)) * (1.0 - 0.5 * length_norm)))

    # Overall complexity score in [0, 1]
    complexity_score = float(
        np.clip(
            0.35 * reasoning_depth
            + 0.30 * arch_depth
            + 0.20 * code_density
            + 0.10 * constraint_signal
            + 0.15 * length_norm
            - 0.25 * simplicity_signal,
            0.02,
            0.99,
        )
    )

    # Domain classification
    if reasoning_depth >= 0.45 and ("prove" in lower or "math" in lower or "integral" in lower or "eigen" in lower or "bayesian" in lower):
        domain = "math_proof"
    elif reasoning_depth >= 0.40:
        domain = "complex_reasoning"
    elif arch_depth >= 0.40:
        domain = "system_design"
    elif code_density >= 0.35:
        domain = "code_simple" if complexity_score < 0.52 else "complex_reasoning"
    elif "summarize" in lower or "summary" in lower or "tl;dr" in lower:
        domain = "summarization"
    else:
        domain = "simple_qa" if complexity_score < 0.45 else "complex_reasoning"

    raw_vec = np.array(
        [
            1.0,
            round(simplicity_signal, 4),
            round(length_norm, 4),
            round(code_density, 4),
            round(reasoning_depth, 4),
            round(arch_depth, 4),
            round(constraint_signal, 4),
            round(conv_depth, 4),
        ],
        dtype=np.float64,
    )
    # Normalize context vector so UCB confidence bounds are well-scaled
    norm = float(np.linalg.norm(raw_vec))
    vec = raw_vec / norm if norm > 1e-8 else raw_vec

    feature_dict = {name: round(float(raw_vec[i]), 4) for i, name in enumerate(FEATURE_NAMES)}

    return ExtractedPromptContext(
        vector=vec,
        feature_dict=feature_dict,
        complexity_score=round(complexity_score, 4),
        detected_domain=domain,
        estimated_prompt_tokens=est_tokens,
    )
