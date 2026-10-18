"""Curated Evaluation Datasets for Semantic Cache NLI Verification and Bandit Routing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CacheEvalPair:
    seed_query: str
    test_query: str
    should_hit: bool
    category: str  # "paraphrase", "antonym_negation", "number_swap", "entity_role_swap", "unrelated"


ADVERSARIAL_CACHE_DATASET: list[CacheEvalPair] = [
    # --- 1. TRUE PARAPHRASES (Should Hit = True) ---
    CacheEvalPair(
        seed_query="How do I enable CORS in FastAPI?",
        test_query="What is the way to turn on cross-origin in FastAPI?",
        should_hit=True,
        category="paraphrase",
    ),
    CacheEvalPair(
        seed_query="Transfer $500 to Alice",
        test_query="Please wire $500 to Alice",
        should_hit=True,
        category="paraphrase",
    ),
    CacheEvalPair(
        seed_query="How to increase connection pool size in Postgres?",
        test_query="How can I raise the connection pool size in PostgreSQL?",
        should_hit=True,
        category="paraphrase",
    ),
    CacheEvalPair(
        seed_query="Write a Python function to encrypt a string",
        test_query="Create a py function for encrypting a string",
        should_hit=True,
        category="paraphrase",
    ),
    CacheEvalPair(
        seed_query="Sort the database records in ascending order",
        test_query="Order the db records ascending",
        should_hit=True,
        category="paraphrase",
    ),
    CacheEvalPair(
        seed_query="Delete the temporary cache files in Linux",
        test_query="How do I remove temporary cache files in Linux?",
        should_hit=True,
        category="paraphrase",
    ),
    CacheEvalPair(
        seed_query="Explain how JWT authentication works",
        test_query="Describe how JWT auth works",
        should_hit=True,
        category="paraphrase",
    ),
    CacheEvalPair(
        seed_query="Upload the backup archive to AWS",
        test_query="How can I push the backup archive to AWS?",
        should_hit=True,
        category="paraphrase",
    ),
    # --- 2. ADVERSARIAL NEGATION & ANTONYM TRAPS (Should Hit = False) ---
    CacheEvalPair(
        seed_query="How do I enable CORS in FastAPI?",
        test_query="How do I disable CORS in FastAPI?",
        should_hit=False,
        category="antonym_negation",
    ),
    CacheEvalPair(
        seed_query="How do I enable CORS in FastAPI?",
        test_query="How do I not enable CORS in FastAPI?",
        should_hit=False,
        category="antonym_negation",
    ),
    CacheEvalPair(
        seed_query="How to increase connection pool size in Postgres?",
        test_query="How to decrease connection pool size in Postgres?",
        should_hit=False,
        category="antonym_negation",
    ),
    CacheEvalPair(
        seed_query="Write a Python function to encrypt a string",
        test_query="Write a Python function to decrypt a string",
        should_hit=False,
        category="antonym_negation",
    ),
    CacheEvalPair(
        seed_query="Sort the database records in ascending order",
        test_query="Sort the database records in descending order",
        should_hit=False,
        category="antonym_negation",
    ),
    CacheEvalPair(
        seed_query="Create the users table in Postgres",
        test_query="Drop the users table in Postgres",
        should_hit=False,
        category="antonym_negation",
    ),
    CacheEvalPair(
        seed_query="Upload the backup archive to AWS",
        test_query="Download the backup archive from AWS",
        should_hit=False,
        category="antonym_negation",
    ),
    CacheEvalPair(
        seed_query="Allow public access to the Redis cluster",
        test_query="Block public access to the Redis cluster",
        should_hit=False,
        category="antonym_negation",
    ),
    # --- 3. ADVERSARIAL NUMBER / QUANTITY SWAPS (Should Hit = False) ---
    CacheEvalPair(
        seed_query="Transfer $500 to Alice",
        test_query="Transfer $5000 to Alice",
        should_hit=False,
        category="number_swap",
    ),
    CacheEvalPair(
        seed_query="Transfer $500 to Alice",
        test_query="Transfer $50 to Alice",
        should_hit=False,
        category="number_swap",
    ),
    CacheEvalPair(
        seed_query="Configure a 30 second timeout in FastAPI",
        test_query="Configure a 120 second timeout in FastAPI",
        should_hit=False,
        category="number_swap",
    ),
    CacheEvalPair(
        seed_query="Limit maximum retries to 3 in Python",
        test_query="Limit maximum retries to 10 in Python",
        should_hit=False,
        category="number_swap",
    ),
    # --- 4. ADVERSARIAL ENTITY & DIRECTIONAL ROLE SWAPS (Should Hit = False) ---
    CacheEvalPair(
        seed_query="Transfer $500 to Alice",
        test_query="Transfer $500 to Bob",
        should_hit=False,
        category="entity_role_swap",
    ),
    CacheEvalPair(
        seed_query="How do I enable CORS in FastAPI?",
        test_query="How do I enable CORS in Flask?",
        should_hit=False,
        category="entity_role_swap",
    ),
    CacheEvalPair(
        seed_query="How do I enable CORS in FastAPI?",
        test_query="How do I enable CSRF in FastAPI?",
        should_hit=False,
        category="entity_role_swap",
    ),
    CacheEvalPair(
        seed_query="Transfer funds from checking to savings",
        test_query="Transfer funds from savings to checking",
        should_hit=False,
        category="entity_role_swap",
    ),
    CacheEvalPair(
        seed_query="Upload the backup archive to AWS",
        test_query="Upload the backup archive to GCP",
        should_hit=False,
        category="entity_role_swap",
    ),
    # --- 5. UNRELATED QUERIES (Should Hit = False) ---
    CacheEvalPair(
        seed_query="How do I enable CORS in FastAPI?",
        test_query="Prove that the halting problem is undecidable",
        should_hit=False,
        category="unrelated",
    ),
    CacheEvalPair(
        seed_query="Transfer $500 to Alice",
        test_query="Design a multi-region Kafka replication architecture",
        should_hit=False,
        category="unrelated",
    ),
]


BANDIT_WORKLOAD_PROMPTS: list[str] = [
    # Easy / Simple QA & Summarization (Fast tier Llama-3.1-8B is optimal)
    "What does HTTP stand for? Explain in one sentence.",
    "Translate 'Hello, how are you?' to Spanish and French.",
    "Summarize in one short sentence what a Python dictionary is.",
    "Give me a short list of 3 popular relational databases.",
    "What is the syntax for printing a variable in Python?",
    "Define what REST API stands for briefly.",
    "Convert 100 kilometers to miles.",
    "What is the capital of Japan?",
    # Medium / Code Simple (Fast or Balanced tier is optimal)
    "Write a Python function `def reverse_words(s: str)` that reverses words in a string.",
    "Show a basic SQL query using `SELECT`, `FROM`, and `WHERE` to filter active users.",
    "Write a simple JavaScript arrow function `const sum = (a, b) =>` and export it.",
    "How do I read a JSON file in Python using `try:` and `except:`?",
    # Hard / Complex Reasoning, Formal Math & Distributed Systems (Frontier tier is required for high quality)
    "Prove by strong induction and derive the exact amortized time complexity O(n log n) for a self-balancing tree with dynamic programming state transitions and invariant verification.",
    "Design a multi-region active-active distributed consensus architecture using Raft and Paxos with sharding, vector clocks, idempotency, backpressure, and CAP theorem trade-offs.",
    "Derive the Bayesian posterior distribution and Hessian eigenvalue bounds for non-convex optimization, proving convergence under martingale noise and formal verification invariants.",
    "Architect a lock-free MVCC database storage engine with write-ahead logging, B-tree / LSM-tree compaction, linearizability guarantees, and deadlock-free distributed transactions.",
]
