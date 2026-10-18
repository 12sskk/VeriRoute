# Interview Preparation Guide: VeriRoute

This guide prepares you to present **VeriRoute** in both **Software Engineering (Backend / Systems / Infra)** and **AI / ML Engineering** interviews.

---

## 1. Resume Bullet Points (Copy & Customize)

**VeriRoute — Entailment-Verified Semantic LLM Gateway & Contextual Bandit Router** | *Python, FastAPI, NumPy, ONNX Runtime, SQLite, Pytest*
* **Architected an OpenAI-compatible AI Gateway** supporting Server-Sent Events (SSE) streaming, per-tenant token-bucket rate limiting, stateful agent loop circuit breaking, and Prometheus observability (`<1ms` P95 gateway overhead).
* **Eliminated Semantic Cache Hallucinations:** Built a two-stage hybrid semantic cache combining dense Bi-Encoder vector search with a Stage-2 NLI Cross-Encoder and deterministic constraint guard (verifying numerical, entity, and negation invariants), reducing adversarial false-positive cache hits from **84.2% to 0.0%** while maintaining **100% paraphrase recall**.
* **Online Contextual Bandit Routing:** Implemented a Disjoint **LinUCB (Linear Upper Confidence Bound)** router over 8-dimensional prompt complexity features, real-time EMA Time-To-First-Token (TTFT) telemetry, and asynchronous shadow-mode evaluations, cutting LLM token costs by **74.5%** while preserving **95.1% quality parity** against frontier models.

---

## 2. The 60-Second Elevator Pitch

> *"When building LLM applications, I noticed two major flaws in existing open-source gateways like LiteLLM and GPTCache. First, standard semantic caches rely purely on Bi-Encoder cosine similarity—meaning queries like `'How do I enable CORS in FastAPI?'` and `'How do I disable CORS in FastAPI?'` have over 0.90 cosine similarity and return catastrophic false-positive cache hits. Second, existing prompt routers use static classifiers trained offline that don't adapt to live provider latency or domain-specific failure modes.*
>
> *So I built **VeriRoute**. For caching, it uses a two-stage architecture: Stage 1 retrieves top candidates via dense vector similarity, and Stage 2 runs an NLI cross-encoder and deterministic constraint guard that checks for bidirectional entailment, negation polarity, numbers, and named entities—dropping false positives on our adversarial benchmark from 84% to 0% in under 1 millisecond.*
>
> *For routing, I implemented an online Disjoint LinUCB Contextual Bandit that extracts an 8D feature vector from each prompt and balances cost, quality, and live TTFT latency. It also mirrors 15% of traffic asynchronously in shadow mode to continuously update its ridge regression matrices without adding user-facing latency."*

---

## 3. Deep-Dive Questions from AI / ML Interviewers

### Q1: Why do Bi-Encoders fail on negations and entity swaps, and why does a Cross-Encoder NLI model fix it?
* **Answer:** Bi-Encoders encode Query $A$ and Query $B$ **independently** into single fixed-width vectors $\mathbf{u}, \mathbf{v}$ and compute $\cos(\mathbf{u}, \mathbf{v})$. Because 90% of the tokens in *"Transfer \$500 to Alice"* and *"Transfer \$500 to Bob"* are identical and `"Alice"`/`"Bob"` occupy the same proper-noun subspace, their cosine similarity exceeds `0.90`.
* A **Cross-Encoder NLI** model processes `[CLS] Query A [SEP] Query B [SEP]` jointly with full cross-attention across token pairs, combined with explicit invariant extraction (numbers, proper nouns, prepositional roles like `from X to Y`, and antonym polarity). It outputs a 3-way probability distribution over `(ENTAILMENT, NEUTRAL, CONTRADICTION)`. We only serve a cache hit if `label == ENTAILMENT`, $P(\text{entailment}) \ge 0.70$, and $P(\text{contradiction}) \le 0.25$.

### Q2: Why did you choose Disjoint LinUCB over a static neural classifier or $\epsilon$-greedy bandit?
* **Answer:**
  1. **Contextual Adaptation:** Unlike multi-armed bandits without context (like standard Thompson Sampling or $\epsilon$-greedy), model performance strongly depends on the prompt's context $\mathbf{x}_t \in \mathbb{R}^d$ (e.g., `llama-3.1-8b-instant` is great at short Q&A, but fails at formal induction proofs).
  2. **Directed Exploration via Uncertainty Ellipsoids:** $\epsilon$-greedy explores blindly by picking random models even when it is already certain a model is bad for a given context. LinUCB maintains the ridge covariance matrix $\mathbf{A}_a = \mathbf{I}_d + \sum \mathbf{x}_t \mathbf{x}_t^\top$ for each arm $a$ and computes the confidence width $\alpha_t \sqrt{\mathbf{x}_t^\top \mathbf{A}_a^{-1} \mathbf{x}_t}$. It only explores arms in regions of the feature space where uncertainty is high.
  3. **Closed-Form $O(d^2)$ Updates:** Because $d = 8$, updating $\mathbf{A}_a \leftarrow \mathbf{A}_a + \mathbf{x}_t \mathbf{x}_t^\top$ and inverting an $8 \times 8$ matrix takes microseconds—requiring no GPU or batch gradient descent.

### Q3: How did you design the Bandit's composite reward function $r_t$?
* **Answer:** If you reward only quality, the bandit routes 100% of traffic to the Frontier model (`claude-3-5-sonnet`). If you reward only cost, it routes 100% to the smallest model. We formulated a multi-objective scalar reward in $[0, 1]$:
  $$r_t = 0.74 \cdot Q_t + 0.26 \cdot \left(1 - \frac{\text{Cost}_a}{\text{Cost}_{\max}}\right) - 1.35 \cdot \max(0, 0.78 - Q_t) - \text{LatencyPenalty}$$
  Notice the **hinge penalty** $\max(0, 0.78 - Q_t)$: if a cheap model produces a low-quality answer below `0.78`, the hinge penalty sharply penalizes the reward so the bandit learns never to route complex reasoning tasks to the small model.

---

## 4. Deep-Dive Questions from SWE / Systems Interviewers

### Q1: How do you prevent Shadow Mode evaluations from slowing down the user's HTTP response?
* **Answer:** In [`app/main.py`](app/main.py), once the primary model completes (or begins streaming), shadow evaluations are scheduled on the event loop via `asyncio.create_task(shadow_evaluator.run_shadow_evaluation(...))` outside the critical request path. In a multi-worker deployment, this task can be pushed to a Redis Stream / Celery queue so gateway workers remain 100% CPU-bound only on proxying.

### Q2: How does your Stateful Agent Loop Circuit Breaker work?
* **Answer:** Autonomous agents frequently get stuck calling the same tool with identical arguments when an external resource fails (burning thousands of tokens). In [`app/guardrails/loop_detector.py`](app/guardrails/loop_detector.py), we normalize tool call AST/JSON signatures (`tool:name({"arg": "val"})` with sorted keys) and compute semantic similarity over a sliding window of the last 6 turns per `session_id`. When `repeat_count >= 3`, the gateway trips the circuit breaker: it appends a high-priority system message instructing the model not to repeat the failing signature and forces the Bandit Router to escalate that turn to the Frontier model.

### Q3: How does the Per-Tenant Rate Limiter work and how would you scale it across multiple gateway nodes?
* **Answer:** [`app/infra/rate_limiter.py`](app/infra/rate_limiter.py) implements an $O(1)$ lazy-refill Token Bucket algorithm using monotonic timestamps (`time.monotonic()`). Each tenant has a burst capacity $C$ and continuous refill rate $r$ tokens/sec. To scale horizontally across $N$ stateless gateway pods, the exact same math (`elapsed * refill_rate`, `min(capacity, tokens)`, and decrement) is executed atomically inside a single **Redis Lua script (`EVALSHA`)** so there are no race conditions across pods.
