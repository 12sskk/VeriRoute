"""Disjoint LinUCB Contextual Multi-Armed Bandit Router.

Implements the Linear Upper Confidence Bound (LinUCB) algorithm (Li et al., WWW 2010)
augmented with real-time Exponential Moving Average (EMA) Time-To-First-Token (TTFT)
latency penalties and SQLite persistence.

For each candidate LLM arm a in A:
  - A_a in R^{d x d}: Ridge covariance matrix (initialized to I_d + warm prior)
  - b_a in R^d: Cumulative reward-weighted context vector
  - theta_hat_a = A_a^{-1} b_a
  - UCB_a(x_t) = theta_hat_a^T x_t + alpha * sqrt(x_t^T A_a^{-1} x_t) - lambda_lat * NormalizedTTFT_a

Reward function r_t in [0, 1]:
  Combines response quality q_t in [0, 1] with token cost efficiency:
  r_t = 0.72 * q_t + 0.28 * (1.0 - cost_ratio_a) - 0.15 * max(0, 0.75 - q_t)
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Optional
import numpy as np

from app.config import DEFAULT_MODEL_ARMS, ModelArmSpec
from app.models import BanditArmScore, ChatMessage, RouterDecisionMetadata
from app.router.features import FEATURE_NAMES, ExtractedPromptContext, extract_prompt_context


class ArmState:
    """Maintains ridge regression parameters (A_a, b_a) and live telemetry for one LLM arm."""

    def __init__(self, spec: ModelArmSpec, dim: int = 8) -> None:
        self.spec = spec
        self.dim = dim
        self.A: np.ndarray = np.eye(dim, dtype=np.float64)
        self.b: np.ndarray = np.zeros(dim, dtype=np.float64)
        self.ema_ttft_ms: float = spec.base_ttft_ms
        self.pulls: int = 0
        self.total_reward: float = 0.0
        self.total_quality: float = 0.0
        self.total_cost_usd: float = 0.0

    @property
    def theta_hat(self) -> np.ndarray:
        return np.linalg.solve(self.A, self.b)


class LinUCBBanditRouter:
    """Online Contextual Multi-Armed Bandit for Cost-Quality-Latency Optimal LLM Routing."""

    def __init__(
        self,
        arms: Optional[list[ModelArmSpec]] = None,
        alpha: float = 0.85,
        latency_penalty_weight: float = 0.15,
        db_path: Optional[str] = None,
        warm_start: bool = True,
    ) -> None:
        self.arm_specs = arms or DEFAULT_MODEL_ARMS
        self.dim = len(FEATURE_NAMES)
        self.alpha = alpha
        self.latency_penalty_weight = latency_penalty_weight
        self.db_path = db_path
        self.arms: dict[str, ArmState] = {
            spec.arm_id: ArmState(spec, dim=self.dim) for spec in self.arm_specs
        }
        self.max_cost_rate = max(s.cost_per_1k_output for s in self.arm_specs)
        self.total_routing_decisions: int = 0
        self.reward_history: list[dict] = []

        if self.db_path:
            self._init_db()
            loaded = self._load_from_db()
            if not loaded and warm_start:
                self._apply_domain_priors()
                self._save_to_db()
        elif warm_start:
            self._apply_domain_priors()

    def _apply_domain_priors(self) -> None:
        """Seed mild ridge regression priors so cold-start routing respects basic capability tiers
        while leaving high uncertainty for online exploration."""
        # Feature order:
        # [0:bias, 1:simplicity, 2:length, 3:code, 4:reasoning_math, 5:systems_arch, 6:constraint, 7:conv_depth]
        prior_profiles = {
            "llama-3.1-8b-instant": [
                (np.array([1.0, 0.90, 0.15, 0.10, 0.0, 0.0, 0.0, 0.1]), 0.92),
                (np.array([1.0, 0.05, 0.70, 0.40, 0.85, 0.75, 0.4, 0.3]), 0.32),
            ],
            "gpt-4o-mini": [
                (np.array([1.0, 0.50, 0.45, 0.65, 0.30, 0.30, 0.5, 0.3]), 0.84),
                (np.array([1.0, 0.10, 0.70, 0.50, 0.80, 0.80, 0.5, 0.4]), 0.64),
            ],
            "claude-3-5-sonnet": [
                (np.array([1.0, 0.90, 0.15, 0.10, 0.0, 0.0, 0.0, 0.1]), 0.68),  # penalized by high cost on trivial tasks
                (np.array([1.0, 0.05, 0.75, 0.65, 0.90, 0.85, 0.6, 0.4]), 0.91),  # highest reward on complex reasoning/arch
            ],
        }
        for arm_id, samples in prior_profiles.items():
            if arm_id not in self.arms:
                continue
            state = self.arms[arm_id]
            for raw_x, reward in samples:
                x = raw_x / np.linalg.norm(raw_x)
                # Multiply by 2 virtual pulls to establish initial slope
                for _ in range(2):
                    state.A += np.outer(x, x)
                    state.b += reward * x

    def reset(self, warm_start: bool = False) -> None:
        """Reset all bandit matrices to identity I_d (or warm priors) for live experimentation."""
        self.arms = {spec.arm_id: ArmState(spec, dim=self.dim) for spec in self.arm_specs}
        self.total_routing_decisions = 0
        self.reward_history.clear()
        if warm_start:
            self._apply_domain_priors()
        if self.db_path:
            self._save_to_db()

    def compute_composite_reward(
        self,
        arm_id: str,
        quality_score: float,
        latency_ms: Optional[float] = None,
    ) -> float:
        """Compute scalar bandit reward r_t balancing quality, token cost savings, and SLA."""
        spec = self.arms[arm_id].spec
        cost_ratio = spec.cost_per_1k_output / max(1e-9, self.max_cost_rate)
        cost_efficiency = 1.0 - cost_ratio

        # Heavy penalty if quality falls below acceptable threshold (0.75)
        quality_shortfall = max(0.0, 0.78 - quality_score) * 1.35

        latency_term = 0.0
        if latency_ms is not None:
            latency_term = min(0.15, (latency_ms / 1500.0) * 0.10)

        reward = (0.74 * quality_score) + (0.26 * cost_efficiency) - quality_shortfall - latency_term
        return float(np.clip(reward, 0.0, 1.0))

    def select_arm(
        self,
        messages: list[ChatMessage],
        requested_model: str = "auto",
        escalate_to_frontier: bool = False,
    ) -> tuple[ModelArmSpec, ExtractedPromptContext, RouterDecisionMetadata]:
        """Score all arms using LinUCB and return the optimal arm + full telemetry."""
        t0 = time.perf_counter()
        ctx = extract_prompt_context(messages)
        x = ctx.vector
        self.total_routing_decisions += 1

        arm_scores: list[BanditArmScore] = []
        best_arm_id = self.arm_specs[0].arm_id
        best_ucb = -1e9

        for arm_id, state in self.arms.items():
            A_inv = np.linalg.inv(state.A)
            theta_hat = A_inv @ state.b
            expected_reward = float(theta_hat @ x)
            effective_alpha = self.alpha / np.sqrt(1.0 + state.pulls / 6.0)
            uncertainty = float(effective_alpha * np.sqrt(max(0.0, float(x.T @ A_inv @ x))))
            norm_ttft = min(1.0, state.ema_ttft_ms / 600.0)
            lat_penalty = float(self.latency_penalty_weight * norm_ttft)
            ucb_val = expected_reward + uncertainty - lat_penalty

            est_cost = (
                (ctx.estimated_prompt_tokens / 1000.0) * state.spec.cost_per_1k_input
                + (220.0 / 1000.0) * state.spec.cost_per_1k_output
            )
            arm_scores.append(
                BanditArmScore(
                    arm_id=arm_id,
                    display_name=state.spec.display_name,
                    tier=state.spec.tier,
                    expected_reward=round(expected_reward, 4),
                    uncertainty_bonus=round(uncertainty, 4),
                    latency_penalty=round(lat_penalty, 4),
                    ucb_score=round(ucb_val, 4),
                    ema_ttft_ms=round(state.ema_ttft_ms, 2),
                    estimated_cost_usd=round(est_cost, 6),
                )
            )
            if ucb_val > best_ucb:
                best_ucb = ucb_val
                best_arm_id = arm_id

        # Check overrides
        if escalate_to_frontier:
            selected_spec = next(
                (s for s in self.arm_specs if s.tier == "frontier"),
                self.arms[best_arm_id].spec,
            )
            policy = "loop_breaker_escalation"
        elif requested_model != "auto" and requested_model in self.arms:
            selected_spec = self.arms[requested_model].spec
            policy = "explicit_override"
        else:
            selected_spec = self.arms[best_arm_id].spec
            policy = "linucb_bandit"

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        decision = RouterDecisionMetadata(
            selected_model=selected_spec.arm_id,
            selected_tier=selected_spec.tier,
            routing_policy=policy,
            complexity_score=ctx.complexity_score,
            detected_domain=ctx.detected_domain,
            context_features=ctx.feature_dict,
            arm_scores=arm_scores,
            router_latency_ms=round(elapsed_ms, 3),
        )
        return selected_spec, ctx, decision

    def update_arm(
        self,
        arm_id: str,
        context_vector: np.ndarray,
        quality_score: float,
        latency_ms: Optional[float] = None,
        cost_usd: float = 0.0,
        signal_source: str = "online_eval",
    ) -> float:
        """Perform online ridge regression update: A_a += x x^T, b_a += r_t x."""
        if arm_id not in self.arms:
            return 0.0

        state = self.arms[arm_id]
        x = context_vector.astype(np.float64)
        reward = self.compute_composite_reward(arm_id, quality_score, latency_ms)

        state.A += np.outer(x, x)
        state.b += reward * x
        state.pulls += 1
        state.total_reward += reward
        state.total_quality += quality_score
        state.total_cost_usd += cost_usd

        if latency_ms is not None and latency_ms > 0:
            state.ema_ttft_ms = 0.85 * state.ema_ttft_ms + 0.15 * latency_ms

        self.reward_history.append(
            {
                "timestamp": time.time(),
                "arm_id": arm_id,
                "quality_score": round(quality_score, 4),
                "composite_reward": round(reward, 4),
                "signal_source": signal_source,
            }
        )
        if len(self.reward_history) > 100:
            self.reward_history.pop(0)

        if self.db_path:
            self._save_to_db()

        return reward

    def get_diagnostics(self) -> dict:
        """Return full bandit state including learned feature weights theta_hat per arm."""
        arms_info = {}
        for arm_id, state in self.arms.items():
            theta = state.theta_hat
            arms_info[arm_id] = {
                "display_name": state.spec.display_name,
                "tier": state.spec.tier,
                "pulls": state.pulls,
                "avg_reward": round(state.total_reward / max(1, state.pulls), 4),
                "avg_quality": round(state.total_quality / max(1, state.pulls), 4),
                "total_cost_usd": round(state.total_cost_usd, 6),
                "ema_ttft_ms": round(state.ema_ttft_ms, 2),
                "learned_theta": {
                    FEATURE_NAMES[i]: round(float(theta[i]), 4) for i in range(self.dim)
                },
            }
        return {
            "algorithm": "Disjoint LinUCB (Linear Upper Confidence Bound)",
            "alpha": self.alpha,
            "latency_penalty_weight": self.latency_penalty_weight,
            "total_decisions": self.total_routing_decisions,
            "arms": arms_info,
            "recent_updates": self.reward_history[-15:],
        }

    def _init_db(self) -> None:
        if not self.db_path:
            return
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bandit_arms (
                    arm_id TEXT PRIMARY KEY,
                    matrix_a TEXT NOT NULL,
                    vector_b TEXT NOT NULL,
                    ema_ttft_ms REAL NOT NULL,
                    pulls INTEGER NOT NULL,
                    total_reward REAL NOT NULL,
                    total_quality REAL NOT NULL,
                    total_cost_usd REAL NOT NULL
                )
                """
            )

    def _save_to_db(self) -> None:
        if not self.db_path:
            return
        try:
            with sqlite3.connect(self.db_path) as conn:
                for arm_id, state in self.arms.items():
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO bandit_arms
                        (arm_id, matrix_a, vector_b, ema_ttft_ms, pulls, total_reward, total_quality, total_cost_usd)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            arm_id,
                            json.dumps(state.A.tolist()),
                            json.dumps(state.b.tolist()),
                            state.ema_ttft_ms,
                            state.pulls,
                            state.total_reward,
                            state.total_quality,
                            state.total_cost_usd,
                        ),
                    )
        except Exception:
            pass

    def _load_from_db(self) -> bool:
        if not self.db_path or not Path(self.db_path).exists():
            return False
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT arm_id, matrix_a, vector_b, ema_ttft_ms, pulls, total_reward, total_quality, total_cost_usd FROM bandit_arms"
                ).fetchall()
                if not rows:
                    return False
                for row in rows:
                    arm_id = row[0]
                    if arm_id in self.arms:
                        state = self.arms[arm_id]
                        state.A = np.array(json.loads(row[1]), dtype=np.float64)
                        state.b = np.array(json.loads(row[2]), dtype=np.float64)
                        state.ema_ttft_ms = float(row[3])
                        state.pulls = int(row[4])
                        state.total_reward = float(row[5])
                        state.total_quality = float(row[6])
                        state.total_cost_usd = float(row[7])
                return True
        except Exception:
            return False
