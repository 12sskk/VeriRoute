"""Unit tests for Disjoint LinUCB Contextual Bandit Router."""

from app.models import ChatMessage
from app.router.bandit import LinUCBBanditRouter
from app.router.features import extract_prompt_context


def test_bandit_routes_easy_vs_hard_prompts():
    router = LinUCBBanditRouter(alpha=0.5, db_path=None, warm_start=True)

    easy_msgs = [ChatMessage(role="user", content="What does HTTP stand for? Explain in one sentence.")]
    spec_easy, ctx_easy, meta_easy = router.select_arm(easy_msgs)
    assert ctx_easy.complexity_score < 0.35
    assert spec_easy.arm_id in ("llama-3.1-8b-instant", "gpt-4o-mini")

    hard_msgs = [
        ChatMessage(
            role="user",
            content=(
                "Prove by strong induction and derive the exact amortized time complexity O(n log n) "
                "for a distributed Raft consensus state machine with formal verification invariants."
            ),
        )
    ]
    spec_hard, ctx_hard, meta_hard = router.select_arm(hard_msgs)
    assert ctx_hard.complexity_score > 0.60
    assert spec_hard.arm_id == "claude-3-5-sonnet"


def test_online_bandit_uncertainty_shrinks_and_weights_adapt(tmp_path):
    db_file = str(tmp_path / "test_bandit.db")
    router = LinUCBBanditRouter(alpha=0.85, db_path=db_file, warm_start=False)

    msgs = [ChatMessage(role="user", content="Write a Python function to parse JSON logs.")]
    _, ctx, meta_before = router.select_arm(msgs)
    initial_bonus = next(a.uncertainty_bonus for a in meta_before.arm_scores if a.arm_id == "gpt-4o-mini")

    # Feed 10 high-reward updates to gpt-4o-mini
    for _ in range(10):
        router.update_arm(
            arm_id="gpt-4o-mini",
            context_vector=ctx.vector,
            quality_score=0.96,
            latency_ms=110.0,
            cost_usd=0.0002,
        )

    _, _, meta_after = router.select_arm(msgs)
    updated_score = next(a for a in meta_after.arm_scores if a.arm_id == "gpt-4o-mini")
    assert updated_score.uncertainty_bonus < initial_bonus
    assert updated_score.expected_reward > 0.65

    # Verify SQLite persistence reloads state accurately
    reloaded_router = LinUCBBanditRouter(alpha=0.85, db_path=db_file, warm_start=False)
    assert reloaded_router.arms["gpt-4o-mini"].pulls == 10
