"""Unit tests for Two-Stage Entailment-Verified Semantic Cache."""

from app.cache.semantic_cache import EntailmentSemanticCache


def test_paraphrase_hits_cache():
    cache = EntailmentSemanticCache(cosine_threshold=0.80)
    cache.store(
        query="How do I enable CORS in FastAPI?",
        response_text="Use CORSMiddleware in FastAPI.",
        model_used="claude-3-5-sonnet",
    )

    entry, meta = cache.lookup("What is the way to turn on cross-origin in FastAPI?")
    assert entry is not None
    assert meta.cache_hit is True
    assert meta.false_positive_prevented is False
    assert meta.stage2_nli is not None
    assert meta.stage2_nli.label == "ENTAILMENT"


def test_antonym_and_negation_blocked():
    cache = EntailmentSemanticCache(cosine_threshold=0.80)
    cache.store(
        query="How do I enable CORS in FastAPI?",
        response_text="Use CORSMiddleware in FastAPI.",
        model_used="claude-3-5-sonnet",
    )

    # 1. Antonym flip ("disable" vs "enable")
    entry, meta = cache.lookup("How do I disable CORS in FastAPI?")
    assert entry is None
    assert meta.cache_hit is False
    assert meta.vanilla_cosine_would_hit is True
    assert meta.false_positive_prevented is True
    assert meta.stage2_nli is not None
    assert meta.stage2_nli.label == "CONTRADICTION"

    # 2. Explicit negation ("not enable")
    entry2, meta2 = cache.lookup("How do I not enable CORS in FastAPI?")
    assert entry2 is None
    assert meta2.cache_hit is False
    assert meta2.false_positive_prevented is True


def test_number_and_entity_swaps_blocked():
    cache = EntailmentSemanticCache(cosine_threshold=0.80)
    cache.store(
        query="Transfer $500 to Alice",
        response_text="Wire transfer of $500 to Alice prepared.",
        model_used="claude-3-5-sonnet",
    )

    # Number swap ($5000 vs $500)
    entry_num, meta_num = cache.lookup("Transfer $5000 to Alice")
    assert entry_num is None
    assert meta_num.cache_hit is False
    assert meta_num.vanilla_cosine_would_hit is True
    assert meta_num.false_positive_prevented is True
    assert meta_num.stage2_nli.constraint_report.numbers_match is False

    # Entity swap (Bob vs Alice)
    entry_ent, meta_ent = cache.lookup("Transfer $500 to Bob")
    assert entry_ent is None
    assert meta_ent.cache_hit is False
    assert meta_ent.vanilla_cosine_would_hit is True
    assert meta_ent.false_positive_prevented is True
    assert meta_ent.stage2_nli.constraint_report.entities_match is False

    # Safe paraphrase ("Please wire $500 to Alice")
    entry_ok, meta_ok = cache.lookup("Please wire $500 to Alice")
    assert entry_ok is not None
    assert meta_ok.cache_hit is True


def test_directional_role_reversal_blocked():
    cache = EntailmentSemanticCache(cosine_threshold=0.80)
    cache.store(
        query="Transfer funds from checking to savings",
        response_text="Transferred from checking to savings.",
        model_used="gpt-4o-mini",
    )

    entry, meta = cache.lookup("Transfer funds from savings to checking")
    assert entry is None
    assert meta.cache_hit is False
    assert meta.false_positive_prevented is True
