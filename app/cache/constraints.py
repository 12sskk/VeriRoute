"""Hard Constraint & Structural Guard for Semantic Caching.

Extracts deterministic invariants that dense embeddings frequently conflate:
1. Numerical quantities, versions, percentages, and currency amounts ($500 vs $50).
2. Proper nouns, named entities, code identifiers, and target arguments (Alice vs Bob).
3. Negation polarity markers (not, never, without, disable vs enable).
4. Antonym / directional opposition pairs (ascending vs descending, encrypt vs decrypt).
"""

from __future__ import annotations

import re
from app.models import Stage1ConstraintReport

# Directional / functional antonyms where swapping one word flips the entire meaning
ANTONYM_PAIRS: list[tuple[set[str], set[str]]] = [
    ({"enable", "enabling", "activate", "activating", "turn on", "switch on", "allow", "permit"},
     {"disable", "disabling", "deactivate", "deactivating", "turn off", "switch off", "block", "forbid", "prevent"}),
    ({"encrypt", "encrypting", "encode", "encoding", "lock"},
     {"decrypt", "decrypting", "decode", "decoding", "unlock"}),
    ({"ascending", "asc", "smallest first", "lowest first", "oldest first"},
     {"descending", "desc", "largest first", "highest first", "newest first"}),
    ({"increase", "increasing", "raise", "maximize", "expand", "increment", "up"},
     {"decrease", "decreasing", "lower", "minimize", "shrink", "decrement", "down"}),
    ({"create", "creating", "add", "adding", "insert", "install"},
     {"delete", "deleting", "remove", "removing", "drop", "dropping", "uninstall", "destroy"}),
    ({"upload", "uploading", "export", "exporting", "push", "send"},
     {"download", "downloading", "import", "importing", "pull", "receive"}),
    ({"public", "external", "open", "mutable", "synchronous", "sync", "stateful", "frontend", "client"},
     {"private", "internal", "closed", "immutable", "asynchronous", "async", "stateless", "backend", "server"}),
    ({"before", "pre", "prefix", "start", "begin", "first", "min", "minimum"},
     {"after", "post", "suffix", "end", "stop", "last", "max", "maximum"}),
]

NEGATION_TOKENS = {
    "not", "no", "never", "none", "neither", "nor", "without", "dont", "don't",
    "doesnt", "doesn't", "cant", "can't", "cannot", "wont", "won't", "shouldnt",
    "shouldn't", "avoid", "skip", "exclude", "except",
}

# Words that are common question starters or non-entity verbs/stopwords
NON_ENTITY_WORDS = {
    "how", "what", "when", "where", "why", "who", "which", "can", "could", "would",
    "should", "will", "do", "does", "did", "is", "are", "was", "were", "please",
    "explain", "tell", "show", "write", "give", "help", "make", "build", "create",
    "enable", "disable", "activate", "deactivate", "turn", "switch", "allow", "block",
    "transfer", "send", "wire", "pay", "remit", "delete", "remove", "drop", "increase",
    "decrease", "sort", "order", "encrypt", "decrypt", "upload", "download",
    "the", "a", "an", "in", "on", "at", "to", "for", "from", "with", "by", "of", "and",
    "or", "if", "then", "into", "using", "use", "best", "way", "simple", "fast",
    "quickly", "easily", "safely", "properly", "correctly", "example", "code", "function",
    "method", "script", "query", "request", "response", "file", "data", "system",
    "service", "application", "app", "user", "account", "money", "funds", "dollars",
    "usd", "amount", "table", "column", "row", "list", "array", "string", "number",
    "value", "key", "token", "setting", "feature", "option", "mode", "error", "issue",
    "bug", "problem", "solution", "fix", "solve", "debug", "configure", "setup",
    "not", "never", "without", "dont", "cannot", "avoid", "ascending", "descending",
    "my", "your", "our", "their", "me", "you", "us", "them", "it", "this", "that",
}

TECH_AND_DOMAIN_ENTITIES = {
    "fastapi", "flask", "django", "express", "nextjs", "react", "vue", "angular",
    "postgres", "postgresql", "mysql", "sqlite", "mongodb", "redis", "qdrant", "kafka",
    "docker", "kubernetes", "aws", "gcp", "azure", "linux", "windows", "macos",
    "python", "javascript", "typescript", "rust", "golang", "java", "cpp", "csharp",
    "jwt", "oauth", "saml", "cors", "csrf", "xss", "sql", "graphql", "grpc", "rest",
    "openai", "anthropic", "claude", "llama", "gemini", "groq", "onnx", "pytorch", "tensorflow",
}

TECH_SYNONYM_MAP = {
    "postgresql": "postgres",
    "k8s": "kubernetes",
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "golang": "go",
    "cross-origin": "cors",
    "cross_origin": "cors",
}


def extract_numbers(text: str) -> list[str]:
    """Extract normalized numerical quantities (including decimals and currency)."""
    raw_matches = re.findall(r"(?<![a-zA-Z])\$?\d+(?:,\d{3})*(?:\.\d+)?%?(?![a-zA-Z])", text)
    normalized: list[str] = []
    for m in raw_matches:
        clean = m.replace(",", "").lstrip("$")
        normalized.append(clean)
    return sorted(normalized)


def extract_entities(text: str) -> list[str]:
    """Extract proper nouns (e.g. Alice, Bob), code symbols, and technical framework identifiers."""
    entities: set[str] = set()
    normalized_text = text.replace("cross-origin", "cors").replace("cross origin", "cors")

    # 1. Backtick or quoted literals
    for match in re.findall(r"`([^`]+)`|'([^']+)'|\"([^\"]+)\"", normalized_text):
        literal = next((m for m in match if m), "").strip().lower()
        if literal and len(literal) <= 32:
            entities.add(TECH_SYNONYM_MAP.get(literal, literal))

    # 2. Capitalized proper nouns mid-sentence (e.g., Alice, Bob, Stripe, GitHub)
    words_with_pos = list(re.finditer(r"\b([A-Za-z][a-zA-Z0-9_-]{1,})\b", normalized_text.strip()))
    for idx, match in enumerate(words_with_pos):
        word = match.group(1)
        if not word[0].isupper():
            continue
        lower = word.lower()
        canonical = TECH_SYNONYM_MAP.get(lower, lower)
        if canonical in TECH_AND_DOMAIN_ENTITIES:
            entities.add(canonical)
            continue
        # Only treat non-sentence-initial capitalized words (idx > 0) as proper noun entities
        if idx > 0 and canonical not in NON_ENTITY_WORDS:
            entities.add(canonical)

    # 3. Lowercase technical entities
    for tok in re.findall(r"\b[a-z0-9_+-]+\b", normalized_text.lower()):
        canonical = TECH_SYNONYM_MAP.get(tok, tok)
        if canonical in TECH_AND_DOMAIN_ENTITIES:
            entities.add(canonical)

    return sorted(entities)


def has_negation(text: str) -> bool:
    """Return True if the query contains explicit negation polarity (mod 2)."""
    tokens = re.findall(r"[a-z']+|\bnot\b", text.lower())
    neg_count = sum(1 for t in tokens if t in NEGATION_TOKENS)
    return (neg_count % 2) == 1


def detect_antonym_conflict(query_a: str, query_b: str) -> tuple[str, str] | None:
    """Detect if query_a and query_b contain opposing directional/action antonyms."""
    lower_a = f" {query_a.lower()} "
    lower_b = f" {query_b.lower()} "

    def _matches_phrase(text: str, phrase: str) -> bool:
        return bool(re.search(rf"\b{re.escape(phrase)}\b", text))

    for set_pos, set_neg in ANTONYM_PAIRS:
        a_pos = next((w for w in set_pos if _matches_phrase(lower_a, w)), None)
        a_neg = next((w for w in set_neg if _matches_phrase(lower_a, w)), None)
        b_pos = next((w for w in set_pos if _matches_phrase(lower_b, w)), None)
        b_neg = next((w for w in set_neg if _matches_phrase(lower_b, w)), None)

        if a_pos and b_neg:
            return (a_pos, b_neg)
        if a_neg and b_pos:
            return (a_neg, b_pos)
    return None


def verify_hard_constraints(query: str, candidate: str) -> Stage1ConstraintReport:
    """Verify deterministic invariants between incoming query and cached candidate."""
    nums_q = extract_numbers(query)
    nums_c = extract_numbers(candidate)
    numbers_match = nums_q == nums_c

    ent_q = extract_entities(query)
    ent_c = extract_entities(candidate)
    entities_match = ent_q == ent_c

    neg_q = has_negation(query)
    neg_c = has_negation(candidate)
    antonym_conflict = detect_antonym_conflict(query, candidate)

    # Note: "not enable" and "disable" have flipped negation AND flipped antonym -> net match!
    # Wait: if one has negation and the other doesn't, AND there is no compensating antonym flip, polarity mismatches.
    if antonym_conflict is not None and (neg_q == neg_c):
        negation_polarity_match = False
    elif antonym_conflict is None and (neg_q != neg_c):
        negation_polarity_match = False
        antonym_conflict = ("negated" if neg_q else "affirmative", "negated" if neg_c else "affirmative")
    else:
        negation_polarity_match = True
        antonym_conflict = None

    passed = numbers_match and entities_match and negation_polarity_match
    reasons: list[str] = []
    if not numbers_match:
        reasons.append(f"numerical_mismatch({nums_q} != {nums_c})")
    if not entities_match:
        reasons.append(f"entity_mismatch({ent_q} != {ent_c})")
    if not negation_polarity_match:
        reasons.append(f"polarity_or_antonym_conflict({antonym_conflict})")

    return Stage1ConstraintReport(
        numbers_query=nums_q,
        numbers_candidate=nums_c,
        numbers_match=numbers_match,
        entities_query=ent_q,
        entities_candidate=ent_c,
        entities_match=entities_match,
        negation_polarity_match=negation_polarity_match,
        antonym_conflict=antonym_conflict,
        passed_hard_constraints=passed,
        rejection_reason="; ".join(reasons) if reasons else None,
    )
