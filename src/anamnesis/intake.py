from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class IntakeDecision:
    action: str  # accept | inbox | reject
    lifecycle: str
    reasons: list[str]
    confidence: float


_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")


_LOW_VALUE_CHAT_FRAGMENT_PATTERNS = [
    r"^\s*(?:\[(?:user|assistant)\]\s*)?(?:ah\s+)?(?:ok|okay|k)(?:\s+(?:go ahead|sure|lah|la|leh))*[.!\s]*$",
    r"^\s*(?:\[(?:user|assistant)\]\s*)?(?:ok(?:ay)?\s+)?go ahead(?:\s+(?:la|lah|leh))?[.!\s]*$",
    r"^\s*(?:\[(?:user|assistant)\]\s*)?(?:please\s+)?(?:fix it|carry on|continue|proceed|go)(?:\s+(?:la|lah|leh))?[.!\s]*$",
    r"^\s*(?:\[(?:user|assistant)\]\s*)?(?:please\s+)?go and do what u need.*$",
    r"^\s*(?:\[(?:user|assistant)\]\s*)?now what[.!?\s]*$",
    r"^\s*(?:\[(?:user|assistant)\]\s*)?carry on\b.*$",
    r"^\s*(?:\[(?:user|assistant)\]\s*)?ok\s+do it\b.*$",
    r"^\s*(?:\[(?:user|assistant)\]\s*)?(?:what|why|how|is|are|do|does|did|can|could|would|should)\b.*$",
    r"^\s*\d+\.\s+.*\b(how|what|why|is|are|do|does|did|can|could|would|should)\b.*$",
    r"^\s*(?:\[(?:user|assistant)\]\s*)?u didn't really answer\b.*$",
    r"^\s*(?:\[(?:user|assistant)\]\s*)?(?:thanks|thank you|thx)[.!\s]*$",
    r"^\s*(?:\[(?:user|assistant)\]\s*)?(?:yes|yep|yeah|no|nope)[.!\s]*$",
]

_TEMPORARY_PATTERNS = [
    r"\[assistant\]",
    r"\[user\]",
    r"review the conversation above",
    r"initiated the conversation with a greeting",
    r"nothing was worth saving",
    r"\bpid\s*[:#]?\s*\d+\b",
    r"\bport\s+\d+\b.*\b(listening|started|running)\b",
    r"\b(currently|now)\s+(running|stuck|wedged|listening)\b",
    r"\bcurrently\b.*\b(port|endpoint|localhost|server)\b",
    r"\b(stuck|wedged)\b.*\b(lock|pid|cron|process|job)\b",
    r"\btemporary\b.*\b(process|server|manual|task|port)\b",
    r"\bphase\s+\d+\s+(completed|done)\b",
    r"\bcommit\s+[0-9a-f]{6,40}\b",
    r"\bpr\s*#?\d+\b",
    r"\bissue\s*#?\d+\b",
    r"\b(question|what|why|how|where|when|who)\b.*\?",
    r"\bu mean\b.*\?",
    r"\bwhy\b.*\b(owner|domain|autopilot|memory|profile|db)\b",
    r"\bwhat is\b.*\b(autopilot|owner|domain|memory|profile|db)\b",
]

_SENSITIVE_PATTERNS = [
    r"\b(password|passcode|otp|one[-\s]?time password|2fa|mfa|api key|secret key|token|credential)\b",
    r"\b(credit card|card number|cvv|nric|passport|ssn|medical|diagnosis)\b",
]

_PLATFORM_LOCAL_PATTERNS = [
    r"\b(only|just)\s+(on|in|for)\s+(whatsapp|telegram|cli|discord|slack)\b",
    r"\b(whatsapp|telegram|cli|discord|slack)[-\s]?only\b",
    r"\b(platform[-\s]?local|chat[-\s]?local|this platform only|only here)\b",
]

_DURABLE_PATTERNS = [
    r"\b(prefers|expects|wants|requires|must|should|includes|named|wife|helper|uses|forbids)\b",
    r"\b(cannot|can not|can|allowed|forbidden|no access|access)\b",
    r"\b(local-only|private-network|private|privacy|permission|policy)\b",
    r"\b(project|uses|runs on|dashboard|service|provider)\b",
    r"\b(timezone|macos username|username|exit by age|exit goal|case study)\b",
    r"\b(c2pa|synthid|fal|grpc|tailscale|llm vision|skill\(s\) not found|script|metadata)\b",
    r"\b(author style|handle-only|draft review|auto-post|confidence/risk)\b",
]

_AMBIGUOUS_PATTERNS = [
    r"\b(maybe|seemed|might|perhaps|possibly|should be able)\b",
    r"\b(if .* says okay|if .* approves)\b",
]

_STABLE_INFRA_PATTERNS = [
    r"\b(runs on|default port|dashboard|service|provider|production|integration|grpc|access prompt|pitfall)\b",
    r"\b(127\.0\.0\.1|lan host|webhook endpoint)\b",
]

_OPERATIONAL_IDENTIFIER_PATTERNS = [
    r"\bport\s+\d+\b",
    r"\bendpoint\b",
    r"\bhttps?://",
    r"\b127\.0\.0\.1\b",
    r"\blocalhost\b",
    r"\bgrpc\b",
]

_STABLE_OPERATIONAL_DOMAINS = {
    "infrastructure",
    "infra",
    "project",
    "system",
    "systems",
    "devops",
    "mlops",
    "integration",
    "integrations",
}


def classify_intake(text: str, *, domain: str = "", user_is_technical: bool = False) -> IntakeDecision:
    normalized = " ".join((text or "").split())
    low = normalized.lower()
    if not low:
        return IntakeDecision(
            action="reject", lifecycle="empty", reasons=["empty"], confidence=1.0
        )

    if _matches_any(low, _TEMPORARY_PATTERNS):
        # Explicit temporary/current-process markers beat operational anchors.
        return IntakeDecision(
            action="reject",
            lifecycle="temporary",
            reasons=["temporary_task_state"],
            confidence=0.92,
        )

    if _matches_any(low, _LOW_VALUE_CHAT_FRAGMENT_PATTERNS):
        return IntakeDecision(
            action="reject",
            lifecycle="low_value_chat_fragment",
            reasons=["low_value_chat_fragment"],
            confidence=0.96,
        )

    if _matches_any(low, _SENSITIVE_PATTERNS):
        return IntakeDecision(
            action="inbox",
            lifecycle="sensitive",
            reasons=["sensitive_content"],
            confidence=0.7,
        )

    if _matches_any(low, _AMBIGUOUS_PATTERNS):
        return IntakeDecision(
            action="inbox",
            lifecycle="ambiguous",
            reasons=["ambiguous_or_sensitive"],
            confidence=0.65,
        )

    operational_identifier = _matches_any(low, _OPERATIONAL_IDENTIFIER_PATTERNS)
    stable_operational_context = _has_stable_operational_context(
        low, domain=domain, user_is_technical=user_is_technical
    )

    if operational_identifier and stable_operational_context:
        return IntakeDecision(
            action="accept",
            lifecycle="stable_infrastructure",
            reasons=["durable_operational_evidence"],
            confidence=0.82,
        )

    if operational_identifier and not stable_operational_context:
        return IntakeDecision(
            action="inbox",
            lifecycle="operational_identifier_needs_context",
            reasons=["operational_identifier_without_durable_context"],
            confidence=0.58,
        )

    if _matches_any(low, _DURABLE_PATTERNS):
        lifecycle = (
            "stable_infrastructure"
            if _matches_any(low, _STABLE_INFRA_PATTERNS)
            else "durable"
        )
        return IntakeDecision(
            action="accept",
            lifecycle=lifecycle,
            reasons=["durable_signal"],
            confidence=0.82,
        )

    return IntakeDecision(
        action="inbox",
        lifecycle="unknown",
        reasons=["no_clear_lifecycle_signal"],
        confidence=0.5,
    )


def is_platform_local_text(text: str) -> bool:
    normalized = " ".join((text or "").split()).lower()
    return _matches_any(normalized, _PLATFORM_LOCAL_PATTERNS)


def extract_durable_facts(text: str) -> list[str]:
    """Extract durable sentence-level facts from mixed transcript/task-state text."""
    facts: list[str] = []
    seen: set[str] = set()
    for match in _SENTENCE_RE.finditer(text or ""):
        sentence = _clean_sentence(match.group(0))
        if not sentence:
            continue
        decision = classify_intake(sentence)
        if decision.action != "accept":
            continue
        key = sentence.lower()
        if key not in seen:
            facts.append(sentence)
            seen.add(key)
    return facts


def _clean_sentence(sentence: str) -> str:
    cleaned = re.sub(
        r"^\s*\[(?:assistant|user|system|tool)\]\s*",
        "",
        sentence.strip(),
        flags=re.IGNORECASE,
    )
    cleaned = cleaned.strip(" \t\n\r-•")
    return cleaned


def _has_stable_operational_context(
    text: str, *, domain: str = "", user_is_technical: bool = False
) -> bool:
    domain_key = domain.lower().replace("-", "_").strip()
    if domain_key in _STABLE_OPERATIONAL_DOMAINS:
        return True
    if _matches_any(text, _STABLE_INFRA_PATTERNS):
        return True
    return user_is_technical and _matches_any(
        text,
        [
            r"\b(config|configuration|default|persistent|permanent|production|service|dashboard)\b",
            r"\b(runs on|hosted at|listens on)\b",
        ],
    )


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)
