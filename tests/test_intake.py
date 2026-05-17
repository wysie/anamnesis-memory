from anamnesis.intake import IntakeDecision, classify_intake, extract_durable_facts


def test_classify_intake_rejects_temporary_operational_state():
    samples = [
        "The server is currently running on PID 12345 and port 8766 is listening.",
        "The cron job is stuck and holding the tick lock via PID 59571.",
        "Phase 2 completed in commit abc1234; PR #12 was submitted.",
        "Temporary manual process started on port 54321.",
        "Review the conversation above and update the skill library.",
        "[ASSISTANT] The temporary server started on port 8766.",
        "U mean u create separate memory for Trusted contact and Helper though it's the same profile?",
        "What is autopilot and why is there owner and domain etc?",
    ]

    for sample in samples:
        decision = classify_intake(sample)
        assert isinstance(decision, IntakeDecision)
        assert decision.action == "reject"
        assert "temporary_task_state" in decision.reasons


def test_classify_intake_rejects_low_value_chat_fragments():
    samples = [
        "Go ahead.",
        "Ok go ahead.",
        "Ok go ahead la",
        "[USER] Ok go ahead",
        "Ah ok",
        "Fix it",
        "Carry on leh",
        "Please go and do what u need to leh keep going",
        "Now what",
        "Carry on leh why u stop working",
        "Go",
        "Ok do it la.",
        "What is. Batch shadow mode for",
        "Is WhatsApp Primary user and telegram Primary user considered same user or no.",
        "3. And how would someone set platform scope all lol",
        "U didn't really answer the platform scope question. If you save smth now does it default to current or all or whatever",
    ]

    for sample in samples:
        decision = classify_intake(sample)
        assert decision.action == "reject"
        assert set(decision.reasons) & {"low_value_chat_fragment", "temporary_task_state"}


def test_classify_intake_accepts_durable_preferences_permissions_and_infra():
    samples = [
        "Primary user prefers WhatsApp summaries to remain local-only and never use cloud fallback.",
        "Helper cannot control smart home devices, buy things, or run commands.",
        "Local memory dashboard runs on 127.0.0.1:8767.",
        "Project Anamnesis uses SQLite and FTS5 for deterministic local recall.",
        "The household includes Trusted contact, Family member, and Helper.",
    ]

    for sample in samples:
        decision = classify_intake(sample)
        assert decision.action == "accept"
        assert decision.lifecycle in {"durable", "stable_infrastructure"}


def test_classify_intake_does_not_overfit_ports_for_nontechnical_users():
    assert classify_intake("The API endpoint is http://localhost:8000.").action == "inbox"
    assert classify_intake("The dev server is currently on port 5173.").action == "reject"


def test_classify_intake_accepts_operational_facts_with_durable_evidence():
    samples = [
        ("The dashboard runs on port 8765.", "infrastructure"),
        ("Draw Things gRPC port 7859 can be open while generation fails due to macOS access prompt.", "infrastructure"),
        ("The production webhook endpoint is /api/webhooks/stripe.", "project"),
    ]

    for text, domain in samples:
        decision = classify_intake(text, domain=domain)
        assert decision.action == "accept"
        assert decision.lifecycle == "stable_infrastructure"
        assert "durable_operational_evidence" in decision.reasons


def test_classify_intake_sends_sensitive_or_ambiguous_items_to_inbox():
    samples = [
        "Maybe Trusted contact should be able to see some summaries if Primary user says okay.",
        "The user seemed to prefer a different model provider today.",
    ]

    for sample in samples:
        decision = classify_intake(sample)
        assert decision.action == "inbox"
        assert decision.reasons


def test_extract_durable_facts_keeps_stable_sentences_from_mixed_memories():
    text = (
        "[ASSISTANT] The gateway restart command failed today. "
        "Primary user forbids Hermes source patches or gateway restarts unless explicit. "
        "Temporary manual server started on port 54321. "
        "Project Anamnesis uses SQLite and FTS5 for deterministic local recall."
    )

    facts = extract_durable_facts(text)

    assert facts == [
        "Primary user forbids Hermes source patches or gateway restarts unless explicit.",
        "Project Anamnesis uses SQLite and FTS5 for deterministic local recall.",
    ]


def test_extract_durable_facts_rejects_mixed_items_without_durable_sentence():
    text = "[ASSISTANT] Phase 2 completed in commit abc1234. Temporary server started on port 54321."

    assert extract_durable_facts(text) == []
