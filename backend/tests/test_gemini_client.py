"""Unit tests for gemini_client.py. No network: responses are faked."""
import typing
from types import SimpleNamespace

import pytest

from app.config import DEPARTMENTS, Settings
from app.gemini_client import (
    GeminiClient, GeminiTriageOutput, estimate_cost, to_triage_result, token_usage,
)

CFG = Settings(gemini_input_price_per_m=0.75, gemini_output_price_per_m=3.75)


def fake_response(text: str, prompt=100, output=20, thoughts=30):
    usage = SimpleNamespace(prompt_token_count=prompt, candidates_token_count=output, thoughts_token_count=thoughts)
    return SimpleNamespace(text=text, usage_metadata=usage)


VALID_JSON = ('{"department": "shipping", "department_confidence": 0.9, "urgency_level": 2, '
              '"policy_violation": false, "human_needed": true, "standard_reply": false}')


def test_estimate_cost():
    # 1M input tokens at $0.75 + 1M output tokens at $3.75
    assert estimate_cost(1_000_000, 1_000_000, CFG) == pytest.approx(4.50)
    assert estimate_cost(1000, 200, CFG) == pytest.approx(0.00075 + 0.00075)


def test_token_usage_counts_thinking_as_output():
    assert token_usage(fake_response("x", prompt=100, output=20, thoughts=30)) == (100, 50)


def test_token_usage_handles_missing_fields():
    assert token_usage(fake_response("x", prompt=None, output=None, thoughts=None)) == (0, 0)
    assert token_usage(SimpleNamespace(usage_metadata=None)) == (0, 0)


def test_schema_departments_match_config():
    allowed = typing.get_args(GeminiTriageOutput.model_fields["department"].annotation)
    assert set(allowed) == set(DEPARTMENTS)


def test_gemini_answer_maps_to_triage_result():
    out = GeminiTriageOutput.model_validate_json(VALID_JSON)
    r = to_triage_result(out, latency_ms=1234.56, input_tokens=100, output_tokens=50, cfg=CFG)
    assert r.source == "gemini_fallback"
    assert r.department == "shipping" and r.department_prob == 0.9
    assert r.urgency_level == 2 and r.urgency_score == 2.0
    assert (r.policy_violation_prob, r.human_needed_prob, r.standard_reply_prob) == (0.0, 1.0, 0.0)
    assert r.cost_usd == pytest.approx(estimate_cost(100, 50, CFG))


def client_returning(*responses) -> GeminiClient:
    """A GeminiClient whose API call returns the given responses in order (no network)."""
    client = object.__new__(GeminiClient)  # skip __init__, which would build a real SDK client
    client.cfg = CFG
    queue = list(responses)

    def fake_generate(prompt, system, json_schema=None):
        response = queue.pop(0)
        return (response, 500.0, *token_usage(response))

    client._generate = fake_generate
    return client


def test_triage_retries_once_on_invalid_json_and_counts_both_calls():
    client = client_returning(fake_response('{"department": "marketing"}'), fake_response(VALID_JSON))
    r = client.gemini_only_triage("Late parcel", "Where is it?")
    assert r.department == "shipping"
    assert r.latency_ms == 1000.0             # both attempts
    assert r.input_tokens == 200 and r.output_tokens == 100
    assert r.cost_usd == pytest.approx(estimate_cost(200, 100, CFG))


def test_triage_gives_up_after_two_invalid_answers():
    client = client_returning(fake_response("not json"), fake_response("{}"))
    with pytest.raises(RuntimeError, match="failed validation"):
        client.gemini_only_triage("x", "y")


def test_empty_draft_is_an_error():
    client = client_returning(fake_response("   "))
    with pytest.raises(RuntimeError, match="empty draft"):
        client.draft_reply("x", "y", "billing")
