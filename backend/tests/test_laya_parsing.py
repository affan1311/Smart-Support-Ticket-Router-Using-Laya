"""Tests that parse_laya_result() reads laya's output correctly.

The fake response below copies the real shape laya 0.3.6 returned in the Phase 1 probe,
so these tests run without loading the model.
"""
from app.laya_client import parse_laya_result

RAW = {
    "model": "laya-rl-agent",
    "answers": {
        "department": {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.9717, "technical": 0.0068, "shipping": 0.0074,
                              "account": 0.0054, "general": 0.0087},
            "confidence": 0.8957,
        },
        "urgency": {
            "type": "score",
            "score": 1.4969,
            "probabilities": {"0": 0.0107, "1": 0.4816, "2": 0.5076},
            "confidence": 0.3221,
        },
        "policy_violation": {"type": "noul", "noul": 0.6965, "confidence": 0.6965},
        "human_needed": {"type": "noul", "noul": 0.2047, "confidence": 0.7953},
        "standard_reply": {"type": "noul", "noul": 0.2879, "confidence": 0.7121},
    },
    "usage": {"input_tokens": 319, "output_tokens": 0},
}


def test_parse_uses_top_probability_not_laya_confidence():
    r = parse_laya_result(RAW, latency_ms=123.456, truncated=False)
    assert r.department == "billing"
    assert r.department_prob == 0.9717
    assert r.department_confidence == 0.8957


def test_parse_urgency_level_is_most_likely_level():
    r = parse_laya_result(RAW, latency_ms=1.0, truncated=False)
    assert r.urgency_level == 2        # "2" has the highest probability (0.5076)
    assert r.urgency_score == 1.4969   # expected value, used for sorting


def test_parse_noul_and_bookkeeping():
    r = parse_laya_result(RAW, latency_ms=123.456, truncated=True)
    assert r.policy_violation_prob == 0.6965
    assert r.human_needed_prob == 0.2047
    assert r.standard_reply_prob == 0.2879
    assert r.source == "laya"
    assert r.latency_ms == 123.5
    assert r.input_tokens == 319
    assert r.truncated is True
