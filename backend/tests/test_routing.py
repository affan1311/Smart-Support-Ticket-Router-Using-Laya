"""Unit tests for routing.route(), using fake Laya answers (no model needed)."""
import pytest

from app.config import ESCALATION_QUEUE, GENERAL_TRIAGE_QUEUE, Settings
from app.routing import route
from app.schemas import TriageResult

# Fixed thresholds, so the tests don't depend on whatever is in .env.
CFG = Settings(min_confidence=0.5, auto_confidence=0.8, noul_threshold=0.5, policy_threshold=0.5)


def fake_triage(**overrides) -> TriageResult:
    """A confident, routine billing ticket. Override any field to test a case."""
    fields = dict(
        department="billing",
        department_prob=0.95,
        department_confidence=0.9,
        department_probs={"billing": 0.95, "general": 0.05},
        urgency_level=1,
        urgency_score=1.0,
        urgency_probs={"0": 0.1, "1": 0.8, "2": 0.1},
        policy_violation_prob=0.05,
        human_needed_prob=0.1,
        standard_reply_prob=0.9,
        source="laya",
        latency_ms=100.0,
    )
    fields.update(overrides)
    return TriageResult(**fields)


def test_confident_routine_ticket_is_auto_assigned_and_drafted():
    d = route(fake_triage(), CFG)
    assert d.queue == "billing"
    assert d.needs_review is False
    assert d.should_draft is True


# --- Rule 1: policy violation ---

def test_policy_violation_escalates_even_when_everything_else_looks_fine():
    d = route(fake_triage(policy_violation_prob=0.9), CFG)
    assert d.queue == ESCALATION_QUEUE
    assert d.policy_violation is True
    assert d.should_draft is False


def test_policy_violation_beats_low_department_confidence():
    d = route(fake_triage(policy_violation_prob=0.9, department_prob=0.2), CFG)
    assert d.queue == ESCALATION_QUEUE


def test_policy_threshold_is_configurable():
    triage = fake_triage(policy_violation_prob=0.7)
    assert route(triage, CFG).queue == ESCALATION_QUEUE
    strict = Settings(policy_threshold=0.8)
    assert route(triage, strict).queue == "billing"


# --- Rules 2-4: department confidence ---

def test_low_confidence_goes_to_general_triage_without_draft():
    d = route(fake_triage(department_prob=0.49), CFG)
    assert d.queue == GENERAL_TRIAGE_QUEUE
    assert d.needs_review is True
    assert d.should_draft is False


def test_exactly_min_confidence_is_assigned_with_review():
    d = route(fake_triage(department_prob=0.5), CFG)
    assert d.queue == "billing"
    assert d.needs_review is True


def test_medium_confidence_is_assigned_with_review_and_can_still_draft():
    d = route(fake_triage(department_prob=0.65), CFG)
    assert d.queue == "billing"
    assert d.needs_review is True
    assert d.should_draft is True


def test_just_below_auto_confidence_needs_review():
    d = route(fake_triage(department_prob=0.79), CFG)
    assert d.needs_review is True


def test_exactly_auto_confidence_is_auto_assigned():
    d = route(fake_triage(department_prob=0.8), CFG)
    assert d.queue == "billing"
    assert d.needs_review is False


# --- Rule 5: when to draft ---

@pytest.mark.parametrize(
    "standard_reply_prob, human_needed_prob, expected_draft",
    [
        (0.9, 0.1, True),   # standard reply, no human -> draft
        (0.9, 0.9, False),  # human needed blocks the draft
        (0.1, 0.1, False),  # no standard reply
        (0.1, 0.9, False),  # neither
    ],
)
def test_draft_only_when_standard_reply_and_no_human(standard_reply_prob, human_needed_prob, expected_draft):
    d = route(fake_triage(standard_reply_prob=standard_reply_prob, human_needed_prob=human_needed_prob), CFG)
    assert d.should_draft is expected_draft


def test_noul_threshold_boundary_counts_as_yes():
    d = route(fake_triage(standard_reply_prob=0.5, human_needed_prob=0.5), CFG)
    assert d.standard_reply is True
    assert d.human_needed is True
    assert d.should_draft is False


def test_every_decision_explains_itself():
    for triage in [fake_triage(), fake_triage(policy_violation_prob=0.9), fake_triage(department_prob=0.1)]:
        assert route(triage, CFG).reasons
