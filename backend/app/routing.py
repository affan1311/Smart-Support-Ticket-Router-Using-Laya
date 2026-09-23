"""Routing rules: turn triage probabilities into an action.

This is a pure function with no model calls, database or network access, so it's
fast, deterministic and easy to unit-test with fake answers.

Rules, in order:
  1. Policy violation            -> escalation queue, no draft
  2. Department prob < MIN       -> general_triage queue, no draft
  3. MIN <= prob < AUTO          -> assign to department, mark for review
  4. prob >= AUTO                -> assign to department automatically
  5. Draft a reply only if standard_reply = yes AND human_needed = no
     (only for tickets that reached a department queue in rules 3-4)
"""
from app.config import ESCALATION_QUEUE, GENERAL_TRIAGE_QUEUE, Settings, settings as default_settings
from app.schemas import RoutingDecision, TriageResult


def route(triage: TriageResult, cfg: Settings = default_settings) -> RoutingDecision:
    # Turn yes/no probabilities into booleans using the configured thresholds.
    policy_violation = triage.policy_violation_prob >= cfg.policy_threshold
    human_needed = triage.human_needed_prob >= cfg.noul_threshold
    standard_reply = triage.standard_reply_prob >= cfg.noul_threshold

    flags = dict(policy_violation=policy_violation, human_needed=human_needed, standard_reply=standard_reply)

    # Rule 1: policy violations always go to a senior person, whatever else the model says.
    if policy_violation:
        return RoutingDecision(
            queue=ESCALATION_QUEUE,
            needs_review=True,
            should_draft=False,
            reasons=[f"policy violation (p={triage.policy_violation_prob:.2f})"],
            **flags,
        )

    # Rule 2: the model isn't sure which department, so a human decides.
    if triage.department_prob < cfg.min_confidence:
        return RoutingDecision(
            queue=GENERAL_TRIAGE_QUEUE,
            needs_review=True,
            should_draft=False,
            reasons=[f"low department confidence ({triage.department} p={triage.department_prob:.2f})"],
            **flags,
        )

    # Rules 3-4: assign to the department; mark for review if not confident enough.
    reasons = [f"assigned to {triage.department} (p={triage.department_prob:.2f})"]
    needs_review = triage.department_prob < cfg.auto_confidence
    if needs_review:
        reasons.append("medium confidence, needs review")

    # Rule 5: draft only when a standard reply works and no human judgment is needed.
    should_draft = standard_reply and not human_needed
    if should_draft:
        reasons.append("standard reply possible, drafting")
    elif human_needed:
        reasons.append("human needed, no draft")
    else:
        reasons.append("no standard reply, no draft")

    return RoutingDecision(
        queue=triage.department,
        needs_review=needs_review,
        should_draft=should_draft,
        reasons=reasons,
        **flags,
    )
