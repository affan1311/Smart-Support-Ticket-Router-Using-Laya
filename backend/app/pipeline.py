"""The ticket pipeline: triage -> route -> save -> draft (only if needed).

This is kept apart from main.py so the API stays thin and the Phase 4 benchmark can
reuse run_triage() without going through HTTP.
"""
import logging
import threading

from app import db as database
from app.config import settings
from app.gemini_client import GeminiClient
from app.laya_client import LayaClient
from app.models import Decision, Draft, Ticket
from app.routing import route
from app.schemas import RoutingDecision, TriageResult

log = logging.getLogger(__name__)

# Background tasks run in a thread pool, but there's only one Laya model in memory.
# The lock lets one ticket use it at a time, and ~2 s per ticket on CPU is fine for a demo.
_laya_lock = threading.Lock()


def run_triage(subject: str, body: str, laya: LayaClient | None,
               gemini: GeminiClient | None) -> tuple[TriageResult, str | None]:
    """Laya first; Gemini-only triage if Laya is off or fails.

    Returns (triage, laya_error). laya_error explains a fallback, or is None.
    """
    laya_error = None
    if laya is not None:
        try:
            with _laya_lock:
                return laya.triage(subject, body), None
        except Exception as e:  # any Laya failure falls back rather than losing the ticket
            log.exception("Laya triage failed, falling back to Gemini")
            laya_error = f"{type(e).__name__}: {e}"
    elif settings.use_laya:
        laya_error = "Laya model not loaded"

    if gemini is None:
        raise RuntimeError(f"No triage model available (Laya: {laya_error or 'disabled'}; Gemini: no API key)")
    return gemini.gemini_only_triage(subject, body), laya_error


def save_decision(db, ticket: Ticket, triage: TriageResult, decision: RoutingDecision,
                  laya_error: str | None) -> None:
    """Store the model's answers (Decision row) and copy the outcome onto the ticket."""
    db.add(Decision(
        ticket_id=ticket.id,
        **triage.model_dump(),
        queue=decision.queue,
        needs_review=decision.needs_review,
        should_draft=decision.should_draft,
        reasons=decision.reasons,
        laya_error=laya_error,
    ))
    ticket.queue = decision.queue
    ticket.department = triage.department
    ticket.department_prob = triage.department_prob
    ticket.urgency_level = triage.urgency_level
    ticket.urgency_score = triage.urgency_score
    ticket.needs_review = decision.needs_review
    ticket.policy_violation = decision.policy_violation
    ticket.human_needed = decision.human_needed
    ticket.standard_reply = decision.standard_reply
    ticket.status = "triaged"
    ticket.error = None


def triage_ticket(ticket_id: int, laya: LayaClient | None, gemini: GeminiClient | None) -> None:
    """Background task: run the full pipeline for one saved ticket."""
    with database.SessionLocal() as db:
        ticket = db.get(Ticket, ticket_id)
        if ticket is None:
            return

        try:
            triage, laya_error = run_triage(ticket.subject, ticket.body, laya, gemini)
        except Exception as e:
            log.exception("Triage failed for ticket %s", ticket_id)
            ticket.status = "failed"
            ticket.error = f"Triage failed: {e}"
            db.commit()
            return

        decision = route(triage)
        save_decision(db, ticket, triage, decision, laya_error)
        db.commit()  # commit routing first, so the ticket shows up in the queue even if drafting is slow

        if not decision.should_draft:
            return
        if gemini is None:
            ticket.error = "Draft skipped: GEMINI_API_KEY not set"
        else:
            try:
                draft = gemini.draft_reply(ticket.subject, ticket.body, decision.queue)
                db.add(Draft(
                    ticket_id=ticket.id,
                    draft_text=draft.text,
                    model=draft.model,
                    latency_ms=draft.latency_ms,
                    input_tokens=draft.input_tokens,
                    output_tokens=draft.output_tokens,
                    cost_usd=draft.cost_usd,
                ))
            except Exception as e:  # a failed draft must not undo the routing
                log.exception("Draft failed for ticket %s", ticket_id)
                ticket.error = f"Draft failed: {e}"
        db.commit()
