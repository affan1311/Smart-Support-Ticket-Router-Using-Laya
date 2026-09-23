"""Numbers for the /metrics endpoint and the dashboard metrics page.

It loads rows and computes in Python, which is simple and fine at portfolio scale.
With millions of tickets you'd push these aggregates into SQL.
"""
import statistics
from collections import Counter

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Decision, Draft, Override, Ticket
from app.schemas import LatencyStats, MetricsOut


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile, e.g. pct=95 -> the value 95% of samples are at or below."""
    ordered = sorted(values)
    index = max(0, round(pct / 100 * len(ordered)) - 1)
    return ordered[index]


def latency_stats(values: list[float]) -> LatencyStats:
    if not values:
        return LatencyStats(count=0, p50_ms=None, p95_ms=None)
    return LatencyStats(count=len(values), p50_ms=round(statistics.median(values), 1),
                        p95_ms=round(percentile(values, 95), 1))


def compute_metrics(db: Session) -> MetricsOut:
    tickets = db.scalars(select(Ticket)).all()
    decisions = db.scalars(select(Decision)).all()
    drafts = db.scalars(select(Draft)).all()
    overridden_ids = set(db.scalars(select(Override.ticket_id)).all())

    triaged = [t for t in tickets if t.status in ("triaged", "resolved")]
    laya_decisions = [d for d in decisions if d.source == "laya"]
    gemini_decisions = [d for d in decisions if d.source == "gemini_fallback"]

    # A ticket "avoided Gemini" if Laya triaged it and no draft was needed.
    gemini_ticket_ids = {d.ticket_id for d in gemini_decisions} | {d.ticket_id for d in drafts}
    avoided = [t for t in triaged if t.id not in gemini_ticket_ids]

    total_cost = sum(d.cost_usd for d in decisions) + sum(d.cost_usd for d in drafts)

    return MetricsOut(
        total_tickets=len(tickets),
        by_status=dict(Counter(t.status for t in tickets)),
        by_queue=dict(Counter(t.queue for t in tickets if t.queue)),
        triage_source=dict(Counter(d.source for d in decisions)),
        laya_latency=latency_stats([d.latency_ms for d in laya_decisions]),
        gemini_triage_latency=latency_stats([d.latency_ms for d in gemini_decisions]),
        draft_latency=latency_stats([d.latency_ms for d in drafts]),
        drafts_created=len(drafts),
        drafts_approved=sum(d.approved for d in drafts),
        drafts_edited=sum(d.approved and d.edited_text is not None for d in drafts),
        gemini_calls_avoided_pct=round(100 * len(avoided) / len(triaged), 1) if triaged else None,
        total_cost_usd=round(total_cost, 6),
        cost_per_1000_tickets_usd=round(1000 * total_cost / len(triaged), 4) if triaged else None,
        needs_review=sum(t.needs_review for t in tickets),
        tickets_overridden=len(overridden_ids),
        correction_rate=round(len(overridden_ids) / len(triaged), 3) if triaged else None,
    )
