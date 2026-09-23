"""Database tables.

Design choice: a Decision row is what the model said and is never edited. The Ticket
row holds the *current* state (queue, department, flags), which agents can override.
Comparing the two is how the correction rate is measured.
"""
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Ticket(Base):
    __tablename__ = "tickets"

    id: Mapped[int] = mapped_column(primary_key=True)
    customer_email: Mapped[str | None] = mapped_column(String(320))
    subject: Mapped[str] = mapped_column(String(300))
    body: Mapped[str] = mapped_column(Text)
    channel: Mapped[str] = mapped_column(String(20), default="api")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    # new -> triaged -> resolved, or new -> failed
    status: Mapped[str] = mapped_column(String(20), default="new", index=True)
    error: Mapped[str | None] = mapped_column(Text)  # last triage/draft error, shown to agents

    # Current state: filled in by triage, changed by agent overrides.
    queue: Mapped[str | None] = mapped_column(String(50), index=True)
    department: Mapped[str | None] = mapped_column(String(50))
    department_prob: Mapped[float | None] = mapped_column(Float)
    urgency_level: Mapped[int | None] = mapped_column(Integer)
    urgency_score: Mapped[float | None] = mapped_column(Float, index=True)  # sort key for the queue
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False)
    policy_violation: Mapped[bool | None] = mapped_column(Boolean)
    human_needed: Mapped[bool | None] = mapped_column(Boolean)
    standard_reply: Mapped[bool | None] = mapped_column(Boolean)

    decisions: Mapped[list["Decision"]] = relationship(back_populates="ticket", order_by="Decision.id")
    drafts: Mapped[list["Draft"]] = relationship(back_populates="ticket", order_by="Draft.id")
    overrides: Mapped[list["Override"]] = relationship(back_populates="ticket", order_by="Override.id")


class Decision(Base):
    """One triage run: the raw model answers plus what routing decided."""

    __tablename__ = "decisions"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticket_id: Mapped[int] = mapped_column(ForeignKey("tickets.id"), index=True)
    source: Mapped[str] = mapped_column(String(20))  # "laya" or "gemini_fallback"

    # Model answers
    department: Mapped[str] = mapped_column(String(50))
    department_prob: Mapped[float] = mapped_column(Float)
    department_confidence: Mapped[float] = mapped_column(Float)
    department_probs: Mapped[dict] = mapped_column(JSON)
    urgency_level: Mapped[int] = mapped_column(Integer)
    urgency_score: Mapped[float] = mapped_column(Float)
    urgency_probs: Mapped[dict] = mapped_column(JSON)
    policy_violation_prob: Mapped[float] = mapped_column(Float)
    human_needed_prob: Mapped[float] = mapped_column(Float)
    standard_reply_prob: Mapped[float] = mapped_column(Float)

    # Routing outcome
    queue: Mapped[str] = mapped_column(String(50))
    needs_review: Mapped[bool] = mapped_column(Boolean)
    should_draft: Mapped[bool] = mapped_column(Boolean)
    reasons: Mapped[list] = mapped_column(JSON)

    # Cost and speed
    latency_ms: Mapped[float] = mapped_column(Float)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    truncated: Mapped[bool] = mapped_column(Boolean, default=False)
    laya_error: Mapped[str | None] = mapped_column(Text)  # why we fell back to Gemini, if we did

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    ticket: Mapped[Ticket] = relationship(back_populates="decisions")


class Draft(Base):
    __tablename__ = "drafts"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticket_id: Mapped[int] = mapped_column(ForeignKey("tickets.id"), index=True)
    draft_text: Mapped[str] = mapped_column(Text)
    model: Mapped[str] = mapped_column(String(100))
    latency_ms: Mapped[float] = mapped_column(Float)
    input_tokens: Mapped[int] = mapped_column(Integer)
    output_tokens: Mapped[int] = mapped_column(Integer)
    cost_usd: Mapped[float] = mapped_column(Float)
    approved: Mapped[bool] = mapped_column(Boolean, default=False)
    edited_text: Mapped[str | None] = mapped_column(Text)  # set if the agent changed the draft
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    ticket: Mapped[Ticket] = relationship(back_populates="drafts")


class Override(Base):
    """An agent correcting the model. Each row is a labeled example for evaluation."""

    __tablename__ = "overrides"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticket_id: Mapped[int] = mapped_column(ForeignKey("tickets.id"), index=True)
    field: Mapped[str] = mapped_column(String(50))
    old_value: Mapped[str | None] = mapped_column(String(100))
    new_value: Mapped[str] = mapped_column(String(100))
    agent_id: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    ticket: Mapped[Ticket] = relationship(back_populates="overrides")
