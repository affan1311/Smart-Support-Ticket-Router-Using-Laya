"""Pydantic data shapes shared across the app.

TriageResult is the contract between the triage step (Laya, or Gemini as a fallback)
and routing.py. Whichever model produced it, routing sees the same fields.

The second half of the file holds the API request/response shapes used by main.py.
"""
from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field


class TriageResult(BaseModel):
    """The raw answers to the 5 triage questions. It holds probabilities, not decisions.

    Turning probabilities into yes/no and choosing a queue is routing.py's job, so all
    thresholds live in one place.
    """

    # department (choice question)
    department: str                       # most likely department
    department_prob: float = Field(ge=0, le=1)  # its probability: what the thresholds use
    department_confidence: float = Field(ge=0, le=1)  # Laya's entropy-based confidence (stored for analysis)
    department_probs: dict[str, float]    # probability for every department

    # urgency (score question, levels 0-2)
    urgency_level: int = Field(ge=0, le=2)  # most likely level: the label shown to agents
    urgency_score: float = Field(ge=0, le=2)  # expected level, e.g. 1.5: used to sort the queue
    urgency_probs: dict[str, float]

    # yes/no (noul) questions: probability that the answer is "yes"
    policy_violation_prob: float = Field(ge=0, le=1)
    human_needed_prob: float = Field(ge=0, le=1)
    standard_reply_prob: float = Field(ge=0, le=1)

    # bookkeeping
    source: Literal["laya", "gemini_fallback"]
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0   # always 0 for Laya (it doesn't generate text)
    cost_usd: float = 0.0    # always 0 for Laya (runs locally)
    truncated: bool = False  # True if the ticket was cut to fit Laya's context


class DraftResult(BaseModel):
    """A reply drafted by Gemini, plus what it cost."""

    text: str
    model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float


class RoutingDecision(BaseModel):
    """What the app should do with a ticket, decided by routing.route()."""

    queue: str              # a department name, "escalation" or "general_triage"
    needs_review: bool      # show a "review" badge in the dashboard
    should_draft: bool      # call Gemini to draft a reply?
    policy_violation: bool
    human_needed: bool
    standard_reply: bool
    reasons: list[str]      # plain-English explanation, shown to agents


# ---------------------------------------------------------------------------
# API shapes
# ---------------------------------------------------------------------------

# SQLite drops timezone info, so a stored UTC time comes back "naive". Mark it as UTC
# again, so the API returns "...Z" (UTC) times and the browser doesn't read them as local time.
UtcDatetime = Annotated[datetime, AfterValidator(lambda d: d if d.tzinfo else d.replace(tzinfo=timezone.utc))]


class TicketCreate(BaseModel):
    subject: str = Field(min_length=1, max_length=300)
    body: str = Field(min_length=1, max_length=20_000)
    customer_email: str | None = None
    channel: Literal["api", "form", "csv"] = "api"


class TicketCreated(BaseModel):
    id: int
    status: str


class BulkCreated(BaseModel):
    count: int
    ids: list[int]


# from_attributes lets these be built straight from SQLAlchemy objects.
class TicketSummary(BaseModel):
    """One row in the dashboard queue."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    subject: str
    customer_email: str | None
    channel: str
    status: str
    queue: str | None
    department: str | None
    department_prob: float | None
    urgency_level: int | None
    urgency_score: float | None
    needs_review: bool
    policy_violation: bool | None
    human_needed: bool | None
    standard_reply: bool | None
    error: str | None
    created_at: UtcDatetime


class DecisionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    source: str
    department: str
    department_prob: float
    department_confidence: float
    department_probs: dict[str, float]
    urgency_level: int
    urgency_score: float
    urgency_probs: dict[str, float]
    policy_violation_prob: float
    human_needed_prob: float
    standard_reply_prob: float
    queue: str
    needs_review: bool
    should_draft: bool
    reasons: list[str]
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    truncated: bool
    laya_error: str | None
    created_at: UtcDatetime


class DraftOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    draft_text: str
    model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    approved: bool
    edited_text: str | None
    approved_at: UtcDatetime | None
    created_at: UtcDatetime


class OverrideOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    field: str
    old_value: str | None
    new_value: str
    agent_id: str
    created_at: UtcDatetime


class TicketDetail(TicketSummary):
    body: str
    decisions: list[DecisionOut]
    drafts: list[DraftOut]
    overrides: list[OverrideOut]


# Fields an agent may correct, and the type each new value must have.
OverrideField = Literal["department", "queue", "urgency_level", "policy_violation", "human_needed"]


class OverrideIn(BaseModel):
    field: OverrideField
    value: str | int | bool
    agent_id: str = "agent"


class DraftApproveIn(BaseModel):
    edited_text: str | None = None  # send the agent's edited version, or nothing to approve as-is


class LatencyStats(BaseModel):
    count: int
    p50_ms: float | None
    p95_ms: float | None


class MetricsOut(BaseModel):
    total_tickets: int
    by_status: dict[str, int]
    by_queue: dict[str, int]
    triage_source: dict[str, int]         # laya vs gemini_fallback
    laya_latency: LatencyStats
    gemini_triage_latency: LatencyStats
    draft_latency: LatencyStats
    drafts_created: int
    drafts_approved: int
    drafts_edited: int
    gemini_calls_avoided_pct: float | None  # triaged tickets that never called Gemini
    total_cost_usd: float
    cost_per_1000_tickets_usd: float | None
    needs_review: int
    tickets_overridden: int
    correction_rate: float | None           # overridden / triaged
