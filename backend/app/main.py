"""FastAPI app: HTTP endpoints only. The real work happens in pipeline.py.

Run from backend/:  uvicorn app.main:app --reload
Interactive docs:   http://127.0.0.1:8000/docs
"""
import csv
import io
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import DEPARTMENTS, ESCALATION_QUEUE, GENERAL_TRIAGE_QUEUE, settings
from app.db import create_tables, get_db
from app.gemini_client import get_gemini_client
from app.laya_client import get_laya_client
from app.metrics import compute_metrics
from app.models import Override, Ticket
from app.pipeline import triage_ticket
from app.schemas import (
    BulkCreated, DraftApproveIn, MetricsOut, OverrideIn, TicketCreate, TicketCreated,
    TicketDetail, TicketSummary,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # Hugging Face + Gemini log every HTTP request otherwise
log = logging.getLogger(__name__)

MAX_CSV_ROWS = 1000
_laya_load_failed = False  # remember a failed load so every request doesn't retry the slow load


# ---------------------------------------------------------------------------
# Model clients as FastAPI dependencies (tests replace these with fakes)
# ---------------------------------------------------------------------------

def get_laya():
    """The shared Laya client, or None if disabled or failed to load (triage then falls back)."""
    global _laya_load_failed
    if not settings.use_laya or _laya_load_failed:
        return None
    try:
        return get_laya_client()
    except Exception:
        log.exception("Could not load Laya; using Gemini-only triage")
        _laya_load_failed = True
        return None


def get_gemini():
    return get_gemini_client()


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    # Load the ~2 GB model once at startup so the first ticket isn't slow.
    if settings.use_laya and settings.preload_laya:
        log.info("Loading Laya model...")
        if get_laya() is not None:
            log.info("Laya loaded in %.1fs", get_laya_client().load_seconds)
    yield


app = FastAPI(title="Smart Support Ticket Router", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_ticket_or_404(db: Session, ticket_id: int) -> Ticket:
    ticket = db.get(Ticket, ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "use_laya": settings.use_laya,
        "laya_loaded": get_laya_client.cache_info().currsize > 0,
        "gemini_configured": bool(settings.gemini_api_key),
        "gemini_model": settings.gemini_model,
    }


@app.post("/tickets", response_model=TicketCreated, status_code=201)
def create_ticket(payload: TicketCreate, background: BackgroundTasks, db: Session = Depends(get_db),
                  laya=Depends(get_laya), gemini=Depends(get_gemini)):
    """Save the ticket and return right away; triage runs after the response is sent."""
    ticket = Ticket(**payload.model_dump())
    db.add(ticket)
    db.commit()
    background.add_task(triage_ticket, ticket.id, laya, gemini)
    return TicketCreated(id=ticket.id, status=ticket.status)


@app.post("/tickets/bulk", response_model=BulkCreated, status_code=201)
async def bulk_upload(file: UploadFile, background: BackgroundTasks, db: Session = Depends(get_db),
                      laya=Depends(get_laya), gemini=Depends(get_gemini)):
    """Upload a CSV with columns subject, body (required) and customer_email (optional)."""
    try:
        text = (await file.read()).decode("utf-8-sig")  # utf-8-sig strips Excel's BOM
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="CSV must be UTF-8")

    reader = csv.DictReader(io.StringIO(text))
    missing = {"subject", "body"} - set(reader.fieldnames or [])
    if missing:
        raise HTTPException(status_code=400, detail=f"CSV is missing columns: {sorted(missing)}")

    rows = list(reader)
    if len(rows) > MAX_CSV_ROWS:
        raise HTTPException(status_code=400, detail=f"CSV has more than {MAX_CSV_ROWS} rows")

    # Validate every row before saving any, so a bad file saves nothing.
    payloads = []
    for line_no, row in enumerate(rows, start=2):  # line 1 is the header
        try:
            payloads.append(TicketCreate(subject=row["subject"], body=row["body"],
                                         customer_email=row.get("customer_email") or None, channel="csv"))
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=f"Line {line_no}: {e.errors()[0]['msg']}")

    tickets = [Ticket(**p.model_dump()) for p in payloads]
    db.add_all(tickets)
    db.commit()
    for t in tickets:
        background.add_task(triage_ticket, t.id, laya, gemini)
    return BulkCreated(count=len(tickets), ids=[t.id for t in tickets])


@app.get("/tickets", response_model=list[TicketSummary])
def list_tickets(
    db: Session = Depends(get_db),
    dept: str | None = Query(None, description="Queue: a department, 'escalation' or 'general_triage'"),
    urgency: int | None = Query(None, ge=0, le=2),
    needs_review: bool | None = None,
    status: str | None = None,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """The dashboard queue: most urgent first, then oldest first."""
    query = select(Ticket)
    if dept is not None:
        query = query.where(Ticket.queue == dept)
    if urgency is not None:
        query = query.where(Ticket.urgency_level == urgency)
    if needs_review is not None:
        query = query.where(Ticket.needs_review == needs_review)
    if status is not None:
        query = query.where(Ticket.status == status)
    query = query.order_by(Ticket.urgency_score.desc().nulls_last(), Ticket.created_at).limit(limit).offset(offset)
    return db.scalars(query).all()


@app.get("/tickets/{ticket_id}", response_model=TicketDetail)
def get_ticket(ticket_id: int, db: Session = Depends(get_db)):
    return get_ticket_or_404(db, ticket_id)


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if str(value).lower() in ("true", "1", "yes"):
        return True
    if str(value).lower() in ("false", "0", "no"):
        return False
    raise HTTPException(status_code=422, detail=f"Expected true/false, got {value!r}")


@app.patch("/tickets/{ticket_id}/decision", response_model=TicketDetail)
def override_decision(ticket_id: int, payload: OverrideIn, db: Session = Depends(get_db)):
    """An agent corrects one field. The change is logged as an Override for evaluation."""
    ticket = get_ticket_or_404(db, ticket_id)
    field, value = payload.field, payload.value

    if field == "department":
        value = str(value)
        if value not in DEPARTMENTS:
            raise HTTPException(status_code=422, detail=f"Unknown department {value!r}")
        old = ticket.department
        ticket.department = value
        if not ticket.policy_violation:  # escalated tickets stay escalated
            ticket.queue = value
    elif field == "queue":
        value = str(value)
        if value not in (*DEPARTMENTS, ESCALATION_QUEUE, GENERAL_TRIAGE_QUEUE):
            raise HTTPException(status_code=422, detail=f"Unknown queue {value!r}")
        old = ticket.queue
        ticket.queue = value
    elif field == "urgency_level":
        try:
            value = int(value)
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail="urgency_level must be 0, 1 or 2")
        if value not in (0, 1, 2):
            raise HTTPException(status_code=422, detail="urgency_level must be 0, 1 or 2")
        old = ticket.urgency_level
        ticket.urgency_level = value
        ticket.urgency_score = float(value)  # re-sort the queue by the agent's judgment
    elif field == "policy_violation":
        value = _parse_bool(value)
        old = ticket.policy_violation
        ticket.policy_violation = value
        if value:
            ticket.queue = ESCALATION_QUEUE
        elif ticket.queue == ESCALATION_QUEUE:
            ticket.queue = ticket.department or GENERAL_TRIAGE_QUEUE
    else:  # human_needed
        value = _parse_bool(value)
        old = ticket.human_needed
        ticket.human_needed = value

    # Same value = the agent confirmed the model was right. That isn't a correction,
    # so it isn't logged (it would wrongly count against the model's correction rate).
    if old != value:
        db.add(Override(ticket_id=ticket.id, field=field, old_value=None if old is None else str(old),
                        new_value=str(value), agent_id=payload.agent_id))
    ticket.needs_review = False  # a human has now looked at it
    db.commit()
    db.refresh(ticket)
    return ticket


@app.post("/tickets/{ticket_id}/draft/approve", response_model=TicketDetail)
def approve_draft(ticket_id: int, payload: DraftApproveIn, db: Session = Depends(get_db)):
    """Approve the latest draft, optionally with the agent's edits. Marks the ticket resolved.

    Nothing is sent to the customer: sending is out of scope for v1.
    """
    ticket = get_ticket_or_404(db, ticket_id)
    if not ticket.drafts:
        raise HTTPException(status_code=404, detail="This ticket has no draft")

    draft = ticket.drafts[-1]
    draft.approved = True
    draft.approved_at = datetime.now(timezone.utc)
    if payload.edited_text and payload.edited_text.strip() != draft.draft_text.strip():
        draft.edited_text = payload.edited_text.strip()
    ticket.status = "resolved"
    db.commit()
    db.refresh(ticket)
    return ticket


@app.get("/metrics", response_model=MetricsOut)
def metrics(db: Session = Depends(get_db)):
    return compute_metrics(db)
