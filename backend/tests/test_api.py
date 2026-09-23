"""End-to-end API tests with fake Laya and Gemini clients.

FastAPI's TestClient runs background tasks before returning the response, so a
ticket is fully triaged by the time the next request is made.
"""
from conftest import FakeGemini, FakeLaya, make_triage


def post_ticket(client, subject="Charged twice", body="Please refund the duplicate charge."):
    r = client.post("/tickets", json={"subject": subject, "body": body, "customer_email": "a@b.com"})
    assert r.status_code == 201
    return r.json()["id"]


# --- Pipeline through the API ---

def test_routine_ticket_is_triaged_by_laya_and_drafted(client, fakes):
    ticket_id = post_ticket(client)
    t = client.get(f"/tickets/{ticket_id}").json()

    assert t["status"] == "triaged"
    assert t["queue"] == "billing"
    assert t["needs_review"] is False
    assert t["decisions"][0]["source"] == "laya"
    assert len(t["drafts"]) == 1
    assert fakes["gemini"].triage_calls == 0  # Gemini only drafted, it didn't triage


def test_policy_violation_escalates_without_calling_gemini(client, fakes):
    fakes["laya"] = FakeLaya(make_triage(policy_violation_prob=0.95))
    t = client.get(f"/tickets/{post_ticket(client)}").json()

    assert t["queue"] == "escalation"
    assert t["policy_violation"] is True
    assert t["drafts"] == []
    assert fakes["gemini"].draft_calls == 0


def test_laya_failure_falls_back_to_gemini_triage(client, fakes):
    fakes["laya"] = FakeLaya(error=RuntimeError("model crashed"))
    t = client.get(f"/tickets/{post_ticket(client)}").json()

    assert t["status"] == "triaged"
    decision = t["decisions"][0]
    assert decision["source"] == "gemini_fallback"
    assert "model crashed" in decision["laya_error"]
    assert decision["cost_usd"] > 0


def test_ticket_fails_cleanly_when_no_model_is_available(client, fakes):
    fakes["laya"] = FakeLaya(error=RuntimeError("down"))
    fakes["gemini"] = None
    t = client.get(f"/tickets/{post_ticket(client)}").json()

    assert t["status"] == "failed"
    assert "Triage failed" in t["error"]


def test_draft_failure_keeps_the_routing(client, fakes):
    fakes["gemini"] = FakeGemini(draft_error=RuntimeError("quota exceeded"))
    t = client.get(f"/tickets/{post_ticket(client)}").json()

    assert t["status"] == "triaged"
    assert t["queue"] == "billing"
    assert t["drafts"] == []
    assert "quota exceeded" in t["error"]


def test_timestamps_are_returned_as_utc(client):
    # SQLite drops the timezone; the API must still say the time is UTC.
    t = client.get(f"/tickets/{post_ticket(client)}").json()
    assert t["created_at"].endswith("Z")
    assert t["decisions"][0]["created_at"].endswith("Z")


def test_get_missing_ticket_returns_404(client):
    assert client.get("/tickets/999").status_code == 404


# --- Queue listing ---

def test_queue_is_sorted_by_urgency_and_filterable(client, fakes):
    urgency_by_subject = {"low": 0.2, "high": 1.9, "mid": 1.0}
    fakes["laya"] = FakeLaya(lambda subject: make_triage(
        urgency_score=urgency_by_subject[subject],
        department="shipping" if subject == "mid" else "billing",
    ))
    for subject in urgency_by_subject:
        post_ticket(client, subject=subject)

    subjects = [t["subject"] for t in client.get("/tickets").json()]
    assert subjects == ["high", "mid", "low"]

    shipping = client.get("/tickets", params={"dept": "shipping"}).json()
    assert [t["subject"] for t in shipping] == ["mid"]


# --- Bulk upload ---

def test_bulk_csv_upload_triages_every_row(client):
    csv_text = "subject,body,customer_email\nA,first ticket,a@x.com\nB,second ticket,\n"
    r = client.post("/tickets/bulk", files={"file": ("t.csv", csv_text, "text/csv")})
    assert r.status_code == 201
    assert r.json()["count"] == 2
    assert all(t["status"] == "triaged" for t in client.get("/tickets").json())


def test_bulk_csv_missing_column_is_rejected(client):
    r = client.post("/tickets/bulk", files={"file": ("t.csv", "subject\nonly a subject\n", "text/csv")})
    assert r.status_code == 400
    assert "body" in r.json()["detail"]


def test_bulk_csv_with_a_bad_row_saves_nothing(client):
    csv_text = "subject,body\nGood,fine\nBad,\n"
    r = client.post("/tickets/bulk", files={"file": ("t.csv", csv_text, "text/csv")})
    assert r.status_code == 400
    assert "Line 3" in r.json()["detail"]
    assert client.get("/tickets").json() == []


# --- Overrides ---

def test_department_override_moves_queue_and_is_logged(client):
    ticket_id = post_ticket(client)
    r = client.patch(f"/tickets/{ticket_id}/decision",
                     json={"field": "department", "value": "shipping", "agent_id": "sam"})
    assert r.status_code == 200
    t = r.json()
    assert t["department"] == "shipping"
    assert t["queue"] == "shipping"
    assert t["needs_review"] is False
    assert t["overrides"][0] | {"id": 0, "created_at": ""} == {
        "id": 0, "field": "department", "old_value": "billing", "new_value": "shipping",
        "agent_id": "sam", "created_at": "",
    }
    # The original model decision is kept unchanged for evaluation.
    assert t["decisions"][0]["department"] == "billing"


def test_confirming_the_same_value_clears_review_but_is_not_a_correction(client, fakes):
    fakes["laya"] = FakeLaya(make_triage(department_prob=0.6))  # medium confidence -> review
    ticket_id = post_ticket(client)
    t = client.patch(f"/tickets/{ticket_id}/decision", json={"field": "department", "value": "billing"}).json()
    assert t["needs_review"] is False
    assert t["overrides"] == []
    assert client.get("/metrics").json()["tickets_overridden"] == 0


def test_policy_override_escalates(client):
    ticket_id = post_ticket(client)
    t = client.patch(f"/tickets/{ticket_id}/decision", json={"field": "policy_violation", "value": True}).json()
    assert t["queue"] == "escalation"


def test_invalid_override_values_are_rejected(client):
    ticket_id = post_ticket(client)
    assert client.patch(f"/tickets/{ticket_id}/decision",
                        json={"field": "department", "value": "marketing"}).status_code == 422
    assert client.patch(f"/tickets/{ticket_id}/decision",
                        json={"field": "urgency_level", "value": 7}).status_code == 422
    assert client.patch(f"/tickets/{ticket_id}/decision",
                        json={"field": "body", "value": "x"}).status_code == 422


# --- Draft approval ---

def test_approve_draft_with_edits_resolves_ticket(client):
    ticket_id = post_ticket(client)
    t = client.post(f"/tickets/{ticket_id}/draft/approve", json={"edited_text": "Edited reply"}).json()
    assert t["status"] == "resolved"
    assert t["drafts"][0]["approved"] is True
    assert t["drafts"][0]["edited_text"] == "Edited reply"


def test_approve_without_draft_returns_404(client, fakes):
    fakes["laya"] = FakeLaya(make_triage(standard_reply_prob=0.1))  # no draft made
    ticket_id = post_ticket(client)
    assert client.post(f"/tickets/{ticket_id}/draft/approve", json={}).status_code == 404


# --- Metrics ---

def test_metrics_count_costs_and_gemini_calls_avoided(client, fakes):
    post_ticket(client, subject="drafted")                        # Laya + Gemini draft
    fakes["laya"] = FakeLaya(make_triage(standard_reply_prob=0.1))
    post_ticket(client, subject="no draft")                       # Laya only
    ticket_id = post_ticket(client, subject="overridden")         # Laya only, then corrected
    client.patch(f"/tickets/{ticket_id}/decision", json={"field": "urgency_level", "value": 2})

    m = client.get("/metrics").json()
    assert m["total_tickets"] == 3
    assert m["triage_source"] == {"laya": 3}
    assert m["laya_latency"] == {"count": 3, "p50_ms": 100.0, "p95_ms": 100.0}
    assert m["drafts_created"] == 1
    assert m["gemini_calls_avoided_pct"] == 66.7
    assert m["total_cost_usd"] == 0.0005
    assert m["tickets_overridden"] == 1
    assert m["correction_rate"] == 0.333


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
