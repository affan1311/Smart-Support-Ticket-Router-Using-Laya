"""Shared test setup: a throwaway SQLite database and fake model clients.

The environment variables are set before any app module is imported, so settings
point at the test database and the real Laya model and Gemini API are never used.
"""
import os
import tempfile
from pathlib import Path

_tmp_dir = Path(tempfile.mkdtemp())
os.environ["DATABASE_URL"] = f"sqlite:///{(_tmp_dir / 'test.db').as_posix()}"
os.environ["USE_LAYA"] = "true"
os.environ["PRELOAD_LAYA"] = "false"
os.environ["GEMINI_API_KEY"] = ""  # load_dotenv never overrides a variable that's already set

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.db import Base, engine  # noqa: E402
from app.main import app, get_gemini, get_laya  # noqa: E402
from app.schemas import DraftResult, TriageResult  # noqa: E402


def make_triage(**overrides) -> TriageResult:
    """A confident, routine billing ticket that should get a draft."""
    fields = dict(
        department="billing", department_prob=0.95, department_confidence=0.9,
        department_probs={"billing": 0.95, "general": 0.05},
        urgency_level=1, urgency_score=1.0, urgency_probs={"0": 0.1, "1": 0.8, "2": 0.1},
        policy_violation_prob=0.05, human_needed_prob=0.1, standard_reply_prob=0.9,
        source="laya", latency_ms=100.0,
    )
    fields.update(overrides)
    return TriageResult(**fields)


class FakeLaya:
    """Returns whatever TriageResult it was given; `result` can also be a function of the subject."""

    def __init__(self, result=None, error: Exception | None = None):
        self.result = result or make_triage()
        self.error = error
        self.calls = 0

    def triage(self, subject, body):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result(subject) if callable(self.result) else self.result


class FakeGemini:
    def __init__(self, draft_error: Exception | None = None):
        self.draft_error = draft_error
        self.draft_calls = 0
        self.triage_calls = 0

    def draft_reply(self, subject, body, department):
        self.draft_calls += 1
        if self.draft_error:
            raise self.draft_error
        return DraftResult(text=f"Hello, about '{subject}'...", model="fake-gemini", latency_ms=800.0,
                           input_tokens=200, output_tokens=100, cost_usd=0.0005)

    def gemini_only_triage(self, subject, body):
        self.triage_calls += 1
        return make_triage(source="gemini_fallback", latency_ms=1500.0, input_tokens=300,
                           output_tokens=50, cost_usd=0.0004)


@pytest.fixture(autouse=True)
def fresh_db():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield


@pytest.fixture
def fakes():
    """Mutable holder so a test can swap in a different fake before posting tickets."""
    return {"laya": FakeLaya(), "gemini": FakeGemini()}


@pytest.fixture
def client(fakes):
    app.dependency_overrides[get_laya] = lambda: fakes["laya"]
    app.dependency_overrides[get_gemini] = lambda: fakes["gemini"]
    with TestClient(app) as c:  # "with" runs the startup code (creates tables)
        yield c
    app.dependency_overrides.clear()
