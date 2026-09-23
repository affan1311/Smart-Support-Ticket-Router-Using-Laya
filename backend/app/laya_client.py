"""Adapter around the laya package.

Only this file touches laya's API. The rest of the app works with TriageResult,
so a laya API change, or swapping in Gemini, stays contained here.

Laya facts this code relies on (checked in laya 0.3.6 source):
  - agent.predict(state, questions) answers every question in one forward pass.
  - Each question becomes its own 512-token sequence: question + options (<= 192
    tokens), then the ticket. So the ticket gets ~320 tokens, and laya silently
    cuts off anything longer. We truncate explicitly so we know when it happens.
  - choice -> {"choice", "probabilities", "confidence"}
    score  -> {"score" (expected level), "probabilities" per level, "confidence"}
    noul   -> {"noul" (= P(yes)), "confidence"}
  - "confidence" is 1 - normalized entropy, NOT the top probability. Thresholds use
    the top probability, which is easier to reason about.
"""
import os
import time
from functools import lru_cache

from app.config import DEPARTMENTS, URGENCY_LEVELS, settings
from app.schemas import TriageResult

# The 5 triage questions. Wording is kept short because it shares the 192-token
# question budget with the option descriptions.
QUESTIONS: dict = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this support ticket?",
        "criteria": DEPARTMENTS,
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this ticket?",
        "criteria": URGENCY_LEVELS,
    },
    "policy_violation": {
        "type": "noul",
        "instructions": "Does the ticket contain abuse, fraud or a legal threat?",
    },
    "human_needed": {
        "type": "noul",
        "instructions": "Does a human agent need to make a judgment call?",
    },
    "standard_reply": {
        "type": "noul",
        "instructions": "Can a standard template reply fully resolve this ticket?",
    },
}


def parse_laya_result(raw: dict, latency_ms: float, truncated: bool) -> TriageResult:
    """Convert laya's raw output dict into our TriageResult."""
    a = raw["answers"]
    dept, urg = a["department"], a["urgency"]
    urgency_probs = urg["probabilities"]  # e.g. {"0": 0.01, "1": 0.48, "2": 0.51}

    return TriageResult(
        department=dept["choice"],
        department_prob=dept["probabilities"][dept["choice"]],
        department_confidence=dept["confidence"],
        department_probs=dept["probabilities"],
        # The label is the most likely level; the expected score sorts the queue.
        urgency_level=int(max(urgency_probs, key=urgency_probs.get)),
        urgency_score=urg["score"],
        urgency_probs=urgency_probs,
        policy_violation_prob=a["policy_violation"]["noul"],
        human_needed_prob=a["human_needed"]["noul"],
        standard_reply_prob=a["standard_reply"]["noul"],
        source="laya",
        latency_ms=round(latency_ms, 1),
        input_tokens=raw.get("usage", {}).get("input_tokens", 0),
        truncated=truncated,
    )


class LayaClient:
    """Holds one loaded Laya model and runs triage with it."""

    def __init__(self, model_id: str = settings.laya_model, device: str | None = settings.laya_device):
        # Must be set before transformers is imported, or laya.load() can hang.
        os.environ.setdefault("USE_TF", "0")
        import laya  # imported here so tests that don't need the model stay fast

        start = time.perf_counter()
        self.agent = laya.load(model_id, device=device)
        self.load_seconds = time.perf_counter() - start

    def _truncate(self, text: str) -> tuple[str, bool]:
        """Keep the first max_ticket_tokens tokens of the ticket, using Laya's own tokenizer."""
        ids = self.agent.tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= settings.max_ticket_tokens:
            return text, False
        return self.agent.tok.decode(ids[: settings.max_ticket_tokens]), True

    def triage(self, subject: str, body: str) -> TriageResult:
        """Answer the 5 triage questions for one ticket."""
        # Subject first: if we have to truncate, the most informative line survives.
        text, truncated = self._truncate(f"Subject: {subject}\n\n{body}")

        start = time.perf_counter()
        raw = self.agent.predict(text, QUESTIONS)
        latency_ms = (time.perf_counter() - start) * 1000

        return parse_laya_result(raw, latency_ms, truncated)


@lru_cache(maxsize=1)
def get_laya_client() -> LayaClient:
    """Return the shared LayaClient, loading the model on the first call only.

    lru_cache keeps the one instance, so the ~2 GB model loads once per process.
    In Phase 2 the FastAPI startup hook calls this so the first request isn't slow.
    """
    return LayaClient()
