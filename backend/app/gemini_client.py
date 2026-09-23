"""Adapter around the google-genai SDK. Only this file talks to Gemini.

Two jobs:
  - draft_reply():        write a reply for tickets that routing says can be drafted
  - gemini_only_triage(): answer the same 5 triage questions as Laya, as validated
                          JSON. Used as the fallback when Laya fails, and as the
                          benchmark baseline.

Every call records latency, tokens and cost. Gemini bills "thinking" tokens as
output, so output_tokens = visible answer tokens + thinking tokens.
"""
import logging
import time
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from app.config import DEPARTMENTS, URGENCY_LEVELS, Settings, settings
from app.schemas import DraftResult, TriageResult

log = logging.getLogger(__name__)

TIMEOUT_MS = 30_000  # give up on a Gemini call after 30 s


class GeminiTriageOutput(BaseModel):
    """The JSON shape Gemini must return for triage. It's sent to Gemini as a schema and
    used again to validate the reply, so a malformed answer never reaches routing."""

    # Keep in sync with config.DEPARTMENTS (a test checks this).
    department: Literal["billing", "technical", "shipping", "account", "general"]
    department_confidence: float = Field(ge=0, le=1, description="How sure you are about the department, 0 to 1")
    urgency_level: int = Field(ge=0, le=2, description="0 = can wait days, 1 = handle today, 2 = act now")
    policy_violation: bool = Field(description="Abuse, fraud or a legal threat")
    human_needed: bool = Field(description="A human agent must make a judgment call")
    standard_reply: bool = Field(description="A standard template reply fully resolves it")


TRIAGE_SYSTEM_PROMPT = (
    "You triage customer support tickets. Answer with JSON only.\n"
    "Departments: " + "; ".join(f"{k} = {v}" for k, v in DEPARTMENTS.items()) + ".\n"
    "Urgency levels: " + "; ".join(f"{i} = {lvl}" for i, lvl in enumerate(URGENCY_LEVELS)) + ".\n"
    "policy_violation: true if the ticket contains abuse, fraud or a legal threat.\n"
    "human_needed: true if a human agent must make a judgment call.\n"
    "standard_reply: true if a standard template reply fully resolves the ticket."
)

DRAFT_SYSTEM_PROMPT = (
    "You are a friendly customer support agent for an online store. Write a reply to the "
    "customer that resolves their issue with standard steps. Keep it under 120 words, plain "
    "text, no subject line. Never invent order details, account facts, prices or policies, "
    "and never promise a refund; use placeholders like [order number] where needed. "
    "Sign off as 'The Support Team'."
)


def estimate_cost(input_tokens: int, output_tokens: int, cfg: Settings = settings) -> float:
    """USD cost of one call, using the per-1M-token prices from config."""
    return (input_tokens * cfg.gemini_input_price_per_m + output_tokens * cfg.gemini_output_price_per_m) / 1_000_000


def token_usage(response) -> tuple[int, int]:
    """(input_tokens, output_tokens) from a Gemini response. Output includes thinking."""
    usage = response.usage_metadata
    if usage is None:
        return 0, 0
    input_tokens = usage.prompt_token_count or 0
    output_tokens = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
    return input_tokens, output_tokens


def ticket_prompt(subject: str, body: str) -> str:
    return f"Subject: {subject}\n\n{body}"


class GeminiClient:
    def __init__(self, cfg: Settings = settings):
        from google import genai
        from google.genai import types

        self.cfg = cfg
        self.types = types
        self.client = genai.Client(api_key=cfg.gemini_api_key, http_options=types.HttpOptions(timeout=TIMEOUT_MS))

    def _generate(self, prompt: str, system: str, json_schema: dict | None = None):
        """One Gemini call. Returns (response, latency_ms, input_tokens, output_tokens)."""
        config = self.types.GenerateContentConfig(
            system_instruction=system,
            thinking_config=self.types.ThinkingConfig(thinking_level=self.cfg.gemini_thinking_level),
            # For triage, force JSON that matches our schema.
            response_mime_type="application/json" if json_schema else None,
            response_json_schema=json_schema,
            # We use no tools, so turn off the SDK's automatic function calling.
            automatic_function_calling=self.types.AutomaticFunctionCallingConfig(disable=True),
        )
        start = time.perf_counter()
        response = self.client.models.generate_content(model=self.cfg.gemini_model, contents=prompt, config=config)
        latency_ms = (time.perf_counter() - start) * 1000
        return (response, latency_ms, *token_usage(response))

    def draft_reply(self, subject: str, body: str, department: str) -> DraftResult:
        prompt = f"Department: {department}\n\nCustomer ticket:\n{ticket_prompt(subject, body)}"
        response, latency_ms, input_tokens, output_tokens = self._generate(prompt, DRAFT_SYSTEM_PROMPT)
        text = (response.text or "").strip()
        if not text:
            raise RuntimeError("Gemini returned an empty draft")
        return DraftResult(
            text=text,
            model=self.cfg.gemini_model,
            latency_ms=round(latency_ms, 1),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=estimate_cost(input_tokens, output_tokens, self.cfg),
        )

    def gemini_only_triage(self, subject: str, body: str, max_attempts: int = 2) -> TriageResult:
        """Triage with Gemini alone. Retries once if the JSON doesn't validate.

        Latency, tokens and cost add up across attempts, because a retry is a real cost.
        """
        schema = GeminiTriageOutput.model_json_schema()
        total_ms = total_in = total_out = 0
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            response, latency_ms, input_tokens, output_tokens = self._generate(
                ticket_prompt(subject, body), TRIAGE_SYSTEM_PROMPT, json_schema=schema
            )
            total_ms += latency_ms
            total_in += input_tokens
            total_out += output_tokens
            try:
                parsed = GeminiTriageOutput.model_validate_json(response.text or "")
                return to_triage_result(parsed, total_ms, total_in, total_out, self.cfg)
            except ValidationError as e:
                last_error = e
                log.warning("Gemini triage JSON invalid (attempt %d/%d): %s", attempt, max_attempts, e)

        raise RuntimeError(f"Gemini triage failed validation after {max_attempts} attempts: {last_error}")


def to_triage_result(out: GeminiTriageOutput, latency_ms: float, input_tokens: int,
                     output_tokens: int, cfg: Settings = settings) -> TriageResult:
    """Map Gemini's answer onto the same TriageResult that Laya produces.

    Gemini gives booleans, not probabilities, so yes/no become 1.0/0.0. Its department
    confidence is self-reported, which is less trustworthy than Laya's probabilities
    (the benchmark measures how well each is calibrated).
    """
    return TriageResult(
        department=out.department,
        department_prob=out.department_confidence,
        department_confidence=out.department_confidence,
        department_probs={out.department: out.department_confidence},
        urgency_level=out.urgency_level,
        urgency_score=float(out.urgency_level),
        urgency_probs={str(out.urgency_level): 1.0},
        policy_violation_prob=1.0 if out.policy_violation else 0.0,
        human_needed_prob=1.0 if out.human_needed else 0.0,
        standard_reply_prob=1.0 if out.standard_reply else 0.0,
        source="gemini_fallback",
        latency_ms=round(latency_ms, 1),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=estimate_cost(input_tokens, output_tokens, cfg),
    )


@lru_cache(maxsize=1)
def get_gemini_client() -> GeminiClient | None:
    """Shared GeminiClient, or None if no API key is configured."""
    if not settings.gemini_api_key:
        log.warning("GEMINI_API_KEY not set: drafts and fallback triage are disabled")
        return None
    return GeminiClient()
