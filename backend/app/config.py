"""All tunable settings in one place.

Values come from environment variables (loaded from the repo-root .env file), with
safe defaults, so thresholds and feature flags can change without touching code.
"""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# .env lives at the repo root: backend/app/config.py -> parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

# Department name -> short description. Laya reads the descriptions as part of the
# question, so they are kept short (all question text must fit in ~192 tokens).
DEPARTMENTS: dict[str, str] = {
    "billing": "charges, invoices, refunds, payments",
    "technical": "bugs, errors, outages, product setup and how-to",
    "shipping": "delivery, tracking, returns, lost or damaged package",
    "account": "login, password, profile, account access",
    "general": "sales questions and anything else",
}

# Urgency levels for Laya's score question: index = level.
URGENCY_LEVELS: list[str] = ["can wait days", "handle today", "act now"]

# Special queues that aren't departments.
ESCALATION_QUEUE = "escalation"
GENERAL_TRIAGE_QUEUE = "general_triage"


def _get_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _get_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _get_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    # Laya
    use_laya: bool = True
    laya_model: str = "convaiinnovations/laya"
    laya_device: str | None = None  # None = let laya pick cuda/mps/cpu
    max_ticket_tokens: int = 300

    # Routing thresholds (all compared against probabilities from 0 to 1)
    min_confidence: float = 0.5    # below -> general_triage
    auto_confidence: float = 0.8   # at/above -> auto-assign; between -> review
    noul_threshold: float = 0.5    # P(yes) needed for human_needed / standard_reply
    policy_threshold: float = 0.5  # P(yes) needed to flag a policy violation

    preload_laya: bool = True      # load the model at API startup, not on the first request

    # Gemini
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"
    gemini_thinking_level: str = "low"
    gemini_input_price_per_m: float = 0.75   # USD per 1M input tokens
    gemini_output_price_per_m: float = 3.75  # USD per 1M output tokens (thinking included)

    # API
    database_url: str = "sqlite:///./tickets.db"
    cors_origins: tuple[str, ...] = ("http://localhost:5173",)


def load_settings() -> Settings:
    """Build Settings from environment variables, using the defaults above."""
    d = Settings()
    return Settings(
        use_laya=_get_bool("USE_LAYA", d.use_laya),
        laya_model=os.getenv("LAYA_MODEL", d.laya_model),
        laya_device=os.getenv("LAYA_DEVICE") or None,
        max_ticket_tokens=_get_int("MAX_TICKET_TOKENS", d.max_ticket_tokens),
        min_confidence=_get_float("MIN_CONFIDENCE", d.min_confidence),
        auto_confidence=_get_float("AUTO_CONFIDENCE", d.auto_confidence),
        noul_threshold=_get_float("NOUL_THRESHOLD", d.noul_threshold),
        policy_threshold=_get_float("POLICY_THRESHOLD", d.policy_threshold),
        preload_laya=_get_bool("PRELOAD_LAYA", d.preload_laya),
        gemini_api_key=os.getenv("GEMINI_API_KEY", d.gemini_api_key),
        gemini_model=os.getenv("GEMINI_MODEL", d.gemini_model),
        gemini_thinking_level=os.getenv("GEMINI_THINKING_LEVEL", d.gemini_thinking_level),
        gemini_input_price_per_m=_get_float("GEMINI_INPUT_PRICE_PER_M", d.gemini_input_price_per_m),
        gemini_output_price_per_m=_get_float("GEMINI_OUTPUT_PRICE_PER_M", d.gemini_output_price_per_m),
        database_url=os.getenv("DATABASE_URL", d.database_url),
        cors_origins=tuple(
            o.strip() for o in os.getenv("CORS_ORIGINS", ",".join(d.cors_origins)).split(",") if o.strip()
        ),
    )


# The app imports this single shared instance.
settings = load_settings()
