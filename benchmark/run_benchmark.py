"""Run both pipelines over the labeled set and save every prediction.

  Pipeline A "gemini": Gemini-only triage (the baseline), then routing
  Pipeline B "laya":   Laya triage (ours), then the same routing

Both use the app's own code (LayaClient, GeminiClient, routing.route), so the benchmark
measures what the app actually does. Predictions go to benchmark/results/predictions_<name>.csv.
Tickets already in that file are skipped, so an interrupted run resumes where it stopped.

Drafts: each pipeline drafts a reply only when routing says so. Drafting the same ticket
costs the same whichever pipeline triaged it, so instead of drafting every ticket twice we
draft a sample (--draft-sample) and use the measured mean draft cost and latency.

Usage (from the repo root):
  .venv\\Scripts\\python benchmark\\run_benchmark.py              # full run, then evaluate
  .venv\\Scripts\\python benchmark\\run_benchmark.py --limit 10   # quick smoke run
"""
import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))  # so "import app..." works from here
os.environ.setdefault("USE_TF", "0")

from app.gemini_client import GeminiClient  # noqa: E402
from app.routing import route  # noqa: E402

LABELED = REPO_ROOT / "data" / "benchmark_labeled.csv"
RESULTS = REPO_ROOT / "benchmark" / "results"

PREDICTION_FIELDS = [
    "id", "department", "department_prob", "urgency_level", "urgency_score",
    "policy_violation_prob", "human_needed_prob", "standard_reply_prob",
    "policy_violation", "human_needed", "standard_reply", "queue", "needs_review", "should_draft",
    "latency_ms", "input_tokens", "output_tokens", "cost_usd", "truncated", "error",
]
DRAFT_FIELDS = ["id", "latency_ms", "input_tokens", "output_tokens", "cost_usd", "text"]


def load_tickets(limit: int | None) -> list[dict]:
    with open(LABELED, newline="", encoding="utf-8") as f:
        tickets = list(csv.DictReader(f))
    return tickets[:limit] if limit else tickets


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {row["id"] for row in csv.DictReader(f)}


def prediction_row(ticket_id: str, triage) -> dict:
    decision = route(triage)
    return {
        "id": ticket_id,
        **{k: getattr(triage, k) for k in (
            "department", "department_prob", "urgency_level", "urgency_score", "policy_violation_prob",
            "human_needed_prob", "standard_reply_prob", "latency_ms", "input_tokens", "output_tokens",
            "cost_usd", "truncated")},
        **{k: getattr(decision, k) for k in (
            "policy_violation", "human_needed", "standard_reply", "queue", "needs_review", "should_draft")},
        "error": "",
    }


def with_retries(call, attempts: int = 6):
    """Retry Gemini calls that fail for temporary reasons (rate limit, overload), with backoff.

    Waiting time isn't counted in latency: the client times only the successful call.
    """
    from google.genai import errors

    for attempt in range(attempts):
        try:
            return call()
        except errors.APIError as e:
            if e.code not in (429, 500, 503) or attempt == attempts - 1:
                raise
            wait = 2 ** attempt * 5
            print(f"  Gemini {e.code}, retrying in {wait}s")
            time.sleep(wait)


class Writer:
    """Appends rows to a CSV, writing the header only for a new file."""

    def __init__(self, path: Path, fields: list[str]):
        new = not path.exists()
        self.file = open(path, "a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=fields)
        if new:
            self.writer.writeheader()

    def write(self, row: dict) -> None:
        self.writer.writerow(row)
        self.file.flush()  # keep progress if the run is interrupted

    def close(self) -> None:
        self.file.close()


def run_laya(tickets: list[dict]) -> None:
    from app.laya_client import LayaClient

    path = RESULTS / "predictions_laya.csv"
    todo = [t for t in tickets if t["id"] not in done_ids(path)]
    if not todo:
        print("Laya: all tickets already done")
        return

    print("Laya: loading model...")
    client = LayaClient()
    print(f"Laya: loaded in {client.load_seconds:.0f}s, triaging {len(todo)} tickets (one at a time)")
    out = Writer(path, PREDICTION_FIELDS)
    for i, t in enumerate(todo, 1):
        try:
            out.write(prediction_row(t["id"], client.triage(t["subject"], t["body"])))
        except Exception as e:  # record the failure and carry on
            out.write({"id": t["id"], "error": f"{type(e).__name__}: {e}"})
        if i % 25 == 0 or i == len(todo):
            print(f"  Laya {i}/{len(todo)}")
    out.close()


def run_gemini(tickets: list[dict], gemini: GeminiClient, workers: int) -> None:
    path = RESULTS / "predictions_gemini.csv"
    todo = [t for t in tickets if t["id"] not in done_ids(path)]
    if not todo:
        print("Gemini: all tickets already done")
        return

    print(f"Gemini: triaging {len(todo)} tickets ({workers} in parallel)")
    out = Writer(path, PREDICTION_FIELDS)

    def one(t):
        try:
            return prediction_row(t["id"], with_retries(lambda: gemini.gemini_only_triage(t["subject"], t["body"])))
        except Exception as e:
            return {"id": t["id"], "error": f"{type(e).__name__}: {e}"}

    # Parallel calls finish sooner; each call's own latency is still timed on its own.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, t) for t in todo]
        for i, future in enumerate(as_completed(futures), 1):
            out.write(future.result())
            if i % 25 == 0 or i == len(todo):
                print(f"  Gemini {i}/{len(todo)}")
    out.close()


def run_draft_sample(tickets: list[dict], gemini: GeminiClient, sample_size: int) -> None:
    """Draft replies for a sample of tickets that Laya-first routing would draft."""
    path = RESULTS / "drafts_sample.csv"
    already = done_ids(path)
    if len(already) >= sample_size:
        print(f"Drafts: sample of {len(already)} already done")
        return

    laya_path = RESULTS / "predictions_laya.csv"
    gemini_path = RESULTS / "predictions_gemini.csv"
    draft_ids = set()
    for p in (laya_path, gemini_path):
        if p.exists():
            with open(p, newline="", encoding="utf-8") as f:
                draft_ids |= {r["id"] for r in csv.DictReader(f) if r["should_draft"] == "True"}

    by_id = {t["id"]: t for t in tickets}
    todo = [by_id[i] for i in sorted(draft_ids - already, key=int) if i in by_id][: sample_size - len(already)]
    print(f"Drafts: writing {len(todo)} sample drafts")
    out = Writer(path, DRAFT_FIELDS)
    for t in todo:
        department = t["department"]
        draft = with_retries(lambda: gemini.draft_reply(t["subject"], t["body"], department))
        out.write({"id": t["id"], **draft.model_dump(include={"latency_ms", "input_tokens", "output_tokens", "cost_usd", "text"})})
    out.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="only the first N tickets (quick test)")
    parser.add_argument("--pipelines", default="laya,gemini", help="comma-separated: laya,gemini")
    parser.add_argument("--workers", type=int, default=4, help="parallel Gemini calls")
    parser.add_argument("--draft-sample", type=int, default=20, help="real drafts used to measure draft cost")
    parser.add_argument("--fresh", action="store_true", help="delete previous results first")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.fresh:
        for f in RESULTS.glob("*.csv"):
            f.unlink()

    tickets = load_tickets(args.limit)
    gemini = GeminiClient()
    pipelines = args.pipelines.split(",")
    started = time.perf_counter()

    if "gemini" in pipelines:
        run_gemini(tickets, gemini, args.workers)
    if "laya" in pipelines:
        run_laya(tickets)
    if args.draft_sample:
        run_draft_sample(tickets, gemini, args.draft_sample)

    print(f"\nCollected predictions in {(time.perf_counter() - started) / 60:.1f} min")
    import evaluate
    evaluate.main(limit=args.limit)


if __name__ == "__main__":
    main()
