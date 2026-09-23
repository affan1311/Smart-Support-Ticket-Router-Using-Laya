"""Run the 10 sample tickets through Laya + routing and print the results.

Usage (from the backend/ folder):
    python -m scripts.run_samples
"""
import csv
import statistics

from app.config import REPO_ROOT
from app.laya_client import get_laya_client
from app.routing import route

CSV_PATH = REPO_ROOT / "data" / "sample_tickets.csv"


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile, good enough for a handful of samples."""
    ordered = sorted(values)
    index = max(0, round(pct / 100 * len(ordered)) - 1)
    return ordered[index]


def main() -> None:
    print("Loading Laya model (first run downloads ~1.7 GB)...")
    client = get_laya_client()
    print(f"Loaded in {client.load_seconds:.1f}s on {client.agent.device}\n")

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        tickets = list(csv.DictReader(f))

    header = f"{'id':>2} {'department':<11}{'p':>5} {'ok':<3}{'urg':>5} {'pol':>5} {'hum':>5} {'std':>5}  {'queue':<15}{'review':<7}{'draft':<6}{'ms':>6}"
    print(header)
    print("-" * len(header))

    latencies, correct = [], 0
    for t in tickets:
        triage = client.triage(t["subject"], t["body"])
        decision = route(triage)
        latencies.append(triage.latency_ms)

        ok = triage.department == t["expected_department"]
        correct += ok
        print(
            f"{t['id']:>2} {triage.department:<11}{triage.department_prob:>5.2f} {'Y' if ok else 'N':<3}"
            f"{triage.urgency_score:>5.2f} {triage.policy_violation_prob:>5.2f} "
            f"{triage.human_needed_prob:>5.2f} {triage.standard_reply_prob:>5.2f}  "
            f"{decision.queue:<15}{'yes' if decision.needs_review else '':<7}"
            f"{'yes' if decision.should_draft else '':<6}{triage.latency_ms:>6.0f}"
            + ("  (truncated)" if triage.truncated else "")
        )

    print("\nColumns: p = department probability, ok = matches expected_department, urg = urgency 0-2,")
    print("pol/hum/std = P(yes) for policy_violation / human_needed / standard_reply.")
    print(f"\nDepartment correct: {correct}/{len(tickets)}")
    print(f"Laya latency: p50 {statistics.median(latencies):.0f} ms, p95 {percentile(latencies, 95):.0f} ms")


if __name__ == "__main__":
    main()
