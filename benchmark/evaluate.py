"""Score the saved predictions and write the report and charts.

Reads  benchmark/results/predictions_{laya,gemini}.csv and drafts_sample.csv
Writes benchmark/results/summary.json, report.md, classification_*.txt and charts/*.png

It never calls a model, so you can re-run it for free after changing a chart or metric:
  .venv\\Scripts\\python benchmark\\evaluate.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix, f1_score  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))
from app.config import DEPARTMENTS  # noqa: E402

RESULTS = REPO_ROOT / "benchmark" / "results"
CHARTS = RESULTS / "charts"
LABELED = REPO_ROOT / "data" / "benchmark_labeled.csv"

PIPELINES = {"laya": "Laya-first", "gemini": "Gemini-only"}
LABELS = list(DEPARTMENTS)

# Chart style: colorblind-safe blue/orange pair (validated), quiet axes, ink-colored text.
COLORS = {"laya": "#2a78d6", "gemini": "#eb6834"}
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": INK_2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlelocation": "left", "axes.titlesize": 12, "axes.titleweight": "bold",
})


def yes(series: pd.Series) -> pd.Series:
    """'yes'/'True' -> True, 'no'/'False' -> False, '' -> NaN (unknown label)."""
    return series.astype(str).str.lower().map({"yes": True, "true": True, "no": False, "false": False})


def load(name: str, labeled: pd.DataFrame) -> pd.DataFrame:
    # Read everything as text ("True"/"False", numbers) and convert explicitly where needed,
    # so pandas' type guessing can't change behavior between runs.
    preds = pd.read_csv(RESULTS / f"predictions_{name}.csv", dtype=str, keep_default_na=False)
    preds["id"] = preds["id"].astype(int)
    df = labeled.merge(preds, on="id", suffixes=("", "_pred"))
    df["ok"] = df["error"] == ""
    return df


def ece(confidence: pd.Series, correct: pd.Series, bins: int = 10) -> float:
    """Expected calibration error: the average gap between confidence and accuracy, weighted by bin size."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(confidence, edges) - 1, 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            total += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
    return float(total)


def score(df: pd.DataFrame, name: str, draft: dict) -> dict:
    ok = df[df["ok"]].copy()
    hard, public = ok[ok["set"] == "hard"], ok[ok["set"] != "hard"]
    ok["dept_correct"] = ok["department"] == ok["department_pred"]

    # Urgency labels exist only for the Tobi technical tickets and the hard set.
    with_urgency = ok[pd.to_numeric(ok["urgency"], errors="coerce").notna()]
    urgency_true = pd.to_numeric(with_urgency["urgency"]).astype(int)
    urgency_pred = with_urgency["urgency_level"].astype(int)

    def dept_accuracy(subset):
        return float((subset["department"] == subset["department_pred"]).mean())

    # Yes/no labels exist only where the dataset has them (see build_dataset.py).
    def yes_no_accuracy(column):
        truth = yes(hard[column])
        return float((truth == hard[f"{column}_pred"].map({"True": True, "False": False})).mean())

    policy_truth = yes(hard["policy_violation"])
    policy_pred = hard["policy_violation_pred"] == "True"
    triage_ms = ok["latency_ms"].astype(float)
    should_draft = ok["should_draft"] == "True"
    triage_cost = ok["cost_usd"].astype(float)

    # Per ticket: triage, plus a draft (at the measured mean cost/latency) when routing asks for one.
    ticket_cost = triage_cost + should_draft * draft["mean_cost_usd"]
    ticket_ms = triage_ms + should_draft * draft["mean_latency_ms"]
    # A ticket "skips Gemini" only if Laya triaged it and no draft was needed.
    gemini_skipped = (~should_draft).mean() if name == "laya" else 0.0

    return {
        "tickets": int(len(df)),
        "errors": int((~df["ok"]).sum()),
        "department_accuracy": float(ok["dept_correct"].mean()),
        "department_accuracy_bitext": dept_accuracy(ok[ok["set"] == "bitext"]),
        "department_accuracy_tobi": dept_accuracy(ok[ok["set"] == "tobi"]),
        "department_accuracy_hard": dept_accuracy(hard),
        "department_macro_f1": float(f1_score(ok["department"], ok["department_pred"], labels=LABELS,
                                              average="macro", zero_division=0)),
        "urgency_labeled_tickets": int(len(with_urgency)),
        "urgency_accuracy_3_level": float((urgency_true == urgency_pred).mean()),
        "urgency_act_now_accuracy": float(((urgency_true == 2) == (urgency_pred == 2)).mean()),
        "policy_recall_hard": float((policy_pred & policy_truth).sum() / max(1, policy_truth.sum())),
        "policy_precision_hard": float((policy_pred & policy_truth).sum() / max(1, policy_pred.sum())),
        "policy_false_alarm_rate_public": float((public["policy_violation_pred"] == "True").mean()),
        "human_needed_accuracy_hard": yes_no_accuracy("human_needed"),
        "standard_reply_accuracy_hard": yes_no_accuracy("standard_reply"),
        "needs_review_rate": float((ok["needs_review"] == "True").mean()),
        "draft_rate": float(should_draft.mean()),
        "gemini_skipped_pct": float(100 * gemini_skipped),
        "triage_p50_ms": float(triage_ms.median()),
        "triage_p95_ms": float(triage_ms.quantile(0.95)),
        "end_to_end_p50_ms": float(ticket_ms.median()),
        "end_to_end_p95_ms": float(ticket_ms.quantile(0.95)),
        "triage_cost_per_1000_usd": float(1000 * triage_cost.mean()),
        "draft_cost_per_1000_usd": float(1000 * (should_draft * draft["mean_cost_usd"]).mean()),
        "cost_per_1000_usd": float(1000 * ticket_cost.mean()),
        "department_ece": ece(ok["department_prob"].astype(float), ok["dept_correct"]),
        "truncated": int((ok["truncated"] == "True").sum()),
    }


def draft_stats() -> dict:
    path = RESULTS / "drafts_sample.csv"
    if not path.exists():
        return {"sample_size": 0, "mean_cost_usd": 0.0, "mean_latency_ms": 0.0}
    drafts = pd.read_csv(path)
    return {"sample_size": int(len(drafts)), "mean_cost_usd": float(drafts["cost_usd"].mean()),
            "mean_latency_ms": float(drafts["latency_ms"].mean()),
            "mean_output_tokens": float(drafts["output_tokens"].mean())}


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def save(fig, name: str) -> None:
    fig.savefig(CHARTS / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def grouped_hbar(ax, categories: list[str], values: dict[str, list[float]], fmt) -> None:
    """Horizontal grouped bars, one bar per pipeline, with a direct label on each bar."""
    y = np.arange(len(categories))
    height = 0.36
    for i, (name, vals) in enumerate(values.items()):
        offset = (i - 0.5) * (height + 0.03)  # a small gap between the paired bars
        bars = ax.barh(y + offset, vals, height=height, color=COLORS[name], label=PIPELINES[name])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, " " + fmt(v),
                    va="center", fontsize=9, color=INK_2)
    ax.set_yticks(y, categories)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2)


def chart_accuracy(s: dict) -> None:
    metrics = [
        ("department_accuracy", "Department (all)"),
        ("department_macro_f1", "Department macro-F1"),
        ("urgency_act_now_accuracy", "Urgency: act now vs not"),
        ("urgency_accuracy_3_level", "Urgency: 3 levels"),
        ("policy_recall_hard", "Policy violations caught (hard set)"),
        ("human_needed_accuracy_hard", "Human needed (hard set)"),
        ("standard_reply_accuracy_hard", "Standard reply (hard set)"),
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    grouped_hbar(ax, [label for _, label in metrics],
                 {p: [s[p][key] for key, _ in metrics] for p in PIPELINES}, lambda v: f"{v:.0%}")
    ax.set_xlim(0, 1.12)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Accuracy by decision", pad=24)
    save(fig, "accuracy.png")


def chart_latency(s: dict) -> None:
    metrics = [("triage_p50_ms", "Triage p50"), ("triage_p95_ms", "Triage p95"),
               ("end_to_end_p50_ms", "With drafting p50"), ("end_to_end_p95_ms", "With drafting p95")]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    grouped_hbar(ax, [label for _, label in metrics],
                 {p: [s[p][key] / 1000 for key, _ in metrics] for p in PIPELINES}, lambda v: f"{v:.1f} s")
    ax.set_xlim(0, max(s[p][k] for p in PIPELINES for k, _ in metrics) / 1000 * 1.18)
    ax.set_xlabel("seconds per ticket")
    ax.set_title("Latency per ticket", pad=24)
    save(fig, "latency.png")


def chart_cost(s: dict) -> None:
    """Stacked bars: triage cost + drafting cost per 1,000 tickets."""
    fig, ax = plt.subplots(figsize=(8, 2.6))
    names = list(PIPELINES)
    y = np.arange(len(names))
    triage = [s[p]["triage_cost_per_1000_usd"] for p in names]
    drafts = [s[p]["draft_cost_per_1000_usd"] for p in names]
    ax.barh(y, triage, height=0.5, color=[COLORS[p] for p in names], edgecolor=SURFACE, linewidth=2)
    ax.barh(y, drafts, left=triage, height=0.5, color=[COLORS[p] for p in names], alpha=0.45,
            edgecolor=SURFACE, linewidth=2)
    for i, p in enumerate(names):
        total = s[p]["cost_per_1000_usd"]
        ax.text(total, i, f"  ${total:.2f}  (triage ${triage[i]:.2f} + drafts ${drafts[i]:.2f})",
                va="center", fontsize=9, color=INK_2)
    ax.set_yticks(y, [PIPELINES[p] for p in names])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(s[p]["cost_per_1000_usd"] for p in names) * 1.9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("USD per 1,000 tickets (solid = triage, light = drafts)")
    ax.set_title("Gemini cost per 1,000 tickets")
    save(fig, "cost.png")


def chart_confusion(df: pd.DataFrame, name: str) -> None:
    ok = df[df["ok"]]
    matrix = confusion_matrix(ok["department"], ok["department_pred"], labels=LABELS)
    row_share = matrix / np.maximum(1, matrix.sum(axis=1, keepdims=True))
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.imshow(row_share, cmap=BLUES, vmin=0, vmax=1)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            if matrix[i, j]:
                ax.text(j, i, matrix[i, j], ha="center", va="center", fontsize=10,
                        color="white" if row_share[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=30, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(f"{PIPELINES[name]}: department confusion (counts)")
    save(fig, f"confusion_{name}.png")


def chart_calibration(frames: dict[str, pd.DataFrame]) -> None:
    """Reliability diagram: when a pipeline says X% sure, how often is it right?"""
    fig, ax = plt.subplots(figsize=(5.6, 5))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="perfectly calibrated")
    edges = np.linspace(0, 1, 11)
    for name, df in frames.items():
        ok = df[df["ok"]]
        conf = ok["department_prob"].astype(float)
        correct = ok["department"] == ok["department_pred"]
        idx = np.clip(np.digitize(conf, edges) - 1, 0, 9)
        xs, ys = [], []
        for b in range(10):
            mask = idx == b
            if mask.sum() >= 5:  # skip bins too small to mean anything
                xs.append(conf[mask].mean())
                ys.append(correct[mask].mean())
        ax.plot(xs, ys, color=COLORS[name], linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=2, label=PIPELINES[name])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("stated department confidence")
    ax.set_ylabel("actually correct")
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Calibration (bins with 5+ tickets)")
    save(fig, "calibration.png")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(s: dict, draft: dict, counts: dict) -> None:
    pct = lambda v: f"{v:.1%}"  # noqa: E731
    sec = lambda v: f"{v / 1000:.2f} s"  # noqa: E731
    usd = lambda v: f"${v:.2f}"  # noqa: E731
    rows = [
        ("Department accuracy (all)", "department_accuracy", pct),
        ("Department accuracy (Bitext, 240)", "department_accuracy_bitext", pct),
        ("Department accuracy (Tobi technical, 60)", "department_accuracy_tobi", pct),
        ("Department accuracy (hard set)", "department_accuracy_hard", pct),
        ("Department macro-F1", "department_macro_f1", lambda v: f"{v:.3f}"),
        ("Urgency: act now vs not (Tobi + hard)", "urgency_act_now_accuracy", pct),
        ("Urgency: exact level 0/1/2 (Tobi + hard)", "urgency_accuracy_3_level", pct),
        ("Policy violations caught (hard set recall)", "policy_recall_hard", pct),
        ("Policy precision (hard set)", "policy_precision_hard", pct),
        ("Policy false alarms (public set)", "policy_false_alarm_rate_public", pct),
        ("Human needed accuracy (hard set)", "human_needed_accuracy_hard", pct),
        ("Standard reply accuracy (hard set)", "standard_reply_accuracy_hard", pct),
        ("Marked for review", "needs_review_rate", pct),
        ("Tickets drafted", "draft_rate", pct),
        ("Tickets that never call Gemini", "gemini_skipped_pct", lambda v: f"{v:.1f}%"),
        ("Triage latency p50", "triage_p50_ms", sec),
        ("Triage latency p95", "triage_p95_ms", sec),
        ("End-to-end p50 (incl. drafts)", "end_to_end_p50_ms", sec),
        ("End-to-end p95 (incl. drafts)", "end_to_end_p95_ms", sec),
        ("Gemini cost per 1,000 tickets", "cost_per_1000_usd", usd),
        ("  of which triage", "triage_cost_per_1000_usd", usd),
        ("  of which drafts", "draft_cost_per_1000_usd", usd),
        ("Department calibration error (ECE, lower is better)", "department_ece", lambda v: f"{v:.3f}"),
        ("Failed tickets", "errors", str),
        ("Tickets truncated for Laya", "truncated", str),
    ]
    table = "\n".join(f"| {label} | {fmt(s['laya'][key])} | {fmt(s['gemini'][key])} |" for label, key, fmt in rows)
    report = f"""# Benchmark: Laya-first vs Gemini-only

Generated by `benchmark/evaluate.py`. Re-run `benchmark/run_benchmark.py` to reproduce.

**Dataset:** {counts['total']} labeled tickets: {counts['bitext']} from Bitext (account, billing, shipping,
general), {counts['tobi']} technical tickets from Tobi-Bueck, and {counts['hard']} hand-written hard cases.
See `benchmark/build_dataset.py` for sources, filtering and label mapping.

| Metric | Laya-first | Gemini-only |
|---|---|---|
{table}

**How cost and latency are counted**
- Laya runs locally on CPU, so its triage costs $0 in API fees (hardware not included).
- Gemini cost uses logged token counts (thinking tokens billed as output) at the prices in `.env`.
- Drafting: each pipeline drafts only where its own routing says so. Draft cost and latency are the
  mean of {draft['sample_size']} real drafts: ${draft['mean_cost_usd']:.5f} and {draft['mean_latency_ms'] / 1000:.1f} s per draft.

**Label caveats**
- Bitext departments come from its intent labels (mapping in `build_dataset.py`). Bitext messages are
  short chat-style requests, easier than real email tickets.
- Tobi-Bueck's queue labels are noisy, so only technical tickets whose tags agree were kept; a few are
  still borderline (e.g. marketing tools). Its priority field is the urgency label.
- Policy / human-needed / standard-reply labels exist only for the hand-written hard set. Public tickets
  are assumed to contain no policy violations (swearing isn't one).
- Hard-set labels were written for this project and are a judgment call on ambiguous tickets.

![Accuracy](charts/accuracy.png)
![Latency](charts/latency.png)
![Cost](charts/cost.png)
![Confusion: Laya](charts/confusion_laya.png)
![Confusion: Gemini](charts/confusion_gemini.png)
![Calibration](charts/calibration.png)
"""
    (RESULTS / "report.md").write_text(report, encoding="utf-8")


def main(limit: int | None = None) -> None:
    CHARTS.mkdir(parents=True, exist_ok=True)
    labeled = pd.read_csv(LABELED, keep_default_na=False)
    if limit:
        labeled = labeled.head(limit)

    draft = draft_stats()
    frames = {name: load(name, labeled) for name in PIPELINES}
    summary = {name: score(df, name, draft) for name, df in frames.items()}
    summary["draft_sample"] = draft

    for name, df in frames.items():
        ok = df[df["ok"]]
        report = classification_report(ok["department"], ok["department_pred"], labels=LABELS, zero_division=0)
        (RESULTS / f"classification_{name}.txt").write_text(report, encoding="utf-8")
        chart_confusion(df, name)

    chart_accuracy(summary)
    chart_latency(summary)
    chart_cost(summary)
    chart_calibration(frames)
    counts = {"total": len(labeled), **{s: int((labeled["set"] == s).sum()) for s in ("bitext", "tobi", "hard")}}
    write_report(summary, draft, counts)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nWrote {RESULTS / 'report.md'} and charts in {CHARTS}")
    for name in PIPELINES:
        s = summary[name]
        print(f"{PIPELINES[name]:>12}: dept {s['department_accuracy']:.1%} | act-now {s['urgency_act_now_accuracy']:.1%}"
              f" | triage p50 {s['triage_p50_ms'] / 1000:.2f}s | ${s['cost_per_1000_usd']:.2f}/1k"
              f" | Gemini skipped {s['gemini_skipped_pct']:.0f}% | errors {s['errors']}")


if __name__ == "__main__":
    main()
