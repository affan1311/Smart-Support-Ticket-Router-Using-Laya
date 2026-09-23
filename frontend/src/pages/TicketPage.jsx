import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { approveDraft, getTicket, overrideDecision } from "../api.js";
import { DEPARTMENTS, QUEUES, URGENCY_LABELS, queueLabel } from "../constants.js";
import usePolling from "../usePolling.js";
import {
  Button, Card, ErrorBox, Pill, ProbBar, UrgencyBadge, formatMs, formatUsd, timeAgo,
} from "../ui.jsx";

export default function TicketPage() {
  const { id } = useParams();
  const [pollMs, setPollMs] = useState(2000);
  const { data: ticket, error, reload } = usePolling(() => getTicket(id), [id], pollMs);

  // Poll only while the ticket is still being triaged (or its draft may still be coming).
  useEffect(() => {
    if (!ticket) return;
    const waitingForDraft = ticket.decisions.at(-1)?.should_draft && ticket.drafts.length === 0 && !ticket.error;
    setPollMs(ticket.status === "new" || waitingForDraft ? 2000 : null);
  }, [ticket]);

  if (error && !ticket) return <ErrorBox message={error} />;
  if (!ticket) return <p className="text-muted">Loading…</p>;

  const decision = ticket.decisions.at(-1); // latest triage run

  return (
    <div className="space-y-4">
      <Link to="/" className="text-sm text-accent hover:underline">← Back to queue</Link>

      <header className="space-y-2">
        <h1 className="text-xl font-semibold">{ticket.subject}</h1>
        <div className="flex flex-wrap items-center gap-3 text-sm text-ink-2">
          <UrgencyBadge level={ticket.urgency_level} score={ticket.urgency_score} />
          <span>Queue: <strong className="text-ink">{queueLabel(ticket.queue)}</strong></span>
          <Pill>{ticket.status}</Pill>
          {ticket.needs_review && <Pill>needs review</Pill>}
          {ticket.policy_violation && <Pill className="border-critical/50 text-critical-text">policy violation</Pill>}
          <span className="text-xs text-muted">
            {ticket.customer_email ?? "no email"} · via {ticket.channel} · {timeAgo(ticket.created_at)}
          </span>
        </div>
      </header>

      <ErrorBox message={ticket.error} />

      <div className="grid gap-4 lg:grid-cols-5">
        <div className="space-y-4 lg:col-span-3">
          <Card title="Message">
            <p className="whitespace-pre-wrap text-sm leading-relaxed">{ticket.body}</p>
          </Card>
          <DraftPanel ticket={ticket} decision={decision} onChange={reload} />
          <OverrideHistory overrides={ticket.overrides} />
        </div>

        <div className="space-y-4 lg:col-span-2">
          {ticket.status === "new" ? (
            <Card title="Model decision"><p className="text-sm text-muted">Triaging…</p></Card>
          ) : (
            decision && <DecisionPanel decision={decision} />
          )}
          {ticket.status !== "new" && <CorrectionPanel ticket={ticket} onChange={reload} />}
        </div>
      </div>
    </div>
  );
}

/** What the model answered, with every probability, so agents can see why it routed this way. */
function DecisionPanel({ decision }) {
  const departments = Object.entries(decision.department_probs).sort((a, b) => b[1] - a[1]);
  const yesNo = [
    ["Policy violation", decision.policy_violation_prob],
    ["Human needed", decision.human_needed_prob],
    ["Standard reply", decision.standard_reply_prob],
  ];

  return (
    <Card title="Model decision">
      <div className="space-y-4 text-sm">
        <div className="flex flex-wrap gap-2">
          <Pill>{decision.source === "laya" ? "Laya" : "Gemini fallback"}</Pill>
          <Pill className="tabular">{formatMs(decision.latency_ms)}</Pill>
          <Pill className="tabular">{formatUsd(decision.cost_usd)}</Pill>
          {decision.truncated && <Pill>ticket truncated</Pill>}
        </div>

        {decision.laya_error && (
          <p className="text-xs text-ink-2">Fell back to Gemini because Laya failed: {decision.laya_error}</p>
        )}

        <Section label="Department">
          {departments.map(([name, prob]) => (
            <Row key={name} label={name} strong={name === decision.department}>
              <ProbBar value={prob} label={name} />
            </Row>
          ))}
        </Section>

        <Section label="Urgency">
          {URGENCY_LABELS.map((label, level) => (
            <Row key={level} label={label} strong={level === decision.urgency_level}>
              <ProbBar value={decision.urgency_probs[String(level)] ?? 0} label={label} />
            </Row>
          ))}
        </Section>

        <Section label="Yes / no (probability of yes)">
          {yesNo.map(([label, prob]) => (
            <Row key={label} label={label}><ProbBar value={prob} label={label} /></Row>
          ))}
        </Section>

        <Section label="Routing">
          <ul className="list-disc space-y-0.5 pl-5 text-ink-2">
            {decision.reasons.map((reason) => <li key={reason}>{reason}</li>)}
          </ul>
        </Section>
      </div>
    </Card>
  );
}

function Section({ label, children }) {
  return (
    <div>
      <h3 className="mb-1.5 text-xs font-medium text-muted">{label}</h3>
      <div className="space-y-1">{children}</div>
    </div>
  );
}

function Row({ label, strong = false, children }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <span className={strong ? "font-medium" : "text-ink-2"}>{label}</span>
      {children}
    </div>
  );
}

/** The Gemini draft: the agent edits and approves it. Nothing is sent to the customer in v1. */
function DraftPanel({ ticket, decision, onChange }) {
  const draft = ticket.drafts.at(-1);
  const [text, setText] = useState("");
  const [error, setError] = useState(null);
  const [saving, setSaving] = useState(false);

  // Load the draft into the editor when it arrives.
  useEffect(() => {
    if (draft) setText(draft.edited_text ?? draft.draft_text);
  }, [draft?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!draft) {
    let reason = "No draft for this ticket.";
    if (ticket.status === "new") reason = "Waiting for triage…";
    else if (decision?.should_draft && !ticket.error) reason = "Drafting a reply…";
    else if (decision && !decision.should_draft) reason = "No draft: routing decided this ticket needs a person to write the reply.";
    return <Card title="Reply draft"><p className="text-sm text-muted">{reason}</p></Card>;
  }

  const edited = text.trim() !== draft.draft_text.trim();

  async function approve() {
    setSaving(true);
    try {
      await approveDraft(ticket.id, edited ? text : null);
      setError(null);
      onChange();
    } catch (e) {
      setError(e.message);
    } finally {
      setSaving(false);
    }
  }

  return (
    <Card title="Reply draft">
      <div className="space-y-3">
        <textarea
          rows={9}
          value={text}
          onChange={(e) => setText(e.target.value)}
          disabled={draft.approved}
          className="w-full rounded-md border border-line bg-surface px-3 py-2 text-sm leading-relaxed text-ink disabled:opacity-70"
        />
        <ErrorBox message={error} />
        <div className="flex flex-wrap items-center justify-between gap-3">
          {draft.approved ? (
            <span className="text-sm text-good-text">
              ✓ Approved {draft.edited_text ? "with edits " : ""}{timeAgo(draft.approved_at)}
            </span>
          ) : (
            <div className="flex gap-2">
              <Button onClick={approve} disabled={saving}>{edited ? "Approve with edits" : "Approve"}</Button>
              {edited && (
                <Button variant="secondary" onClick={() => setText(draft.draft_text)}>Reset</Button>
              )}
            </div>
          )}
          <span className="tabular text-xs text-muted">
            {draft.model} · {formatMs(draft.latency_ms)} · {draft.input_tokens + draft.output_tokens} tokens · {formatUsd(draft.cost_usd)}
          </span>
        </div>
      </div>
    </Card>
  );
}

// Each field an agent can correct: its options, and how to read its current value.
const CORRECTABLE = [
  { field: "department", label: "Department", options: DEPARTMENTS.map((d) => [d, d]) },
  { field: "queue", label: "Queue", options: QUEUES.map((q) => [q, queueLabel(q)]) },
  { field: "urgency_level", label: "Urgency", options: URGENCY_LABELS.map((l, i) => [String(i), l]) },
  { field: "policy_violation", label: "Policy violation", options: [["true", "Yes"], ["false", "No"]] },
  { field: "human_needed", label: "Human needed", options: [["true", "Yes"], ["false", "No"]] },
];

/** Agent corrections. Each saved change is logged on the server as a labeled example. */
function CorrectionPanel({ ticket, onChange }) {
  const [error, setError] = useState(null);

  async function save(field, value) {
    try {
      await overrideDecision(ticket.id, field, value);
      setError(null);
      onChange();
    } catch (e) {
      setError(e.message);
    }
  }

  return (
    <Card title="Correct the model">
      <div className="space-y-2 text-sm">
        {CORRECTABLE.map((item) => (
          <CorrectionRow key={`${item.field}-${ticket[item.field]}`} item={item}
                         current={String(ticket[item.field])} onSave={save} />
        ))}
        {ticket.needs_review && (
          // Confirming the department clears "review" without counting as a correction.
          <Button variant="secondary" className="mt-2 w-full" onClick={() => save("department", ticket.department)}>
            Model is right: mark reviewed
          </Button>
        )}
        <ErrorBox message={error} />
      </div>
    </Card>
  );
}

function CorrectionRow({ item, current, onSave }) {
  const [value, setValue] = useState(current);
  return (
    <div className="flex items-center gap-2">
      <span className="w-32 shrink-0 text-ink-2">{item.label}</span>
      <select value={value} onChange={(e) => setValue(e.target.value)}
              className="min-w-0 flex-1 rounded-md border border-line bg-surface px-2 py-1 text-ink">
        {item.options.map(([optionValue, optionLabel]) => (
          <option key={optionValue} value={optionValue}>{optionLabel}</option>
        ))}
      </select>
      <Button variant="secondary" disabled={value === current} onClick={() => onSave(item.field, value)}>
        Save
      </Button>
    </div>
  );
}

function OverrideHistory({ overrides }) {
  if (overrides.length === 0) return null;
  return (
    <Card title="Corrections">
      <ul className="space-y-1 text-sm">
        {overrides.map((o) => (
          <li key={o.id} className="text-ink-2">
            <strong className="text-ink">{o.agent_id}</strong> changed {o.field.replace("_", " ")} from{" "}
            <code>{o.old_value ?? "—"}</code> to <code>{o.new_value}</code>{" "}
            <span className="text-xs text-muted">{timeAgo(o.created_at)}</span>
          </li>
        ))}
      </ul>
    </Card>
  );
}
