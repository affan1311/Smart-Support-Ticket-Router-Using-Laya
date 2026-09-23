import { useRef, useState } from "react";
import { Link } from "react-router-dom";
import { createTicket, listTickets, uploadCsv } from "../api.js";
import { QUEUES, STATUSES, URGENCY_LABELS, queueLabel } from "../constants.js";
import usePolling from "../usePolling.js";
import { Button, Card, ErrorBox, Pill, ProbBar, UrgencyBadge, timeAgo } from "../ui.jsx";

const EMPTY_FILTERS = { dept: "", urgency: "", needs_review: "", status: "" };

export default function QueuePage() {
  const [filters, setFilters] = useState(EMPTY_FILTERS);
  const [showForm, setShowForm] = useState(false);
  // The server already sorts by urgency (highest first), then oldest first.
  const { data: tickets, error, reload } = usePolling(() => listTickets(filters), [filters], 5000);

  const setFilter = (key) => (event) => setFilters({ ...filters, [key]: event.target.value });

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-xl font-semibold">
          Queue {tickets && <span className="tabular text-base font-normal text-muted">({tickets.length})</span>}
        </h1>
        <div className="flex gap-2">
          <CsvUpload onUploaded={reload} />
          <Button onClick={() => setShowForm(!showForm)}>{showForm ? "Close" : "New ticket"}</Button>
        </div>
      </div>

      {showForm && (
        <NewTicketForm
          onCreated={() => {
            setShowForm(false);
            reload();
          }}
        />
      )}

      {/* Filters sit in one row above the table */}
      <div className="flex flex-wrap items-end gap-3 text-sm">
        <Select label="Queue" value={filters.dept} onChange={setFilter("dept")}
                options={QUEUES.map((q) => [q, queueLabel(q)])} />
        <Select label="Urgency" value={filters.urgency} onChange={setFilter("urgency")}
                options={URGENCY_LABELS.map((label, level) => [String(level), label])} />
        <Select label="Status" value={filters.status} onChange={setFilter("status")}
                options={STATUSES.map((s) => [s, s])} />
        <Select label="Review" value={filters.needs_review} onChange={setFilter("needs_review")}
                options={[["true", "Needs review"], ["false", "No review"]]} />
        {Object.values(filters).some(Boolean) && (
          <Button variant="secondary" onClick={() => setFilters(EMPTY_FILTERS)}>Clear</Button>
        )}
      </div>

      <ErrorBox message={error} />

      <Card className="overflow-x-auto p-0">
        <table className="w-full min-w-[760px] text-left text-sm">
          <thead className="border-b border-line text-xs text-muted">
            <tr>
              <th className="px-4 py-2 font-medium">Urgency</th>
              <th className="px-4 py-2 font-medium">Subject</th>
              <th className="px-4 py-2 font-medium">Queue</th>
              <th className="px-4 py-2 font-medium">Confidence</th>
              <th className="px-4 py-2 font-medium">Flags</th>
              <th className="px-4 py-2 font-medium">Received</th>
            </tr>
          </thead>
          <tbody>
            {tickets?.map((t) => (
              <tr key={t.id} className="border-b border-line last:border-0 hover:bg-line/30">
                <td className="px-4 py-2.5"><UrgencyBadge level={t.urgency_level} score={t.urgency_score} /></td>
                <td className="max-w-xs px-4 py-2.5">
                  <Link to={`/tickets/${t.id}`} className="block truncate font-medium hover:text-accent">
                    {t.subject}
                  </Link>
                  <span className="block truncate text-xs text-muted">{t.customer_email ?? `#${t.id}`}</span>
                </td>
                <td className="px-4 py-2.5 whitespace-nowrap">{queueLabel(t.queue)}</td>
                <td className="px-4 py-2.5"><ProbBar value={t.department_prob} label="Department confidence" width="w-16" /></td>
                <td className="px-4 py-2.5"><Flags ticket={t} /></td>
                <td className="px-4 py-2.5 whitespace-nowrap text-xs text-ink-2">{timeAgo(t.created_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {tickets?.length === 0 && (
          <p className="px-4 py-8 text-center text-sm text-muted">
            No tickets{Object.values(filters).some(Boolean) ? " match these filters" : " yet. Add one or upload a CSV"}.
          </p>
        )}
        {!tickets && !error && <p className="px-4 py-8 text-center text-sm text-muted">Loading…</p>}
      </Card>
    </div>
  );
}

function Flags({ ticket }) {
  return (
    <div className="flex flex-wrap gap-1">
      {ticket.status === "new" && <Pill>triaging…</Pill>}
      {ticket.status === "failed" && <Pill className="border-critical/50 text-critical-text">failed</Pill>}
      {ticket.status === "resolved" && <Pill className="text-good-text">resolved</Pill>}
      {ticket.policy_violation && <Pill className="border-critical/50 text-critical-text">policy</Pill>}
      {ticket.needs_review && <Pill>review</Pill>}
      {ticket.human_needed && <Pill>human</Pill>}
    </div>
  );
}

function Select({ label, value, onChange, options }) {
  return (
    <label className="flex flex-col gap-1 text-xs text-muted">
      {label}
      <select value={value} onChange={onChange}
              className="rounded-md border border-line bg-surface px-2 py-1.5 text-sm text-ink">
        <option value="">All</option>
        {options.map(([optionValue, optionLabel]) => (
          <option key={optionValue} value={optionValue}>{optionLabel}</option>
        ))}
      </select>
    </label>
  );
}

function NewTicketForm({ onCreated }) {
  const [form, setForm] = useState({ subject: "", body: "", customer_email: "" });
  const [error, setError] = useState(null);
  const [saving, setSaving] = useState(false);

  async function submit(event) {
    event.preventDefault();
    setSaving(true);
    try {
      await createTicket({ ...form, customer_email: form.customer_email || null });
      onCreated();
    } catch (e) {
      setError(e.message);
    } finally {
      setSaving(false);
    }
  }

  const field = (key) => ({ value: form[key], onChange: (e) => setForm({ ...form, [key]: e.target.value }) });
  const inputClass = "w-full rounded-md border border-line bg-surface px-3 py-2 text-sm text-ink";

  return (
    <Card title="New ticket">
      <form onSubmit={submit} className="space-y-3">
        <div className="grid gap-3 sm:grid-cols-2">
          <input required placeholder="Subject" className={inputClass} {...field("subject")} />
          <input type="email" placeholder="Customer email (optional)" className={inputClass} {...field("customer_email")} />
        </div>
        <textarea required rows={4} placeholder="Message" className={inputClass} {...field("body")} />
        <ErrorBox message={error} />
        <Button type="submit" disabled={saving}>{saving ? "Submitting…" : "Submit ticket"}</Button>
      </form>
    </Card>
  );
}

function CsvUpload({ onUploaded }) {
  const inputRef = useRef(null);
  const [message, setMessage] = useState(null);

  async function upload(event) {
    const file = event.target.files[0];
    if (!file) return;
    try {
      const result = await uploadCsv(file);
      setMessage(`Uploaded ${result.count} tickets`);
      onUploaded();
    } catch (e) {
      setMessage(`Upload failed: ${e.message}`);
    }
    event.target.value = ""; // allow uploading the same file again
  }

  return (
    <div className="flex items-center gap-2">
      {message && <span className="text-xs text-ink-2">{message}</span>}
      <input ref={inputRef} type="file" accept=".csv,text/csv" onChange={upload} className="hidden" />
      <Button variant="secondary" onClick={() => inputRef.current.click()} title="Columns: subject, body, customer_email (optional)">
        Upload CSV
      </Button>
    </div>
  );
}
