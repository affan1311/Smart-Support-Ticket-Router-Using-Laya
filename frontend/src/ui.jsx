// Small presentational pieces shared across pages.
import { URGENCY_LABELS } from "./constants.js";

export function Card({ title, children, className = "" }) {
  return (
    <section className={`rounded-lg border border-line bg-surface p-4 ${className}`}>
      {title && <h2 className="mb-3 text-sm font-semibold text-ink-2">{title}</h2>}
      {children}
    </section>
  );
}

export function Button({ variant = "primary", className = "", ...props }) {
  const styles = {
    primary: "bg-accent text-white hover:opacity-90",
    secondary: "border border-line bg-surface text-ink hover:bg-line/40",
  };
  return (
    <button
      className={`rounded-md px-3 py-1.5 text-sm font-medium disabled:cursor-not-allowed disabled:opacity-50 ${styles[variant]} ${className}`}
      {...props}
    />
  );
}

export function ErrorBox({ message }) {
  if (!message) return null;
  return (
    <p role="alert" className="rounded-md border border-critical/40 bg-critical/10 px-3 py-2 text-sm text-critical-text">
      {message}
    </p>
  );
}

/** A neutral pill for labels like "review" or "Laya". */
export function Pill({ children, className = "" }) {
  return (
    <span className={`inline-flex items-center gap-1 rounded-full border border-line px-2 py-0.5 text-xs text-ink-2 ${className}`}>
      {children}
    </span>
  );
}

/**
 * Urgency uses status colors (critical / warning / neutral), and the dot is always
 * paired with a text label, so meaning never depends on color alone.
 */
export function UrgencyBadge({ level, score }) {
  if (level === null || level === undefined) return <span className="text-muted">—</span>;
  const dot = ["bg-muted", "bg-warning", "bg-critical"][level];
  return (
    <span className="inline-flex items-center gap-1.5 whitespace-nowrap text-sm" title={`Urgency score ${score?.toFixed(2)} of 2`}>
      <span className={`h-2 w-2 rounded-full ${dot}`} aria-hidden="true" />
      {URGENCY_LABELS[level]}
      {score !== null && score !== undefined && <span className="tabular text-xs text-muted">{score.toFixed(1)}</span>}
    </span>
  );
}

/** A thin bar for a probability from 0 to 1, with the number beside it. */
export function ProbBar({ value, label, width = "w-24" }) {
  if (value === null || value === undefined) return <span className="text-muted">—</span>;
  return (
    <span className="inline-flex items-center gap-2" title={label ? `${label}: ${formatPct(value)}` : formatPct(value)}>
      <span className={`h-1.5 ${width} overflow-hidden rounded-full bg-line`}>
        <span className="block h-full rounded-full bg-accent" style={{ width: `${Math.round(value * 100)}%` }} />
      </span>
      <span className="tabular w-9 text-right text-xs text-ink-2">{formatPct(value)}</span>
    </span>
  );
}

export const formatPct = (value, digits = 0) =>
  value === null || value === undefined ? "—" : `${(value * 100).toFixed(digits)}%`;

export const formatUsd = (value) =>
  value === null || value === undefined ? "—"
    : value === 0 ? "$0"
    : `$${value < 0.01 ? value.toFixed(5) : value.toFixed(2)}`;

export const formatMs = (ms) =>
  ms === null || ms === undefined ? "—" : ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`;

const rtf = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });

/** "3 minutes ago" style time. The API returns UTC times ending in "Z". */
export function timeAgo(isoString) {
  const seconds = (new Date(isoString).getTime() - Date.now()) / 1000;
  const steps = [
    [60, "second"], [60, "minute"], [24, "hour"], [7, "day"], [4.35, "week"], [12, "month"], [Infinity, "year"],
  ];
  let value = seconds;
  for (const [size, unit] of steps) {
    if (Math.abs(value) < size) return rtf.format(Math.round(value), unit);
    value /= size;
  }
}
