import { useEffect, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, LabelList, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { getMetrics } from "../api.js";
import { queueLabel } from "../constants.js";
import usePolling from "../usePolling.js";
import { Card, ErrorBox, formatMs, formatPct, formatUsd } from "../ui.jsx";

/**
 * Recharts draws SVG and needs real color values, so read the CSS tokens from index.css
 * and read them again when the OS switches between light and dark mode.
 */
function useThemeColors() {
  const read = () => {
    const style = getComputedStyle(document.documentElement);
    const get = (name) => style.getPropertyValue(`--color-${name}`).trim();
    return {
      series1: get("accent"), series2: get("series-2"), ink: get("ink"), ink2: get("ink-2"),
      muted: get("muted"), line: get("line"), axis: get("axis"), surface: get("surface"),
    };
  };
  const [colors, setColors] = useState(read);
  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const update = () => setColors(read());
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);
  return colors;
}

export default function MetricsPage() {
  const { data: m, error } = usePolling(getMetrics, [], 10000);
  const colors = useThemeColors();

  if (error && !m) return <ErrorBox message={error} />;
  if (!m) return <p className="text-muted">Loading…</p>;
  if (m.total_tickets === 0) return <p className="text-muted">No tickets yet. Metrics appear once tickets are triaged.</p>;

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">Metrics</h1>
      <ErrorBox message={error} />

      {/* Headline numbers: single values are stat tiles, not charts */}
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat label="Gemini skipped" value={m.gemini_calls_avoided_pct === null ? "—" : `${m.gemini_calls_avoided_pct}%`}
              note="of triaged tickets never called Gemini" />
        <Stat label="Cost per 1,000 tickets" value={formatUsd(m.cost_per_1000_tickets_usd)}
              note={`${formatUsd(m.total_cost_usd)} total so far`} />
        <Stat label="Laya triage, median" value={formatMs(m.laya_latency.p50_ms)}
              note={`p95 ${formatMs(m.laya_latency.p95_ms)}`} />
        <Stat label="Correction rate" value={formatPct(m.correction_rate, 1)}
              note={`${m.tickets_overridden} of ${m.total_tickets} tickets corrected`} />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Latency by step (seconds)">
          <LatencyChart m={m} colors={colors} />
        </Card>
        <Card title="Tickets by queue">
          <QueueChart byQueue={m.by_queue} colors={colors} />
        </Card>
      </div>

      <MetricsTable m={m} />
    </div>
  );
}

function Stat({ label, value, note }) {
  return (
    <Card>
      <p className="text-xs text-muted">{label}</p>
      <p className="mt-1 text-2xl font-semibold">{value}</p>
      <p className="mt-1 text-xs text-ink-2">{note}</p>
    </Card>
  );
}

/** A tooltip in the app's own colors. Text uses ink tokens; the swatch carries the series color. */
function ChartTooltip({ active, payload, label, unit = "" }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-md border border-line bg-surface px-3 py-2 text-xs shadow-sm">
      <p className="mb-1 font-medium text-ink">{label}</p>
      {payload.map((entry) => (
        <p key={entry.dataKey} className="flex items-center gap-2 text-ink-2">
          <span className="h-2 w-2 rounded-sm" style={{ background: entry.color }} aria-hidden="true" />
          {entry.name}: <span className="tabular text-ink">{entry.value}{unit}</span>
        </p>
      ))}
    </div>
  );
}

/** p50 vs p95 per step. Two series: blue = p50, orange = p95, with a legend and value labels. */
function LatencyChart({ m, colors }) {
  const toSeconds = (ms) => (ms === null ? null : Number((ms / 1000).toFixed(2)));
  const data = [
    ["Laya triage", m.laya_latency],
    ["Gemini triage", m.gemini_triage_latency],
    ["Gemini draft", m.draft_latency],
  ]
    .filter(([, stats]) => stats.count > 0) // skip steps that never ran
    .map(([step, stats]) => ({ step, p50: toSeconds(stats.p50_ms), p95: toSeconds(stats.p95_ms), n: stats.count }));

  if (data.length === 0) return <p className="text-sm text-muted">No calls yet.</p>;

  const axisTick = { fill: colors.muted, fontSize: 12 };
  const label = { position: "top", fill: colors.ink2, fontSize: 11 };
  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} margin={{ top: 20, right: 8, left: -12, bottom: 0 }} barGap={2}>
        <CartesianGrid vertical={false} stroke={colors.line} />
        <XAxis dataKey="step" tick={axisTick} tickLine={false} axisLine={{ stroke: colors.axis }} />
        <YAxis tick={axisTick} tickLine={false} axisLine={false} unit="s" />
        <Tooltip content={<ChartTooltip unit=" s" />} cursor={{ fill: colors.line, opacity: 0.4 }} />
        <Legend iconType="square" iconSize={10}
                formatter={(value) => <span style={{ color: colors.ink2, fontSize: 12 }}>{value}</span>} />
        <Bar dataKey="p50" name="p50 (median)" fill={colors.series1} radius={[4, 4, 0, 0]} maxBarSize={36} isAnimationActive={false}>
          <LabelList dataKey="p50" {...label} />
        </Bar>
        <Bar dataKey="p95" name="p95" fill={colors.series2} radius={[4, 4, 0, 0]} maxBarSize={36} isAnimationActive={false}>
          <LabelList dataKey="p95" {...label} />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/** One series (ticket count), so one color and no legend; the title names it. */
function QueueChart({ byQueue, colors }) {
  const data = Object.entries(byQueue)
    .map(([queue, count]) => ({ queue: queueLabel(queue), count }))
    .sort((a, b) => b.count - a.count);

  return (
    <ResponsiveContainer width="100%" height={Math.max(160, data.length * 34 + 20)}>
      <BarChart data={data} layout="vertical" margin={{ top: 0, right: 32, left: 8, bottom: 0 }}>
        <XAxis type="number" hide allowDecimals={false} />
        <YAxis type="category" dataKey="queue" width={100} tick={{ fill: colors.ink2, fontSize: 12 }}
               tickLine={false} axisLine={{ stroke: colors.axis }} />
        <Tooltip content={<ChartTooltip />} cursor={{ fill: colors.line, opacity: 0.4 }} />
        <Bar dataKey="count" name="Tickets" fill={colors.series1} radius={[0, 4, 4, 0]} maxBarSize={22} isAnimationActive={false}>
          <LabelList dataKey="count" position="right" fill={colors.ink2} fontSize={11} />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Every number as a table: the accessible view of the charts, and handy for screenshots. */
function MetricsTable({ m }) {
  const latency = (stats) => `${formatMs(stats.p50_ms)} / ${formatMs(stats.p95_ms)} (n=${stats.count})`;
  const rows = [
    ["Tickets", m.total_tickets],
    ["By status", Object.entries(m.by_status).map(([k, v]) => `${k} ${v}`).join(", ")],
    ["Triage source", Object.entries(m.triage_source).map(([k, v]) => `${k} ${v}`).join(", ")],
    ["Laya latency p50 / p95", latency(m.laya_latency)],
    ["Gemini triage latency p50 / p95", latency(m.gemini_triage_latency)],
    ["Gemini draft latency p50 / p95", latency(m.draft_latency)],
    ["Drafts created / approved / edited", `${m.drafts_created} / ${m.drafts_approved} / ${m.drafts_edited}`],
    ["Needs review", m.needs_review],
    ["Total Gemini cost", formatUsd(m.total_cost_usd)],
  ];
  return (
    <Card title="All metrics">
      <table className="w-full text-sm">
        <tbody>
          {rows.map(([label, value]) => (
            <tr key={label} className="border-b border-line last:border-0">
              <th scope="row" className="py-1.5 pr-4 text-left font-normal text-ink-2">{label}</th>
              <td className="tabular py-1.5 text-right">{value}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  );
}
