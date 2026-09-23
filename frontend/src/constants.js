// Must match backend/app/config.py (DEPARTMENTS, URGENCY_LEVELS, special queues).
export const DEPARTMENTS = ["billing", "technical", "shipping", "account", "general"];
export const SPECIAL_QUEUES = ["escalation", "general_triage"];
export const QUEUES = [...DEPARTMENTS, ...SPECIAL_QUEUES];

export const URGENCY_LABELS = ["Can wait", "Today", "Act now"];
export const STATUSES = ["new", "triaged", "failed", "resolved"];

// Nicer display names for queue ids like "general_triage".
export const queueLabel = (queue) =>
  queue ? queue.replace("_", " ").replace(/^\w/, (c) => c.toUpperCase()) : "—";
