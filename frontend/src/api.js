// All calls to the FastAPI backend live here, so components never build URLs themselves.
// "/api" is forwarded to the backend by the Vite dev server (see vite.config.js).
const BASE = import.meta.env.VITE_API_URL ?? "/api";

async function request(path, options = {}) {
  const response = await fetch(BASE + path, options);
  if (!response.ok) {
    // FastAPI puts the error in "detail": a string, or a list of validation errors.
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") message = body.detail;
      else if (Array.isArray(body.detail)) message = body.detail.map((e) => e.msg).join("; ");
    } catch {
      // not JSON: keep the status text
    }
    throw new Error(message);
  }
  return response.json();
}

function sendJson(method, path, data) {
  return request(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

export function listTickets(filters = {}) {
  // Drop empty filters so they aren't sent as "?dept=".
  const params = new URLSearchParams(
    Object.entries(filters).filter(([, value]) => value !== "" && value !== null && value !== undefined),
  );
  return request(`/tickets?${params}`);
}

export const getTicket = (id) => request(`/tickets/${id}`);
export const getMetrics = () => request("/metrics");
export const createTicket = (ticket) => sendJson("POST", "/tickets", { ...ticket, channel: "form" });
export const overrideDecision = (id, field, value, agentId = "agent") =>
  sendJson("PATCH", `/tickets/${id}/decision`, { field, value, agent_id: agentId });
export const approveDraft = (id, editedText = null) =>
  sendJson("POST", `/tickets/${id}/draft/approve`, { edited_text: editedText });

export function uploadCsv(file) {
  const form = new FormData();
  form.append("file", file);
  return request("/tickets/bulk", { method: "POST", body: form });
}
