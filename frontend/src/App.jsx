import { NavLink, Route, Routes } from "react-router-dom";
import QueuePage from "./pages/QueuePage.jsx";
import TicketPage from "./pages/TicketPage.jsx";
import MetricsPage from "./pages/MetricsPage.jsx";

function NavItem({ to, children }) {
  return (
    <NavLink
      to={to}
      end
      className={({ isActive }) =>
        `rounded-md px-3 py-1.5 text-sm ${isActive ? "bg-accent/10 text-accent font-medium" : "text-ink-2 hover:text-ink"}`
      }
    >
      {children}
    </NavLink>
  );
}

export default function App() {
  return (
    <div className="min-h-screen">
      <header className="border-b border-line bg-surface">
        <div className="mx-auto flex max-w-6xl items-center gap-6 px-4 py-3">
          <span className="font-semibold">Ticket Router</span>
          <nav className="flex gap-1">
            <NavItem to="/">Queue</NavItem>
            <NavItem to="/metrics">Metrics</NavItem>
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-4 py-6">
        <Routes>
          <Route path="/" element={<QueuePage />} />
          <Route path="/tickets/:id" element={<TicketPage />} />
          <Route path="/metrics" element={<MetricsPage />} />
          <Route path="*" element={<p className="text-ink-2">Page not found.</p>} />
        </Routes>
      </main>
    </div>
  );
}
