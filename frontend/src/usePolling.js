import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Load data with `fetcher`, then reload it every `intervalMs` (pass null to stop polling).
 * Triage runs in the background on the server, so polling is how new results appear
 * without a refresh. WebSockets would be the upgrade for a real product.
 *
 * `deps` works like useEffect deps: when they change (e.g. filters), data reloads right away.
 */
export default function usePolling(fetcher, deps, intervalMs = null) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher; // always call the latest fetcher, without restarting timers

  const reload = useCallback(async () => {
    try {
      setData(await fetcherRef.current());
      setError(null);
    } catch (e) {
      setError(e.message);
    }
  }, []);

  // Reload immediately whenever deps change.
  useEffect(() => {
    reload();
  }, deps); // eslint-disable-line react-hooks/exhaustive-deps

  // Poll on a timer while intervalMs is set.
  useEffect(() => {
    if (!intervalMs) return;
    const timer = setInterval(reload, intervalMs);
    return () => clearInterval(timer);
  }, [intervalMs, reload]);

  return { data, error, reload, loading: data === null && error === null };
}
