import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";
import { Alerts } from "./views/Alerts";
import { Campaigns } from "./views/Campaigns";
import { Heatmap } from "./views/Heatmap";
import { Overview } from "./views/Overview";

const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 60_000, retry: 1 } },
});

const TABS = ["Overview", "Campaigns", "Alerts", "ATT&CK Heatmap"] as const;
type Tab = (typeof TABS)[number];

export default function App() {
  const [tab, setTab] = useState<Tab>("Overview");
  return (
    <QueryClientProvider client={queryClient}>
      <div className="shell">
        <header className="masthead">
          <h1>SENTINEL</h1>
          <span>threat intelligence fusion — OSINT × NLP × IDS in one ATT&CK graph</span>
        </header>
        <nav className="tabs">
          {TABS.map((t) => (
            <button key={t} className={t === tab ? "active" : ""} onClick={() => setTab(t)}>
              {t}
            </button>
          ))}
        </nav>
        {tab === "Overview" && <Overview />}
        {tab === "Campaigns" && <Campaigns />}
        {tab === "Alerts" && <Alerts />}
        {tab === "ATT&CK Heatmap" && <Heatmap />}
      </div>
    </QueryClientProvider>
  );
}
