import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";
import { Calibrate } from "./views/Calibrate";
import { Landscape } from "./views/Landscape";
import { ReportCard } from "./views/ReportCard";
import { ThreatFeed } from "./views/ThreatFeed";

const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 60_000, retry: 1 } },
});

const TABS = [
  { id: "feed", label: "Threat feed", sub: "what should I look at now?" },
  { id: "land", label: "Landscape", sub: "what's happening in the world?" },
  { id: "card", label: "Model report card", sub: "how much should I trust this?" },
  { id: "cal", label: "Calibrate", sub: "teach it this network" },
] as const;
type Tab = (typeof TABS)[number]["id"];

export default function App() {
  const [tab, setTab] = useState<Tab>("feed");
  return (
    <QueryClientProvider client={queryClient}>
      <div className="shell">
        <header className="masthead">
          <div className="brand">
            <svg className="brand-mark" viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
              <circle cx="12" cy="12" r="9.2" fill="none" stroke="currentColor" strokeWidth="1.4" opacity="0.45" />
              <circle cx="12" cy="12" r="4.6" fill="none" stroke="currentColor" strokeWidth="1.4" opacity="0.85" />
              <path d="M12 12 L19 6.6" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
              <circle cx="12" cy="12" r="1.7" fill="currentColor" />
            </svg>
            <h1>SENTINEL</h1>
          </div>
          <p className="tagline">Is anything on the network tied to a known real-world threat?</p>
        </header>
        <nav className="tabs">
          {TABS.map((t) => (
            <button
              key={t.id}
              className={t.id === tab ? "active" : ""}
              onClick={() => setTab(t.id)}
            >
              {t.label}
              <em>{t.sub}</em>
            </button>
          ))}
        </nav>
        {tab === "feed" && <ThreatFeed />}
        {tab === "land" && <Landscape />}
        {tab === "card" && <ReportCard />}
        {tab === "cal" && <Calibrate />}
      </div>
    </QueryClientProvider>
  );
}
