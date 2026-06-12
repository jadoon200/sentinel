import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";
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
] as const;
type Tab = (typeof TABS)[number]["id"];

export default function App() {
  const [tab, setTab] = useState<Tab>("feed");
  return (
    <QueryClientProvider client={queryClient}>
      <div className="shell">
        <header className="masthead">
          <h1>SENTINEL</h1>
          <span>Is anything on the network tied to a known real-world threat?</span>
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
      </div>
    </QueryClientProvider>
  );
}
