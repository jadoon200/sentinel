import { useQuery } from "@tanstack/react-query";
import { api } from "../api";

const API = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export function Landscape() {
  const stats = useQuery({ queryKey: ["stats"], queryFn: api.stats });
  const campaigns = useQuery({ queryKey: ["campaigns"], queryFn: api.campaigns });
  const trending = useQuery({ queryKey: ["trending"], queryFn: api.trending });
  const drift = useQuery({ queryKey: ["drift"], queryFn: api.drift });

  if (stats.error) return <div className="error">API unreachable.</div>;
  const kevCampaigns = (campaigns.data ?? []).filter((c) => c.kev_cves.length > 0).length;

  return (
    <>
      <div className="metric-row">
        <div className="metric">
          <div className="ml">Feed status</div>
          <div className="mv" style={{ color: drift.data?.verdict === "stable" ? "var(--good)" : "var(--warn)" }}>
            {drift.data ? drift.data.verdict : "—"}
          </div>
          <div className="mh">
            source-mix PSI {drift.data ? drift.data.population_stability_index.toFixed(1) : "—"}
          </div>
        </div>
        <div className="metric">
          <div className="ml">Active campaigns</div>
          <div className="mv">
            {stats.data?.campaigns ?? "—"}{" "}
            <span style={{ fontSize: 13, color: "var(--warn)" }}>· {kevCampaigns} KEV</span>
          </div>
          <div className="mh">clustered from {stats.data?.threat_reports ?? "—"} reports</div>
        </div>
        <div className="metric">
          <div className="ml">Knowledge graph</div>
          <div className="mv">{(stats.data?.attack_techniques ?? 0).toLocaleString()}</div>
          <div className="mh">ATT&CK techniques · {stats.data?.vulnerabilities ?? 0} CVEs</div>
        </div>
      </div>

      <section className="panel">
        <h2>Trending techniques — mention rate this week vs last</h2>
        {(trending.data ?? []).slice(0, 8).map((t) => (
          <div key={t.technique_id} className="trend-row">
            <span className="badge tech">{t.technique_id}</span>
            <span className="trend-name">{t.name ?? ""}</span>
            <span className="hint">
              {t.recent_count} vs {t.prior_count}
            </span>
            <span className="trend-lift">×{t.lift.toFixed(1)}</span>
          </div>
        ))}
      </section>

      <section className="panel">
        <h2>
          Daily briefing
          <a className="navlink" href={`${API}/attack-navigator-layer`} target="_blank" rel="noreferrer">
            export ATT&CK Navigator layer <i className="ti ti-external-link" aria-hidden="true" />
          </a>
        </h2>
        <Briefing />
      </section>
    </>
  );
}

function Briefing() {
  const briefing = useQuery({ queryKey: ["briefing"], queryFn: api.briefing });
  if (!briefing.data) return <div className="muted">generating…</div>;
  return <pre className="briefing">{briefing.data}</pre>;
}
