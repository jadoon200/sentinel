import { useQuery } from "@tanstack/react-query";
import { api, type CampaignSummary } from "../api";

const API = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

function freshness(ageDays: number | null): string {
  if (ageDays === null) return "undated";
  if (ageDays < 1) return "today";
  if (ageDays < 2) return "yesterday";
  return `${Math.round(ageDays)}d ago`;
}

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
            source-mix PSI {drift.data ? drift.data.population_stability_index.toFixed(2) : "—"}
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
        <h2>Active campaigns — ranked by exploited CVEs &amp; recency</h2>
        {(campaigns.data ?? []).length === 0 && (
          <div className="muted">No campaigns yet — run ingestion + enrichment.</div>
        )}
        {(campaigns.data ?? []).slice(0, 8).map((c) => (
          <CampaignRow key={c.campaign_id} c={c} />
        ))}
      </section>

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

function CampaignRow({ c }: { c: CampaignSummary }) {
  return (
    <div className="camp-row">
      <code className="camp-id">{c.campaign_id}</code>
      <div className="camp-mid">
        {c.techniques.slice(0, 4).map((t) => (
          <span key={t.technique_id} className="badge tech">
            {t.technique_id}
          </span>
        ))}
        {c.techniques.length === 0 && <span className="hint">no techniques tagged yet</span>}
      </div>
      {c.kev_cves.length > 0 && <span className="badge kev">{c.kev_cves.length} KEV</span>}
      <span className="hint camp-meta">
        {c.report_count} reports · {freshness(c.age_days)}
      </span>
    </div>
  );
}

function Briefing() {
  const briefing = useQuery({ queryKey: ["briefing"], queryFn: api.briefing });
  if (!briefing.data) return <div className="muted">generating…</div>;
  return <pre className="briefing">{briefing.data}</pre>;
}
