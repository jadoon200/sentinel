import { useQuery } from "@tanstack/react-query";
import { api } from "../api";

const TACTIC_ORDER = [
  "reconnaissance",
  "resource-development",
  "initial-access",
  "execution",
  "persistence",
  "privilege-escalation",
  "defense-evasion",
  "credential-access",
  "discovery",
  "lateral-movement",
  "collection",
  "command-and-control",
  "exfiltration",
  "impact",
];

function shade(count: number, max: number): string {
  if (count === 0) return "transparent";
  const t = Math.min(count / Math.max(max, 1), 1);
  return `rgba(56, 189, 248, ${0.12 + 0.55 * t})`;
}

export function Heatmap() {
  const techniques = useQuery({ queryKey: ["techniques"], queryFn: api.techniques });
  const campaigns = useQuery({ queryKey: ["campaigns"], queryFn: api.campaigns });
  const reports = useQuery({ queryKey: ["reports"], queryFn: () => api.reports(200) });
  const alerts = useQuery({ queryKey: ["alerts", "all"], queryFn: () => api.alerts(undefined, 200) });

  if (techniques.error) return <div className="error">failed to load techniques</div>;
  if (!techniques.data) return <div className="muted">loading catalog…</div>;

  // Evidence per technique across all three layers of the graph.
  const evidence = new Map<string, number>();
  const bump = (id: string) => evidence.set(id, (evidence.get(id) ?? 0) + 1);
  for (const c of campaigns.data ?? []) for (const t of c.techniques) bump(t.technique_id);
  for (const r of reports.data ?? []) for (const t of r.techniques) bump(t.technique_id);
  for (const a of alerts.data ?? []) for (const t of a.techniques) bump(t);

  const max = Math.max(...evidence.values(), 1);
  const columns = TACTIC_ORDER.map((tactic) => ({
    tactic,
    cells: techniques.data
      .filter((t) => t.tactics.includes(tactic) && (evidence.get(t.technique_id) ?? 0) > 0)
      .sort((a, b) => (evidence.get(b.technique_id) ?? 0) - (evidence.get(a.technique_id) ?? 0))
      .slice(0, 12),
  })).filter((c) => c.cells.length > 0);

  return (
    <section className="panel">
      <h2>
        ATT&CK heatmap — technique evidence fused across reports, campaigns, and IDS alerts
      </h2>
      {columns.length === 0 && <div className="muted">no technique evidence yet — run make enrich / make replay</div>}
      <div className="heat">
        {columns.map((col) => (
          <div className="col" key={col.tactic}>
            <h4>{col.tactic.replace(/-/g, " ")}</h4>
            {col.cells.map((t) => {
              const n = evidence.get(t.technique_id) ?? 0;
              return (
                <div
                  key={t.technique_id}
                  className="cell"
                  style={{ background: shade(n, max) }}
                  title={`${t.technique_id} ${t.name} — evidence ×${n}`}
                >
                  <span className="tid">{t.technique_id}</span> ×{n}
                  <br />
                  <span className="tname">{t.name}</span>
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </section>
  );
}
