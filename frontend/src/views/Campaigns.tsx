import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "../api";

export function Campaigns() {
  const campaigns = useQuery({ queryKey: ["campaigns"], queryFn: api.campaigns });
  const [selected, setSelected] = useState<string | null>(null);
  const detail = useQuery({
    queryKey: ["campaign", selected],
    queryFn: () => api.campaign(selected as string),
    enabled: selected !== null,
  });

  if (campaigns.error) return <div className="error">failed to load campaigns</div>;

  return (
    <section className="panel">
      <h2>Campaigns — reports clustered by shared CVE mentions</h2>
      {(campaigns.data ?? []).map((c) => (
        <div key={c.campaign_id}>
          <div
            className="row"
            onClick={() => setSelected(selected === c.campaign_id ? null : c.campaign_id)}
          >
            <span className="title">
              <code>{c.campaign_id}</code> — {c.cve_ids.join(", ")}
            </span>
            {c.kev_cves.length > 0 && (
              <span className="badge kev">
                {c.kev_cves.length}/{c.cve_ids.length} KEV — actively exploited
              </span>
            )}
            <span className="meta">{c.report_count} reports</span>
            <span className="meta">
              {c.techniques.slice(0, 3).map((t) => (
                <span key={t.technique_id} className="badge tech">
                  {t.technique_id}
                </span>
              ))}
            </span>
          </div>
          {selected === c.campaign_id && detail.data && (
            <div className="detail">
              <h3>Fused technique evidence</h3>
              {detail.data.techniques.length === 0 && (
                <div className="muted">no technique evidence above threshold</div>
              )}
              {detail.data.techniques.map((t) => (
                <span key={t.technique_id} className="badge tech">
                  {t.technique_id} {t.name ?? ""} ×{t.corroborations}
                </span>
              ))}
              <h3 style={{ marginTop: 14 }}>Member reports</h3>
              {detail.data.reports.map((r) => (
                <div key={r.report_id} style={{ marginBottom: 6, fontSize: 13 }}>
                  {r.url ? (
                    <a href={r.url} target="_blank" rel="noreferrer">
                      {r.title}
                    </a>
                  ) : (
                    r.title
                  )}{" "}
                  <span className="muted">({r.source})</span>
                </div>
              ))}
            </div>
          )}
        </div>
      ))}
    </section>
  );
}
