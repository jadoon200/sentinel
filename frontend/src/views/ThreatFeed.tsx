import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { api, type AlertRef, type HostThreat } from "../api";

const DETECTOR_LABEL: Record<string, string> = {
  "lightgbm-multiclass": "supervised",
  autoencoder: "anomaly",
  sequence: "sequence",
  profile: "profile",
  beacon: "beacon (C2)",
  sqli: "SQLi (WAF)",
};

// The five flow detectors; SQLi is a separate application-layer modality.
const FLOW_DETECTORS = 5;
const flowDetectors = (t: HostThreat) => t.detectors.filter((d) => d !== "sqli");

function riskClass(r: number) {
  return r >= 85 ? "risk-crit" : r >= 70 ? "risk-high" : "risk-med";
}

const pct = (x: number) => `${Math.round(x * 100)}%`;

function story(t: HostThreat): string {
  const labels = t.true_labels.filter((l) => l.toUpperCase() !== "BENIGN");
  const what = labels.length ? labels.join(", ").toLowerCase() : "anomalous activity";
  const flows = flowDetectors(t);
  const hasSqli = t.detectors.includes("sqli");
  let base: string;
  if (flows.length === 0 && hasSqli) {
    base = "SQL injection attempts — application-layer (WAF)";
  } else {
    base = `${what} — flagged by ${flows.length} of ${FLOW_DETECTORS} detectors`;
    if (hasSqli) base += " + WAF";
  }
  if (t.fused.length > 0) {
    // Tie the badge into the sentence: which campaign, how strong, expand to see
    // which specific detection drove it.
    return `${base}; ${pct(t.fused[0].fusion.strength)} match to active campaign ${t.fused[0].campaign_id}`;
  }
  return base;
}

function freshness(ageDays: number | null): string {
  if (ageDays === null) return "undated";
  if (ageDays < 1) return "reported today";
  if (ageDays < 2) return "reported yesterday";
  return `${Math.round(ageDays)} days old`;
}

function DetectionLine({ alert }: { alert: AlertRef }) {
  const [open, setOpen] = useState(false);
  // Per-detection campaign context, fetched only when the analyst drills in.
  const ctx = useQuery({
    queryKey: ["alertctx", alert.alert_id],
    queryFn: () => api.alertContext(alert.alert_id),
    enabled: open,
  });
  return (
    <div className="det-line">
      <button className="det-toggle" onClick={() => setOpen(!open)}>
        <i className={`ti ti-chevron-${open ? "down" : "right"}`} aria-hidden="true" />{" "}
        {DETECTOR_LABEL[alert.model] ?? alert.model}
        {alert.techniques.length > 0 && (
          <span className="hint"> · {alert.techniques.join(", ")}</span>
        )}
      </button>
      {open && (
        <div className="det-ctx">
          {!ctx.data && <span className="hint">checking intel…</span>}
          {ctx.data && ctx.data.matched_campaigns.length === 0 && (
            <span className="hint">no campaign correlation for this detection</span>
          )}
          {ctx.data?.matched_campaigns.map((m) => (
            <div key={m.campaign_id} className="hint">
              <code>{m.campaign_id}</code> — {pct(m.fusion.strength)} match
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function EvidenceChain({ t }: { t: HostThreat }) {
  const flows = flowDetectors(t).length;
  return (
    <div className="chain">
      <div className="stage">
        <h4>{flows > 0 ? `${flows} of ${FLOW_DETECTORS} detectors agree` : "application-layer (WAF)"}</h4>
        {t.alerts.map((a) => (
          <DetectionLine key={a.alert_id} alert={a} />
        ))}
      </div>
      <i className="ti ti-arrow-right arrow" aria-hidden="true" />
      <div className="stage">
        <h4>host</h4>
        <div className="mono">{t.host}</div>
        <div style={{ marginTop: 6 }}>
          {t.techniques.map((tc) => (
            <span key={tc} className="badge tech">
              {tc}
            </span>
          ))}
        </div>
      </div>
      <i className="ti ti-arrow-right arrow" aria-hidden="true" />
      {t.fused.length > 0 ? (
        <div className="stage stage-fused">
          <h4>matched real-world intel</h4>
          {t.fused.map((f) => (
            <div key={f.campaign_id} style={{ marginBottom: 10 }}>
              <code>{f.campaign_id}</code>
              <div className="fusion-strength" title="rarity x recency x corroboration">
                <div className="fusion-bar">
                  <span style={{ width: pct(f.fusion.strength) }} />
                </div>
                <span className="fusion-val">{pct(f.fusion.strength)} match</span>
              </div>
              <div className="hint">
                specific {pct(f.fusion.specificity)} · {freshness(f.fusion.age_days)} ·{" "}
                {f.report_count} corroborating reports
              </div>
              {f.kev_cves.map((c) => (
                <span key={c} className="badge kev">
                  {c} KEV
                </span>
              ))}
            </div>
          ))}
        </div>
      ) : (
        <div className="stage">
          <h4>intel match</h4>
          <div className="hint">No current campaign — internal-only activity.</div>
        </div>
      )}
    </div>
  );
}

function ThreatRow({ t }: { t: HostThreat }) {
  const [open, setOpen] = useState(false);
  return (
    <div className={`threat ${t.fused.length ? "fused" : ""} ${t.simulated ? "fresh" : ""}`}>
      <div className="threat-head" onClick={() => setOpen(!open)}>
        <div className={`risk ${riskClass(t.risk)}`}>
          <span className="rv">{t.risk}</span>
          <span className="rk">risk</span>
        </div>
        <div className="threat-main">
          <div className="threat-title">
            <span className="mono">{t.host}</span>
            {t.simulated && <span className="badge live">live</span>}
            {t.fused.length > 0 && (
              <span
                className="badge fusedtag"
                title="rarity x recency x corroboration — see Model report card"
              >
                {pct(t.fused[0].fusion.strength)} intel match
              </span>
            )}
          </div>
          <div className="threat-story">{story(t)}</div>
          <div>
            {t.detectors.map((d) => (
              <span key={d} className="badge det">
                {DETECTOR_LABEL[d] ?? d}
              </span>
            ))}
            {t.techniques.map((tc) => (
              <span key={tc} className="badge tech">
                {tc}
              </span>
            ))}
          </div>
        </div>
        <i className={`ti ti-chevron-${open ? "up" : "down"} chev`} aria-hidden="true" />
      </div>
      {open && <EvidenceChain t={t} />}
    </div>
  );
}

export function ThreatFeed() {
  const hosts = useQuery({ queryKey: ["hosts"], queryFn: api.hosts });
  const sim = useQuery({ queryKey: ["sim"], queryFn: api.simulatedHosts });
  const [revealed, setRevealed] = useState<HostThreat[]>([]);

  if (hosts.error)
    return <div className="error">API unreachable — run `make up`, `make replay`, `make api`.</div>;
  if (!hosts.data) return <div className="muted">correlating detections…</div>;

  const queue = (sim.data ?? []).filter((s) => !revealed.some((r) => r.host === s.host));
  const all = [...revealed, ...hosts.data].sort((a, b) => b.risk - a.risk);

  return (
    <>
      <div className="feed-bar">
        <span className="muted">
          {all.length} host threats · ranked by risk ·{" "}
          <span style={{ color: "var(--warn)" }}>amber = scored intel match</span>
        </span>
        <button
          onClick={() => queue.length && setRevealed([{ ...queue[0] }, ...revealed])}
          disabled={queue.length === 0}
        >
          <i className="ti ti-player-play" aria-hidden="true" />
          &nbsp;Simulate detection {queue.length > 0 ? `(${queue.length})` : ""}
        </button>
      </div>
      {all.map((t) => (
        <ThreatRow key={t.host} t={t} />
      ))}
    </>
  );
}
