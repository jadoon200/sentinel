import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { api, type HostThreat } from "../api";

const DETECTOR_LABEL: Record<string, string> = {
  "lightgbm-multiclass": "supervised",
  autoencoder: "anomaly",
  sequence: "sequence",
  profile: "profile",
};

function riskClass(r: number) {
  return r >= 85 ? "risk-crit" : r >= 70 ? "risk-high" : "risk-med";
}

function story(t: HostThreat): string {
  const labels = t.true_labels.filter((l) => l.toUpperCase() !== "BENIGN");
  const what = labels.length ? labels.join(", ").toLowerCase() : "anomalous activity";
  const n = t.detectors.length;
  return `${what} — flagged by ${n} of 4 detectors`;
}

function EvidenceChain({ t }: { t: HostThreat }) {
  return (
    <div className="chain">
      <div className="stage">
        <h4>{t.detectors.length} of 4 detectors agree</h4>
        {t.detectors.map((d) => (
          <div key={d} className="ev-line">
            <i className="ti ti-check" aria-hidden="true" /> {DETECTOR_LABEL[d] ?? d}
          </div>
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
            <div key={f.campaign_id} style={{ marginBottom: 6 }}>
              <code>{f.campaign_id}</code>
              <div className="hint">{f.report_count} corroborating reports</div>
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
            {t.fused.length > 0 && <span className="badge fusedtag">fused with intel</span>}
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
          <span style={{ color: "var(--warn)" }}>amber = fused with intel</span>
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
