import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "../api";

const MODELS = ["all", "lightgbm-multiclass", "autoencoder", "sequence", "profile"] as const;

export function Alerts() {
  const [model, setModel] = useState<(typeof MODELS)[number]>("all");
  const [selected, setSelected] = useState<number | null>(null);
  const alerts = useQuery({
    queryKey: ["alerts", model],
    queryFn: () => api.alerts(model === "all" ? undefined : model, 100),
  });
  const context = useQuery({
    queryKey: ["context", selected],
    queryFn: () => api.alertContext(selected as number),
    enabled: selected !== null,
  });

  if (alerts.error) return <div className="error">failed to load alerts</div>;

  return (
    <>
      <section className="panel">
        <h2>
          IDS alert feed{" "}
          {MODELS.map((m) => (
            <button
              key={m}
              className={`badge ${model === m ? "ok" : "model"}`}
              style={{ marginLeft: 6, cursor: "pointer", background: "none" }}
              onClick={() => setModel(m)}
            >
              {m}
            </button>
          ))}
        </h2>
        <table className="alerts">
          <thead>
            <tr>
              <th>id</th>
              <th>detector</th>
              <th>day</th>
              <th>score</th>
              <th>prediction</th>
              <th>ground truth</th>
              <th>ATT&CK</th>
            </tr>
          </thead>
          <tbody>
            {(alerts.data ?? []).map((a) => (
              <tr
                key={a.alert_id}
                className={selected === a.alert_id ? "sel" : ""}
                onClick={() => setSelected(selected === a.alert_id ? null : a.alert_id)}
              >
                <td>{a.alert_id}</td>
                <td>
                  <span className="badge model">{a.model}</span>
                </td>
                <td>{a.day}</td>
                <td>{a.score.toFixed(3)}</td>
                <td>{a.predicted_label ?? "—"}</td>
                <td>{a.true_label}</td>
                <td>
                  {a.techniques.map((t) => (
                    <span key={t} className="badge tech">
                      {t}
                    </span>
                  ))}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
      {selected !== null && context.data && (
        <section className="panel">
          <h2>Fusion context — alert #{selected}</h2>
          {context.data.matched_campaigns.length === 0 ? (
            <div className="muted">
              no campaign currently shares this alert's techniques — no known active threat-intel
              context
            </div>
          ) : (
            context.data.matched_campaigns.map((m) => (
              <div key={m.campaign_id} className="detail">
                <code>{m.campaign_id}</code> · matched on{" "}
                {m.matched_techniques.map((t) => (
                  <span key={t} className="badge tech">
                    {t}
                  </span>
                ))}{" "}
                · {m.report_count} reports
                {m.kev_cves.length > 0 && (
                  <span className="badge kev" style={{ marginLeft: 8 }}>
                    involves actively-exploited CVEs: {m.kev_cves.join(", ")}
                  </span>
                )}
              </div>
            ))
          )}
        </section>
      )}
    </>
  );
}
