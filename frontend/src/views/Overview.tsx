import { useQuery } from "@tanstack/react-query";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api";

const PALETTE = ["#38bdf8", "#818cf8", "#34d399", "#fbbf24", "#f87171"];

export function Overview() {
  const stats = useQuery({ queryKey: ["stats"], queryFn: api.stats });
  const campaigns = useQuery({ queryKey: ["campaigns"], queryFn: api.campaigns });

  if (stats.error) return <div className="error">API unreachable — run `make up` and `make api`.</div>;
  if (!stats.data) return <div className="muted">loading…</div>;

  const s = stats.data;
  const cards = [
    { k: "CVEs ingested", v: s.vulnerabilities },
    { k: "KEV (exploited)", v: s.kev_entries },
    { k: "ATT&CK techniques", v: s.attack_techniques },
    { k: "Threat reports", v: s.threat_reports },
    { k: "Report↔technique edges", v: s.report_technique_edges },
    { k: "Campaigns", v: s.campaigns },
    { k: "IDS alerts", v: s.alerts },
  ];

  const campaignBars = (campaigns.data ?? []).map((c) => ({
    name: c.campaign_id.replace("camp:", "").slice(0, 6),
    reports: c.report_count,
    kev: c.kev_cves.length,
  }));

  const modelSlices = Object.entries(s.alerts_by_model).map(([name, value]) => ({ name, value }));

  return (
    <>
      <div className="cards">
        {cards.map((c) => (
          <div className="card" key={c.k}>
            <div className="v">{c.v.toLocaleString()}</div>
            <div className="k">{c.k}</div>
          </div>
        ))}
      </div>
      <div className="grid-2">
        <section className="panel">
          <h2>Campaigns by corroborating reports (KEV overlap in red)</h2>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={campaignBars}>
              <CartesianGrid stroke="#1f2937" vertical={false} />
              <XAxis dataKey="name" stroke="#8094ab" fontSize={11} />
              <YAxis stroke="#8094ab" fontSize={11} allowDecimals={false} />
              <Tooltip contentStyle={{ background: "#121821", border: "1px solid #1f2937" }} />
              <Bar dataKey="reports" fill="#38bdf8" radius={[4, 4, 0, 0]} isAnimationActive={false} />
              <Bar dataKey="kev" fill="#f87171" radius={[4, 4, 0, 0]} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </section>
        <section className="panel">
          <h2>Alerts by detector</h2>
          <ResponsiveContainer width="100%" height={260}>
            <PieChart>
              <Pie data={modelSlices} dataKey="value" nameKey="name" innerRadius={55} outerRadius={95} label isAnimationActive={false}>
                {modelSlices.map((entry, i) => (
                  <Cell key={entry.name} fill={PALETTE[i % PALETTE.length]} />
                ))}
              </Pie>
              <Tooltip contentStyle={{ background: "#121821", border: "1px solid #1f2937" }} />
            </PieChart>
          </ResponsiveContainer>
        </section>
      </div>
    </>
  );
}
