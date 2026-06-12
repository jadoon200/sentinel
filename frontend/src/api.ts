/** Typed client for the SENTINEL knowledge-graph API. */

const BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export interface Stats {
  vulnerabilities: number;
  kev_entries: number;
  attack_techniques: number;
  threat_reports: number;
  report_technique_edges: number;
  campaigns: number;
  alerts: number;
  alerts_by_model: Record<string, number>;
}

export interface TechniqueEvidence {
  technique_id: string;
  name: string | null;
  score: number;
  corroborations: number;
}

export interface CampaignSummary {
  campaign_id: string;
  cve_ids: string[];
  kev_cves: string[];
  report_count: number;
  techniques: TechniqueEvidence[];
}

export interface ReportSummary {
  report_id: string;
  source: string;
  title: string;
  url: string | null;
  published: string | null;
  techniques: TechniqueEvidence[];
}

export interface CampaignDetail extends CampaignSummary {
  reports: ReportSummary[];
}

export interface AlertOut {
  alert_id: number;
  model: string;
  day: string | null;
  score: number;
  predicted_label: string | null;
  true_label: string | null;
  techniques: string[];
}

export interface CampaignMatch {
  campaign_id: string;
  cve_ids: string[];
  kev_cves: string[];
  report_count: number;
  matched_techniques: string[];
}

export interface AlertContext {
  alert: AlertOut;
  matched_campaigns: CampaignMatch[];
}

export interface TechniqueListItem {
  technique_id: string;
  name: string;
  tactics: string[];
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  return res.json() as Promise<T>;
}

export const api = {
  stats: () => get<Stats>("/stats"),
  campaigns: () => get<CampaignSummary[]>("/campaigns"),
  campaign: (id: string) => get<CampaignDetail>(`/campaigns/${id}`),
  reports: (limit = 100) => get<ReportSummary[]>(`/reports?limit=${limit}`),
  alerts: (model?: string, limit = 200) =>
    get<AlertOut[]>(`/alerts?limit=${limit}${model ? `&model=${model}` : ""}`),
  alertContext: (id: number) => get<AlertContext>(`/alerts/${id}/context`),
  techniques: () => get<TechniqueListItem[]>("/techniques"),
};
