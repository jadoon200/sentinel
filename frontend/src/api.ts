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

export interface FusionScore {
  strength: number;
  specificity: number;
  recency: number;
  corroboration: number;
  age_days: number | null;
}

export interface CampaignMatch {
  campaign_id: string;
  cve_ids: string[];
  kev_cves: string[];
  report_count: number;
  matched_techniques: string[];
  fusion: FusionScore;
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

export interface CampaignLink {
  campaign_id: string;
  matched_techniques: string[];
  report_count: number;
  kev_cves: string[];
  fusion: FusionScore;
}

export interface HostThreat {
  host: string;
  risk: number;
  detectors: string[];
  techniques: string[];
  predicted_labels: string[];
  true_labels: string[];
  alert_count: number;
  fused: CampaignLink[];
  simulated: boolean;
}

export interface TrendingItem {
  technique_id: string;
  name: string | null;
  recent_count: number;
  prior_count: number;
  lift: number;
}

export interface DriftOut {
  population_stability_index: number;
  verdict: string;
  top_shifts: [string, number][];
}

export const api = {
  stats: () => get<Stats>("/stats"),
  hosts: () => get<HostThreat[]>("/hosts"),
  simulatedHosts: () => get<HostThreat[]>("/hosts/simulated"),
  campaigns: () => get<CampaignSummary[]>("/campaigns"),
  campaign: (id: string) => get<CampaignDetail>(`/campaigns/${id}`),
  reports: (limit = 100) => get<ReportSummary[]>(`/reports?limit=${limit}`),
  alerts: (model?: string, limit = 200) =>
    get<AlertOut[]>(`/alerts?limit=${limit}${model ? `&model=${model}` : ""}`),
  alertContext: (id: number) => get<AlertContext>(`/alerts/${id}/context`),
  techniques: () => get<TechniqueListItem[]>("/techniques"),
  trending: () => get<TrendingItem[]>("/trending"),
  drift: () => get<DriftOut>("/feed-drift"),
  briefing: async () => {
    const res = await fetch(`${BASE}/briefing`);
    return res.text();
  },
};
