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
  age_days: number | null;
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

export interface MappedTechnique {
  technique_id: string;
  name: string;
  score: number;
  corroborations: number;
  tactics: string[];
  url: string | null;
}

export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function apiError(path: string, res: Response): Promise<ApiError> {
  let detail = `${path}: ${res.status}`;
  try {
    const body = (await res.json()) as { detail?: string };
    if (body.detail) detail = body.detail;
  } catch {
    // Preserve the status fallback when an upstream proxy returns non-JSON.
  }
  return new ApiError(detail, res.status);
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw await apiError(path, res);
  return res.json() as Promise<T>;
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    ...(body === undefined
      ? {}
      : { headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }),
  });
  if (!res.ok) throw await apiError(path, res);
  return res.json() as Promise<T>;
}

export type CalibrationLabel = "benign" | "attack";

export interface CalibrationFlow {
  id: number;
  pool_row: number;
  features: Record<string, number>;
  model_score: number;
  operator_label?: CalibrationLabel;
  true_label?: CalibrationLabel;
  labelled_at?: string;
}

export interface CalibrationRun {
  id: number;
  created_at: string;
  recall_before: number;
  recall_after: number;
  fpr_after: number;
  auc_after: number;
  n_labels_used: number;
  operator_accuracy: number;
  metrics: {
    per_family_recall?: Record<string, number>;
    baseline_per_family_recall?: Record<string, number>;
  };
}

export interface CalibrationBatch {
  id: number;
  created_at: string;
  strategy: string;
  seed: number;
  n_flows: number;
  n_labelled: number;
  status: string;
  notes?: string;
  flows: CalibrationFlow[];
  runs: CalibrationRun[];
}

export interface CalibrationCurvePoint {
  n: number;
  mean_recall: number;
  families: Record<string, number>;
}

export interface CalibrationCurve {
  strategy: string;
  points: CalibrationCurvePoint[];
}

export interface CampaignLink {
  campaign_id: string;
  matched_techniques: string[];
  report_count: number;
  kev_cves: string[];
  fusion: FusionScore;
}

export interface AlertRef {
  alert_id: number;
  model: string;
  score: number;
  predicted_label: string | null;
  techniques: string[];
}

export interface HostThreat {
  host: string;
  risk: number;
  detectors: string[];
  techniques: string[];
  predicted_labels: string[];
  true_labels: string[];
  alert_count: number;
  alerts: AlertRef[];
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
  mapTechniques: (text: string) => post<MappedTechnique[]>("/map-techniques", { text }),
  createCalibrationBatch: (n = 50, seed = 13) =>
    post<CalibrationBatch>("/calibration/batches", {
      n,
      seed,
      strategy: "stratified",
    }),
  calibrationBatch: (id: number) => get<CalibrationBatch>(`/calibration/batches/${id}`),
  labelCalibrationFlow: (id: number, label: CalibrationLabel) =>
    post<CalibrationFlow>(`/calibration/flows/${id}/label`, { label }),
  simulateCalibrationLabel: (id: number) =>
    post<CalibrationFlow>(`/calibration/flows/${id}/simulate-label`),
  retrainCalibrationBatch: (id: number) =>
    post<CalibrationRun>(`/calibration/batches/${id}/retrain`),
  calibrationCurve: () => get<CalibrationCurve>("/calibration/curve"),
  trending: () => get<TrendingItem[]>("/trending"),
  drift: () => get<DriftOut>("/feed-drift"),
  briefing: async () => {
    const res = await fetch(`${BASE}/briefing`);
    if (!res.ok) throw new Error(`/briefing: ${res.status}`);
    return res.text();
  },
};
