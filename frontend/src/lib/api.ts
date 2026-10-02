const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

function authHeaders(): HeadersInit {
  if (typeof window === "undefined") return {};
  const token = window.localStorage.getItem("dp_token");
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      // response had no JSON body
    }
    throw new ApiError(detail, res.status);
  }
  return res.json() as Promise<T>;
}

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

export async function login(username: string, password: string) {
  const res = await fetch(`${API_BASE}/api/v1/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  return handle<{ access_token: string; token_type: string; username: string }>(res);
}

export async function getMe() {
  const res = await fetch(`${API_BASE}/api/v1/auth/me`, { headers: authHeaders() });
  return handle<{ username: string }>(res);
}

export async function health() {
  const res = await fetch(`${API_BASE}/health`);
  return handle<{ status: string }>(res);
}

async function uploadForm<T>(path: string, form: FormData): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: authHeaders(),
    body: form,
  });
  return handle<T>(res);
}

export async function ingestDataset(file: File) {
  const form = new FormData();
  form.append("file", file);
  return uploadForm<{ reference: DatasetReference; profile: DatasetProfile }>(
    "/api/v1/datasets/ingest",
    form,
  );
}

export async function analyzeQuality(file: File, targetColumn?: string) {
  const form = new FormData();
  form.append("file", file);
  if (targetColumn) form.append("target_column", targetColumn);
  return uploadForm<QualityReport>("/api/v1/datasets/quality", form);
}

export async function analyzeEda(file: File) {
  const form = new FormData();
  form.append("file", file);
  return uploadForm<EdaReport>("/api/v1/datasets/eda", form);
}

export async function runModeling(file: File, objective: string, forecastHorizon = 1) {
  const form = new FormData();
  form.append("file", file);
  form.append("objective", objective);
  form.append("forecast_horizon", String(forecastHorizon));
  return uploadForm<ModelingSpec>("/api/v1/modeling/run", form);
}

export async function submitModelingJob(file: File, objective: string, forecastHorizon = 1) {
  const form = new FormData();
  form.append("file", file);
  form.append("objective", objective);
  form.append("forecast_horizon", String(forecastHorizon));
  return uploadForm<JobRecord>("/api/v1/jobs/modeling", form);
}

export async function getJob(jobId: string) {
  const res = await fetch(`${API_BASE}/api/v1/jobs/${jobId}`, { headers: authHeaders() });
  return handle<JobRecord>(res);
}

export async function queryAnalytics(experimentIds: string[], sql: string) {
  const res = await fetch(`${API_BASE}/api/v1/analytics/experiments/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ experiment_ids: experimentIds, sql }),
  });
  return handle<{ rows: Record<string, unknown>[] }>(res);
}

// ---- types (mirroring the backend's own Pydantic response shapes) --------

export interface DatasetReference {
  dataset_id: string;
  original_filename: string;
  source_format: string;
  size_bytes: number;
  sha256: string;
  created_at: string;
}

export interface DatasetProfile {
  dataset_id: string;
  n_rows: number;
  n_columns: number;
  column_names: string[];
  duplicate_row_count: number;
  numeric_columns: string[];
  categorical_columns: string[];
  datetime_columns: string[];
}

export interface QualityFinding {
  finding_type: string;
  column?: string | null;
  severity: string;
  message: string;
  suggested_action?: string | null;
}

export interface QualityReport {
  dataset_id: string;
  target_column?: string | null;
  summary: Record<string, unknown>;
  findings: QualityFinding[];
}

export interface QuantileValue {
  quantile: number;
  value: number | null;
}

export interface NumericColumnAnalysis {
  column: string;
  count: number;
  missing_count: number;
  missing_percentage: number;
  mean: number | null;
  median: number | null;
  std: number | null;
  minimum: number | null;
  maximum: number | null;
  quantiles: QuantileValue[];
}

export interface TopValue {
  value: string;
  count: number;
  frequency: number;
}

export interface CategoricalColumnAnalysis {
  column: string;
  count: number;
  missing_count: number;
  missing_percentage: number;
  unique_count: number;
  cardinality_ratio: number | null;
  top_values: TopValue[];
}

export interface DatetimeColumnAnalysis {
  column: string;
  count: number;
  missing_count: number;
  missing_percentage: number;
  minimum: string | null;
  maximum: string | null;
  unique_count: number;
}

export interface MissingnessAnalysis {
  total_cells: number;
  total_missing_cells: number;
  missing_percentage: number;
  columns: { column: string; missing_count: number; missing_percentage: number }[];
}

export interface UnivariateAnalysis {
  numeric: NumericColumnAnalysis[];
  categorical: CategoricalColumnAnalysis[];
  datetime: DatetimeColumnAnalysis[];
  missingness: MissingnessAnalysis;
}

export interface HistogramBin {
  left_edge: number;
  right_edge: number;
  count: number;
}

export interface Histogram {
  status: "completed" | "unavailable";
  reason?: string | null;
  n_bins: number | null;
  bin_edges: number[];
  bins: HistogramBin[];
  total_count: number | null;
}

export interface NumericDistribution {
  column: string;
  status: "completed" | "unavailable";
  reason?: string | null;
  count: number;
  minimum: number | null;
  maximum: number | null;
  mean: number | null;
  median: number | null;
  std: number | null;
  histogram: Histogram;
}

export interface DistributionAnalysis {
  columns: NumericDistribution[];
  notes: string[];
}

export interface EdaReport {
  dataset_id: string;
  n_rows: number;
  n_columns: number;
  column_names: string[];
  column_kinds: Record<string, "numeric" | "categorical" | "datetime">;
  univariate: UnivariateAnalysis;
  distribution: DistributionAnalysis;
  [key: string]: unknown;
}

export interface ModelingSpec {
  status: string;
  reason?: string | null;
  selection: {
    status: string;
    selected_family?: string | null;
    selected_estimator?: string | null;
    selection_metric?: string | null;
    selected_score?: number | null;
    ranking: {
      family: string;
      estimator_name?: string | null;
      rank?: number | null;
      score?: number | null;
    }[];
  };
  training: {
    status: string;
    runs: { family: string; status: string; metrics: Record<string, number> }[];
  };
}

export interface JobRecord {
  job_id: string;
  kind: string;
  status: "pending" | "running" | "completed" | "failed";
  created_at: string;
  updated_at: string;
  result: ModelingSpec | null;
  error: string | null;
}
