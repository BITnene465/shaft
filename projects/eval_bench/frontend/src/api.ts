export type BenchmarkSummary = {
  benchmark_id: string;
  tasks: string[];
  labels: string[];
  layers: string[];
  split: string;
  sample_count: number;
  root: string;
  manifest_path: string;
  created_at: string | null;
  source_manifest_path: string | null;
};

export type BenchmarkListFilters = {
  offset?: number;
  limit?: number;
  task?: string;
  layer?: string;
  split?: string;
  query?: string;
};

export type BenchmarkListResponse = {
  benchmarks: BenchmarkSummary[];
  total?: number;
  offset?: number;
  limit?: number;
  filters?: Record<string, string>;
};

export type RunSummary = {
  run_id: string;
  status: string;
  benchmark_id: string;
  tasks: string[];
  spec_task: string;
  target_labels: string[];
  model_id: string;
  model_path: string;
  prompt_id: string;
  prompt_path: string | null;
  prompt_hash: string | null;
  prompt_metadata: Record<string, unknown>;
  parser: string;
  metric_profile: string;
  visualization_profile: string;
  inference: Record<string, unknown>;
  created_at: string | null;
  prediction_count: number;
  report_count: number;
  manifest_path: string;
  report_path: string | null;
  note: string;
  note_updated_at: string | null;
  note_max_length: number;
  precision_iou50: number | null;
  recall_iou50: number | null;
  mean_iou: number | null;
};

export type RunListFilters = {
  offset?: number;
  limit?: number;
  task?: string;
  benchmarkId?: string;
  status?: string;
  label?: string;
  modelId?: string;
  promptId?: string;
  metricProfile?: string;
  query?: string;
};

export type RunListResponse = {
  runs: RunSummary[];
  total?: number;
  offset?: number;
  limit?: number;
  filters?: Record<string, string>;
};

export type RunNote = {
  run_id: string;
  note: string;
  updated_at: string | null;
  path: string;
  max_length: number;
};

export type RankBoardEntry = {
  rank: number;
  f1_iou50: number | null;
  run_id: string;
  score: number | null;
  status: string;
  benchmark_id: string;
  task: string;
  target_labels: string[];
  model_id: string;
  prompt_id: string;
  metric_profile: string;
  prediction_count: number;
  precision_iou50: number | null;
  recall_iou50: number | null;
  mean_iou: number | null;
  created_at: string | null;
  note: string;
  score_components: Array<Record<string, unknown>>;
};

export type RankBoard = {
  offset: number;
  limit: number;
  total: number;
  evaluated_count: number;
  filters: Record<string, string>;
  primary_metric: string;
  primary_metric_label: string;
  sort_by: string;
  sort_order: string;
  score_formula: string;
  rank_scheme: Record<string, unknown> | null;
  facets: Record<string, Array<{ value: string; count: number }>>;
  entries: RankBoardEntry[];
};

export type JobSummary = {
  job_id: string;
  kind: string;
  status: string;
  payload: Record<string, unknown>;
  created_at: string | null;
  updated_at: string | null;
  error: string | null;
  metadata: Record<string, unknown>;
};

export type JobListFilters = {
  offset?: number;
  limit?: number;
  kind?: string;
  status?: string;
  query?: string;
};

export type JobListResponse = {
  jobs: JobSummary[];
  total?: number;
  offset?: number;
  limit?: number;
  filters?: Record<string, string>;
};

export type ServiceSummary = {
  service_id: string;
  kind: string;
  status: string;
  config: Record<string, unknown>;
  runtime: Record<string, unknown>;
  created_at: string | null;
  updated_at: string | null;
  error: string | null;
  metadata: Record<string, unknown>;
};

export type ServiceListFilters = {
  offset?: number;
  limit?: number;
  kind?: string;
  status?: string;
  query?: string;
};

export type ServiceListResponse = {
  services: ServiceSummary[];
  total?: number;
  offset?: number;
  limit?: number;
  filters?: Record<string, string>;
};

export type ServiceLog = {
  service_id: string;
  log_path: string | null;
  lines: string[];
  text: string;
};

export type JobLog = {
  job_id: string;
  log_path: string | null;
  lines: string[];
  text: string;
};

export type SchedulerStatus = {
  enabled: boolean;
  loop_alive?: boolean;
  max_concurrent_jobs?: number;
  interval_s?: number;
  live_running_jobs?: string[];
  live_running_count?: number;
  active_worker_threads?: string[];
  reserved_cuda_devices?: string[];
  reserved_runtime_ports?: number[];
};

export type EvalInstance = {
  label: string;
  bbox: number[];
  keypoints?: number[][] | null;
  linestrip?: number[][] | null;
  score?: number | null;
  extra?: Record<string, unknown>;
};

export type SampleDiagnostics = {
  matched_count: number;
  false_negative_count: number;
  false_positive_count: number;
  mean_iou: number;
  matches: Array<{ label: string; gt_index: number; pred_index: number; iou: number }>;
  false_negatives: Array<{ index: number; label: string; bbox: number[] }>;
  false_positives: Array<{ index: number; label: string; bbox: number[] }>;
  labels: Record<
    string,
    {
      gt_count: number;
      pred_count: number;
      matched_count: number;
      false_negative_count: number;
      false_positive_count: number;
      mean_iou: number;
    }
  >;
};

export type RunSampleSummary = {
  index: number;
  image: string;
  json_path: string;
  image_width: number | null;
  image_height: number | null;
  gt_instance_count: number;
  pred_instance_count: number;
  labels: string[];
  has_prediction: boolean;
  prediction_path: string | null;
  image_url: string;
  image_preview_url?: string;
  image_tile_url_template?: string;
  image_tile_size?: number;
  diagnostics: SampleDiagnostics | null;
};

export type BenchmarkSampleSummary = {
  index: number;
  image: string;
  json_path: string;
  image_width: number | null;
  image_height: number | null;
  instance_count: number;
  labels: string[];
  image_url: string;
  image_preview_url?: string;
  image_tile_url_template?: string;
  image_tile_size?: number;
};

export type RunSampleDetail = {
  run_id: string;
  sample: RunSampleSummary;
  gt_instances: EvalInstance[];
  pred_instances: EvalInstance[];
  raw_payload: Record<string, unknown>;
  prediction_payload: Record<string, unknown> | null;
  diagnostics: SampleDiagnostics | null;
};

export type BenchmarkSampleDetail = {
  benchmark_id: string;
  sample: BenchmarkSampleSummary;
  gt_instances: EvalInstance[];
  raw_payload: Record<string, unknown>;
};

export type SamplePage<T> = {
  offset: number;
  limit: number;
  total: number;
  labels: string[];
  samples: T[];
};

export type DashboardState = {
  store_root: string;
  benchmark_count: number;
  run_count: number;
  total_benchmark_samples: number;
  prediction_count: number;
  benchmarks: BenchmarkSummary[];
  runs: RunSummary[];
};

export type CreateBenchmarkPayload = {
  benchmark_id: string;
  source_root: string;
  source_manifest: string;
  split: string;
  tasks: string[];
  layers?: string[];
  overwrite?: boolean;
};

export type ComparisonReport = {
  comparison_id?: string;
  baseline_run_id: string;
  candidate_run_id: string;
  task: string;
  metric_profile?: string;
  target_labels?: string[];
  warnings?: string[];
  sample_count: number;
  delta: {
    precision_iou50: number;
    recall_iou50: number;
    mean_iou: number;
    mean_keypoint_distance: number;
    matched_count: number;
    keypoint_pair_count: number;
    false_positive_count: number;
    false_negative_count: number;
  };
  summary: {
    improved_samples: number;
    regressed_samples: number;
    changed_samples: number;
    unchanged_samples: number;
    missing_in_baseline: number;
    missing_in_candidate: number;
  };
  labels?: ComparisonLabelDelta[];
  top_improvements: ComparisonSample[];
  top_regressions: ComparisonSample[];
};

export type ComparisonSummary = {
  comparison_id: string;
  baseline_run_id: string;
  candidate_run_id: string;
  task: string;
  metric_profile?: string;
  target_labels?: string[];
  target_labels_source?: string | null;
  sample_count: number;
  created_at: string | null;
  path: string;
  delta: ComparisonReport["delta"];
  summary: ComparisonReport["summary"];
};

export type ComparisonListFilters = {
  task?: string;
  label?: string;
  query?: string;
  offset?: number;
  limit?: number;
};

export type ComparisonSampleDetail = {
  baseline_run_id: string;
  candidate_run_id: string;
  sample_index: number;
  baseline: RunSampleDetail;
  candidate: RunSampleDetail;
};

export type ComparisonSample = {
  key: string;
  image: string | null;
  sample_index: number | null;
  baseline_index: number | null;
  candidate_index: number | null;
  status: string;
  labels?: Record<
    string,
    {
      matched_count: number;
      false_positive_count: number;
      false_negative_count: number;
      mean_iou: number;
      keypoint_pair_count: number;
      mean_keypoint_distance: number;
    }
  >;
  delta_score: number;
  delta: {
    matched_count: number;
    false_positive_count: number;
    false_negative_count: number;
    mean_iou: number;
    keypoint_pair_count: number;
    mean_keypoint_distance: number;
  };
};

export type ComparisonLabelDelta = {
  label: string;
  delta_score: number;
  baseline: Record<string, number>;
  candidate: Record<string, number>;
  delta: {
    precision_iou50: number;
    recall_iou50: number;
    mean_iou: number;
    mean_keypoint_distance: number;
    matched_count: number;
    keypoint_pair_count: number;
    false_positive_count: number;
    false_negative_count: number;
  };
};

export type ImportPredictionPayload = {
  run_id: string;
  benchmark_id: string;
  prediction_root: string;
  task: string;
  model_id: string;
  model_path?: string;
  prompt_id?: string;
  spec_id?: string;
  target_labels?: string[];
  strict?: boolean;
  overwrite?: boolean;
  evaluate?: boolean;
};

export type ImportPredictionResult = {
  run_id: string;
  run_manifest_path: string;
  report_path: string | null;
  imported_predictions: number;
  missing_predictions: string[];
  missing_prediction_count: number;
};

export type DeleteResult = {
  deleted: boolean;
  trash_path: string | null;
};

export type JobTemplate = {
  label: string;
  description: string;
  manifest: Record<string, unknown>;
};

export type JobTemplatesResponse = {
  templates: Record<string, JobTemplate>;
};

export type PromptTemplate = {
  prompt_id: string;
  label: string;
  task: "detection" | "keypoint" | string;
  system_prompt: string;
  user_prompt: string;
  parser: string | null;
  metric_profile: string | null;
  visualization_profile: string | null;
  generation: Record<string, unknown>;
  data: Record<string, unknown>;
  metadata: Record<string, unknown>;
  created_at: string | null;
  updated_at: string | null;
};

export type PromptTemplatesResponse = {
  templates: PromptTemplate[];
  by_id: Record<string, PromptTemplate>;
};

export type JobPreflightResult = {
  ok: boolean;
  errors: string[];
  warnings: string[];
  kind?: string;
  resolved_manifest?: Record<string, unknown>;
  resolved_payload?: Record<string, unknown>;
  runtime_command?: string[] | null;
};

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { Accept: "application/json", ...(init?.headers ?? {}) }
  });
  if (!response.ok) {
    let detail = "";
    try {
      const errorPayload = (await response.json()) as { detail?: unknown };
      detail = typeof errorPayload.detail === "string" ? `: ${errorPayload.detail}` : "";
    } catch {
      detail = "";
    }
    const requestId = response.headers.get("x-eval-bench-request-id");
    const message = `${response.status} ${response.statusText}${detail}${
      requestId ? ` (request ${requestId})` : ""
    }`;
    notifyApiError(message);
    throw new Error(message);
  }
  return (await response.json()) as T;
}

function notifyApiError(message: string) {
  if (typeof window === "undefined") {
    return;
  }
  window.dispatchEvent(
    new CustomEvent("eval-bench-api-error", {
      detail: { message }
    })
  );
}

export function fetchState(): Promise<DashboardState> {
  return fetchJson<DashboardState>("/api/state");
}

export function fetchRankBoard(options: {
  offset: number;
  limit: number;
  task?: string;
  benchmarkId?: string;
  status?: string;
  label?: string;
  modelId?: string;
  promptId?: string;
  metricProfile?: string;
  minScore?: string;
  sortBy?: string;
  sortOrder?: string;
  query?: string;
  rankScheme?: string;
}): Promise<RankBoard> {
  const params = new URLSearchParams({
    offset: String(options.offset),
    limit: String(options.limit)
  });
  if (options.task && options.task !== "all") {
    params.set("task", options.task);
  }
  if (options.benchmarkId && options.benchmarkId !== "all") {
    params.set("benchmark_id", options.benchmarkId);
  }
  if (options.status && options.status !== "all") {
    params.set("status", options.status);
  }
  if (options.label && options.label !== "all") {
    params.set("label", options.label);
  }
  if (options.modelId && options.modelId !== "all") {
    params.set("model_id", options.modelId);
  }
  if (options.promptId && options.promptId !== "all") {
    params.set("prompt_id", options.promptId);
  }
  if (options.metricProfile && options.metricProfile !== "all") {
    params.set("metric_profile", options.metricProfile);
  }
  if (options.minScore?.trim()) {
    params.set("min_score", options.minScore.trim());
  }
  if (options.sortBy) {
    params.set("sort_by", options.sortBy);
  }
  if (options.sortOrder) {
    params.set("sort_order", options.sortOrder);
  }
  if (options.query?.trim()) {
    params.set("query", options.query.trim());
  }
  if (options.rankScheme?.trim()) {
    params.set("rank_scheme", options.rankScheme.trim());
  }
  return fetchJson<RankBoard>(`/api/rank-board?${params.toString()}`);
}

export function fetchJobs(filters: JobListFilters = {}): Promise<JobListResponse> {
  const params = new URLSearchParams();
  Object.entries(filters).forEach(([key, value]) => {
    if (value !== undefined && value !== null && String(value).trim() !== "") {
      params.set(key, String(value));
    }
  });
  const query = params.toString();
  return fetchJson<JobListResponse>(`/api/jobs${query ? `?${query}` : ""}`);
}

export function fetchSchedulerStatus(): Promise<SchedulerStatus> {
  return fetchJson<SchedulerStatus>("/api/scheduler/status");
}

export function fetchJobTemplates(): Promise<JobTemplatesResponse> {
  return fetchJson<JobTemplatesResponse>("/api/job-templates");
}

export function fetchPromptTemplates(): Promise<PromptTemplatesResponse> {
  return fetchJson<PromptTemplatesResponse>("/api/prompt-templates");
}

export function upsertPromptTemplate(payload: Partial<PromptTemplate>): Promise<PromptTemplate> {
  return fetchJson<PromptTemplate>("/api/prompt-templates", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

export function fetchServices(filters: ServiceListFilters = {}): Promise<ServiceListResponse> {
  const params = new URLSearchParams();
  Object.entries(filters).forEach(([key, value]) => {
    if (value !== undefined && value !== null && String(value).trim() !== "") {
      params.set(key, String(value));
    }
  });
  const query = params.toString();
  return fetchJson<ServiceListResponse>(`/api/services${query ? `?${query}` : ""}`);
}

export function fetchBenchmarks(
  filters: BenchmarkListFilters = {}
): Promise<BenchmarkListResponse> {
  const params = new URLSearchParams();
  Object.entries(filters).forEach(([key, value]) => {
    if (value !== undefined && value !== null && String(value).trim() !== "") {
      params.set(key, String(value));
    }
  });
  const query = params.toString();
  return fetchJson<BenchmarkListResponse>(`/api/benchmarks${query ? `?${query}` : ""}`);
}

export function createBenchmark(payload: CreateBenchmarkPayload): Promise<BenchmarkSummary> {
  return fetchJson<BenchmarkSummary>("/api/benchmarks", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

export function createService(payload: Record<string, unknown>): Promise<ServiceSummary> {
  return fetchJson<ServiceSummary>("/api/services", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

export function startService(serviceId: string): Promise<ServiceSummary> {
  return fetchJson<ServiceSummary>(`/api/services/${encodeURIComponent(serviceId)}/start`, {
    method: "POST"
  });
}

export function checkServiceHealth(serviceId: string): Promise<ServiceSummary> {
  return fetchJson<ServiceSummary>(`/api/services/${encodeURIComponent(serviceId)}/health`, {
    method: "POST"
  });
}

export function fetchServiceLogs(serviceId: string): Promise<ServiceLog> {
  return fetchJson<ServiceLog>(`/api/services/${encodeURIComponent(serviceId)}/logs`);
}

export function stopService(serviceId: string): Promise<ServiceSummary> {
  return fetchJson<ServiceSummary>(`/api/services/${encodeURIComponent(serviceId)}/stop`, {
    method: "POST"
  });
}

export function deleteService(
  serviceId: string
): Promise<{ service: ServiceSummary; trash_path: string | null }> {
  return fetchJson<{ service: ServiceSummary; trash_path: string | null }>(
    `/api/services/${encodeURIComponent(serviceId)}`,
    { method: "DELETE" }
  );
}

export function createJob(payload: Record<string, unknown>): Promise<JobSummary> {
  return fetchJson<JobSummary>("/api/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

export function preflightJob(payload: Record<string, unknown>): Promise<JobPreflightResult> {
  return fetchJson<JobPreflightResult>("/api/jobs/preflight", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

export function processNextJob(): Promise<{
  processed: boolean;
  job: JobSummary | null;
  background?: boolean;
  message?: string;
}> {
  return fetchJson<{
    processed: boolean;
    job: JobSummary | null;
    background?: boolean;
    message?: string;
  }>("/api/jobs/process-next", { method: "POST" });
}

export function cancelJob(jobId: string): Promise<JobSummary> {
  return fetchJson<JobSummary>(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, {
    method: "POST"
  });
}

export function fetchJobLogs(jobId: string, maxLines = 200): Promise<JobLog> {
  const params = new URLSearchParams({ max_lines: String(maxLines) });
  return fetchJson<JobLog>(`/api/jobs/${encodeURIComponent(jobId)}/logs?${params.toString()}`);
}

export function deleteJob(jobId: string): Promise<DeleteResult & { job_id: string }> {
  return fetchJson<DeleteResult & { job_id: string }>(`/api/jobs/${encodeURIComponent(jobId)}`, {
    method: "DELETE"
  });
}

export function fetchRuns(filters: RunListFilters = {}): Promise<RunListResponse> {
  const params = new URLSearchParams();
  if (filters.offset !== undefined) {
    params.set("offset", String(filters.offset));
  }
  if (filters.limit !== undefined) {
    params.set("limit", String(filters.limit));
  }
  if (filters.task && filters.task !== "all") {
    params.set("task", filters.task);
  }
  if (filters.benchmarkId && filters.benchmarkId !== "all") {
    params.set("benchmark_id", filters.benchmarkId);
  }
  if (filters.status && filters.status !== "all") {
    params.set("status", filters.status);
  }
  if (filters.label && filters.label !== "all") {
    params.set("label", filters.label);
  }
  if (filters.modelId && filters.modelId !== "all") {
    params.set("model_id", filters.modelId);
  }
  if (filters.promptId && filters.promptId !== "all") {
    params.set("prompt_id", filters.promptId);
  }
  if (filters.metricProfile && filters.metricProfile !== "all") {
    params.set("metric_profile", filters.metricProfile);
  }
  if (filters.query?.trim()) {
    params.set("query", filters.query.trim());
  }
  const query = params.toString();
  return fetchJson<RunListResponse>(`/api/runs${query ? `?${query}` : ""}`);
}

export function evaluateRun(runId: string): Promise<{ run_id: string; report_path: string }> {
  return fetchJson<{ run_id: string; report_path: string }>(`/api/runs/${runId}/evaluate`, {
    method: "POST"
  });
}

export function archiveRun(runId: string): Promise<{ run_id: string; status: string }> {
  return fetchJson<{ run_id: string; status: string }>(
    `/api/runs/${encodeURIComponent(runId)}/archive`,
    { method: "POST" }
  );
}

export function deleteRun(runId: string): Promise<DeleteResult & { run_id: string }> {
  return fetchJson<DeleteResult & { run_id: string }>(`/api/runs/${encodeURIComponent(runId)}`, {
    method: "DELETE"
  });
}

export function fetchRunNote(runId: string): Promise<RunNote> {
  return fetchJson<RunNote>(`/api/runs/${encodeURIComponent(runId)}/note`);
}

export function updateRunNote(runId: string, note: string): Promise<RunNote> {
  return fetchJson<RunNote>(`/api/runs/${encodeURIComponent(runId)}/note`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ note })
  });
}

export function importPredictions(
  payload: ImportPredictionPayload
): Promise<ImportPredictionResult> {
  return fetchJson<ImportPredictionResult>("/api/runs/import-predictions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

export function fetchRunSamples(
  runId: string,
  options: { offset: number; limit: number; label?: string; errorFilter?: string }
): Promise<SamplePage<RunSampleSummary>> {
  const params = new URLSearchParams({
    offset: String(options.offset),
    limit: String(options.limit)
  });
  if (options.label && options.label !== "all") {
    params.set("label", options.label);
  }
  if (options.errorFilter && options.errorFilter !== "all") {
    params.set("error_filter", options.errorFilter);
  }
  return fetchJson<SamplePage<RunSampleSummary>>(`/api/runs/${runId}/samples?${params.toString()}`);
}

export function fetchRunSampleDetail(runId: string, index: number): Promise<RunSampleDetail> {
  return fetchJson<RunSampleDetail>(`/api/runs/${runId}/samples/${index}`);
}

export function fetchBenchmarkSamples(
  benchmarkId: string,
  options: { offset: number; limit: number; label?: string }
): Promise<SamplePage<BenchmarkSampleSummary>> {
  const params = new URLSearchParams({
    offset: String(options.offset),
    limit: String(options.limit)
  });
  if (options.label && options.label !== "all") {
    params.set("label", options.label);
  }
  return fetchJson<SamplePage<BenchmarkSampleSummary>>(
    `/api/benchmarks/${benchmarkId}/samples?${params.toString()}`
  );
}

export function fetchBenchmarkSampleDetail(
  benchmarkId: string,
  index: number
): Promise<BenchmarkSampleDetail> {
  return fetchJson<BenchmarkSampleDetail>(`/api/benchmarks/${benchmarkId}/samples/${index}`);
}

export function fetchSettingsPreviewSample(): Promise<BenchmarkSampleDetail> {
  return fetchJson<BenchmarkSampleDetail>("/api/settings/preview-sample");
}

export function fetchComparison(
  baselineRunId: string,
  candidateRunId: string
): Promise<ComparisonReport> {
  const params = new URLSearchParams({
    baseline_run_id: baselineRunId,
    candidate_run_id: candidateRunId
  });
  return fetchJson<ComparisonReport>(`/api/comparisons?${params.toString()}`);
}

export function fetchComparisons(
  filters: ComparisonListFilters = {}
): Promise<{
  comparisons: ComparisonSummary[];
  total?: number;
  offset?: number;
  limit?: number;
  filters?: Record<string, string>;
}> {
  const params = new URLSearchParams();
  Object.entries(filters).forEach(([key, value]) => {
    if (value !== undefined && value !== null && String(value).trim() !== "") {
      params.set(key, String(value));
    }
  });
  const query = params.toString();
  return fetchJson<{
    comparisons: ComparisonSummary[];
    total?: number;
    offset?: number;
    limit?: number;
    filters?: Record<string, string>;
  }>(`/api/comparisons${query ? `?${query}` : ""}`);
}

export function fetchComparisonSample(
  baselineRunId: string,
  candidateRunId: string,
  sampleIndex: number
): Promise<ComparisonSampleDetail> {
  const params = new URLSearchParams({
    baseline_run_id: baselineRunId,
    candidate_run_id: candidateRunId,
    sample_index: String(sampleIndex)
  });
  return fetchJson<ComparisonSampleDetail>(`/api/comparisons/sample?${params.toString()}`);
}
