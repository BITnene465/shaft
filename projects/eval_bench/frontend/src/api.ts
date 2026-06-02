import type {
  BenchmarkSummary,
  BenchmarkManifest,
  FacetBucket,
  FacetBuckets,
  BenchmarkListFilters,
  BenchmarkListResponse,
  BenchmarkDetailResponse,
  RunSummary,
  RunListFilters,
  RunListResponse,
  RunNote,
  RankBoardEntry,
  RankBoard,
  SuiteTaskSplitSummary,
  SuiteSummary,
  SuiteListResponse,
  SuiteDetailResponse,
  CampaignSummary,
  CampaignListResponse,
  CampaignDetailResponse,
  SuiteRankEntry,
  SuiteRankBoard,
  JobSummary,
  JobListFilters,
  JobListResponse,
  ServiceSummary,
  ServiceListFilters,
  ServiceListResponse,
  ServiceLog,
  JobLog,
  SchedulerStatus,
  EvalInstance,
  SampleDiagnostics,
  RunSampleSummary,
  BenchmarkSampleSummary,
  RunSampleDetail,
  CompositeSampleLayer,
  CompositeLayerStatus,
  CompositeSampleView,
  BenchmarkSampleDetail,
  SamplePage,
  DashboardState,
  CreateBenchmarkPayload,
  CreateBenchmarkSlicePayload,
  ComparisonRunMetrics,
  ComparisonDelta,
  ComparisonOverview,
  ComparisonSampleMetrics,
  ComparisonSummary,
  ComparisonListFilters,
  ComparisonSampleDetail,
  ComparisonSample,
  ComparisonLabelDelta,
  ComparisonReport,
  ImportPredictionPayload,
  ImportPredictionResult,
  DeleteResult,
  ArchiveRunResult,
  DeleteJobResult,
  JobTemplate,
  JobTemplatesResponse,
  PromptTemplate,
  PromptTemplatesResponse,
  TargetLabelResolution,
  TargetLabelResolutionParams,
  JobPreflightResult
} from "./apiTypes";

export type * from "./apiTypes";

export class ApiError extends Error {
  status: number;
  statusText: string;
  detail: string;
  requestId: string | null;

  constructor({
    status,
    statusText,
    detail,
    requestId
  }: {
    status: number;
    statusText: string;
    detail: string;
    requestId: string | null;
  }) {
    const message = `${status} ${statusText}${detail}${requestId ? ` (request ${requestId})` : ""}`;
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.statusText = statusText;
    this.detail = detail;
    this.requestId = requestId;
  }
}

export function isApiError(error: unknown): error is ApiError {
  return error instanceof ApiError;
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { Accept: "application/json", ...(init?.headers ?? {}) }
  });
  if (!response.ok) {
    let detail = "";
    try {
      const errorPayload = (await response.json()) as unknown;
      const detailText = apiErrorDetailText(errorDetailPayload(errorPayload));
      detail = detailText ? `: ${detailText}` : "";
    } catch {
      detail = "";
    }
    const requestId = response.headers.get("x-eval-bench-request-id");
    const error = new ApiError({
      status: response.status,
      statusText: response.statusText,
      detail,
      requestId
    });
    notifyApiError(error.message);
    throw error;
  }
  return (await response.json()) as T;
}

export function apiErrorDetailText(detail: unknown): string {
  if (typeof detail === "string") {
    return detail.trim();
  }
  if (Array.isArray(detail)) {
    return truncateErrorDetail(detail.map(errorItemText).filter(Boolean).join("; "));
  }
  if (!isPlainRecord(detail)) {
    return "";
  }
  const lines = [
    ...stringList(detail.errors),
    ...stringList(detail.warnings).map((warning) => `warning: ${warning}`)
  ];
  if (lines.length) {
    return truncateErrorDetail(lines.join("; "));
  }
  for (const key of ["detail", "message", "error"]) {
    const value = detail[key];
    if (typeof value === "string" && value.trim()) {
      return truncateErrorDetail(value.trim());
    }
  }
  return truncateErrorDetail(JSON.stringify(detail));
}

function errorDetailPayload(payload: unknown): unknown {
  return isPlainRecord(payload) && "detail" in payload ? payload.detail : payload;
}

function errorItemText(item: unknown): string {
  if (typeof item === "string") {
    return item.trim();
  }
  if (!isPlainRecord(item)) {
    return String(item).trim();
  }
  const message = typeof item.msg === "string" ? item.msg.trim() : "";
  const location = Array.isArray(item.loc) ? item.loc.map(String).join(".") : "";
  if (message && location) {
    return `${location}: ${message}`;
  }
  if (message) {
    return message;
  }
  return JSON.stringify(item);
}

function stringList(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.map((item) => String(item).trim()).filter(Boolean);
}

function truncateErrorDetail(value: string): string {
  return value.length > 1200 ? `${value.slice(0, 1197)}...` : value;
}

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
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
  benchmarkSplit?: string;
  status?: string;
  label?: string;
  modelId?: string;
  promptId?: string;
  metricProfile?: string;
  minScore?: string;
  sortBy?: string;
  sortOrder?: string;
  query?: string;
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
  if (options.benchmarkSplit && options.benchmarkSplit !== "all") {
    params.set("benchmark_split", options.benchmarkSplit);
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
  return fetchJson<RankBoard>(`/api/rank-board?${params.toString()}`);
}

export function fetchSuiteRankBoard(options: {
  offset: number;
  limit: number;
  suiteId?: string;
  modelId?: string;
  promptId?: string;
  sortBy?: string;
  sortOrder?: string;
}): Promise<SuiteRankBoard> {
  const params = new URLSearchParams({
    offset: String(options.offset),
    limit: String(options.limit)
  });
  if (options.suiteId && options.suiteId !== "all") {
    params.set("suite_id", options.suiteId);
  }
  if (options.modelId && options.modelId !== "all") {
    params.set("model_id", options.modelId);
  }
  if (options.promptId && options.promptId !== "all") {
    params.set("prompt_id", options.promptId);
  }
  if (options.sortBy) {
    params.set("sort_by", options.sortBy);
  }
  if (options.sortOrder) {
    params.set("sort_order", options.sortOrder);
  }
  return fetchJson<SuiteRankBoard>(`/api/suite-rank-board?${params.toString()}`);
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

export function fetchTargetLabelResolution(
  options: TargetLabelResolutionParams = {}
): Promise<TargetLabelResolution> {
  const params = new URLSearchParams();
  if (options.benchmarkId?.trim()) {
    params.set("benchmark_id", options.benchmarkId.trim());
  }
  if (options.task?.trim()) {
    params.set("task", options.task.trim());
  }
  if (options.promptId?.trim()) {
    params.set("prompt_id", options.promptId.trim());
  }
  for (const label of options.targetLabels ?? []) {
    const value = label.trim();
    if (value) {
      params.append("target_label", value);
    }
  }
  const query = params.toString();
  return fetchJson<TargetLabelResolution>(`/api/target-labels${query ? `?${query}` : ""}`);
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

export function fetchBenchmark(benchmarkId: string): Promise<BenchmarkDetailResponse> {
  return fetchJson<BenchmarkDetailResponse>(`/api/benchmarks/${encodeURIComponent(benchmarkId)}`);
}

export function fetchSuites(): Promise<SuiteListResponse> {
  return fetchJson<SuiteListResponse>("/api/suites");
}

export function fetchSuite(suiteId: string): Promise<SuiteDetailResponse> {
  return fetchJson<SuiteDetailResponse>(`/api/suites/${encodeURIComponent(suiteId)}`);
}

export function fetchCampaigns(): Promise<CampaignListResponse> {
  return fetchJson<CampaignListResponse>("/api/campaigns");
}

export function fetchCampaign(campaignId: string): Promise<CampaignDetailResponse> {
  return fetchJson<CampaignDetailResponse>(`/api/campaigns/${encodeURIComponent(campaignId)}`);
}

export function createBenchmark(payload: CreateBenchmarkPayload): Promise<BenchmarkManifest> {
  return fetchJson<BenchmarkManifest>("/api/benchmarks", {
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

export function deleteJob(jobId: string): Promise<DeleteJobResult> {
  return fetchJson<DeleteJobResult>(`/api/jobs/${encodeURIComponent(jobId)}`, {
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
  if (filters.benchmarkSplit && filters.benchmarkSplit !== "all") {
    params.set("benchmark_split", filters.benchmarkSplit);
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

export function evaluateRun(
  runId: string
): Promise<{ run_id: string; report_path: string; summary_path: string }> {
  return fetchJson<{ run_id: string; report_path: string; summary_path: string }>(
    `/api/runs/${encodeURIComponent(runId)}/evaluate`,
    { method: "POST" }
  );
}

export function archiveRun(runId: string): Promise<ArchiveRunResult> {
  return fetchJson<ArchiveRunResult>(
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

export function updateRunNote(
  runId: string,
  note: string,
  expectedUpdatedAt?: string | null
): Promise<RunNote> {
  return fetchJson<RunNote>(`/api/runs/${encodeURIComponent(runId)}/note`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ note, expected_updated_at: expectedUpdatedAt ?? null })
  });
}

export function appendRunNote(
  runId: string,
  note: string,
  heading?: string,
  expectedUpdatedAt?: string | null
): Promise<RunNote> {
  const payload: { note: string; heading?: string; expected_updated_at?: string | null } = {
    note,
    heading
  };
  if (expectedUpdatedAt !== undefined) {
    payload.expected_updated_at = expectedUpdatedAt;
  }
  return fetchJson<RunNote>(`/api/runs/${encodeURIComponent(runId)}/note/append`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
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
  return fetchJson<SamplePage<RunSampleSummary>>(
    `/api/runs/${encodeURIComponent(runId)}/samples?${params.toString()}`
  );
}

export function fetchRunSampleDetail(runId: string, index: number): Promise<RunSampleDetail> {
  return fetchJson<RunSampleDetail>(`/api/runs/${encodeURIComponent(runId)}/samples/${index}`);
}

export function fetchCompositeSample(options: {
  sampleIndex: number;
  layerRuns?: Record<string, string>;
  layoutRunId?: string;
  arrowRunId?: string;
  shapeRunId?: string;
  iconRunId?: string;
}): Promise<CompositeSampleView> {
  const params = new URLSearchParams({ sample_index: String(options.sampleIndex) });
  Object.entries(options.layerRuns ?? {}).forEach(([layer, runId]) => {
    if (layer && runId) {
      params.append("layer_run", `${layer}:${runId}`);
    }
  });
  if (options.layoutRunId) {
    params.set("layout_run_id", options.layoutRunId);
  }
  if (options.arrowRunId) {
    params.set("arrow_run_id", options.arrowRunId);
  }
  if (options.shapeRunId) {
    params.set("shape_run_id", options.shapeRunId);
  }
  if (options.iconRunId) {
    params.set("icon_run_id", options.iconRunId);
  }
  return fetchJson<CompositeSampleView>(`/api/composite-samples?${params.toString()}`);
}

export function fetchBenchmarkSamples(
  benchmarkId: string,
  options: { offset: number; limit: number; label?: string; split?: string }
): Promise<SamplePage<BenchmarkSampleSummary>> {
  const params = new URLSearchParams({
    offset: String(options.offset),
    limit: String(options.limit)
  });
  if (options.label && options.label !== "all") {
    params.set("label", options.label);
  }
  if (options.split && options.split !== "all") {
    params.set("split", options.split);
  }
  return fetchJson<SamplePage<BenchmarkSampleSummary>>(
    `/api/benchmarks/${encodeURIComponent(benchmarkId)}/samples?${params.toString()}`
  );
}

export function fetchBenchmarkSampleDetail(
  benchmarkId: string,
  index: number,
  options: { split?: string } = {}
): Promise<BenchmarkSampleDetail> {
  const params = new URLSearchParams();
  if (options.split && options.split !== "all") {
    params.set("split", options.split);
  }
  const query = params.toString();
  return fetchJson<BenchmarkSampleDetail>(
    `/api/benchmarks/${encodeURIComponent(benchmarkId)}/samples/${index}${query ? `?${query}` : ""}`
  );
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
  params.set("list", "1");
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
  if (filters.benchmarkSplit && filters.benchmarkSplit !== "all") {
    params.set("benchmark_split", filters.benchmarkSplit);
  }
  if (filters.baselineRunId?.trim()) {
    params.set("baseline_run_id", filters.baselineRunId.trim());
  }
  if (filters.candidateRunId?.trim()) {
    params.set("candidate_run_id", filters.candidateRunId.trim());
  }
  if (filters.label && filters.label !== "all") {
    params.set("label", filters.label);
  }
  if (filters.query?.trim()) {
    params.set("query", filters.query.trim());
  }
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
  sampleIndex: number,
  indexes: { baselineIndex?: number | null; candidateIndex?: number | null } = {}
): Promise<ComparisonSampleDetail> {
  const params = new URLSearchParams({
    baseline_run_id: baselineRunId,
    candidate_run_id: candidateRunId,
    sample_index: String(sampleIndex)
  });
  if (indexes.baselineIndex !== undefined && indexes.baselineIndex !== null) {
    params.set("baseline_index", String(indexes.baselineIndex));
  }
  if (indexes.candidateIndex !== undefined && indexes.candidateIndex !== null) {
    params.set("candidate_index", String(indexes.candidateIndex));
  }
  return fetchJson<ComparisonSampleDetail>(`/api/comparisons/sample?${params.toString()}`);
}
