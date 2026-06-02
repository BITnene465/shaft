export type BenchmarkSummary = {
  benchmark_id: string;
  tasks: string[];
  labels: string[];
  layers: string[];
  split: string;
  sample_count: number;
  root: string;
  manifest_path: string;
  benchmark_type: string;
  official: boolean;
  created_at: string | null;
  source_manifest_path: string | null;
  split_manifests?: Record<string, string>;
  sample_counts?: Record<string, number>;
  metadata?: Record<string, unknown>;
};

export type BenchmarkManifest = BenchmarkSummary & {
  source_raw_root: string | null;
  source_manifest_path: string | null;
  split_manifests: Record<string, string>;
  sample_counts: Record<string, number>;
  metadata: Record<string, unknown>;
};

export type FacetBucket = Array<{ value: string; count: number }>;
export type FacetBuckets = Record<string, FacetBucket>;

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
  facets?: FacetBuckets;
};

export type BenchmarkDetailResponse = {
  benchmark: BenchmarkSummary;
};

export type RunSummary = {
  run_id: string;
  status: string;
  benchmark_id: string;
  benchmark_split: string;
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
  benchmark_type: string;
  benchmark_official: boolean;
  integrity_status: string;
  integrity_reason: string;
  suite_ids: string[];
  note: string;
  note_updated_at: string | null;
  note_max_length: number;
  f1_iou50: number | null;
  precision_iou50: number | null;
  recall_iou50: number | null;
  mean_iou: number | null;
};

export type RunListFilters = {
  offset?: number;
  limit?: number;
  task?: string;
  benchmarkId?: string;
  benchmarkSplit?: string;
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
  facets?: FacetBuckets;
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
  score_delta?: number | null;
  status: string;
  benchmark_id: string;
  benchmark_split: string;
  suite_id: string | null;
  benchmark_type: string;
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
  facets: FacetBuckets;
  entries: RankBoardEntry[];
};

export type SuiteTaskSplitSummary = {
  split: string;
  benchmark_id: string;
  manifest_path: string;
  sample_count: number;
  tasks: string[];
  layers: string[];
  target_labels: string[];
  run_count: number;
};

export type SuiteSummary = {
  suite_id: string;
  version: string;
  benchmark_id: string;
  benchmark_type: string;
  official: boolean;
  task_splits: SuiteTaskSplitSummary[];
  sample_universe: Record<string, unknown>;
  metric_profile: string;
  run_count: number;
  integrity_status: string;
  integrity_reason: string;
  validation_errors: string[];
  created_at: string | null;
  manifest_path: string | null;
  metadata: Record<string, unknown>;
};

export type SuiteListResponse = {
  total: number;
  suites: SuiteSummary[];
};

export type SuiteDetailResponse = {
  suite: SuiteSummary;
};

export type CampaignSummary = {
  campaign_id: string;
  suite_id: string;
  model_id: string;
  checkpoint: string;
  prompt_set: string[];
  pixel_budget: number | null;
  decoding_config: Record<string, unknown>;
  run_ids: string[];
  task_splits: string[];
  aggregate_report: Record<string, unknown>;
  created_at: string | null;
  manifest_path: string | null;
  metadata: Record<string, unknown>;
};

export type CampaignListResponse = {
  total: number;
  campaigns: CampaignSummary[];
};

export type CampaignDetailResponse = {
  campaign: CampaignSummary;
};

export type SuiteRankEntry = {
  rank: number;
  campaign_id: string;
  suite_id: string;
  model_id: string;
  checkpoint: string;
  prompt_set: string[];
  pixel_budget: number | null;
  task_splits: string[];
  aggregate_score: number | null;
  f1_iou50: number | null;
  run_count: number;
  created_at: string | null;
  per_split: Record<string, unknown>;
  score_delta: number | null;
};

export type SuiteRankBoard = {
  offset: number;
  limit: number;
  total: number;
  evaluated_count: number;
  filters: Record<string, string>;
  primary_metric: string;
  sort_by: string;
  sort_order: string;
  facets: FacetBuckets;
  entries: SuiteRankEntry[];
};

export type JobSummary = {
  job_id: string;
  run_id: string | null;
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
  facets?: FacetBuckets;
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
  facets?: FacetBuckets;
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

export type CompositeSampleLayer = RunSampleDetail & {
  layer: string;
  sample_index: number;
  status: "ready";
  available: true;
  missing_reason: "";
  image_key: string;
  benchmark_id: string;
  benchmark_split: string;
  task: string;
  target_labels: string[];
  diagnostic_summary: {
    matched_count: number;
    false_positive_count: number;
    false_negative_count: number;
    labels: string[];
  };
};

export type CompositeLayerStatus = {
  layer: string;
  run_id: string;
  status: "ready" | "image_missing" | "prediction_missing";
  available: boolean;
  missing_reason: string;
  image_key: string;
  sample_index: number | null;
  sample: RunSampleSummary | null;
  benchmark_id: string;
  benchmark_split: string;
  task: string;
  target_labels: string[];
  diagnostic_summary: CompositeSampleLayer["diagnostic_summary"];
};

export type CompositeSampleView = {
  kind: "composite_sample_view";
  sample_index: number;
  image_index: number;
  image_count: number;
  image_key: string;
  image_keys: string[];
  image: string;
  benchmark_id: string;
  layer_options: string[];
  view_modes: Array<"gt" | "prediction" | "diff">;
  layers: CompositeSampleLayer[];
  layer_statuses: CompositeLayerStatus[];
  diagnostics: {
    warnings: string[];
    per_layer: Record<string, CompositeSampleLayer["diagnostic_summary"]>;
  };
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
  filters: Record<string, string>;
  labels: string[];
  samples: T[];
};

export type DashboardState = {
  store_root: string;
  benchmark_count: number;
  suite_count: number;
  campaign_count: number;
  run_count: number;
  total_benchmark_samples: number;
  prediction_count: number;
  benchmarks: BenchmarkSummary[];
  runs: RunSummary[];
};

export type CreateBenchmarkPayload = {
  benchmark_id: string;
  source_root: string;
  source_manifest?: string;
  split: string;
  tasks: string[];
  layers?: string[];
  slices?: CreateBenchmarkSlicePayload[];
  default_slice?: string;
  flatten?: boolean;
  overwrite?: boolean;
  metadata?: Record<string, unknown>;
};

export type CreateBenchmarkSlicePayload = {
  split: string;
  source_manifest?: string;
  entries?: string[];
  tasks?: string[];
  layers?: string[];
  target_labels?: string[];
};

export type ComparisonRunMetrics = {
  precision_iou50: number;
  recall_iou50: number;
  mean_iou: number;
  keypoint_pair_count: number;
  mean_keypoint_distance: number;
  matched_count: number;
  gt_instance_count: number;
  pred_instance_count: number;
};

export type ComparisonDelta = {
  precision_iou50: number;
  recall_iou50: number;
  mean_iou: number;
  mean_keypoint_distance: number;
  matched_count: number;
  keypoint_pair_count: number;
  false_positive_count: number;
  false_negative_count: number;
};

export type ComparisonOverview = {
  improved_samples: number;
  regressed_samples: number;
  changed_samples: number;
  unchanged_samples: number;
  missing_in_baseline: number;
  missing_in_candidate: number;
  improved_labels: number;
  regressed_labels: number;
};

export type ComparisonSampleMetrics = {
  matched_count: number;
  false_positive_count: number;
  false_negative_count: number;
  mean_iou: number;
  keypoint_pair_count: number;
  mean_keypoint_distance: number;
};

export type ComparisonSummary = {
  comparison_id: string;
  baseline_run_id: string;
  candidate_run_id: string;
  benchmark_id?: string;
  benchmark_split?: string;
  task: string;
  metric_profile?: string;
  target_labels?: string[];
  target_labels_source?: string | null;
  warnings?: string[];
  sample_count: number;
  created_at: string | null;
  path: string;
  delta: ComparisonDelta;
  summary: ComparisonOverview;
};

export type ComparisonListFilters = {
  task?: string;
  benchmarkId?: string;
  benchmarkSplit?: string;
  baselineRunId?: string;
  candidateRunId?: string;
  label?: string;
  query?: string;
  offset?: number;
  limit?: number;
};

export type ComparisonSampleDetail = {
  baseline_run_id: string;
  candidate_run_id: string;
  sample_index: number;
  baseline_index?: number;
  candidate_index?: number;
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
  labels?: Record<string, ComparisonSampleMetrics>;
  delta_score: number;
  delta: ComparisonSampleMetrics;
  baseline?: ComparisonSampleMetrics | null;
  candidate?: ComparisonSampleMetrics | null;
};

export type ComparisonLabelDelta = {
  label: string;
  delta_score: number;
  baseline: Record<string, number>;
  candidate: Record<string, number>;
  delta: ComparisonDelta;
};

export type ComparisonReport = {
  comparison_id?: string;
  baseline_run_id: string;
  candidate_run_id: string;
  benchmark_id?: string;
  benchmark_split?: string;
  task: string;
  metric_profile?: string;
  target_labels?: string[];
  target_labels_source?: string | null;
  warnings?: string[];
  sample_count: number;
  created_at?: string | null;
  baseline: ComparisonRunMetrics;
  candidate: ComparisonRunMetrics;
  delta: ComparisonDelta;
  summary: ComparisonOverview;
  labels?: ComparisonLabelDelta[];
  samples?: ComparisonSample[];
  top_improvements: ComparisonSample[];
  top_regressions: ComparisonSample[];
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
  split?: string;
  target_labels?: string[];
  strict?: boolean;
  overwrite?: boolean;
  evaluate?: boolean;
};

export type ImportPredictionResult = {
  run_id: string;
  run_manifest_path: string;
  report_path: string | null;
  summary_path: string | null;
  imported_predictions: number;
  missing_predictions: string[];
  missing_prediction_count: number;
};

export type DeleteResult = {
  deleted: boolean;
  trash_path: string | null;
};

export type ArchiveRunResult = {
  run_id: string;
  status: string;
  manifest_path: string;
};

export type DeleteJobResult = DeleteResult & {
  job_id: string;
  job: JobSummary;
  trash_path: string;
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

export type TargetLabelResolution = {
  task: string;
  benchmark_id: string;
  prompt_id: string;
  target_labels: string[];
  target_labels_source: string;
  candidate_labels: string[];
  benchmark_labels: string[];
  prompt_target_labels: string[];
  explicit_target_labels: string[];
  label_subtasks_supported: boolean;
  valid: boolean;
  errors: string[];
  warnings: string[];
};

export type TargetLabelResolutionParams = {
  benchmarkId?: string;
  task?: string;
  promptId?: string;
  targetLabels?: string[];
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

