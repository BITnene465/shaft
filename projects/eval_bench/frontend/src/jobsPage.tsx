import { useEffect, useMemo, useState } from "react";
import type { FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { Trash2, X } from "lucide-react";

import type { BenchmarkSummary, JobLog, JobSummary, PromptTemplate, RunSummary, SchedulerStatus } from "./api";
import {
  cancelJob,
  createJob,
  deleteJob,
  fetchJobLogs,
  fetchJobs,
  fetchJobTemplates,
  fetchPromptTemplates,
  fetchSchedulerStatus,
  preflightJob,
  upsertPromptTemplate
} from "./api";
import { CompactSelectControl } from "./controlPrimitives";
import { useDashboardState } from "./dashboardState";
import { basename, formatDate, formatMetric, jobTarget, stringValue, unique } from "./formatters";
import { AdvancedFilterBar } from "./filterControls";
import { AppIcon } from "./iconLibrary";
import {
  applyBenchmarkDefault,
  applyPromptTemplateToManifest,
  formatManifest,
  manifestBenchmarkId,
  manifestEvalTask,
  manifestTargetLabels,
  promptTemplateFromManifest,
  targetLabelsFromPrompt,
  updateManifestTargetLabels
} from "./manifestTools";
import {
  canCancelJob,
  canDeleteJob,
  jobProgress,
  progressPhaseText
} from "./statusModel";
import { PagerControl, clampListPageOffset } from "./samplePager";
import {
  ActionButton,
  Badge,
  CommandButton,
  DangerConfirmDialog,
  IconActionButton,
  OptionChipButton,
  PanelTitle,
  WorkspaceDialog
} from "./ui";
import { ResizableSplit } from "./workspaceLayout";

const JOB_PAGE_SIZE = 80;

export function JobsPage() {
  const { data } = useDashboardState();
  const [createOpen, setCreateOpen] = useState(false);
  const recentRuns = data?.runs.slice(0, 12) ?? [];
  return (
    <section className="page-stack density-page">
      <div className="page-command-row">
        <div>
          <h2>评测中心</h2>
          <span>队列、runtime log 和最近结果</span>
        </div>
        <CommandButton icon={<AppIcon name="createEval" size={17} />} onClick={() => setCreateOpen(true)}>
          新建评测
        </CommandButton>
      </div>
      <div className="job-activity-grid">
        <div className="workspace-card fill">
          <PanelTitle title="任务队列" meta="执行、失败排障和 runtime log" />
          <JobQueuePanel />
        </div>
        <div className="workspace-card fill">
          <PanelTitle title="最近结果" meta="完整结果在结果库页面" />
          <RecentRunList runs={recentRuns} />
        </div>
      </div>
      <WorkspaceDialog
        open={createOpen}
        title="新建评测任务"
        meta="模板 manifest + 后端预检查"
        wide
        onClose={() => setCreateOpen(false)}
      >
        <JobCreatePanel benchmarks={data?.benchmarks ?? []} bare />
      </WorkspaceDialog>
    </section>
  );
}

function RecentRunList({ runs }: { runs: RunSummary[] }) {
  if (runs.length === 0) {
    return <div className="empty-panel">还没有评测结果。</div>;
  }
  return (
    <div className="recent-run-list">
      {runs.map((run) => (
        <Link
          className="recent-run-card"
          key={run.run_id}
          to="/runs/$runId"
          params={{ runId: run.run_id }}
        >
          <span className="recent-run-head">
            <strong title={run.run_id}>{run.run_id}</strong>
            <Badge value={run.status} domain="run" />
          </span>
          <span className="recent-run-meta" title={run.model_id}>
            {run.model_id || "unknown model"}
          </span>
          <span className="recent-run-metrics">
            <em>P {formatMetric(run.precision_iou50)}</em>
            <em>R {formatMetric(run.recall_iou50)}</em>
            <em>IoU {formatMetric(run.mean_iou)}</em>
          </span>
        </Link>
      ))}
    </div>
  );
}

export function JobQueuePanel({ compact = false }: { compact?: boolean }) {
  const queryClient = useQueryClient();
  const [selectedJobId, setSelectedJobId] = useState<string>("");
  const [deleteJobTarget, setDeleteJobTarget] = useState<JobSummary | null>(null);
  const [searchText, setSearchText] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [kindFilter, setKindFilter] = useState("all");
  const [pageOffset, setPageOffset] = useState(0);
  const jobFilters = useMemo(
    () => ({
      offset: compact ? 0 : pageOffset,
      limit: compact ? 12 : JOB_PAGE_SIZE,
      status: !compact && statusFilter !== "all" ? statusFilter : undefined,
      kind: !compact && kindFilter !== "all" ? kindFilter : undefined,
      query: !compact && searchText.trim() ? searchText.trim() : undefined
    }),
    [compact, kindFilter, pageOffset, searchText, statusFilter]
  );
  const { data, isLoading, error } = useQuery({
    queryKey: ["jobs", jobFilters],
    queryFn: () => fetchJobs(jobFilters),
    refetchInterval: 2_000
  });
  const allJobsQuery = useQuery({
    queryKey: ["jobs", "facets"],
    queryFn: () => fetchJobs({ limit: 500 }),
    refetchInterval: 2_000,
    enabled: !compact
  });
  const schedulerQuery = useQuery({
    queryKey: ["scheduler-status"],
    queryFn: fetchSchedulerStatus,
    refetchInterval: 2_000
  });
  const jobsForSummary = allJobsQuery.data?.jobs ?? data?.jobs ?? [];
  const runningJobs = jobsForSummary.filter((job) => job.status === "running");
  const selectedJob = data?.jobs.find((job) => job.job_id === selectedJobId) ?? null;
  const statuses = unique([
    "queued",
    "running",
    "succeeded",
    "failed",
    "cancelled",
    ...jobsForSummary.map((job) => job.status).filter(Boolean)
  ]);
  const kinds = unique(["eval", "preannotate", ...jobsForSummary.map((job) => job.kind).filter(Boolean)]);
  const filteredJobs = data?.jobs ?? [];
  const totalJobs = data?.total ?? filteredJobs.length;
  const selectedRuntimeLogPath =
    selectedJob && typeof selectedJob.metadata.runtime_log_path === "string"
      ? selectedJob.metadata.runtime_log_path
      : "";
  const jobLogsQuery = useQuery({
    queryKey: ["job-logs", selectedJob?.job_id ?? ""],
    queryFn: () => fetchJobLogs(selectedJob?.job_id ?? "", 0),
    enabled: Boolean(selectedJob?.job_id && selectedRuntimeLogPath),
    refetchInterval: selectedJob?.status === "running" ? 3_000 : false
  });
  const cancelMutation = useMutation({
    mutationFn: cancelJob,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["jobs"] });
      void queryClient.invalidateQueries({ queryKey: ["scheduler-status"] });
    }
  });
  const deleteMutation = useMutation({
    mutationFn: deleteJob,
    onSuccess: (_result, jobId) => {
      setDeleteJobTarget(null);
      setSelectedJobId((current) => (current === jobId ? "" : current));
      void queryClient.invalidateQueries({ queryKey: ["jobs"] });
    }
  });
  useEffect(() => {
    if (!compact) {
      setPageOffset(0);
    }
  }, [compact, searchText, statusFilter, kindFilter]);
  useEffect(() => {
    if (compact) {
      return;
    }
    const nextOffset = clampListPageOffset(pageOffset, totalJobs, JOB_PAGE_SIZE);
    if (nextOffset !== pageOffset) {
      setPageOffset(nextOffset);
    }
  }, [compact, pageOffset, totalJobs]);
  useEffect(() => {
    if (selectedJobId && data && !data.jobs.some((job) => job.job_id === selectedJobId)) {
      setSelectedJobId("");
    }
  }, [data, selectedJobId]);
  if (isLoading) {
    return <div className="empty-panel">正在加载队列状态</div>;
  }
  if (error || !data) {
    return <div className="empty-panel danger-text">队列状态加载失败</div>;
  }
  return (
    <div className={compact ? "queue-stack compact" : "queue-stack"}>
      <SchedulerStrip
        jobs={jobsForSummary}
        scheduler={schedulerQuery.data ?? { enabled: false }}
      />
      {!compact ? (
        <AdvancedFilterBar
          title="任务高级检索"
          meta={`${filteredJobs.length.toLocaleString()} / ${totalJobs.toLocaleString()} 条 job`}
          controls={[
            {
              type: "search",
              id: "job-query",
              label: "全文检索",
              value: searchText,
              onChange: setSearchText,
              placeholder: "搜索 job、模型、benchmark、错误、日志"
            },
            {
              type: "select",
              id: "job-status",
              label: "状态",
              value: statusFilter,
              values: ["all", ...statuses],
              labels: { all: "全部" },
              onChange: setStatusFilter
            },
            {
              type: "select",
              id: "job-kind",
              label: "类型",
              value: kindFilter,
              values: ["all", ...kinds],
              labels: { all: "全部" },
              onChange: setKindFilter
            }
          ]}
        />
      ) : null}
      {totalJobs === 0 ? (
        <div className="empty-panel">当前没有任务。</div>
      ) : filteredJobs.length === 0 ? (
        <div className="empty-panel">没有符合高级检索条件的任务。</div>
      ) : (
        <div className={compact ? "table-shell compact" : "table-shell"}>
          <table>
            <thead>
              <tr>
                <th>任务</th>
                <th>类型</th>
                <th>状态</th>
                <th>目标</th>
                <th>创建时间</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {filteredJobs.map((job) => (
                <tr
                  key={job.job_id}
                  className={job.job_id === selectedJob?.job_id ? "selectable-row selected" : "selectable-row"}
                  onClick={() => setSelectedJobId(job.job_id)}
                >
                  <td>{job.job_id}</td>
                  <td>{job.kind}</td>
                  <td>
                    <Badge value={job.status} domain="job" />
                    <JobProgressInline job={job} />
                  </td>
                  <td>
                    <div className="job-target-cell">
                      <span>{jobTarget(job.payload)}</span>
                      {job.error ? <em title={job.error}>{job.error}</em> : null}
                      {typeof job.metadata.runtime_log_path === "string" ? (
                        <small title={job.metadata.runtime_log_path}>
                          runtime log: {basename(job.metadata.runtime_log_path)}
                        </small>
                      ) : null}
                    </div>
                  </td>
                  <td>{formatDate(job.created_at)}</td>
                  <td>
                    <div className="row-actions">
                      <IconActionButton
                        icon={<X size={14} />}
                        disabled={!canCancelJob(job) || cancelMutation.isPending}
                        title={job.status === "running" ? "终止运行中评测" : "取消排队任务"}
                        onClick={(event) => {
                          event.stopPropagation();
                          cancelMutation.mutate(job.job_id);
                        }}
                      />
                      <IconActionButton
                        icon={<Trash2 size={14} />}
                        danger
                        disabled={!canDeleteJob(job) || deleteMutation.isPending}
                        title="删除任务记录"
                        onClick={(event) => {
                          event.stopPropagation();
                          setDeleteJobTarget(job);
                        }}
                      />
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {!compact ? (
        <PagerControl
          className="rank-board-pager job-list-pager"
          offset={data.offset ?? pageOffset}
          limit={data.limit ?? JOB_PAGE_SIZE}
          total={totalJobs}
          onPageChange={setPageOffset}
        />
      ) : null}
      {selectedJob ? <JobDetailPanel job={selectedJob} logs={jobLogsQuery.data ?? null} /> : null}
      <DangerConfirmDialog
        open={Boolean(deleteJobTarget)}
        title="删除任务记录"
        subject={deleteJobTarget?.job_id ?? ""}
        description="任务记录会移入回收站，队列页、运行日志入口和任务详情面板会同步移除。"
        confirmLabel="删除记录"
        pending={deleteMutation.isPending}
        onCancel={() => setDeleteJobTarget(null)}
        onConfirm={() => {
          if (deleteJobTarget) {
            deleteMutation.mutate(deleteJobTarget.job_id);
          }
        }}
      />
    </div>
  );
}

function SchedulerStrip({
  jobs,
  scheduler
}: {
  jobs: JobSummary[];
  scheduler: SchedulerStatus;
}) {
  const queued = jobs.filter((job) => job.status === "queued").length;
  const running = jobs.filter((job) => job.status === "running").length;
  const failed = jobs.filter((job) => job.status === "failed").length;
  const reservedDevices = scheduler.reserved_cuda_devices ?? [];
  const reservedPorts = scheduler.reserved_runtime_ports ?? [];
  return (
    <div className="scheduler-strip">
      <div>
        <span className={scheduler.enabled ? "status-dot live" : "status-dot"} />
        <strong>{scheduler.enabled ? "自动调度运行中" : "自动调度未启用"}</strong>
      </div>
      <span>运行 {running}</span>
      <span>排队 {queued}</span>
      {failed > 0 ? <span className="danger-text">失败 {failed}</span> : null}
      <span>并发上限 {scheduler.max_concurrent_jobs ?? "-"}</span>
      {reservedDevices.length > 0 ? <span>占用 CUDA {reservedDevices.join(",")}</span> : null}
      {reservedPorts.length > 0 ? <span>占用端口 {reservedPorts.join(",")}</span> : null}
    </div>
  );
}

function JobDetailPanel({ job, logs }: { job: JobSummary; logs: JobLog | null }) {
  const progress = jobProgress(job);
  const lines = logs?.lines ?? [];
  const linkedRunId = stringValue(job.metadata.run_id);
  return (
    <div className="job-detail-panel">
      <div className="job-monitor-header">
        <div>
          <div className="eyebrow">任务详情</div>
          <strong>{job.job_id}</strong>
        </div>
        <div className="job-monitor-actions">
          {linkedRunId ? (
            <Link className="mini-link" to="/runs/$runId" params={{ runId: linkedRunId }}>
              打开结果
            </Link>
          ) : null}
          <Badge value={job.status} domain="job" />
        </div>
      </div>
      <div className="job-progress-row">
        <div className="job-progress-track" aria-label="任务进度">
          <span style={{ width: `${progress.percent ?? 8}%` }} />
        </div>
        <span>{progress.text}</span>
      </div>
      <div className="job-monitor-meta">
        <span>{progressPhaseText(progress.phase)}</span>
        {progress.message ? <span>{progress.message}</span> : null}
        {progress.currentSample ? <span title={progress.currentSample}>{progress.currentSample}</span> : null}
      </div>
      <div className="job-detail-grid">
        <span>目标</span>
        <strong>{jobTarget(job.payload)}</strong>
        <span>创建</span>
        <strong>{formatDate(job.created_at)}</strong>
        <span>更新</span>
        <strong>{formatDate(job.updated_at)}</strong>
        <span>日志</span>
        <strong>
          {typeof job.metadata.runtime_log_path === "string"
            ? job.metadata.runtime_log_path
            : "runtime log 尚未创建"}
        </strong>
      </div>
      {lines.length > 0 ? (
        <pre className="job-log-tail">{lines.join("")}</pre>
      ) : (
        <div className="job-log-empty">
          {logs?.log_path ? "runtime log 还没有新内容。" : "等待 runtime log。"}
        </div>
      )}
    </div>
  );
}

function JobProgressInline({ job }: { job: JobSummary }) {
  if (job.status !== "running" && job.status !== "failed" && job.status !== "succeeded") {
    return null;
  }
  const progress = jobProgress(job);
  return (
    <div className="job-progress-inline">
      <div className="job-progress-mini">
        <span style={{ width: `${progress.percent ?? (job.status === "succeeded" ? 100 : 0)}%` }} />
      </div>
      <small>{progress.text}</small>
    </div>
  );
}

export function JobCreatePanel({ benchmarks, bare }: { benchmarks: BenchmarkSummary[]; bare?: boolean }) {
  const queryClient = useQueryClient();
  const templatesQuery = useQuery({ queryKey: ["job-templates"], queryFn: fetchJobTemplates });
  const promptTemplatesQuery = useQuery({
    queryKey: ["prompt-templates"],
    queryFn: fetchPromptTemplates
  });
  const templates = templatesQuery.data?.templates ?? {};
  const promptTemplates = promptTemplatesQuery.data?.templates ?? [];
  const templateIds = Object.keys(templates);
  const promptIds = promptTemplates.map((template) => template.prompt_id);
  const [templateId, setTemplateId] = useState("eval_job");
  const [promptId, setPromptId] = useState("grounding_arrow.latest");
  const [manifestText, setManifestText] = useState("");
  const [parseError, setParseError] = useState<string | null>(null);
  const mutation = useMutation({
    mutationFn: createJob,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["jobs"] });
    }
  });
  const promptMutation = useMutation({
    mutationFn: upsertPromptTemplate,
    onSuccess: (record) => {
      setPromptId(record.prompt_id);
      void queryClient.invalidateQueries({ queryKey: ["prompt-templates"] });
    }
  });
  const preflightMutation = useMutation({ mutationFn: preflightJob });
  const selectedTemplate = templates[templateId] ?? templates[templateIds[0] ?? ""];
  const selectedPrompt =
    promptTemplates.find((template) => template.prompt_id === promptId) ?? promptTemplates[0];
  const manifestDraft = useMemo(() => parseManifestDraft(manifestText), [manifestText]);
  const manifestTaskValue = manifestEvalTask(manifestDraft);
  const manifestBenchmarkValue = manifestBenchmarkId(manifestDraft);
  const selectedBenchmark = benchmarks.find((benchmark) => benchmark.benchmark_id === manifestBenchmarkValue);
  const selectedTargetLabels = manifestTargetLabels(manifestDraft);
  const labelOptions = unique([
    ...(selectedBenchmark?.labels ?? []),
    ...targetLabelsFromPrompt(selectedPrompt),
    ...selectedTargetLabels
  ]);

  useEffect(() => {
    if (!manifestText && selectedTemplate?.manifest) {
      setManifestText(formatManifest(applyBenchmarkDefault(selectedTemplate.manifest, benchmarks)));
    }
  }, [benchmarks, manifestText, selectedTemplate]);

  useEffect(() => {
    if (promptIds.length > 0 && !promptIds.includes(promptId)) {
      setPromptId(promptIds[0]);
    }
  }, [promptId, promptIds.join("|")]);

  function loadTemplate(nextTemplateId = templateId) {
    const template = templates[nextTemplateId];
    if (!template) {
      return;
    }
    setTemplateId(nextTemplateId);
    setManifestText(formatManifest(applyBenchmarkDefault(template.manifest, benchmarks)));
    setParseError(null);
    preflightMutation.reset();
  }

  function applySelectedPrompt(nextPromptId = promptId) {
    const promptTemplate =
      promptTemplates.find((template) => template.prompt_id === nextPromptId) ?? selectedPrompt;
    if (!promptTemplate) {
      return;
    }
    const manifest = parseManifest() ?? applyBenchmarkDefault(selectedTemplate?.manifest ?? {}, benchmarks);
    setPromptId(promptTemplate.prompt_id);
    setManifestText(
      formatManifest(applyBenchmarkDefault(applyPromptTemplateToManifest(manifest, promptTemplate), benchmarks))
    );
    setParseError(null);
    preflightMutation.reset();
  }

  function savePromptFromManifest() {
    const manifest = parseManifest();
    if (!manifest) {
      return;
    }
    const draft = promptTemplateFromManifest(manifest, selectedPrompt);
    promptMutation.mutate(draft);
  }

  function parseManifest(): Record<string, unknown> | null {
    try {
      const parsed = JSON.parse(manifestText) as unknown;
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        setParseError("Manifest 必须是 JSON object。");
        return null;
      }
      setParseError(null);
      return parsed as Record<string, unknown>;
    } catch (error) {
      setParseError(error instanceof Error ? error.message : String(error));
      return null;
    }
  }

  function validateManifest() {
    const manifest = parseManifest();
    if (!manifest) {
      return;
    }
    preflightMutation.mutate({ manifest });
  }

  function updateTargetLabels(nextLabels: string[]) {
    const manifest = parseManifest();
    if (!manifest) {
      return;
    }
    setManifestText(formatManifest(updateManifestTargetLabels(manifest, nextLabels)));
    setParseError(null);
    preflightMutation.reset();
  }

  function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const manifest = parseManifest();
    if (!manifest) {
      return;
    }
    mutation.mutate({ manifest });
  }

  return (
    <div className={bare ? "manifest-card bare" : "workspace-card manifest-card"}>
      {bare ? null : <PanelTitle title="新建评测任务" meta="模板 manifest + 后端预检查" />}
      <form className="manifest-job-form" onSubmit={submit}>
        <div className="manifest-toolbar">
          <CompactSelectControl
            label="模板"
            value={templateId}
            onChange={loadTemplate}
            disabled={templatesQuery.isLoading}
            options={
              templateIds.length === 0
                ? [{ value: "eval_job", label: "加载中" }]
                : templateIds.map((id) => ({ value: id, label: templates[id]?.label ?? id }))
            }
          />
          <CompactSelectControl
            label="Prompt"
            value={selectedPrompt?.prompt_id ?? promptId}
            onChange={applySelectedPrompt}
            disabled={promptTemplatesQuery.isLoading || promptTemplates.length === 0}
            options={
              promptTemplates.length === 0
                ? [{ value: promptId, label: "加载中" }]
                : promptTemplates.map((template) => ({
                    value: template.prompt_id,
                    label: template.label || template.prompt_id
                  }))
            }
          />
          <ActionButton
            variant="secondary"
            icon={<AppIcon name="restoreTemplate" size={16} />}
            onClick={() => loadTemplate()}
          >
            恢复模板
          </ActionButton>
          <ActionButton
            variant="secondary"
            icon={<AppIcon name="applyPrompt" size={16} />}
            onClick={() => applySelectedPrompt()}
            disabled={!selectedPrompt}
          >
            应用 Prompt
          </ActionButton>
          <ActionButton
            variant="secondary"
            icon={<AppIcon name="preflightValidate" size={16} />}
            onClick={validateManifest}
            disabled={preflightMutation.isPending}
          >
            {preflightMutation.isPending ? "检查中" : "预检查"}
          </ActionButton>
          <ActionButton
            variant="primary"
            type="submit"
            icon={<AppIcon name="enqueueJob" size={16} />}
            disabled={mutation.isPending}
          >
            {mutation.isPending ? "加入中" : "加入队列"}
          </ActionButton>
        </div>
        <LabelSubtaskPanel
          task={manifestTaskValue}
          benchmarkId={manifestBenchmarkValue}
          labelOptions={labelOptions}
          selectedLabels={selectedTargetLabels}
          onChange={updateTargetLabels}
        />
        <ResizableSplit
          className="manifest-split"
          storageKey="eval_bench_manifest_result_width"
          fixedPane="second"
          defaultSize={360}
          minSize={240}
          maxSize={820}
          first={
            <div className="manifest-editor-pane">
              {selectedTemplate ? (
                <p className="manifest-template-note">{selectedTemplate.description}</p>
              ) : null}
              <label className="manifest-editor-field">
                <span>可编辑任务 Manifest</span>
                <textarea
                  spellCheck={false}
                  value={manifestText}
                  onChange={(event) => {
                    setManifestText(event.target.value);
                    setParseError(null);
                    preflightMutation.reset();
                  }}
                />
              </label>
            </div>
          }
          second={
            <div className="manifest-result-pane">
              <PanelTitle title="预检查" meta="提交前的参数与运行时校验" />
              {selectedPrompt ? (
                <PromptTemplatePanel
                  prompt={selectedPrompt}
                  onSaveFromManifest={savePromptFromManifest}
                  saving={promptMutation.isPending}
                  saveError={promptMutation.isError}
                />
              ) : null}
              {parseError ? <div className="form-error">JSON 解析错误：{parseError}</div> : null}
              {preflightMutation.data ? <PreflightPanel result={preflightMutation.data} /> : null}
              {preflightMutation.isError ? (
                <div className="form-error">预检查请求失败。</div>
              ) : null}
              {mutation.isError ? <div className="form-error">任务入队失败。</div> : null}
              {!parseError && !preflightMutation.data && !preflightMutation.isError && !mutation.isError ? (
                <div className="manifest-placeholder">
                  编辑 manifest 后执行预检查；通过后再加入队列。
                </div>
              ) : null}
            </div>
          }
        />
      </form>
    </div>
  );
}

function parseManifestDraft(value: string): Record<string, unknown> | null {
  try {
    const parsed = JSON.parse(value) as unknown;
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : null;
  } catch {
    return null;
  }
}

function LabelSubtaskPanel({
  task,
  benchmarkId,
  labelOptions,
  selectedLabels,
  onChange
}: {
  task: string;
  benchmarkId: string;
  labelOptions: string[];
  selectedLabels: string[];
  onChange: (labels: string[]) => void;
}) {
  const [draftLabel, setDraftLabel] = useState("");
  if (task !== "detection") {
    return null;
  }
  const selectedSet = new Set(selectedLabels);

  function toggleLabel(label: string) {
    if (selectedSet.has(label)) {
      onChange(selectedLabels.filter((item) => item !== label));
      return;
    }
    onChange(unique([...selectedLabels, label]));
  }

  function addDraftLabel(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const value = draftLabel.trim();
    if (!value) {
      return;
    }
    onChange(unique([...selectedLabels, value]));
    setDraftLabel("");
  }

  return (
    <div className="label-subtask-panel">
      <div className="label-subtask-head">
        <div>
          <strong>Detection 子任务</strong>
          <span>{benchmarkId || "未选择 benchmark"}</span>
        </div>
        <div className="label-subtask-actions">
          <ActionButton variant="mini" onClick={() => onChange(labelOptions)}>
            全部候选
          </ActionButton>
          <ActionButton variant="mini" onClick={() => onChange([])}>
            默认策略
          </ActionButton>
        </div>
      </div>
      <div className="label-subtask-chips">
        {labelOptions.map((label) => (
          <OptionChipButton
            key={label}
            active={selectedSet.has(label)}
            onClick={() => toggleLabel(label)}
          >
            {label}
          </OptionChipButton>
        ))}
        {labelOptions.length === 0 ? <span className="label-subtask-empty">暂无 label 索引</span> : null}
      </div>
      <form className="label-subtask-add" onSubmit={addDraftLabel}>
        <input
          value={draftLabel}
          onChange={(event) => setDraftLabel(event.target.value)}
          placeholder="自定义 label"
        />
        <ActionButton variant="mini" type="submit">
          添加
        </ActionButton>
      </form>
    </div>
  );
}

function PreflightPanel({ result }: { result: { ok: boolean; errors: string[]; warnings: string[]; runtime_command?: string[] | null } }) {
  return (
    <div className={result.ok ? "preflight-panel ok" : "preflight-panel failed"}>
      <div className="preflight-heading">
        <strong>{result.ok ? "预检查通过" : "预检查失败"}</strong>
        <span>{result.errors.length} 个错误 / {result.warnings.length} 个警告</span>
      </div>
      {result.errors.length > 0 ? (
        <ul>
          {result.errors.map((error) => (
            <li key={error}>{error}</li>
          ))}
        </ul>
      ) : null}
      {result.warnings.length > 0 ? (
        <ul>
          {result.warnings.map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      ) : null}
      {result.runtime_command && result.runtime_command.length > 0 ? (
        <pre>{result.runtime_command.join(" ")}</pre>
      ) : null}
    </div>
  );
}

function PromptTemplatePanel({
  prompt,
  onSaveFromManifest,
  saving,
  saveError
}: {
  prompt: PromptTemplate;
  onSaveFromManifest: () => void;
  saving: boolean;
  saveError: boolean;
}) {
  const targetLabels = targetLabelsFromPrompt(prompt);
  return (
    <details className="prompt-template-panel" open>
      <summary>
        <span>{prompt.label || prompt.prompt_id}</span>
        <Badge value={prompt.task} />
      </summary>
      <div className="prompt-template-meta">
        <span>{prompt.prompt_id}</span>
        <span>{prompt.parser ?? "parser 未设置"}</span>
        <span>{prompt.metric_profile ?? "metric 未设置"}</span>
        <span>目标 {targetLabels.length ? targetLabels.join(" / ") : "全部 label"}</span>
      </div>
      <div className="prompt-template-text">
        <strong>System</strong>
        <p>{prompt.system_prompt || "-"}</p>
        <strong>User</strong>
        <p>{prompt.user_prompt || "-"}</p>
      </div>
      <ActionButton
        variant="secondary"
        compact
        icon={<AppIcon name="applyPrompt" size={16} />}
        onClick={onSaveFromManifest}
        disabled={saving}
      >
        {saving ? "保存中" : "将当前 Manifest 的 Prompt 保存为模板"}
      </ActionButton>
      {saveError ? <div className="form-error">Prompt 模板保存失败。</div> : null}
    </details>
  );
}
