import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import * as Tabs from "@radix-ui/react-tabs";
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
import { useDashboardState } from "./dashboardState";
import { basename, formatDate, formatMetric, jobTarget, stringValue } from "./formatters";
import {
  applyBenchmarkDefault,
  applyPromptTemplateToManifest,
  formatManifest,
  promptTemplateFromManifest,
  targetLabelsFromPrompt
} from "./manifestTools";
import { RunTable } from "./runTables";
import {
  canCancelJob,
  canDeleteJob,
  jobProgress,
  progressPhaseText
} from "./statusModel";
import { Badge, PanelTitle, WorkspaceTabs } from "./ui";
import { ResizableSplit } from "./workspaceLayout";

export function JobsPage() {
  const { data } = useDashboardState();
  const recentRuns = data?.runs.slice(0, 12) ?? [];
  return (
    <section className="page-stack">
      <WorkspaceTabs defaultValue="activity" label="评测中心">
        <Tabs.List className="workspace-tab-list">
          <Tabs.Trigger value="activity">活动流</Tabs.Trigger>
          <Tabs.Trigger value="new">新建评测</Tabs.Trigger>
          <Tabs.Trigger value="runs">结果库</Tabs.Trigger>
        </Tabs.List>
        <Tabs.Content value="activity" className="workspace-tab-panel">
          <div className="job-activity-grid">
            <div className="workspace-card fill">
              <PanelTitle title="任务队列" meta="创建、执行、失败排障和 runtime log" />
              <JobQueuePanel />
            </div>
            <div className="workspace-card fill">
              <PanelTitle title="最近结果" meta="任务完成后会沉淀为可复查 run" />
              <RecentRunList runs={recentRuns} />
            </div>
          </div>
        </Tabs.Content>
        <Tabs.Content value="new" className="workspace-tab-panel">
          <JobCreatePanel benchmarks={data?.benchmarks ?? []} />
        </Tabs.Content>
        <Tabs.Content value="runs" className="workspace-tab-panel">
          <div className="workspace-card fill">
            <PanelTitle title="结果库" meta={`${(data?.runs.length ?? 0).toLocaleString()} 条记录`} />
            <RunTable runs={data?.runs ?? []} />
          </div>
        </Tabs.Content>
      </WorkspaceTabs>
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
  const { data, isLoading, error } = useQuery({
    queryKey: ["jobs"],
    queryFn: fetchJobs,
    refetchInterval: 2_000
  });
  const schedulerQuery = useQuery({
    queryKey: ["scheduler-status"],
    queryFn: fetchSchedulerStatus,
    refetchInterval: 2_000
  });
  const runningJobs = data?.jobs.filter((job) => job.status === "running") ?? [];
  const selectedJob = data?.jobs.find((job) => job.job_id === selectedJobId) ?? null;
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
    }
  });
  const deleteMutation = useMutation({
    mutationFn: deleteJob,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["jobs"] });
    }
  });
  if (isLoading) {
    return <div className="empty-panel">正在加载队列状态</div>;
  }
  if (error || !data) {
    return <div className="empty-panel danger-text">队列状态加载失败</div>;
  }
  return (
    <div className={compact ? "queue-stack compact" : "queue-stack"}>
      <SchedulerStrip
        jobs={data.jobs}
        scheduler={schedulerQuery.data ?? { enabled: false }}
      />
      {data.jobs.length === 0 ? (
        <div className="empty-panel">当前没有任务。</div>
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
              {data.jobs.map((job) => (
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
                      <button
                        className="icon-button dense"
                        type="button"
                        disabled={!canCancelJob(job) || cancelMutation.isPending}
                        title="取消排队任务"
                        onClick={(event) => {
                          event.stopPropagation();
                          cancelMutation.mutate(job.job_id);
                        }}
                      >
                        <X size={14} />
                      </button>
                      <button
                        className="icon-button dense danger"
                        type="button"
                        disabled={!canDeleteJob(job) || deleteMutation.isPending}
                        title="删除任务记录"
                        onClick={(event) => {
                          event.stopPropagation();
                          if (confirm(`删除任务记录 ${job.job_id}？`)) {
                            deleteMutation.mutate(job.job_id);
                          }
                        }}
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {selectedJob ? <JobDetailPanel job={selectedJob} logs={jobLogsQuery.data ?? null} /> : null}
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

export function JobCreatePanel({ benchmarks }: { benchmarks: BenchmarkSummary[] }) {
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
  const [promptId, setPromptId] = useState("grounding_layout.latest");
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
    setManifestText(formatManifest(applyPromptTemplateToManifest(manifest, promptTemplate)));
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

  function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const manifest = parseManifest();
    if (!manifest) {
      return;
    }
    mutation.mutate({ manifest });
  }

  return (
    <div className="workspace-card manifest-card">
      <PanelTitle title="新建评测任务" meta="模板 manifest + 后端预检查" />
      <form className="manifest-job-form" onSubmit={submit}>
        <div className="manifest-toolbar">
          <label className="filter-select compact">
            <span>模板</span>
            <select
              value={templateId}
              onChange={(event) => loadTemplate(event.target.value)}
              disabled={templatesQuery.isLoading}
            >
              {templateIds.length === 0 ? <option value="eval_job">加载中</option> : null}
              {templateIds.map((id) => (
                <option key={id} value={id}>
                  {templates[id]?.label ?? id}
                </option>
              ))}
            </select>
          </label>
          <label className="filter-select compact">
            <span>Prompt</span>
            <select
              value={selectedPrompt?.prompt_id ?? promptId}
              onChange={(event) => applySelectedPrompt(event.target.value)}
              disabled={promptTemplatesQuery.isLoading || promptTemplates.length === 0}
            >
              {promptTemplates.length === 0 ? <option value={promptId}>加载中</option> : null}
              {promptTemplates.map((template) => (
                <option key={template.prompt_id} value={template.prompt_id}>
                  {template.label || template.prompt_id}
                </option>
              ))}
            </select>
          </label>
          <button className="secondary-button" type="button" onClick={() => loadTemplate()}>
            恢复模板
          </button>
          <button
            className="secondary-button"
            type="button"
            onClick={() => applySelectedPrompt()}
            disabled={!selectedPrompt}
          >
            应用 Prompt
          </button>
          <button
            className="secondary-button"
            type="button"
            onClick={validateManifest}
            disabled={preflightMutation.isPending}
          >
            {preflightMutation.isPending ? "检查中" : "预检查"}
          </button>
          <button className="primary-button" type="submit" disabled={mutation.isPending}>
            {mutation.isPending ? "加入中" : "加入队列"}
          </button>
        </div>
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
      <button
        className="secondary-button dense"
        type="button"
        onClick={onSaveFromManifest}
        disabled={saving}
      >
        {saving ? "保存中" : "将当前 Manifest 的 Prompt 保存为模板"}
      </button>
      {saveError ? <div className="form-error">Prompt 模板保存失败。</div> : null}
    </details>
  );
}

