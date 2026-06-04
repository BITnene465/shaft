import { useEffect, useMemo, useState } from "react";
import type { CSSProperties } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import type { FacetBucket, JobLog, JobSummary, SchedulerStatus } from "./api";
import {
  cancelJob,
  deleteJob,
  fetchJobLogs,
  fetchJobs,
  fetchSchedulerStatus
} from "./api";
import { AdvancedFilterBar } from "./filterControls";
import {
  errorMessage,
  facetValues,
  formatDate,
  jobTarget,
  stringValue
} from "./formatters";
import { JobQueueTable, jobRunId } from "./jobsQueueTable";
import {
  DEFAULT_JOBS_VIEW_STATE,
  JOBS_VIEW_STATE_RESET_EVENT,
  readJobsViewState,
  writeJobsViewState
} from "./jobsViewState";
import { PagerControl, clampListPageOffset, updatePagedFilterValue } from "./samplePager";
import { jobProgress, progressPhaseText } from "./statusModel";
import {
  Badge,
  DangerConfirmDialog,
  InlineNavLink
} from "./ui";

const JOB_PAGE_SIZE = 80;

export function JobQueuePanel({ compact = false }: { compact?: boolean }) {
  const queryClient = useQueryClient();
  const [initialViewState] = useState(readJobsViewState);
  const [selectedJobId, setSelectedJobId] = useState<string>(
    compact ? "" : initialViewState.selectedJobId
  );
  const [deleteJobTarget, setDeleteJobTarget] = useState<JobSummary | null>(null);
  const [searchText, setSearchText] = useState(compact ? "" : initialViewState.searchText);
  const [statusFilter, setStatusFilter] = useState(
    compact ? "all" : initialViewState.statusFilter
  );
  const [kindFilter, setKindFilter] = useState(compact ? "all" : initialViewState.kindFilter);
  const [pageOffset, setPageOffset] = useState(compact ? 0 : initialViewState.pageOffset);
  function resetViewState() {
    if (compact) {
      return;
    }
    setSelectedJobId(DEFAULT_JOBS_VIEW_STATE.selectedJobId);
    setSearchText(DEFAULT_JOBS_VIEW_STATE.searchText);
    setStatusFilter(DEFAULT_JOBS_VIEW_STATE.statusFilter);
    setKindFilter(DEFAULT_JOBS_VIEW_STATE.kindFilter);
    setPageOffset(DEFAULT_JOBS_VIEW_STATE.pageOffset);
  }
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
  const { data, isLoading, isPlaceholderData, error } = useQuery({
    queryKey: ["jobs", jobFilters],
    queryFn: () => fetchJobs(jobFilters),
    refetchInterval: 2_000,
    placeholderData: (previousData) => previousData
  });
  const schedulerQuery = useQuery({
    queryKey: ["scheduler-status"],
    queryFn: fetchSchedulerStatus,
    refetchInterval: 2_000
  });
  const selectedJob = data?.jobs.find((job) => job.job_id === selectedJobId) ?? null;
  const facets = data?.facets;
  const filteredJobs = data?.jobs ?? [];
  const statuses = facetValues(facets, "statuses", [
    "queued",
    "running",
    "succeeded",
    "failed",
    "cancelled",
    ...filteredJobs.map((job) => job.status)
  ]);
  const kinds = facetValues(facets, "kinds", [
    "eval",
    "preannotate",
    ...filteredJobs.map((job) => job.kind)
  ]);
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
  useEffect(() => {
    if (compact) {
      return;
    }
    writeJobsViewState({
      selectedJobId,
      searchText,
      statusFilter,
      kindFilter,
      pageOffset
    });
  }, [compact, selectedJobId, searchText, statusFilter, kindFilter, pageOffset]);
  useEffect(() => {
    if (compact) {
      return undefined;
    }
    window.addEventListener(JOBS_VIEW_STATE_RESET_EVENT, resetViewState);
    return () => window.removeEventListener(JOBS_VIEW_STATE_RESET_EVENT, resetViewState);
  }, [compact]);
  if (isLoading) {
    return <div className="empty-panel">正在加载队列状态</div>;
  }
  if (error || !data) {
    return <div className="empty-panel danger-text">队列状态加载失败：{errorMessage(error)}</div>;
  }
  return (
    <div className={compact ? "queue-stack compact" : "queue-stack"}>
      <SchedulerStrip
        statusFacets={facets?.statuses ?? []}
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
              onChange: (value) =>
                updatePagedFilterValue(searchText, value, setSearchText, setPageOffset),
              placeholder: "搜索 job、模型、benchmark、错误、日志"
            },
            {
              type: "select",
              id: "job-status",
              label: "状态",
              value: statusFilter,
              values: ["all", ...statuses],
              labels: { all: "全部" },
              onChange: (value) =>
                updatePagedFilterValue(statusFilter, value, setStatusFilter, setPageOffset)
            },
            {
              type: "select",
              id: "job-kind",
              label: "类型",
              value: kindFilter,
              values: ["all", ...kinds],
              labels: { all: "全部" },
              onChange: (value) =>
                updatePagedFilterValue(kindFilter, value, setKindFilter, setPageOffset)
            }
          ]}
        />
      ) : null}
      {totalJobs === 0 ? (
        <div className="empty-panel">当前没有任务。</div>
      ) : filteredJobs.length === 0 ? (
        <div className="empty-panel">没有符合高级检索条件的任务。</div>
      ) : (
        <JobQueueTable
          jobs={filteredJobs}
          selectedJobId={selectedJob?.job_id ?? ""}
          compact={compact}
          refreshing={Boolean(isPlaceholderData && data)}
          cancelPending={cancelMutation.isPending}
          deletePending={deleteMutation.isPending}
          onSelectJob={setSelectedJobId}
          onCancelJob={(jobId) => cancelMutation.mutate(jobId)}
          onDeleteJob={setDeleteJobTarget}
        />
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
  statusFacets,
  scheduler
}: {
  statusFacets: FacetBucket;
  scheduler: SchedulerStatus;
}) {
  const queued = facetCount(statusFacets, "queued");
  const running = facetCount(statusFacets, "running");
  const failed = facetCount(statusFacets, "failed");
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

function facetCount(facets: FacetBucket, value: string) {
  return facets.find((item) => item.value === value)?.count ?? 0;
}

function JobDetailPanel({ job, logs }: { job: JobSummary; logs: JobLog | null }) {
  const progress = jobProgress(job);
  const lines = logs?.lines ?? [];
  const linkedRunId = stringValue(job.metadata.run_manifest_path) ? jobRunId(job) : "";
  const percent = progress.percent ?? 8;
  return (
    <div className="job-detail-panel">
      <div className="job-monitor-header">
        <div>
          <div className="eyebrow">任务详情</div>
          <strong>{job.job_id}</strong>
        </div>
        <div className="job-monitor-actions">
          {linkedRunId ? (
            <InlineNavLink to="/runs/$runId" params={{ runId: linkedRunId }}>
              打开结果
            </InlineNavLink>
          ) : null}
          <Badge value={job.status} domain="job" />
        </div>
      </div>
      <div
        className="job-progress-row"
        style={{ "--job-progress": (percent / 100).toFixed(4) } as CSSProperties}
      >
        <div className="job-progress-track" aria-label="任务进度">
          <span />
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
