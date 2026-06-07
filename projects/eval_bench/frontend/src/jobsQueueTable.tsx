import { Trash2, X } from "lucide-react";
import type { CSSProperties } from "react";

import type { JobSummary } from "./api";
import {
  basename,
  formatDate,
  jobTarget,
  stringValue
} from "./formatters";
import { canCancelJob, canDeleteJob, jobProgress } from "./statusModel";
import {
  Badge,
  IconActionButton,
  SelectableTableRow
} from "./ui";
import { tableColumnClassName } from "./uiDataTable";

export const JOB_QUEUE_COLUMN_CLASS_NAMES = {
  identity: tableColumnClassName({ width: "id", wrap: "wrap" }),
  kind: tableColumnClassName({ width: "compact" }),
  status: tableColumnClassName({ width: "status", wrap: "wrap" }),
  target: tableColumnClassName({ width: "wide", wrap: "wrap" }),
  createdAt: tableColumnClassName({ width: "date" }),
  actions: tableColumnClassName({ width: "actions", wrap: "wrap" })
};

export function JobQueueTable({
  jobs,
  selectedJobId,
  compact,
  refreshing,
  cancelPending,
  deletePending,
  onSelectJob,
  onCancelJob,
  onDeleteJob
}: {
  jobs: JobSummary[];
  selectedJobId: string;
  compact: boolean;
  refreshing: boolean;
  cancelPending: boolean;
  deletePending: boolean;
  onSelectJob: (jobId: string) => void;
  onCancelJob: (jobId: string) => void;
  onDeleteJob: (job: JobSummary) => void;
}) {
  return (
    <div
      className={[
        "table-shell",
        compact ? "compact" : "",
        refreshing ? "refreshing" : ""
      ].filter(Boolean).join(" ")}
    >
      {!compact && refreshing ? (
        <span className="table-refresh-indicator" aria-live="polite">
          队列更新中
        </span>
      ) : null}
      <table>
        <thead>
          <tr>
            <th className={JOB_QUEUE_COLUMN_CLASS_NAMES.identity}>评测</th>
            <th className={JOB_QUEUE_COLUMN_CLASS_NAMES.kind}>类型</th>
            <th className={JOB_QUEUE_COLUMN_CLASS_NAMES.status}>状态</th>
            <th className={JOB_QUEUE_COLUMN_CLASS_NAMES.target}>目标</th>
            <th className={JOB_QUEUE_COLUMN_CLASS_NAMES.createdAt}>创建时间</th>
            <th className={JOB_QUEUE_COLUMN_CLASS_NAMES.actions}></th>
          </tr>
        </thead>
        <tbody>
          {jobs.map((job) => (
            <JobQueueRow
              key={job.job_id}
              job={job}
              selected={job.job_id === selectedJobId}
              cancelPending={cancelPending}
              deletePending={deletePending}
              onSelectJob={onSelectJob}
              onCancelJob={onCancelJob}
              onDeleteJob={onDeleteJob}
            />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function JobQueueRow({
  job,
  selected,
  cancelPending,
  deletePending,
  onSelectJob,
  onCancelJob,
  onDeleteJob
}: {
  job: JobSummary;
  selected: boolean;
  cancelPending: boolean;
  deletePending: boolean;
  onSelectJob: (jobId: string) => void;
  onCancelJob: (jobId: string) => void;
  onDeleteJob: (job: JobSummary) => void;
}) {
  const runId = jobRunId(job);
  return (
    <SelectableTableRow
      selected={selected}
      onClick={() => onSelectJob(job.job_id)}
    >
      <td className={JOB_QUEUE_COLUMN_CLASS_NAMES.identity}>
        <div className="job-eval-cell">
          <strong className="run-id-text" title={runId || job.job_id}>
            {runId || job.job_id}
          </strong>
          {runId && runId !== job.job_id ? (
            <small title={job.job_id}>job {job.job_id}</small>
          ) : null}
        </div>
      </td>
      <td className={JOB_QUEUE_COLUMN_CLASS_NAMES.kind}>{job.kind}</td>
      <td className={JOB_QUEUE_COLUMN_CLASS_NAMES.status}>
        <Badge value={job.status} domain="job" />
        <JobProgressInline job={job} />
      </td>
      <td className={JOB_QUEUE_COLUMN_CLASS_NAMES.target}>
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
      <td className={JOB_QUEUE_COLUMN_CLASS_NAMES.createdAt}>
        {formatDate(job.created_at)}
      </td>
      <td className={JOB_QUEUE_COLUMN_CLASS_NAMES.actions}>
        <div className="row-actions">
          <IconActionButton
            icon={<X size={14} />}
            disabled={!canCancelJob(job) || cancelPending}
            title={job.status === "running" ? "终止运行中评测" : "取消排队任务"}
            onClick={(event) => {
              event.stopPropagation();
              onCancelJob(job.job_id);
            }}
          />
          <IconActionButton
            icon={<Trash2 size={14} />}
            danger
            disabled={!canDeleteJob(job) || deletePending}
            title="删除任务记录"
            onClick={(event) => {
              event.stopPropagation();
              onDeleteJob(job);
            }}
          />
        </div>
      </td>
    </SelectableTableRow>
  );
}

export function jobRunId(job: JobSummary) {
  return job.run_id || stringValue(job.metadata.run_id) || stringValue(job.payload.run_id);
}

function JobProgressInline({ job }: { job: JobSummary }) {
  if (job.status !== "running" && job.status !== "failed" && job.status !== "succeeded") {
    return null;
  }
  const progress = jobProgress(job);
  const percent = progress.percent ?? (job.status === "succeeded" ? 100 : 0);
  return (
    <div
      className="job-progress-inline"
      style={{ "--job-progress": (percent / 100).toFixed(4) } as CSSProperties}
    >
      <div className="job-progress-mini" aria-hidden="true">
        <span />
      </div>
      <small>{progress.text}</small>
    </div>
  );
}
