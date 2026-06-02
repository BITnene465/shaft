import { useState } from "react";
import { Link } from "@tanstack/react-router";

import type { RunSummary } from "./api";
import { useDashboardState } from "./dashboardState";
import { AppIcon } from "./iconLibrary";
import { JobCreatePanel } from "./jobsCreatePanel";
import { JobQueuePanel } from "./jobsQueuePanel";
import { recentRunsByCreatedAt, runArtifactReadiness } from "./runArtifactSignals";
import { Badge, CommandButton, PanelTitle, WorkspaceDialog } from "./ui";

import "./jobsPage.css";

export function JobsPage() {
  const { data } = useDashboardState();
  const [createOpen, setCreateOpen] = useState(false);
  const recentRuns = recentRunsByCreatedAt(data?.runs ?? [], 12);
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
      {runs.map((run) => {
        const readiness = runArtifactReadiness(run);
        return (
          <Link
            className={["recent-run-card", readiness.tone].join(" ")}
            key={run.run_id}
            to="/runs/$runId"
            params={{ runId: run.run_id }}
          >
            <span className="recent-run-head">
              <strong className="run-id-text" title={run.run_id}>
                {run.run_id}
              </strong>
              <Badge value={run.status} domain="run" />
            </span>
            <span
              className="recent-run-meta"
              title={`${run.benchmark_id}:${run.benchmark_split || "-"} / ${run.model_id}`}
            >
              {run.benchmark_id || "-"}:{run.benchmark_split || "-"} /{" "}
              {run.model_id || "unknown model"}
            </span>
            <span className="recent-run-artifacts" aria-label="run 产物状态">
              <i>
                <b style={{ width: `${readiness.percent}%` }} />
              </i>
              <span>
                <em>{run.prediction_count.toLocaleString()} pred</em>
                <em>{run.report_count > 0 ? `${run.report_count.toLocaleString()} report` : "待评"}</em>
                {run.note.trim() ? <em>note</em> : null}
              </span>
            </span>
          </Link>
        );
      })}
    </div>
  );
}
