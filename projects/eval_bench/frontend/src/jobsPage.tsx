import { useState } from "react";

import { useDashboardState } from "./dashboardState";
import { AppIcon } from "./iconLibrary";
import { JobCreatePanel } from "./jobsCreatePanel";
import { JobQueuePanel } from "./jobsQueuePanel";
import { CommandButton, PanelTitle } from "./ui";
import { WorkspaceDialog } from "./uiDialog";

import "./jobsPage.css";

export function JobsPage() {
  const { data } = useDashboardState();
  const [createOpen, setCreateOpen] = useState(false);
  return (
    <section className="page-stack density-page jobs-page">
      <div className="page-command-row">
        <div>
          <h2>评测中心</h2>
          <span>队列、runtime log 和任务排障</span>
        </div>
        <CommandButton icon={<AppIcon name="createEval" size={17} />} onClick={() => setCreateOpen(true)}>
          新建评测
        </CommandButton>
      </div>
      <div className="job-activity-grid">
        <div className="workspace-card fill job-queue-card">
          <PanelTitle title="任务队列" meta="执行、失败排障和 runtime log" />
          <JobQueuePanel />
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
