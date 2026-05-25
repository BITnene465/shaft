import React from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  Clock3,
  Gauge,
  Layers3,
  PlayCircle,
  Radio,
  Server,
  Trophy
} from "lucide-react";

import type { JobSummary, RunSummary, ServiceSummary } from "./api";
import { fetchJobs, fetchSchedulerStatus, fetchServices } from "./api";
import { useDashboardState } from "./dashboardState";
import { AppIcon } from "./iconLibrary";
import { Badge, EmptyState, PanelTitle } from "./ui";

type OverviewRoute = "/" | "/rank-board" | "/runs" | "/jobs" | "/services" | "/benchmarks";
type OverviewTone = "idle" | "live" | "warm" | "good" | "danger";
type OverviewActivityLane = {
  label: string;
  tone: "run" | "job" | "service";
  rows: OverviewCountRow[];
  total: number;
};
type OverviewCountRow = { key: string; count: number };
type OverviewPipelineStage = {
  label: string;
  value: number;
  total: number;
  meta: string;
  to: OverviewRoute;
  tone: OverviewTone;
};
type OverviewSignal = {
  label: string;
  value: string;
  detail: string;
  to: OverviewRoute;
  tone: OverviewTone;
  icon: React.ReactNode;
  progress: number;
};
type OverviewAction = {
  label: string;
  detail: string;
  to: OverviewRoute;
  tone: OverviewTone;
  icon: React.ReactNode;
};
type OverviewReadinessItem = OverviewAction & {
  state: string;
  value: number;
  total: number;
};

export function OverviewPage() {
  const { data, isLoading, error } = useDashboardState();
  const jobsQuery = useQuery({
    queryKey: ["overview-jobs"],
    queryFn: () => fetchJobs({ limit: 500 }),
    refetchInterval: 2_000
  });
  const servicesQuery = useQuery({
    queryKey: ["overview-services"],
    queryFn: () => fetchServices({ limit: 500 }),
    refetchInterval: 5_000
  });
  const schedulerQuery = useQuery({
    queryKey: ["overview-scheduler"],
    queryFn: fetchSchedulerStatus,
    refetchInterval: 2_000
  });
  if (isLoading) {
    return <EmptyState title="正在加载看板状态" />;
  }
  if (error || !data) {
    return <EmptyState title="看板状态加载失败" tone="danger" />;
  }

  const jobs = jobsQuery.data?.jobs ?? [];
  const services = servicesQuery.data?.services ?? [];
  const schedulerStatus = schedulerQuery.data;
  const evaluatedRuns = data.runs.filter((run) => run.report_path).length;
  const runsWithPredictions = data.runs.filter((run) => run.prediction_count > 0).length;
  const waitingEvaluation = data.runs.filter(
    (run) => !run.report_path && run.prediction_count > 0
  ).length;
  const queuedJobs = jobs.filter((job) => job.status === "queued").length;
  const runningJobs = jobs.filter((job) => job.status === "running").length;
  const failedJobs = jobs.filter((job) => job.status === "failed").length;
  const liveServices = services.filter((service) => service.status === "running").length;
  const overviewSyncing =
    jobsQuery.isFetching || servicesQuery.isFetching || schedulerQuery.isFetching;
  const totalRuns = Math.max(data.run_count, 1);
  const totalJobs = Math.max(jobs.length, 1);
  const totalServices = Math.max(services.length, 1);
  const coveragePercent = percent(evaluatedRuns, totalRuns);
  const activityLanes = overviewActivityLanes(data.runs, jobs, services, 12);
  const volumeTotal = Math.max(data.total_benchmark_samples, data.prediction_count, 1);
  const activeQueue = queuedJobs + runningJobs;
  const schedulerEnabled = Boolean(schedulerStatus?.enabled);
  const postureLine = overviewPostureLine({
    failedJobs,
    waitingEvaluation,
    activeQueue,
    liveServices,
    serviceCount: services.length,
    evaluatedRuns
  });
  const nextAction = overviewNextAction({
    failedJobs,
    waitingEvaluation,
    activeQueue,
    liveServices,
    serviceCount: services.length,
    evaluatedRuns
  });

  const pipelineStages: OverviewPipelineStage[] = [
    {
      label: "Benchmarks",
      value: data.benchmark_count,
      total: Math.max(data.benchmark_count, 1),
      meta: `${data.total_benchmark_samples.toLocaleString()} samples`,
      to: "/benchmarks" as OverviewRoute,
      tone: data.benchmark_count > 0 ? "good" : "warm"
    },
    {
      label: "Predictions",
      value: data.prediction_count,
      total: volumeTotal,
      meta: `${runsWithPredictions.toLocaleString()} runs with pred`,
      to: "/runs",
      tone: data.prediction_count > 0 ? "live" : "idle"
    },
    {
      label: "Evaluated",
      value: evaluatedRuns,
      total: totalRuns,
      meta: `${coveragePercent}% complete`,
      to: "/runs",
      tone: waitingEvaluation > 0 ? "warm" : "good"
    },
    {
      label: "Rank Ready",
      value: evaluatedRuns,
      total: totalRuns,
      meta: evaluatedRuns > 0 ? "rank board available" : "need report",
      to: "/rank-board",
      tone: evaluatedRuns > 0 ? "good" : "idle"
    }
  ];

  const signalItems: OverviewSignal[] = [
    {
      label: "报告覆盖",
      value: `${coveragePercent}%`,
      detail: `${evaluatedRuns.toLocaleString()} / ${data.run_count.toLocaleString()} runs`,
      to: "/runs",
      tone: evaluatedRuns > 0 ? "good" : "idle",
      icon: <CheckCircle2 size={16} />,
      progress: trackPercent(evaluatedRuns, totalRuns)
    },
    {
      label: "待评估",
      value: waitingEvaluation.toLocaleString(),
      detail: `${runsWithPredictions.toLocaleString()} runs have predictions`,
      to: "/runs",
      tone: waitingEvaluation > 0 ? "warm" : "idle",
      icon: <Gauge size={16} />,
      progress: trackPercent(waitingEvaluation, totalRuns)
    },
    {
      label: "队列压力",
      value: activeQueue.toLocaleString(),
      detail:
        failedJobs > 0
          ? `${failedJobs.toLocaleString()} failed jobs`
          : `${queuedJobs.toLocaleString()} queued / ${runningJobs.toLocaleString()} running`,
      to: "/jobs",
      tone: failedJobs > 0 ? "danger" : activeQueue > 0 ? "live" : "idle",
      icon: activeQueue > 0 ? <Activity size={16} /> : <Clock3 size={16} />,
      progress: trackPercent(activeQueue + failedJobs, totalJobs)
    },
    {
      label: "在线服务",
      value: `${liveServices}/${services.length}`,
      detail: schedulerEnabled ? "scheduler auto" : "manual mode",
      to: "/services",
      tone: liveServices > 0 ? "live" : services.length > 0 ? "warm" : "idle",
      icon: <Server size={16} />,
      progress: trackPercent(liveServices, totalServices)
    }
  ];
  const readinessItems = overviewReadinessItems({
    failedJobs,
    waitingEvaluation,
    queuedJobs,
    runningJobs,
    liveServices,
    serviceCount: services.length,
    evaluatedRuns,
    totalRuns,
    schedulerEnabled
  });
  const recentRuns = overviewRecentRuns(data.runs, 6);

  return (
    <section className="page-stack dashboard-home">
      <div className="overview-console">
        <div className="overview-console-main">
          <div className="overview-title-block">
            <div className="eyebrow">Eval Bench Control</div>
            <h2>总览</h2>
            <p>{postureLine}</p>
          </div>
          <div className="overview-stat-row">
            <OverviewStat label="Coverage" value={`${coveragePercent}%`} />
            <OverviewStat
              label="Pending"
              value={waitingEvaluation}
              tone={waitingEvaluation > 0 ? "live" : "idle"}
            />
            <OverviewStat
              label="Queue"
              value={activeQueue}
              tone={activeQueue > 0 ? "live" : "idle"}
            />
            <OverviewStat
              label="Services"
              value={`${liveServices}/${services.length}`}
              tone={liveServices > 0 ? "live" : "idle"}
            />
          </div>
        </div>
        <div className="overview-console-side">
          <div className={overviewSyncing ? "overview-sync-pill syncing" : "overview-sync-pill"}>
            <span />
            <strong>{overviewSyncing ? "同步中" : "已同步"}</strong>
          </div>
          <div className="overview-console-links">
            <Link to="/rank-board">
              <AppIcon name="rankBoard" size={14} />
              排行榜
            </Link>
            <Link to="/runs">
              <AppIcon name="runResults" size={14} />
              结果库
            </Link>
            <Link to="/jobs">
              <Activity size={14} />
              任务
            </Link>
          </div>
        </div>
      </div>

      <div className="overview-command-deck">
        <section className="workspace-card overview-focus-panel">
          <div className="overview-focus-head">
            <PanelTitle title="当前决策" meta="action / signal / flow" />
            <OverviewNextAction action={nextAction} />
          </div>
          <OverviewSignalStrip signals={signalItems} />
          <OverviewPipeline stages={pipelineStages} />
          <OverviewActivityMatrix lanes={activityLanes} />
        </section>

        <aside className="overview-side-stack">
          <OverviewReadinessPanel schedulerEnabled={schedulerEnabled} items={readinessItems} />
          <OverviewRecentRunsPanel runs={recentRuns} />
        </aside>
      </div>
    </section>
  );
}

function OverviewSignalStrip({ signals }: { signals: OverviewSignal[] }) {
  return (
    <div className="overview-signal-strip overview-operational-grid">
      {signals.map((signal) => (
        <Link className={`overview-signal-card ${signal.tone}`} to={signal.to} key={signal.label}>
          <span>{signal.icon}</span>
          <div>
            <strong>{signal.value}</strong>
            <em>{signal.label}</em>
          </div>
          <small>{signal.detail}</small>
          <i aria-hidden="true">
            <b style={{ width: `${signal.progress}%` }} />
          </i>
        </Link>
      ))}
    </div>
  );
}

function OverviewStat({
  label,
  value,
  tone = "idle"
}: {
  label: string;
  value: number | string;
  tone?: "idle" | "live";
}) {
  return (
    <div className={tone === "live" ? "overview-stat live" : "overview-stat"}>
      <span>{label}</span>
      <strong>{typeof value === "number" ? value.toLocaleString() : value}</strong>
    </div>
  );
}

function OverviewNextAction({ action }: { action: OverviewAction }) {
  return (
    <Link className={`overview-next-action ${action.tone}`} to={action.to}>
      <span>{action.icon}</span>
      <div>
        <strong>{action.label}</strong>
        <em>{action.detail}</em>
      </div>
      <ArrowRight size={16} />
    </Link>
  );
}

function OverviewPipeline({ stages }: { stages: OverviewPipelineStage[] }) {
  return (
    <div className="overview-pipeline" aria-label="Eval Bench 数据管线">
      {stages.map((stage, index) => (
        <Link className={`overview-pipeline-stage ${stage.tone}`} to={stage.to} key={stage.label}>
          <div className="overview-pipeline-node">
            <span>{String(index + 1).padStart(2, "0")}</span>
            <strong>{stage.label}</strong>
          </div>
          <div className="overview-pipeline-value">
            <strong>{stage.value.toLocaleString()}</strong>
            <em>{stage.meta}</em>
          </div>
          <div className="overview-pipeline-rail" aria-hidden="true">
            <i style={{ width: `${trackPercent(stage.value, stage.total)}%` }} />
          </div>
        </Link>
      ))}
    </div>
  );
}

function OverviewReadinessPanel({
  schedulerEnabled,
  items
}: {
  schedulerEnabled: boolean;
  items: OverviewReadinessItem[];
}) {
  return (
    <section className="workspace-card overview-action-panel">
      <PanelTitle title="Readiness" meta={schedulerEnabled ? "scheduler auto" : "scheduler manual"} />
      <div className="overview-action-list">
        {items.map((item) => (
          <Link className={`overview-action-link ${item.tone}`} to={item.to} key={item.label}>
            <span>{item.icon}</span>
            <div>
              <strong>{item.label}</strong>
              <em>{item.detail}</em>
              <div className="overview-action-meter" aria-hidden="true">
                <i style={{ width: `${trackPercent(item.value, item.total)}%` }} />
              </div>
            </div>
            <b>{item.state}</b>
          </Link>
        ))}
      </div>
    </section>
  );
}

function OverviewActivityMatrix({ lanes }: { lanes: OverviewActivityLane[] }) {
  const maxCount = Math.max(
    1,
    ...lanes.flatMap((lane) => lane.rows.map((row) => row.count))
  );
  const total = lanes.reduce((sum, lane) => sum + lane.total, 0);
  const bucketCount = lanes[0]?.rows.length ?? 0;
  const latest = lanes[0]?.rows.at(-1);
  return (
    <div className="overview-activity-matrix">
      <div className="overview-activity-header">
        <span>
          <Radio size={14} />
          活动节奏
        </span>
        <strong>{total.toLocaleString()} events / {bucketCount}d</strong>
      </div>
      <div className="overview-activity-lanes" aria-label="最近日期桶的 run job service 活动">
        {lanes.map((lane) => (
          <div className={`overview-activity-lane ${lane.tone}`} key={lane.label}>
            <span>{lane.label}</span>
            <div className="overview-activity-cells">
              {lane.rows.map((row) => (
                <i
                  key={row.key}
                  style={
                    {
                      opacity: row.count > 0 ? Math.max(0.32, row.count / maxCount) : 0.12
                    } as React.CSSProperties
                  }
                  title={`${lane.label} ${row.key}: ${row.count.toLocaleString()}`}
                />
              ))}
            </div>
            <strong>{lane.total.toLocaleString()}</strong>
          </div>
        ))}
      </div>
      <div className="overview-activity-footer">
        <span>{latest?.key ?? "-"}</span>
        <ArrowRight size={13} />
      </div>
    </div>
  );
}

function OverviewRunList({ runs }: { runs: RunSummary[] }) {
  if (runs.length === 0) {
    return <div className="empty-inline">暂无 run。</div>;
  }
  return (
    <div className="overview-run-list">
      {runs.map((run, index) => (
        <Link key={run.run_id} to="/runs/$runId" params={{ runId: run.run_id }}>
          <em>{String(index + 1).padStart(2, "0")}</em>
          <span>
            <strong>{run.run_id}</strong>
            <small>
              {run.model_id || "-"} · {run.prediction_count.toLocaleString()} pred /{" "}
              {run.report_count.toLocaleString()} report
            </small>
          </span>
          <Badge value={run.status} domain="run" />
        </Link>
      ))}
    </div>
  );
}

function OverviewRecentRunsPanel({ runs }: { runs: RunSummary[] }) {
  return (
    <section className="workspace-card overview-recent-card">
      <PanelTitle title="最近 run" meta={`latest ${runs.length}`} />
      <OverviewRunList runs={runs} />
    </section>
  );
}

function overviewNextAction({
  failedJobs,
  waitingEvaluation,
  activeQueue,
  liveServices,
  serviceCount,
  evaluatedRuns
}: {
  failedJobs: number;
  waitingEvaluation: number;
  activeQueue: number;
  liveServices: number;
  serviceCount: number;
  evaluatedRuns: number;
}): OverviewAction {
  if (failedJobs > 0) {
    return {
      label: "检查失败任务",
      detail: `${failedJobs.toLocaleString()} failed job records`,
      to: "/jobs",
      tone: "danger",
      icon: <AlertTriangle size={16} />
    };
  }
  if (waitingEvaluation > 0) {
    return {
      label: "处理待评估 run",
      detail: `${waitingEvaluation.toLocaleString()} runs have predictions`,
      to: "/runs",
      tone: "warm",
      icon: <Gauge size={16} />
    };
  }
  if (activeQueue > 0) {
    return {
      label: "查看运行队列",
      detail: `${activeQueue.toLocaleString()} queued or running`,
      to: "/jobs",
      tone: "live",
      icon: <Activity size={16} />
    };
  }
  if (serviceCount > 0 && liveServices === 0) {
    return {
      label: "启动模型服务",
      detail: "registered services are idle",
      to: "/services",
      tone: "warm",
      icon: <Server size={16} />
    };
  }
  return {
    label: evaluatedRuns > 0 ? "查看排行榜" : "创建评测任务",
    detail: evaluatedRuns > 0 ? "reports are ready for ranking" : "no evaluated run yet",
    to: evaluatedRuns > 0 ? "/rank-board" : "/jobs",
    tone: evaluatedRuns > 0 ? "good" : "idle",
    icon: evaluatedRuns > 0 ? <Trophy size={16} /> : <Layers3 size={16} />
  };
}

function overviewPostureLine({
  failedJobs,
  waitingEvaluation,
  activeQueue,
  liveServices,
  serviceCount,
  evaluatedRuns
}: {
  failedJobs: number;
  waitingEvaluation: number;
  activeQueue: number;
  liveServices: number;
  serviceCount: number;
  evaluatedRuns: number;
}) {
  if (failedJobs > 0) {
    return `${failedJobs.toLocaleString()} failed jobs need inspection`;
  }
  if (waitingEvaluation > 0) {
    return `${waitingEvaluation.toLocaleString()} runs are ready for evaluation`;
  }
  if (activeQueue > 0) {
    return `${activeQueue.toLocaleString()} jobs are moving through the queue`;
  }
  if (serviceCount > 0 && liveServices === 0) {
    return "rank data is ready; model service is idle";
  }
  if (evaluatedRuns > 0) {
    return `${evaluatedRuns.toLocaleString()} evaluated runs are ready for ranking`;
  }
  return "no evaluated run yet; create a job to start the loop";
}

function overviewReadinessItems({
  failedJobs,
  waitingEvaluation,
  queuedJobs,
  runningJobs,
  liveServices,
  serviceCount,
  evaluatedRuns,
  totalRuns,
  schedulerEnabled
}: {
  failedJobs: number;
  waitingEvaluation: number;
  queuedJobs: number;
  runningJobs: number;
  liveServices: number;
  serviceCount: number;
  evaluatedRuns: number;
  totalRuns: number;
  schedulerEnabled: boolean;
}): OverviewReadinessItem[] {
  const activeJobs = queuedJobs + runningJobs;
  const jobTotal = Math.max(activeJobs + failedJobs, 1);
  return [
    {
      label: "Services",
      detail:
        serviceCount > 0
          ? `${liveServices}/${serviceCount} model services live`
          : "no service registered",
      to: "/services",
      tone: liveServices > 0 ? "good" : serviceCount > 0 ? "warm" : "idle",
      icon: <Server size={15} />,
      state: liveServices > 0 ? "LIVE" : serviceCount > 0 ? "IDLE" : "NONE",
      value: liveServices,
      total: Math.max(serviceCount, 1)
    },
    {
      label: "Queue",
      detail:
        failedJobs > 0
          ? `${failedJobs} failed jobs`
          : schedulerEnabled
            ? "scheduler is armed"
            : "manual worker mode",
      to: "/jobs",
      tone: failedJobs > 0 ? "danger" : activeJobs > 0 ? "live" : "good",
      icon: <PlayCircle size={15} />,
      state: failedJobs > 0 ? "CHECK" : activeJobs > 0 ? "ACTIVE" : "CLEAR",
      value: activeJobs + failedJobs,
      total: jobTotal
    },
    {
      label: "Evaluation",
      detail: waitingEvaluation > 0 ? `${waitingEvaluation} runs waiting` : `${evaluatedRuns} evaluated runs`,
      to: "/runs",
      tone: waitingEvaluation > 0 ? "warm" : evaluatedRuns > 0 ? "good" : "idle",
      icon: <Gauge size={15} />,
      state: waitingEvaluation > 0 ? "WAIT" : evaluatedRuns > 0 ? "DONE" : "EMPTY",
      value: evaluatedRuns,
      total: totalRuns
    },
    {
      label: "Rank Board",
      detail: evaluatedRuns > 0 ? "global ranking is available" : "needs evaluated reports",
      to: "/rank-board",
      tone: evaluatedRuns > 0 ? "good" : "idle",
      icon: <Trophy size={15} />,
      state: evaluatedRuns > 0 ? "READY" : "LOCKED",
      value: evaluatedRuns,
      total: totalRuns
    }
  ];
}

function overviewRecentRuns(runs: RunSummary[], limit: number) {
  return [...runs]
    .sort((left, right) => {
      const leftTime = Date.parse(left.created_at ?? "");
      const rightTime = Date.parse(right.created_at ?? "");
      return (
        (Number.isFinite(rightTime) ? rightTime : 0) -
        (Number.isFinite(leftTime) ? leftTime : 0)
      );
    })
    .slice(0, limit);
}

function percent(value: number, total: number) {
  if (total <= 0) {
    return 0;
  }
  return Math.round((value / total) * 100);
}

function trackPercent(value: number, total: number) {
  if (total <= 0 || value <= 0) {
    return 0;
  }
  return Math.max(5, Math.min(100, (value / total) * 100));
}

function overviewActivityLanes(
  runs: RunSummary[],
  jobs: JobSummary[],
  services: ServiceSummary[],
  bucketCount: number
): OverviewActivityLane[] {
  const keys = overviewTimelineKeys(
    latestActivityDate(runs, jobs, services) ?? new Date(),
    bucketCount
  );
  return [
    {
      label: "Runs",
      tone: "run",
      rows: timelineRowsForItems(runs, keys, (run) => run.created_at),
      total: runs.length
    },
    {
      label: "Jobs",
      tone: "job",
      rows: timelineRowsForItems(jobs, keys, (job) => job.created_at),
      total: jobs.length
    },
    {
      label: "Svc",
      tone: "service",
      rows: timelineRowsForItems(
        services,
        keys,
        (service) => service.updated_at ?? service.created_at
      ),
      total: services.length
    }
  ];
}

function timelineRowsForItems<T>(
  items: T[],
  keys: string[],
  timestampForItem: (item: T) => string | null | undefined
) {
  const counts = countBy(items, (item) => {
    const timestamp = timestampForItem(item);
    return timestamp ? timestamp.slice(0, 10) : "unknown";
  });
  const countMap = new Map(counts.map((row) => [row.key, row.count]));
  return keys.map((key) => ({ key, count: countMap.get(key) ?? 0 }));
}

function countBy<T>(items: T[], keyForItem: (item: T) => string) {
  const counts = new Map<string, number>();
  for (const item of items) {
    const key = keyForItem(item).trim() || "unknown";
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return Array.from(counts.entries())
    .map(([key, count]) => ({ key, count }))
    .sort((left, right) => right.count - left.count || left.key.localeCompare(right.key));
}

function overviewTimelineKeys(endDate: Date, bucketCount: number) {
  return Array.from({ length: bucketCount }, (_, index) => {
    const date = new Date(endDate);
    date.setUTCDate(endDate.getUTCDate() - (bucketCount - 1 - index));
    return date.toISOString().slice(0, 10);
  });
}

function latestActivityDate(
  runs: RunSummary[],
  jobs: JobSummary[],
  services: ServiceSummary[]
) {
  const timestamps = [
    ...runs.map((run) => run.created_at),
    ...jobs.map((job) => job.created_at),
    ...services.map((service) => service.updated_at ?? service.created_at)
  ]
    .map((timestamp) => (timestamp ? Date.parse(timestamp) : Number.NaN))
    .filter(Number.isFinite);
  if (timestamps.length === 0) {
    return null;
  }
  return new Date(Math.max(...timestamps));
}
