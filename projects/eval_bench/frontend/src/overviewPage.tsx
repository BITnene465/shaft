import React from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BarChart3,
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
type OverviewTrack = {
  label: string;
  value: number;
  total: number;
  meta: string;
  tone: OverviewTone;
};
type OverviewPipelineStage = {
  label: string;
  value: number;
  total: number;
  meta: string;
  to: OverviewRoute;
  tone: OverviewTone;
};
type OverviewAction = {
  label: string;
  detail: string;
  to: OverviewRoute;
  tone: OverviewTone;
  icon: React.ReactNode;
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
  const activeRuns = data.runs.filter((run) =>
    ["created", "queued", "running"].includes(run.status)
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

  const runTracks: OverviewTrack[] = [
    {
      label: "已评估",
      value: evaluatedRuns,
      total: totalRuns,
      meta: `${coveragePercent}%`,
      tone: "good"
    },
    {
      label: "有预测待评估",
      value: waitingEvaluation,
      total: totalRuns,
      meta: `${runsWithPredictions.toLocaleString()} with pred`,
      tone: waitingEvaluation > 0 ? "warm" : "idle"
    },
    {
      label: "活跃 run",
      value: activeRuns,
      total: totalRuns,
      meta: activeRuns > 0 ? "live" : "idle",
      tone: activeRuns > 0 ? "live" : "idle"
    }
  ];
  const opsTracks: OverviewTrack[] = [
    {
      label: "Queued jobs",
      value: queuedJobs,
      total: totalJobs,
      meta: `${jobs.length.toLocaleString()} jobs`,
      tone: queuedJobs > 0 ? "warm" : "idle"
    },
    {
      label: "Running jobs",
      value: runningJobs,
      total: totalJobs,
      meta: schedulerEnabled ? "auto scheduler" : "manual",
      tone: runningJobs > 0 ? "live" : "idle"
    },
    {
      label: "Live services",
      value: liveServices,
      total: totalServices,
      meta: `${liveServices}/${services.length}`,
      tone: liveServices > 0 ? "live" : "idle"
    }
  ];
  const volumeTracks: OverviewTrack[] = [
    {
      label: "GT samples",
      value: data.total_benchmark_samples,
      total: volumeTotal,
      meta: `${data.benchmark_count.toLocaleString()} bench`,
      tone: "good"
    },
    {
      label: "Predictions",
      value: data.prediction_count,
      total: volumeTotal,
      meta: `${data.run_count.toLocaleString()} runs`,
      tone: data.prediction_count > 0 ? "live" : "idle"
    },
    {
      label: "Failed jobs",
      value: failedJobs,
      total: totalJobs,
      meta: failedJobs > 0 ? "needs check" : "clear",
      tone: failedJobs > 0 ? "danger" : "idle"
    }
  ];

  return (
    <section className="page-stack dashboard-home">
      <div className="overview-console">
        <div className="overview-console-main">
          <div className="overview-title-block">
            <div className="eyebrow">Eval Bench Control</div>
            <h2>总览</h2>
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
            <PanelTitle title="任务控制台" meta="pipeline / pressure / action" />
            <OverviewNextAction action={nextAction} />
          </div>
          <OverviewPipeline stages={pipelineStages} />
          <div className="overview-operational-grid">
            <OverviewTrackGroup icon={<Gauge size={15} />} title="Run" tracks={runTracks} />
            <OverviewTrackGroup icon={<Activity size={15} />} title="Ops" tracks={opsTracks} />
            <OverviewTrackGroup icon={<BarChart3 size={15} />} title="Volume" tracks={volumeTracks} />
          </div>
          <OverviewActivityMatrix lanes={activityLanes} />
        </section>

        <aside className="overview-side-stack">
          <OverviewActionPanel
            schedulerEnabled={schedulerEnabled}
            liveServices={liveServices}
            serviceCount={services.length}
            activeQueue={activeQueue}
          />
          <OverviewRecentRunsPanel runs={data.runs.slice(0, 6)} />
        </aside>
      </div>
    </section>
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

function OverviewActionPanel({
  schedulerEnabled,
  liveServices,
  serviceCount,
  activeQueue
}: {
  schedulerEnabled: boolean;
  liveServices: number;
  serviceCount: number;
  activeQueue: number;
}) {
  const actions = [
    {
      label: "Rank Board",
      detail: "查看当前可入榜结果",
      to: "/rank-board" as OverviewRoute,
      icon: <Trophy size={15} />
    },
    {
      label: "Create Job",
      detail: activeQueue > 0 ? `${activeQueue} active jobs` : "队列空闲",
      to: "/jobs" as OverviewRoute,
      icon: <PlayCircle size={15} />
    },
    {
      label: "Services",
      detail: serviceCount > 0 ? `${liveServices}/${serviceCount} running` : "未登记服务",
      to: "/services" as OverviewRoute,
      icon: <Server size={15} />
    }
  ];
  return (
    <section className="workspace-card overview-action-panel">
      <PanelTitle title="行动入口" meta={schedulerEnabled ? "scheduler auto" : "scheduler manual"} />
      <div className="overview-action-list">
        {actions.map((action) => (
          <Link className="overview-action-link" to={action.to} key={action.label}>
            <span>{action.icon}</span>
            <div>
              <strong>{action.label}</strong>
              <em>{action.detail}</em>
            </div>
            <ArrowRight size={14} />
          </Link>
        ))}
      </div>
    </section>
  );
}

function OverviewTrackGroup({
  icon,
  title,
  tracks
}: {
  icon: React.ReactNode;
  title: string;
  tracks: OverviewTrack[];
}) {
  return (
    <div className="overview-track-group">
      <div className="overview-track-heading">
        {icon}
        <strong>{title}</strong>
      </div>
      <div className="overview-track-list">
        {tracks.map((track) => (
          <div className={`overview-track ${track.tone}`} key={track.label}>
            <div>
              <span>{track.label}</span>
              <strong>{track.value.toLocaleString()}</strong>
              <em>{track.meta}</em>
            </div>
            <div className="overview-track-rail" aria-hidden="true">
              <i style={{ width: `${trackPercent(track.value, track.total)}%` }} />
            </div>
          </div>
        ))}
      </div>
    </div>
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
