import React from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BarChart3,
  CheckCircle2,
  Clock3,
  Database,
  FileCheck2,
  Gauge,
  Layers3,
  PlayCircle,
  Server,
  Trophy
} from "lucide-react";

import type { RunSummary } from "./api";
import { fetchJobs, fetchSchedulerStatus, fetchServices } from "./api";
import { useDashboardState } from "./dashboardState";
import { AppIcon } from "./iconLibrary";
import { recentRunsByCreatedAt, runAgeLabel, runArtifactReadiness } from "./runArtifactSignals";
import { Badge, EmptyState } from "./ui";

type OverviewRoute = "/" | "/rank-board" | "/runs" | "/jobs" | "/services" | "/benchmarks" | "/compare";
type OverviewTone = "idle" | "live" | "warm" | "good" | "danger";
type OverviewAction = {
  label: string;
  detail: string;
  to: OverviewRoute;
  tone: OverviewTone;
  icon: React.ReactNode;
};
type OverviewSignal = OverviewAction & {
  value: string;
  progress: number;
};
type OverviewFlowStage = {
  label: string;
  value: string;
  detail: string;
  to: OverviewRoute;
  tone: OverviewTone;
  progress: number;
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
  const queuedJobs = jobs.filter((job) => job.status === "queued").length;
  const runningJobs = jobs.filter((job) => job.status === "running").length;
  const failedJobs = jobs.filter((job) => job.status === "failed").length;
  const liveServices = services.filter((service) => service.status === "running").length;
  const activeQueue = queuedJobs + runningJobs;
  const totalRuns = Math.max(data.run_count, 1);
  const totalJobs = Math.max(jobs.length, 1);
  const totalServices = Math.max(services.length, 1);
  const coveragePercent = percent(evaluatedRuns, totalRuns);
  const volumeTotal = Math.max(data.total_benchmark_samples, data.prediction_count, 1);
  const schedulerEnabled = Boolean(schedulerStatus?.enabled);
  const syncing = jobsQuery.isFetching || servicesQuery.isFetching || schedulerQuery.isFetching;
  const nextAction = overviewNextAction({
    failedJobs,
    waitingEvaluation,
    activeQueue,
    liveServices,
    serviceCount: services.length,
    evaluatedRuns
  });
  const postureLine = overviewPostureLine({
    failedJobs,
    waitingEvaluation,
    activeQueue,
    liveServices,
    serviceCount: services.length,
    evaluatedRuns
  });
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
      detail:
        waitingEvaluation > 0
          ? `${runsWithPredictions.toLocaleString()} runs 有预测`
          : "当前无待评估产物",
      to: "/runs",
      tone: waitingEvaluation > 0 ? "warm" : "idle",
      icon: <Gauge size={16} />,
      progress: trackPercent(waitingEvaluation, totalRuns)
    },
    {
      label: "任务队列",
      value: activeQueue.toLocaleString(),
      detail:
        failedJobs > 0
          ? `${failedJobs.toLocaleString()} failed`
          : `${queuedJobs.toLocaleString()} queued / ${runningJobs.toLocaleString()} running`,
      to: "/jobs",
      tone: failedJobs > 0 ? "danger" : activeQueue > 0 ? "live" : "idle",
      icon: activeQueue > 0 ? <Activity size={16} /> : <Clock3 size={16} />,
      progress: trackPercent(activeQueue + failedJobs, totalJobs)
    },
    {
      label: "模型服务",
      value: `${liveServices}/${services.length}`,
      detail:
        services.length > 0
          ? `${services.length.toLocaleString()} services registered`
          : schedulerEnabled
            ? "调度器等待服务"
            : "手动模式",
      to: "/services",
      tone: liveServices > 0 ? "live" : services.length > 0 ? "warm" : "idle",
      icon: <Server size={16} />,
      progress: trackPercent(liveServices, totalServices)
    }
  ];
  const flowStages: OverviewFlowStage[] = [
    {
      label: "基准样本",
      value: data.total_benchmark_samples.toLocaleString(),
      detail: `${data.benchmark_count.toLocaleString()} benchmarks`,
      to: "/benchmarks",
      tone: data.benchmark_count > 0 ? "good" : "warm",
      icon: <Database size={17} />,
      progress: trackPercent(data.total_benchmark_samples, volumeTotal)
    },
    {
      label: "预测产物",
      value: data.prediction_count.toLocaleString(),
      detail: `${runsWithPredictions.toLocaleString()} runs ready`,
      to: "/runs",
      tone: data.prediction_count > 0 ? "live" : "idle",
      icon: <Layers3 size={17} />,
      progress: trackPercent(data.prediction_count, volumeTotal)
    },
    {
      label: "评估报告",
      value: evaluatedRuns.toLocaleString(),
      detail: `${coveragePercent}% complete`,
      to: "/runs",
      tone: waitingEvaluation > 0 ? "warm" : evaluatedRuns > 0 ? "good" : "idle",
      icon: <FileCheck2 size={17} />,
      progress: trackPercent(evaluatedRuns, totalRuns)
    },
    {
      label: "排行榜",
      value: evaluatedRuns > 0 ? "ready" : "empty",
      detail: evaluatedRuns > 0 ? "进入主指标排行" : "等待报告",
      to: "/rank-board",
      tone: evaluatedRuns > 0 ? "good" : "idle",
      icon: <AppIcon name="rankBoard" size={17} />,
      progress: trackPercent(evaluatedRuns, totalRuns)
    }
  ];
  const routeActions = overviewRouteActions({
    nextAction,
    activeQueue,
    failedJobs,
    waitingEvaluation,
    evaluatedRuns,
    liveServices,
    serviceCount: services.length
  });
  const recentRuns = recentRunsByCreatedAt(data.runs, 5);

  return (
    <section
      className="page-stack dashboard-home overview-home-v12"
      onPointerMove={updateOverviewPointer}
    >
      <div className="overview-workband primary">
        <section className={`overview-hero-board ${nextAction.tone}`}>
          <div className="overview-hero-copy">
            <div className="overview-kicker-row">
              <span className="overview-live-dot" />
              <span>Eval Bench Control</span>
              <i className={syncing ? "overview-sync-pill syncing" : "overview-sync-pill"}>
                {syncing ? "同步中" : "已同步"}
              </i>
            </div>
            <h2>{overviewHeroTitle(nextAction)}</h2>
            <p>{postureLine}</p>
            <OverviewNextAction action={nextAction} />
          </div>
          <OverviewFlowSpine stages={flowStages} />
        </section>

        <aside className="overview-signal-board" aria-label="运行信号">
          <div className="overview-section-head">
            <div>
              <span>Live Signals</span>
              <h3>{activeQueue > 0 ? "推进中" : failedJobs > 0 ? "有阻塞" : "可调度"}</h3>
            </div>
            <strong>{syncing ? "sync" : "steady"}</strong>
          </div>
          <OverviewSignalStack signals={signalItems} />
        </aside>
      </div>

      <div className="overview-workband secondary">
        <OverviewRecentRunsPanel runs={recentRuns} />
        <section className="overview-route-panel" aria-label="下一步工作区">
          <div className="overview-section-head compact">
            <div>
              <span>Routes</span>
              <h3>下一步工作区</h3>
            </div>
            <strong>{routeActions.length}</strong>
          </div>
          <OverviewRouteList actions={routeActions} />
        </section>
      </div>
    </section>
  );
}

function updateOverviewPointer(event: React.PointerEvent<HTMLElement>) {
  const bounds = event.currentTarget.getBoundingClientRect();
  if (bounds.width <= 0 || bounds.height <= 0) {
    return;
  }
  const x = ((event.clientX - bounds.left) / bounds.width) * 100;
  const y = ((event.clientY - bounds.top) / bounds.height) * 100;
  event.currentTarget.style.setProperty("--overview-pointer-x", `${x.toFixed(2)}%`);
  event.currentTarget.style.setProperty("--overview-pointer-y", `${y.toFixed(2)}%`);
}

function OverviewSignalStack({ signals }: { signals: OverviewSignal[] }) {
  return (
    <div className="overview-signal-stack">
      {signals.map((signal) => (
        <Link className={`overview-signal-card ${signal.tone}`} to={signal.to} key={signal.label}>
          <span>{signal.icon}</span>
          <div>
            <strong>{signal.value}</strong>
            <em>{signal.label}</em>
            <small>{signal.detail}</small>
          </div>
          <i aria-hidden="true">
            <b style={{ width: `${signal.progress}%` }} />
          </i>
        </Link>
      ))}
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

function OverviewFlowSpine({ stages }: { stages: OverviewFlowStage[] }) {
  return (
    <nav className="overview-flow-spine" aria-label="评测闭环">
      {stages.map((stage) => (
        <Link className={`overview-flow-node ${stage.tone}`} to={stage.to} key={stage.label}>
          <span>{stage.icon}</span>
          <div>
            <strong>{stage.label}</strong>
            <em>{stage.value}</em>
            <small>{stage.detail}</small>
          </div>
          <i aria-hidden="true">
            <b style={{ width: `${stage.progress}%` }} />
          </i>
        </Link>
      ))}
    </nav>
  );
}

function OverviewRouteList({ actions }: { actions: OverviewAction[] }) {
  return (
    <div className="overview-route-list">
      {actions.map((action) => (
        <Link className={`overview-route-link ${action.tone}`} to={action.to} key={action.label}>
          <span>{action.icon}</span>
          <div>
            <strong>{action.label}</strong>
            <em>{action.detail}</em>
          </div>
          <ArrowRight size={15} />
        </Link>
      ))}
    </div>
  );
}

function OverviewRunList({ runs }: { runs: RunSummary[] }) {
  if (runs.length === 0) {
    return <div className="empty-inline">暂无 run。</div>;
  }
  return (
    <div className="overview-run-list">
      {runs.map((run, index) => {
        const readiness = runArtifactReadiness(run);
        return (
          <Link
            className={readiness.tone}
            key={run.run_id}
            to="/runs/$runId"
            params={{ runId: run.run_id }}
          >
            <em>{String(index + 1).padStart(2, "0")}</em>
            <span className="overview-run-main">
              <strong>{run.run_id}</strong>
              <small>
                {run.benchmark_id || "-"} · {run.model_id || "-"}
              </small>
            </span>
            <span className="overview-run-artifacts" aria-label="run 产物完成度">
              <i>
                <b style={{ width: `${readiness.percent}%` }} />
              </i>
              <small>{readiness.label}</small>
            </span>
            <span className="overview-run-counts" aria-label="run 产物规模">
              <b>{run.prediction_count.toLocaleString()} pred</b>
              <b>{run.report_count > 0 ? `${run.report_count.toLocaleString()} report` : "待评"}</b>
            </span>
            <span className="overview-run-state">
              <small>{runAgeLabel(run.created_at)}</small>
              <Badge value={run.status} domain="run" />
            </span>
          </Link>
        );
      })}
    </div>
  );
}

function OverviewRecentRunsPanel({ runs }: { runs: RunSummary[] }) {
  return (
    <section className="overview-recent-card">
      <div className="overview-section-head compact">
        <div>
          <span>Latest Runs</span>
          <h3>最近产物</h3>
        </div>
        <strong>{runs.length}</strong>
      </div>
      <OverviewRunList runs={runs} />
    </section>
  );
}

function overviewHeroTitle(action: OverviewAction) {
  if (action.tone === "danger") {
    return "先处理阻塞";
  }
  if (action.tone === "warm") {
    return "补齐评估闭环";
  }
  if (action.tone === "live") {
    return "队列正在推进";
  }
  if (action.tone === "good") {
    return "可以看排行";
  }
  return "创建首个评测";
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
      detail: `${failedJobs.toLocaleString()} 个任务需要处理`,
      to: "/jobs",
      tone: "danger",
      icon: <AlertTriangle size={16} />
    };
  }
  if (waitingEvaluation > 0) {
    return {
      label: "补评估报告",
      detail: `${waitingEvaluation.toLocaleString()} 个 run 已有预测`,
      to: "/runs",
      tone: "warm",
      icon: <Gauge size={16} />
    };
  }
  if (activeQueue > 0) {
    return {
      label: "查看运行队列",
      detail: `${activeQueue.toLocaleString()} 个任务在推进`,
      to: "/jobs",
      tone: "live",
      icon: <Activity size={16} />
    };
  }
  if (serviceCount > 0 && liveServices === 0) {
    return {
      label: "启动模型服务",
      detail: "已登记服务当前空闲",
      to: "/services",
      tone: "warm",
      icon: <Server size={16} />
    };
  }
  return {
    label: evaluatedRuns > 0 ? "查看排行榜" : "创建评测任务",
    detail: evaluatedRuns > 0 ? "报告已可排名" : "还没有可用报告",
    to: evaluatedRuns > 0 ? "/rank-board" : "/jobs",
    tone: evaluatedRuns > 0 ? "good" : "idle",
    icon: evaluatedRuns > 0 ? <Trophy size={16} /> : <PlayCircle size={16} />
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
    return `${failedJobs.toLocaleString()} 个失败任务正在阻塞产物更新。`;
  }
  if (waitingEvaluation > 0) {
    return `${waitingEvaluation.toLocaleString()} 个 run 已有预测，下一步是生成评估报告。`;
  }
  if (activeQueue > 0) {
    return `${activeQueue.toLocaleString()} 个任务正在排队或运行，关注队列吞吐即可。`;
  }
  if (serviceCount > 0 && liveServices === 0) {
    return "模型服务已登记但当前空闲，发起任务前先确认运行时。";
  }
  if (evaluatedRuns > 0) {
    return `${evaluatedRuns.toLocaleString()} 个 run 已有报告，可以进入排行或对比。`;
  }
  return "还没有评估报告，创建任务后首页会跟踪从样本到排行的闭环。";
}

function overviewRouteActions({
  nextAction,
  activeQueue,
  failedJobs,
  waitingEvaluation,
  evaluatedRuns,
  liveServices,
  serviceCount
}: {
  nextAction: OverviewAction;
  activeQueue: number;
  failedJobs: number;
  waitingEvaluation: number;
  evaluatedRuns: number;
  liveServices: number;
  serviceCount: number;
}): OverviewAction[] {
  const actions: OverviewAction[] = [
    nextAction,
    {
      label: "评测中心",
      detail:
        failedJobs > 0
          ? `${failedJobs.toLocaleString()} failed`
          : `${activeQueue.toLocaleString()} active jobs`,
      to: "/jobs",
      tone: failedJobs > 0 ? "danger" : activeQueue > 0 ? "live" : "idle",
      icon: <PlayCircle size={16} />
    },
    {
      label: "结果库",
      detail:
        waitingEvaluation > 0
          ? `${waitingEvaluation.toLocaleString()} runs 待评估`
          : `${evaluatedRuns.toLocaleString()} reports`,
      to: "/runs",
      tone: waitingEvaluation > 0 ? "warm" : evaluatedRuns > 0 ? "good" : "idle",
      icon: <FileCheck2 size={16} />
    },
    {
      label: evaluatedRuns > 0 ? "排行榜" : "模型服务",
      detail:
        evaluatedRuns > 0
          ? "主指标排行与差距"
          : `${liveServices}/${serviceCount} services online`,
      to: evaluatedRuns > 0 ? "/rank-board" : "/services",
      tone: evaluatedRuns > 0 ? "good" : liveServices > 0 ? "live" : "warm",
      icon: evaluatedRuns > 0 ? <Trophy size={16} /> : <Server size={16} />
    },
    {
      label: "对比分析",
      detail: evaluatedRuns > 1 ? "复查模型差异" : "等待更多报告",
      to: "/compare",
      tone: evaluatedRuns > 1 ? "good" : "idle",
      icon: <BarChart3 size={16} />
    }
  ];
  const deduped = new Map<string, OverviewAction>();
  for (const action of actions) {
    deduped.set(`${action.to}:${action.label}`, action);
  }
  return [...deduped.values()].slice(0, 4);
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
