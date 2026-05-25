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
  Server,
  Trophy
} from "lucide-react";

import type { RunSummary } from "./api";
import { fetchJobs, fetchSchedulerStatus, fetchServices } from "./api";
import { useDashboardState } from "./dashboardState";
import { AppIcon } from "./iconLibrary";
import { Badge, EmptyState } from "./ui";

type OverviewRoute = "/" | "/rank-board" | "/runs" | "/jobs" | "/services" | "/benchmarks";
type OverviewTone = "idle" | "live" | "warm" | "good" | "danger";
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
      label: "基准样本",
      value: data.benchmark_count,
      total: Math.max(data.benchmark_count, 1),
      meta: `${data.total_benchmark_samples.toLocaleString()} samples`,
      to: "/benchmarks",
      tone: data.benchmark_count > 0 ? "good" : "warm"
    },
    {
      label: "预测产物",
      value: data.prediction_count,
      total: volumeTotal,
      meta: `${runsWithPredictions.toLocaleString()} runs ready`,
      to: "/runs",
      tone: data.prediction_count > 0 ? "live" : "idle"
    },
    {
      label: "评估报告",
      value: evaluatedRuns,
      total: totalRuns,
      meta: `${coveragePercent}% complete`,
      to: "/runs",
      tone: waitingEvaluation > 0 ? "warm" : "good"
    },
    {
      label: "排行就绪",
      value: evaluatedRuns,
      total: totalRuns,
      meta: evaluatedRuns > 0 ? "rank board ready" : "need report",
      to: "/rank-board",
      tone: evaluatedRuns > 0 ? "good" : "idle"
    }
  ];

  const signalItems: OverviewSignal[] = [
    {
      label: "覆盖",
      value: `${coveragePercent}%`,
      detail: `${evaluatedRuns.toLocaleString()} / ${data.run_count.toLocaleString()} reports`,
      to: "/runs",
      tone: evaluatedRuns > 0 ? "good" : "idle",
      icon: <CheckCircle2 size={16} />,
      progress: trackPercent(evaluatedRuns, totalRuns)
    },
    {
      label: "待评",
      value: waitingEvaluation.toLocaleString(),
      detail:
        waitingEvaluation > 0
          ? `${runsWithPredictions.toLocaleString()} runs with predictions`
          : "no report backlog",
      to: "/runs",
      tone: waitingEvaluation > 0 ? "warm" : "idle",
      icon: <Gauge size={16} />,
      progress: trackPercent(waitingEvaluation, totalRuns)
    },
    {
      label: "队列",
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
      label: "服务",
      value: `${liveServices}/${services.length}`,
      detail:
        services.length > 0
          ? `${services.length.toLocaleString()} services registered`
          : schedulerEnabled
            ? "auto scheduler waiting"
            : "manual mode",
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
  const recentRuns = overviewRecentRuns(data.runs, 3);

  return (
    <section className="page-stack dashboard-home overview-home-v9">
      <div className="overview-command-center">
        <section className={`overview-priority-stage ${nextAction.tone}`}>
          <div className="overview-priority-copy">
            <div className="overview-kicker-row">
              <span className="overview-live-dot" />
              <span>Eval Bench Control</span>
              <i className={overviewSyncing ? "overview-sync-pill syncing" : "overview-sync-pill"}>
                {overviewSyncing ? "同步中" : "已同步"}
              </i>
            </div>
            <h2>{overviewHeroTitle(nextAction)}</h2>
            <p>{postureLine}</p>
            <div className="overview-decision-strip">
              <OverviewNextAction action={nextAction} />
              <div className="overview-impact-strip" aria-label="首页关键规模">
                <span>
                  <b>{data.total_benchmark_samples.toLocaleString()}</b>
                  samples
                </span>
                <span>
                  <b>{runsWithPredictions.toLocaleString()}</b>
                  ready runs
                </span>
                <span>
                  <b>{activeQueue.toLocaleString()}</b>
                  active jobs
                </span>
              </div>
            </div>
          </div>
          <OverviewPipeline stages={pipelineStages} />
        </section>

        <aside className="overview-command-rail" aria-label="首页核心状态">
          <div className="overview-rail-head">
            <div>
              <span>Live Signals</span>
              <strong>{activeQueue > 0 ? "推进中" : failedJobs > 0 ? "有阻塞" : "可调度"}</strong>
            </div>
            <i>{overviewSyncing ? "sync" : "steady"}</i>
          </div>
          <OverviewSignalStack signals={signalItems} />
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
        </aside>
      </div>

      <div className="overview-workbench">
        <section className="overview-ops-surface">
          <div className="overview-section-head">
            <div>
              <span>{schedulerEnabled ? "Auto Scheduler" : "Manual Mode"}</span>
              <h3>下一步控制台</h3>
            </div>
            <strong>{coveragePercent}% complete</strong>
          </div>
          <OverviewReadinessList items={readinessItems} />
        </section>

        <OverviewRecentRunsPanel runs={recentRuns} />
      </div>
    </section>
  );
}

function OverviewSignalStack({ signals }: { signals: OverviewSignal[] }) {
  return (
    <div className="overview-signal-stack overview-operational-grid">
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

function overviewHeroTitle(action: OverviewAction) {
  if (action.tone === "danger") {
    return "先处理阻塞";
  }
  if (action.tone === "warm") {
    return "有待推进项";
  }
  if (action.tone === "live") {
    return "队列正在推进";
  }
  if (action.tone === "good") {
    return "可进入排行";
  }
  return "等待首个评测";
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

function OverviewReadinessList({ items }: { items: OverviewReadinessItem[] }) {
  return (
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
              {run.benchmark_id || "-"} · {run.model_id || "-"}
            </small>
          </span>
          <span className="overview-run-counts" aria-label="run 产物规模">
            <b>{run.prediction_count.toLocaleString()} pred</b>
            <b>{run.report_count.toLocaleString()} report</b>
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
      label: "处理待评估 run",
      detail: `${waitingEvaluation.toLocaleString()} 个 run 已有预测`,
      to: "/runs",
      tone: "warm",
      icon: <Gauge size={16} />
    };
  }
  if (activeQueue > 0) {
    return {
      label: "查看运行队列",
      detail: `${activeQueue.toLocaleString()} 个任务在队列中`,
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
    return `${failedJobs.toLocaleString()} 个失败任务需要先处理。`;
  }
  if (waitingEvaluation > 0) {
    return `${waitingEvaluation.toLocaleString()} 个 run 已有预测，可以立即补评估报告。`;
  }
  if (activeQueue > 0) {
    return `${activeQueue.toLocaleString()} 个任务正在排队或运行，先关注队列吞吐。`;
  }
  if (serviceCount > 0 && liveServices === 0) {
    return "已登记模型服务处于空闲状态，下一次评测前先确认服务。";
  }
  if (evaluatedRuns > 0) {
    return `${evaluatedRuns.toLocaleString()} 个 run 已有报告，可以进入排行或对比。`;
  }
  return "还没有评估报告，创建任务后首页会跟踪闭环进度。";
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
      label: "模型服务",
      detail:
        serviceCount > 0
          ? `${liveServices}/${serviceCount} 个服务在线`
          : "还未登记服务",
      to: "/services",
      tone: liveServices > 0 ? "good" : serviceCount > 0 ? "warm" : "idle",
      icon: <Server size={15} />,
      state: liveServices > 0 ? "在线" : serviceCount > 0 ? "空闲" : "空",
      value: liveServices,
      total: Math.max(serviceCount, 1)
    },
    {
      label: "任务队列",
      detail:
        failedJobs > 0
          ? `${failedJobs} 个失败任务`
          : schedulerEnabled
            ? "自动推进已开启"
            : "需要手动推进",
      to: "/jobs",
      tone: failedJobs > 0 ? "danger" : activeJobs > 0 ? "live" : "good",
      icon: <PlayCircle size={15} />,
      state: failedJobs > 0 ? "检查" : activeJobs > 0 ? "运行" : "清空",
      value: activeJobs + failedJobs,
      total: jobTotal
    },
    {
      label: "评估报告",
      detail: waitingEvaluation > 0 ? `${waitingEvaluation} 个 run 待评估` : `${evaluatedRuns} 个报告可用`,
      to: "/runs",
      tone: waitingEvaluation > 0 ? "warm" : evaluatedRuns > 0 ? "good" : "idle",
      icon: <Gauge size={15} />,
      state: waitingEvaluation > 0 ? "待评" : evaluatedRuns > 0 ? "完成" : "空",
      value: evaluatedRuns,
      total: totalRuns
    },
    {
      label: "排行榜",
      detail: evaluatedRuns > 0 ? "可以查看全局排名" : "需要先生成报告",
      to: "/rank-board",
      tone: evaluatedRuns > 0 ? "good" : "idle",
      icon: <Trophy size={15} />,
      state: evaluatedRuns > 0 ? "就绪" : "未就绪",
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
