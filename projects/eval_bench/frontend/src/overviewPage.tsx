import React from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  Database,
  FileCheck2,
  Gauge,
  Layers3,
  PlayCircle,
  Server,
  Trophy
} from "lucide-react";

import type { JobListResponse, RunSummary, SchedulerStatus, ServiceListResponse } from "./api";
import { fetchJobs, fetchSchedulerStatus, fetchServices } from "./api";
import { useDashboardState } from "./dashboardState";
import { formatMetric, runF1Score } from "./formatters";
import { AppIcon } from "./iconLibrary";
import { recentRunsByCreatedAt, runAgeLabel, runArtifactReadiness } from "./runArtifactSignals";
import { Badge, EmptyState } from "./ui";

type OverviewRoute =
  | "/"
  | "/rank-board"
  | "/runs"
  | "/jobs"
  | "/services"
  | "/benchmarks"
  | "/compare";
type OverviewTone = "idle" | "live" | "warm" | "good" | "danger";
type OverviewAction = {
  label: string;
  value: string;
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
type OverviewTelemetryItem = {
  label: string;
  value: string;
  detail: string;
  level: number;
  tone: OverviewTone;
};
type BestRun = { run: RunSummary; f1: number };

export function OverviewPage() {
  const { data, isLoading, error } = useDashboardState();
  const jobTotalQuery = useQuery({
    queryKey: ["overview-jobs-total"],
    queryFn: () => fetchJobs({ limit: 1 }),
    refetchInterval: 2_000
  });
  const queuedJobsQuery = useQuery({
    queryKey: ["overview-jobs-queued"],
    queryFn: () => fetchJobs({ status: "queued", limit: 1 }),
    refetchInterval: 2_000
  });
  const runningJobsQuery = useQuery({
    queryKey: ["overview-jobs-running"],
    queryFn: () => fetchJobs({ status: "running", limit: 1 }),
    refetchInterval: 2_000
  });
  const failedJobsQuery = useQuery({
    queryKey: ["overview-jobs-failed"],
    queryFn: () => fetchJobs({ status: "failed", limit: 1 }),
    refetchInterval: 2_000
  });
  const serviceTotalQuery = useQuery({
    queryKey: ["overview-services-total"],
    queryFn: () => fetchServices({ limit: 1 }),
    refetchInterval: 5_000
  });
  const runningServicesQuery = useQuery({
    queryKey: ["overview-services-running"],
    queryFn: () => fetchServices({ status: "running", limit: 1 }),
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

  const schedulerStatus = schedulerQuery.data;
  const evaluatedRuns = data.runs.filter((run) => run.report_path).length;
  const runsWithPredictions = data.runs.filter((run) => run.prediction_count > 0).length;
  const waitingEvaluation = data.runs.filter(
    (run) => !run.report_path && run.prediction_count > 0
  ).length;
  const queuedJobs = jobPageTotal(queuedJobsQuery.data);
  const runningJobs = jobPageTotal(runningJobsQuery.data);
  const failedJobs = jobPageTotal(failedJobsQuery.data);
  const totalJobRecords = Math.max(
    jobPageTotal(jobTotalQuery.data),
    queuedJobs + runningJobs + failedJobs
  );
  const liveServices = servicePageTotal(runningServicesQuery.data);
  const serviceCount = Math.max(servicePageTotal(serviceTotalQuery.data), liveServices);
  const activeQueue = queuedJobs + runningJobs;
  const totalRuns = Math.max(data.run_count, 1);
  const totalJobs = Math.max(totalJobRecords, 1);
  const totalServices = Math.max(serviceCount, 1);
  const coveragePercent = percent(evaluatedRuns, totalRuns);
  const volumeTotal = Math.max(data.total_benchmark_samples, data.prediction_count, 1);
  const syncing =
    jobTotalQuery.isFetching ||
    queuedJobsQuery.isFetching ||
    runningJobsQuery.isFetching ||
    failedJobsQuery.isFetching ||
    serviceTotalQuery.isFetching ||
    runningServicesQuery.isFetching ||
    schedulerQuery.isFetching;
  const bestRun = bestF1Run(data.runs);
  const nextAction = overviewNextAction({
    failedJobs,
    waitingEvaluation,
    activeQueue,
    liveServices,
    serviceCount,
    evaluatedRuns
  });
  const postureLine = overviewPostureLine({
    failedJobs,
    waitingEvaluation,
    activeQueue,
    liveServices,
    serviceCount,
    evaluatedRuns
  });
  const decisionMetrics: OverviewSignal[] = [
    {
      label: "当前最佳",
      value: bestRun ? formatMetric(bestRun.f1) : "-",
      detail: bestRun ? bestRun.run.run_id : "暂无 F1 报告",
      to: "/rank-board",
      tone: bestRun ? "good" : "idle",
      icon: <Trophy size={16} />,
      progress: bestRun ? trackPercent(bestRun.f1, 1) : 0
    },
    {
      label: "报告闭环",
      value: `${coveragePercent}%`,
      detail: `${evaluatedRuns.toLocaleString()} / ${data.run_count.toLocaleString()} runs`,
      to: "/runs",
      tone: evaluatedRuns > 0 ? "good" : "idle",
      icon: <CheckCircle2 size={16} />,
      progress: trackPercent(evaluatedRuns, totalRuns)
    },
    {
      label: "待处理",
      value: waitingEvaluation.toLocaleString(),
      detail:
        waitingEvaluation > 0
          ? `${runsWithPredictions.toLocaleString()} runs 已有预测`
          : "暂无待补报告",
      to: "/runs",
      tone: waitingEvaluation > 0 ? "warm" : "idle",
      icon: <Gauge size={16} />,
      progress: trackPercent(waitingEvaluation, totalRuns)
    },
    {
      label: failedJobs > 0 ? "任务阻塞" : "运行压力",
      value: failedJobs > 0 ? failedJobs.toLocaleString() : `${activeQueue}/${serviceCount}`,
      detail:
        failedJobs > 0
          ? `${queuedJobs.toLocaleString()} queued / ${runningJobs.toLocaleString()} running`
          : `${liveServices.toLocaleString()} services online`,
      to: failedJobs > 0 || activeQueue > 0 ? "/jobs" : "/services",
      tone: failedJobs > 0 ? "danger" : activeQueue > 0 || liveServices > 0 ? "live" : "idle",
      icon: failedJobs > 0 ? <AlertTriangle size={16} /> : <Activity size={16} />,
      progress: trackPercent(activeQueue + failedJobs + liveServices, totalJobs + totalServices)
    }
  ];
  const telemetryItems: OverviewTelemetryItem[] = [
    {
      label: "报告覆盖",
      value: `${coveragePercent}%`,
      detail: `${evaluatedRuns.toLocaleString()} reports / ${data.run_count.toLocaleString()} runs`,
      level: coveragePercent,
      tone: evaluatedRuns > 0 ? "good" : "idle"
    },
    {
      label: "队列负载",
      value: activeQueue.toLocaleString(),
      detail: `${runningJobs.toLocaleString()} running / ${queuedJobs.toLocaleString()} queued`,
      level: trackPercent(
        activeQueue,
        Math.max(totalJobRecords, schedulerStatus?.max_concurrent_jobs ?? 1)
      ),
      tone: failedJobs > 0 ? "danger" : activeQueue > 0 ? "live" : "idle"
    },
    {
      label: "服务容量",
      value: `${liveServices}/${serviceCount}`,
      detail: serviceCount > 0 ? "registered model services" : "no service registered",
      level: trackPercent(liveServices, totalServices),
      tone: liveServices > 0 ? "live" : serviceCount > 0 ? "warm" : "idle"
    },
    {
      label: "预测积压",
      value: waitingEvaluation.toLocaleString(),
      detail: `${data.prediction_count.toLocaleString()} prediction artifacts`,
      level: trackPercent(waitingEvaluation, Math.max(runsWithPredictions, 1)),
      tone: waitingEvaluation > 0 ? "warm" : "good"
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
      label: "报告评分",
      value: evaluatedRuns.toLocaleString(),
      detail: evaluatedRuns > 0 ? "F1 populated" : "no report yet",
      to: "/rank-board",
      tone: evaluatedRuns > 0 ? "good" : "idle",
      icon: <AppIcon name="rankBoard" size={17} />,
      progress: trackPercent(evaluatedRuns, totalRuns)
    }
  ];
  const recentRuns = recentRunsByCreatedAt(data.runs, 4);

  return (
    <section
      className="page-stack dashboard-home overview-home-v17"
      onPointerMove={updateOverviewPointer}
    >
      <section className={`overview-ops-board ${nextAction.tone}`}>
        <div className="overview-decision-stage">
          <div className="overview-command-kicker">
            <span className="overview-live-dot" />
            <span>Eval Bench Command</span>
            <i className={syncing ? "overview-sync-pill syncing" : "overview-sync-pill"}>
              {syncing ? "同步中" : "已同步"}
            </i>
          </div>
          <div className="overview-decision-copy">
            <OverviewStateStrip
              evaluatedRuns={evaluatedRuns}
              waitingEvaluation={waitingEvaluation}
              activeQueue={activeQueue}
              liveServices={liveServices}
              serviceCount={serviceCount}
            />
            <p>{postureLine}</p>
            <OverviewFlowSpine stages={flowStages} />
          </div>
          <div className="overview-decision-bottom">
            <OverviewOpsSignal action={nextAction} />
            <OverviewRunFocus bestRun={bestRun} />
          </div>
        </div>
        <aside className="overview-rank-console" aria-label="主指标与系统态">
          <OverviewScoreDial bestRun={bestRun} coveragePercent={coveragePercent} />
          <OverviewDecisionMetrics metrics={decisionMetrics} />
          <OverviewTelemetryTrace items={telemetryItems} schedulerStatus={schedulerStatus} />
        </aside>
      </section>

      <OverviewRecentRunsPanel runs={recentRuns} />
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

function jobPageTotal(page?: JobListResponse) {
  return page?.total ?? page?.jobs.length ?? 0;
}

function servicePageTotal(page?: ServiceListResponse) {
  return page?.total ?? page?.services.length ?? 0;
}

function OverviewDecisionMetrics({ metrics }: { metrics: OverviewSignal[] }) {
  return (
    <div className="overview-decision-metrics" aria-label="首页核心判断指标">
      {metrics.map((metric) => (
        <Link
          className={`overview-decision-metric ${metric.tone}`}
          key={metric.label}
          to={metric.to}
          style={{ "--metric-progress": `${metric.progress}%` } as React.CSSProperties}
        >
          <span className="overview-decision-icon">{metric.icon}</span>
          <span>{metric.label}</span>
          <strong>{metric.value}</strong>
          <em>{metric.detail}</em>
          <i aria-hidden="true">
            <b />
          </i>
        </Link>
      ))}
    </div>
  );
}

function OverviewTelemetryTrace({
  items,
  schedulerStatus
}: {
  items: OverviewTelemetryItem[];
  schedulerStatus?: SchedulerStatus;
}) {
  const resourceItems = [
    {
      label: "workers",
      value: String(schedulerStatus?.active_worker_threads?.length ?? 0)
    },
    {
      label: "devices",
      value: String(schedulerStatus?.reserved_cuda_devices?.length ?? 0)
    },
    {
      label: "ports",
      value: String(schedulerStatus?.reserved_runtime_ports?.length ?? 0)
    },
    {
      label: "loop",
      value: schedulerStatus?.loop_alive ? "live" : schedulerStatus?.enabled ? "idle" : "off"
    }
  ];
  return (
    <section className="overview-telemetry-trace" aria-label="实时吞吐与资源轨迹">
      <div className="overview-telemetry-head">
        <span>Realtime Trace</span>
        <strong>{schedulerStatus?.max_concurrent_jobs ?? 0} slots</strong>
      </div>
      <div className="overview-telemetry-bars">
        {items.map((item, index) => (
          <span
            className={`overview-telemetry-bar ${item.tone}`}
            key={item.label}
            style={
              {
                "--telemetry-level": `${item.level}%`,
                "--telemetry-delay": `${index * 45}ms`
              } as React.CSSProperties
            }
          >
            <b>{item.label}</b>
            <em>{item.value}</em>
            <small>{item.detail}</small>
            <i aria-hidden="true">
              <b />
            </i>
          </span>
        ))}
      </div>
      <div className="overview-resource-chips" aria-label="scheduler resources">
        {resourceItems.map((item) => (
          <span key={item.label}>
            <b>{item.value}</b>
            <em>{item.label}</em>
          </span>
        ))}
      </div>
    </section>
  );
}

function OverviewStateStrip({
  evaluatedRuns,
  waitingEvaluation,
  activeQueue,
  liveServices,
  serviceCount
}: {
  evaluatedRuns: number;
  waitingEvaluation: number;
  activeQueue: number;
  liveServices: number;
  serviceCount: number;
}) {
  const items = [
    { label: "reports", value: evaluatedRuns.toLocaleString() },
    { label: "pending", value: waitingEvaluation.toLocaleString() },
    { label: "queue", value: activeQueue.toLocaleString() },
    { label: "services", value: `${liveServices}/${serviceCount}` }
  ];
  return (
    <div className="overview-state-strip" aria-label="总览运行状态">
      {items.map((item) => (
        <span key={item.label}>
          <b>{item.value}</b>
          <em>{item.label}</em>
        </span>
      ))}
    </div>
  );
}

function OverviewScoreDial({
  bestRun,
  coveragePercent
}: {
  bestRun: BestRun | null;
  coveragePercent: number;
}) {
  const scorePercent = bestRun ? Math.round(bestRun.f1 * 100) : 0;
  return (
    <Link
      className="overview-score-dial"
      to={bestRun ? "/rank-board" : "/runs"}
      style={
        {
          "--overview-score": `${scorePercent}%`,
          "--overview-coverage": `${coveragePercent}%`
        } as React.CSSProperties
      }
    >
      <span className="overview-score-ring">
        <b>{bestRun ? formatMetric(bestRun.f1) : "-"}</b>
        <em>F1</em>
      </span>
      <div>
        <strong>{bestRun ? bestRun.run.run_id : "等待评估报告"}</strong>
        <small>
          {bestRun ? `${bestRun.run.model_id} · ${bestRun.run.benchmark_id}` : "等待 F1 报告"}
        </small>
        <i aria-hidden="true">
          <b />
        </i>
      </div>
      <ArrowRight size={15} />
    </Link>
  );
}

function OverviewOpsSignal({ action }: { action: OverviewAction }) {
  return (
    <Link className={`overview-ops-signal ${action.tone}`} to={action.to}>
      <span>{action.icon}</span>
      <div>
        <em>{action.label}</em>
        <strong>{action.value}</strong>
        <small>{action.detail}</small>
      </div>
      <ArrowRight size={16} />
    </Link>
  );
}

function OverviewRunFocus({ bestRun }: { bestRun: BestRun | null }) {
  if (!bestRun) {
    return (
      <Link className="overview-run-focus empty" to="/runs">
        <span>
          <AppIcon name="runResults" size={18} />
        </span>
        <div>
          <strong>暂无评估报告</strong>
          <em>导入预测并完成评估后显示主指标。</em>
        </div>
      </Link>
    );
  }
  const run = bestRun.run;
  return (
    <Link className="overview-run-focus" to="/runs/$runId" params={{ runId: run.run_id }}>
      <span>
        <Trophy size={18} />
      </span>
      <div>
        <strong>{run.run_id}</strong>
        <em>{run.model_id || "-"} · {run.benchmark_id || "-"}</em>
      </div>
      <dl>
        <div>
          <dt>pred</dt>
          <dd>{run.prediction_count.toLocaleString()}</dd>
        </div>
        <div>
          <dt>report</dt>
          <dd>{run.report_count.toLocaleString()}</dd>
        </div>
        <div>
          <dt>age</dt>
          <dd>{runAgeLabel(run.created_at)}</dd>
        </div>
      </dl>
    </Link>
  );
}

function OverviewFlowSpine({ stages }: { stages: OverviewFlowStage[] }) {
  return (
    <nav className="overview-flow-spine" aria-label="评测闭环">
      {stages.map((stage, index) => (
        <Link
          className={`overview-flow-node ${stage.tone}`}
          to={stage.to}
          key={stage.label}
          style={
            {
              "--stage-progress": `${stage.progress}%`,
              "--stage-delay": `${index * 55}ms`
            } as React.CSSProperties
          }
        >
          <span>{stage.icon}</span>
          <div>
            <strong>{stage.label}</strong>
            <em>{stage.value}</em>
            <small>{stage.detail}</small>
          </div>
          <i aria-hidden="true">
            <b />
          </i>
        </Link>
      ))}
    </nav>
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
        const f1 = runF1Score(run);
        return (
          <Link
            className={readiness.tone}
            key={run.run_id}
            to="/runs/$runId"
            params={{ runId: run.run_id }}
            style={
              {
                "--run-readiness": `${readiness.percent}%`,
                "--run-delay": `${index * 45}ms`
              } as React.CSSProperties
            }
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
                <b />
              </i>
              <small>{readiness.label}</small>
            </span>
            <span className="overview-run-counts" aria-label="run 指标与产物规模">
              <b>{f1 === null ? "F1 -" : `F1 ${formatMetric(f1)}`}</b>
              <b>{run.prediction_count.toLocaleString()} pred</b>
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
      label: "失败任务",
      value: failedJobs.toLocaleString(),
      detail: `${failedJobs.toLocaleString()} 个任务需要处理`,
      to: "/jobs",
      tone: "danger",
      icon: <AlertTriangle size={16} />
    };
  }
  if (waitingEvaluation > 0) {
    return {
      label: "待评估 run",
      value: waitingEvaluation.toLocaleString(),
      detail: `${waitingEvaluation.toLocaleString()} 个 run 已有预测`,
      to: "/runs",
      tone: "warm",
      icon: <Gauge size={16} />
    };
  }
  if (activeQueue > 0) {
    return {
      label: "运行队列",
      value: activeQueue.toLocaleString(),
      detail: `${activeQueue.toLocaleString()} 个任务在推进`,
      to: "/jobs",
      tone: "live",
      icon: <Activity size={16} />
    };
  }
  if (evaluatedRuns > 0) {
    return {
      label: "评估报告",
      value: evaluatedRuns.toLocaleString(),
      detail: "报告指标已写入",
      to: "/rank-board",
      tone: "good",
      icon: <Trophy size={16} />
    };
  }
  if (serviceCount > 0 && liveServices === 0) {
    return {
      label: "模型服务",
      value: `${liveServices}/${serviceCount}`,
      detail: "已登记服务当前空闲",
      to: "/services",
      tone: "warm",
      icon: <Server size={16} />
    };
  }
  return {
    label: "评测任务",
    value: "0",
    detail: "还没有可用报告",
    to: "/jobs",
    tone: "idle",
    icon: <PlayCircle size={16} />
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
  if (evaluatedRuns > 0) {
    return `${evaluatedRuns.toLocaleString()} 份报告已生成，当前最佳与覆盖状态正在刷新。`;
  }
  if (serviceCount > 0 && liveServices === 0) {
    return "模型服务已登记但当前空闲，发起任务前先确认运行时。";
  }
  return "还没有评估报告，创建任务后首页会跟踪样本、预测和报告闭环。";
}

function bestF1Run(runs: RunSummary[]) {
  return runs.reduce<BestRun | null>((best, run) => {
    const f1 = runF1Score(run);
    if (f1 === null) {
      return best;
    }
    if (!best || f1 > best.f1) {
      return { run, f1 };
    }
    return best;
  }, null);
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
