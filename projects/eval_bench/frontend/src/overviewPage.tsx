import React from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  ClipboardList,
  Database,
  FileCheck2,
  Layers3,
  PlayCircle,
  Server,
  Trophy
} from "lucide-react";

import type { JobListResponse, RunSummary, ServiceListResponse } from "./api";
import { fetchJobs, fetchSchedulerStatus, fetchServices } from "./api";
import { useDashboardState } from "./dashboardState";
import { errorMessage, formatMetric, runF1Score } from "./formatters";
import { recentRunsByCreatedAt, runAgeLabel, runArtifactReadiness } from "./runArtifactSignals";
import { Badge, EmptyState, OptionChipButton } from "./ui";

type OverviewRoute =
  | "/"
  | "/rank-board"
  | "/runs"
  | "/jobs"
  | "/services"
  | "/benchmarks"
  | "/compare";
type OverviewTone = "idle" | "live" | "warm" | "good" | "danger";
type OverviewLens = "health" | "quality" | "throughput";
type BestRun = { run: RunSummary; f1: number };
type OverviewAction = {
  label: string;
  value: string;
  detail: string;
  to: OverviewRoute;
  tone: OverviewTone;
  icon: React.ReactNode;
};

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
    return <EmptyState title={`看板状态加载失败：${errorMessage(error)}`} tone="danger" />;
  }

  const evaluatedRuns = data.runs.filter((run) => run.report_path || run.report_count > 0).length;
  const runsWithPredictions = data.runs.filter((run) => run.prediction_count > 0).length;
  const waitingEvaluation = data.runs.filter(
    (run) => !run.report_path && run.report_count === 0 && run.prediction_count > 0
  ).length;
  const queuedJobs = jobPageTotal(queuedJobsQuery.data);
  const runningJobs = jobPageTotal(runningJobsQuery.data);
  const failedJobs = jobPageTotal(failedJobsQuery.data);
  const totalJobs = Math.max(jobPageTotal(jobTotalQuery.data), queuedJobs + runningJobs + failedJobs);
  const liveServices = servicePageTotal(runningServicesQuery.data);
  const serviceCount = Math.max(servicePageTotal(serviceTotalQuery.data), liveServices);
  const activeQueue = queuedJobs + runningJobs;
  const bestRun = bestF1Run(data.runs);
  const recentRuns = recentRunsByCreatedAt(data.runs, 6);
  const reportCoverage = percent(evaluatedRuns, Math.max(data.run_count, 1));
  const nextAction = overviewNextAction({
    failedJobs,
    waitingEvaluation,
    activeQueue,
    liveServices,
    serviceCount,
    evaluatedRuns
  });
  return (
    <section className="page-stack dashboard-home overview-home-v18">
      <div className="overview-v18-grid">
        <OverviewPrimaryCard
          action={nextAction}
          bestRun={bestRun}
          reportCoverage={reportCoverage}
          benchmarkCount={data.benchmark_count}
          benchmarkSamples={data.total_benchmark_samples}
          predictionCount={data.prediction_count}
          evaluatedRuns={evaluatedRuns}
          queuedJobs={queuedJobs}
          runningJobs={runningJobs}
          failedJobs={failedJobs}
          activeQueue={activeQueue}
          liveServices={liveServices}
          serviceCount={serviceCount}
          schedulerLive={schedulerQuery.data?.loop_alive ?? false}
        />
        <OverviewQueueCard
          activeQueue={activeQueue}
          queuedJobs={queuedJobs}
          runningJobs={runningJobs}
          failedJobs={failedJobs}
          totalJobs={totalJobs}
        />
        <OverviewRecentRunsPanel runs={recentRuns} />
        <OverviewResourceCard
          liveServices={liveServices}
          serviceCount={serviceCount}
          workerCount={schedulerQuery.data?.active_worker_threads?.length ?? 0}
          reservedDevices={schedulerQuery.data?.reserved_cuda_devices?.length ?? 0}
          reservedPorts={schedulerQuery.data?.reserved_runtime_ports?.length ?? 0}
          schedulerEnabled={schedulerQuery.data?.enabled ?? false}
          schedulerLive={schedulerQuery.data?.loop_alive ?? false}
        />
      </div>
    </section>
  );
}

function OverviewPrimaryCard({
  action,
  bestRun,
  reportCoverage,
  benchmarkCount,
  benchmarkSamples,
  predictionCount,
  evaluatedRuns,
  queuedJobs,
  runningJobs,
  failedJobs,
  activeQueue,
  liveServices,
  serviceCount,
  schedulerLive
}: {
  action: OverviewAction;
  bestRun: BestRun | null;
  reportCoverage: number;
  benchmarkCount: number;
  benchmarkSamples: number;
  predictionCount: number;
  evaluatedRuns: number;
  queuedJobs: number;
  runningJobs: number;
  failedJobs: number;
  activeQueue: number;
  liveServices: number;
  serviceCount: number;
  schedulerLive: boolean;
}) {
  const actionLine = bestRun
    ? `当前最佳 ${bestRun.run.run_id}，F1 ${formatMetric(bestRun.f1)}。`
    : "还没有可用 F1 报告。";
  const [lens, setLens] = React.useState<OverviewLens>("health");
  const signalNodes = overviewSignalNodes({
    lens,
    action,
    bestRun,
    benchmarkCount,
    benchmarkSamples,
    predictionCount,
    evaluatedRuns,
    reportCoverage,
    queuedJobs,
    runningJobs,
    failedJobs,
    activeQueue,
    liveServices,
    serviceCount,
    schedulerLive
  });
  const [selectedSignalId, setSelectedSignalId] = React.useState(signalNodes[0]?.id ?? "");
  const selectedSignal = signalNodes.find((node) => node.id === selectedSignalId) ?? signalNodes[0];
  React.useEffect(() => {
    if (!signalNodes.some((node) => node.id === selectedSignalId)) {
      setSelectedSignalId(signalNodes[0]?.id ?? "");
    }
  }, [signalNodes, selectedSignalId]);
  return (
    <section className={`overview-v18-card overview-v18-primary ${action.tone}`}>
      <div className="overview-v18-card-head">
        <div>
          <span>Next Action</span>
          <h3>{action.label}</h3>
        </div>
        <Link to={action.to} className="overview-v18-icon-link" aria-label={action.label}>
          {action.icon}
          <ArrowRight size={15} />
        </Link>
      </div>
      <div className="overview-v18-primary-body">
        <div className="overview-v18-action-copy">
          <strong>{action.value}</strong>
          <p>{action.detail}</p>
          <small>{actionLine}</small>
        </div>
        <Link
          to={bestRun ? "/rank-board" : "/runs"}
          className="overview-v18-score"
          style={{ "--overview-score": `${bestRun ? Math.round(bestRun.f1 * 100) : 0}%` } as React.CSSProperties}
        >
          <span>
            <b>{bestRun ? formatMetric(bestRun.f1) : "-"}</b>
            <em>F1</em>
          </span>
          <i>
            <b />
          </i>
        </Link>
      </div>
      <div className="overview-v18-console" aria-label="首页控制台">
        <div className="overview-v18-console-head">
          <span>Control surface</span>
          <div role="group" aria-label="切换总览视角">
            {(["health", "quality", "throughput"] as OverviewLens[]).map((mode) => (
              <OptionChipButton
                key={mode}
                active={lens === mode}
                className="overview-v18-surface-tab"
                onClick={() => setLens(mode)}
              >
                {overviewLensLabel(mode)}
              </OptionChipButton>
            ))}
          </div>
        </div>
        <div className="overview-v18-surface-body">
          <div className="overview-v18-signal-map" role="list" aria-label={`${overviewLensLabel(lens)}状态地图`}>
            {signalNodes.map((node, index) => (
              <OptionChipButton
                key={node.id}
                active={selectedSignal?.id === node.id}
                className={`overview-v18-signal-node ${node.tone}`}
                onClick={() => setSelectedSignalId(node.id)}
              >
                <span>{node.label}</span>
                <strong>{node.value}</strong>
                <em>{node.caption}</em>
                {index < signalNodes.length - 1 ? <i aria-hidden="true" /> : null}
              </OptionChipButton>
            ))}
          </div>
          <div className={`overview-v18-signal-inspector ${selectedSignal?.tone ?? "idle"}`}>
            <span>{selectedSignal?.label ?? "总览"}</span>
            <strong>{selectedSignal?.value ?? "-"}</strong>
            <p>{selectedSignal?.detail ?? "暂无可展示的全局信号。"}</p>
            <Link to={selectedSignal?.to ?? action.to}>
              打开
              <ArrowRight size={13} />
            </Link>
          </div>
        </div>
      </div>
      <div className="overview-v18-flow" aria-label="评测闭环">
        <OverviewFlowItem
          icon={<Database size={16} />}
          label="Benchmarks"
          value={benchmarkCount.toLocaleString()}
          detail={`${benchmarkSamples.toLocaleString()} samples`}
          to="/benchmarks"
        />
        <OverviewFlowItem
          icon={<Layers3 size={16} />}
          label="Predictions"
          value={predictionCount.toLocaleString()}
          detail="prediction snapshots"
          to="/runs"
        />
        <OverviewFlowItem
          icon={<FileCheck2 size={16} />}
          label="Reports"
          value={evaluatedRuns.toLocaleString()}
          detail={`${reportCoverage}% coverage`}
          to="/rank-board"
        />
      </div>
    </section>
  );
}

function overviewLensLabel(mode: OverviewLens) {
  if (mode === "quality") {
    return "质量";
  }
  if (mode === "throughput") {
    return "吞吐";
  }
  return "健康";
}

function overviewSignalNodes({
  lens,
  action,
  bestRun,
  benchmarkCount,
  benchmarkSamples,
  predictionCount,
  evaluatedRuns,
  reportCoverage,
  queuedJobs,
  runningJobs,
  failedJobs,
  activeQueue,
  liveServices,
  serviceCount,
  schedulerLive
}: {
  lens: OverviewLens;
  action: OverviewAction;
  bestRun: BestRun | null;
  benchmarkCount: number;
  benchmarkSamples: number;
  predictionCount: number;
  evaluatedRuns: number;
  reportCoverage: number;
  queuedJobs: number;
  runningJobs: number;
  failedJobs: number;
  activeQueue: number;
  liveServices: number;
  serviceCount: number;
  schedulerLive: boolean;
}) {
  if (lens === "quality") {
    return [
      {
        id: "benchmark",
        label: "Benchmark",
        value: benchmarkCount.toLocaleString(),
        caption: `${benchmarkSamples.toLocaleString()} samples`,
        detail: "基准集决定评测覆盖面，先确认数据版本与样本规模。",
        to: "/benchmarks" as OverviewRoute,
        tone: "idle" as OverviewTone
      },
      {
        id: "prediction",
        label: "Prediction",
        value: predictionCount.toLocaleString(),
        caption: "snapshots",
        detail: "预测快照是进入报告生成与排行榜的前置资产。",
        to: "/runs" as OverviewRoute,
        tone: predictionCount > 0 ? "live" as OverviewTone : "idle" as OverviewTone
      },
      {
        id: "report",
        label: "Report",
        value: evaluatedRuns.toLocaleString(),
        caption: `${reportCoverage}% coverage`,
        detail: bestRun
          ? `当前最佳 run 是 ${bestRun.run.run_id}，F1 ${formatMetric(bestRun.f1)}。`
          : "还没有可用于全局质量判断的 F1 报告。",
        to: "/rank-board" as OverviewRoute,
        tone: evaluatedRuns > 0 ? "good" as OverviewTone : "warm" as OverviewTone
      }
    ];
  }
  if (lens === "throughput") {
    return [
      {
        id: "running",
        label: "Running",
        value: runningJobs.toLocaleString(),
        caption: "active jobs",
        detail: "正在执行的任务会持续占用 worker、端口和设备资源。",
        to: "/jobs" as OverviewRoute,
        tone: runningJobs > 0 ? "live" as OverviewTone : "idle" as OverviewTone
      },
      {
        id: "queued",
        label: "Queued",
        value: queuedJobs.toLocaleString(),
        caption: "waiting",
        detail: "排队任务反映当前吞吐压力，必要时先扩容 runtime。",
        to: "/jobs" as OverviewRoute,
        tone: queuedJobs > 0 ? "warm" as OverviewTone : "idle" as OverviewTone
      },
      {
        id: "failed",
        label: "Failed",
        value: failedJobs.toLocaleString(),
        caption: "needs review",
        detail: failedJobs > 0 ? "失败任务会阻断全局统计可信度，优先查看日志。" : "当前没有失败任务。",
        to: "/jobs" as OverviewRoute,
        tone: failedJobs > 0 ? "danger" as OverviewTone : "good" as OverviewTone
      }
    ];
  }
  return [
    {
      id: "priority",
      label: "Priority",
      value: action.value,
      caption: action.label,
      detail: action.detail,
      to: action.to,
      tone: action.tone
    },
    {
      id: "runtime",
      label: "Runtime",
      value: `${liveServices}/${serviceCount}`,
      caption: schedulerLive ? "scheduler live" : "scheduler idle",
      detail: "服务与调度器决定评测任务能否稳定推进。",
      to: "/services" as OverviewRoute,
      tone: schedulerLive && liveServices > 0 ? "good" as OverviewTone : "warm" as OverviewTone
    },
    {
      id: "queue",
      label: "Queue",
      value: activeQueue.toLocaleString(),
      caption: "running + queued",
      detail: activeQueue > 0 ? "队列仍在推进，观察运行日志和资源占用。" : "当前没有排队或运行中的任务。",
      to: "/jobs" as OverviewRoute,
      tone: activeQueue > 0 ? "live" as OverviewTone : "idle" as OverviewTone
    }
  ];
}

function OverviewFlowItem({
  icon,
  label,
  value,
  detail,
  to
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  detail: string;
  to: OverviewRoute;
}) {
  return (
    <Link className="overview-v18-flow-item" to={to}>
      <span>{icon}</span>
      <strong>{value}</strong>
      <em>{label}</em>
      <small>{detail}</small>
    </Link>
  );
}

function OverviewQueueCard({
  activeQueue,
  queuedJobs,
  runningJobs,
  failedJobs,
  totalJobs
}: {
  activeQueue: number;
  queuedJobs: number;
  runningJobs: number;
  failedJobs: number;
  totalJobs: number;
}) {
  return (
    <section className={failedJobs > 0 ? "overview-v18-card overview-v18-queue danger" : "overview-v18-card overview-v18-queue"}>
      <div className="overview-v18-card-head">
        <div>
          <span>Queue</span>
          <h3>任务队列</h3>
        </div>
        <Link to="/jobs" className="overview-v18-icon-link" aria-label="打开任务队列">
          <ClipboardList size={17} />
          <ArrowRight size={15} />
        </Link>
      </div>
      <strong className="overview-v18-big-number">{activeQueue.toLocaleString()}</strong>
      <div className="overview-v18-meter" style={{ "--meter": `${trackPercent(activeQueue, Math.max(totalJobs, 1))}%` } as React.CSSProperties}>
        <b />
      </div>
      <dl className="overview-v18-pairs">
        <div>
          <dt>running</dt>
          <dd>{runningJobs.toLocaleString()}</dd>
        </div>
        <div>
          <dt>queued</dt>
          <dd>{queuedJobs.toLocaleString()}</dd>
        </div>
        <div>
          <dt>failed</dt>
          <dd>{failedJobs.toLocaleString()}</dd>
        </div>
      </dl>
    </section>
  );
}

function OverviewRecentRunsPanel({ runs }: { runs: RunSummary[] }) {
  return (
    <section className="overview-v18-card overview-v18-recent">
      <div className="overview-v18-card-head">
        <div>
          <span>Latest Runs</span>
          <h3>最近 run</h3>
        </div>
        <Link to="/runs" className="overview-v18-icon-link" aria-label="打开 run 列表">
          <Activity size={17} />
          <ArrowRight size={15} />
        </Link>
      </div>
      {runs.length === 0 ? <div className="empty-inline">暂无 run。</div> : <OverviewRunList runs={runs} />}
    </section>
  );
}

function OverviewRunList({ runs }: { runs: RunSummary[] }) {
  return (
    <div className="overview-v18-run-list">
      {runs.map((run) => {
        const readiness = runArtifactReadiness(run);
        const f1 = runF1Score(run);
        return (
          <Link key={run.run_id} to="/runs/$runId" params={{ runId: run.run_id }} className={readiness.tone}>
            <span className="overview-v18-run-id">
              <strong className="run-id-text">{run.run_id}</strong>
              <small>{run.model_id || "-"}</small>
            </span>
            <span className="overview-v18-run-artifacts">
              <i style={{ "--run-readiness": `${readiness.percent}%` } as React.CSSProperties}>
                <b />
              </i>
              <small>{readiness.label}</small>
            </span>
            <span className="overview-v18-run-score">
              <b>{f1 === null ? "F1 -" : `F1 ${formatMetric(f1)}`}</b>
              <small>{runAgeLabel(run.created_at)}</small>
            </span>
            <Badge value={run.status} domain="run" />
          </Link>
        );
      })}
    </div>
  );
}

function OverviewResourceCard({
  liveServices,
  serviceCount,
  workerCount,
  reservedDevices,
  reservedPorts,
  schedulerEnabled,
  schedulerLive
}: {
  liveServices: number;
  serviceCount: number;
  workerCount: number;
  reservedDevices: number;
  reservedPorts: number;
  schedulerEnabled: boolean;
  schedulerLive: boolean;
}) {
  return (
    <section className="overview-v18-card overview-v18-resources">
      <div className="overview-v18-card-head">
        <div>
          <span>Runtime</span>
          <h3>服务与资源</h3>
        </div>
        <Link to="/services" className="overview-v18-icon-link" aria-label="打开服务管理">
          <Server size={17} />
          <ArrowRight size={15} />
        </Link>
      </div>
      <div className="overview-v18-service-line">
        <strong>{liveServices}/{serviceCount}</strong>
        <span>services running</span>
      </div>
      <dl className="overview-v18-pairs">
        <div>
          <dt>workers</dt>
          <dd>{workerCount}</dd>
        </div>
        <div>
          <dt>devices</dt>
          <dd>{reservedDevices}</dd>
        </div>
        <div>
          <dt>ports</dt>
          <dd>{reservedPorts}</dd>
        </div>
      </dl>
      <Link to="/jobs" className={schedulerLive ? "overview-v18-scheduler live" : "overview-v18-scheduler"}>
        {schedulerLive ? <CheckCircle2 size={15} /> : <PlayCircle size={15} />}
        <span>{schedulerLive ? "scheduler live" : schedulerEnabled ? "scheduler idle" : "scheduler off"}</span>
      </Link>
    </section>
  );
}

function jobPageTotal(page?: JobListResponse) {
  return page?.total ?? page?.jobs.length ?? 0;
}

function servicePageTotal(page?: ServiceListResponse) {
  return page?.total ?? page?.services.length ?? 0;
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
      label: "处理失败任务",
      value: failedJobs.toLocaleString(),
      detail: `${failedJobs.toLocaleString()} 个任务失败，先看错误日志。`,
      to: "/jobs",
      tone: "danger",
      icon: <AlertTriangle size={17} />
    };
  }
  if (waitingEvaluation > 0) {
    return {
      label: "补齐评估报告",
      value: waitingEvaluation.toLocaleString(),
      detail: `${waitingEvaluation.toLocaleString()} 个 run 已有预测但缺少 report。`,
      to: "/runs",
      tone: "warm",
      icon: <FileCheck2 size={17} />
    };
  }
  if (activeQueue > 0) {
    return {
      label: "观察运行队列",
      value: activeQueue.toLocaleString(),
      detail: `${activeQueue.toLocaleString()} 个任务正在排队或运行。`,
      to: "/jobs",
      tone: "live",
      icon: <Activity size={17} />
    };
  }
  if (evaluatedRuns > 0) {
    return {
      label: "查看最佳结果",
      value: evaluatedRuns.toLocaleString(),
      detail: "报告已生成，继续比较模型或 prompt。",
      to: "/rank-board",
      tone: "good",
      icon: <Trophy size={17} />
    };
  }
  if (serviceCount > 0 && liveServices === 0) {
    return {
      label: "启动模型服务",
      value: `${liveServices}/${serviceCount}`,
      detail: "已有服务配置，发起任务前先确认 endpoint。",
      to: "/services",
      tone: "warm",
      icon: <Server size={17} />
    };
  }
  return {
    label: "创建评测任务",
    value: "0",
    detail: "还没有评估报告，从 benchmark 和 job 开始。",
    to: "/jobs",
    tone: "idle",
    icon: <PlayCircle size={17} />
  };
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
