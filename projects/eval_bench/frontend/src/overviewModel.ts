import { useQuery } from "@tanstack/react-query";

import type { JobListResponse, RunSummary, ServiceListResponse } from "./api";
import { fetchJobs, fetchSchedulerStatus, fetchServices } from "./api";
import { useDashboardState } from "./dashboardState";
import { formatMetric, runF1Score } from "./formatters";
import { recentRunsByCreatedAt } from "./runArtifactSignals";

export type OverviewRoute =
  | "/"
  | "/rank-board"
  | "/runs"
  | "/jobs"
  | "/services"
  | "/benchmarks"
  | "/compare";
export type OverviewTone = "idle" | "live" | "warm" | "good" | "danger";
export type OverviewLens = "health" | "quality" | "throughput";
export type OverviewActionIcon = "alert" | "report" | "activity" | "trophy" | "server" | "play";
export type BestRun = { run: RunSummary; f1: number };
export type OverviewAction = {
  label: string;
  value: string;
  detail: string;
  to: OverviewRoute;
  tone: OverviewTone;
  icon: OverviewActionIcon;
};

type OverviewSignalNode = {
  id: string;
  label: string;
  value: string;
  caption: string;
  detail: string;
  to: OverviewRoute;
  tone: OverviewTone;
};

export function useOverviewModel() {
  const stateQuery = useDashboardState();
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
  const data = stateQuery.data;
  const queuedJobs = jobPageTotal(queuedJobsQuery.data);
  const runningJobs = jobPageTotal(runningJobsQuery.data);
  const failedJobs = jobPageTotal(failedJobsQuery.data);
  const totalJobs = Math.max(jobPageTotal(jobTotalQuery.data), queuedJobs + runningJobs + failedJobs);
  const liveServices = servicePageTotal(runningServicesQuery.data);
  const serviceCount = Math.max(servicePageTotal(serviceTotalQuery.data), liveServices);
  const activeQueue = queuedJobs + runningJobs;
  const evaluatedRuns = data?.runs.filter((run) => run.report_path || run.report_count > 0).length ?? 0;
  const waitingEvaluation =
    data?.runs.filter((run) => !run.report_path && run.report_count === 0 && run.prediction_count > 0).length ?? 0;
  const bestRun = data ? bestF1Run(data.runs) : null;
  const recentRuns = data ? recentRunsByCreatedAt(data.runs, 6) : [];
  const reportCoverage = percent(evaluatedRuns, Math.max(data?.run_count ?? 0, 1));
  const nextAction = overviewNextAction({
    failedJobs,
    waitingEvaluation,
    activeQueue,
    liveServices,
    serviceCount,
    evaluatedRuns
  });

  return {
    data,
    isLoading: stateQuery.isLoading,
    error: stateQuery.error,
    queuedJobs,
    runningJobs,
    failedJobs,
    totalJobs,
    liveServices,
    serviceCount,
    activeQueue,
    evaluatedRuns,
    bestRun,
    recentRuns,
    reportCoverage,
    nextAction,
    schedulerEnabled: schedulerQuery.data?.enabled ?? false,
    schedulerLive: schedulerQuery.data?.loop_alive ?? false,
    workerCount: schedulerQuery.data?.active_worker_threads?.length ?? 0,
    reservedDevices: schedulerQuery.data?.reserved_cuda_devices?.length ?? 0,
    reservedPorts: schedulerQuery.data?.reserved_runtime_ports?.length ?? 0
  };
}

export function overviewLensLabel(mode: OverviewLens) {
  if (mode === "quality") {
    return "质量";
  }
  if (mode === "throughput") {
    return "吞吐";
  }
  return "健康";
}

export function overviewSignalNodes({
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
}): OverviewSignalNode[] {
  if (lens === "quality") {
    return [
      {
        id: "benchmark",
        label: "Benchmark",
        value: benchmarkCount.toLocaleString(),
        caption: `${benchmarkSamples.toLocaleString()} samples`,
        detail: "基准集决定评测覆盖面，先确认数据版本与样本规模。",
        to: "/benchmarks",
        tone: "idle"
      },
      {
        id: "prediction",
        label: "Prediction",
        value: predictionCount.toLocaleString(),
        caption: "snapshots",
        detail: "预测快照是进入报告生成与排行榜的前置资产。",
        to: "/runs",
        tone: predictionCount > 0 ? "live" : "idle"
      },
      {
        id: "report",
        label: "Report",
        value: evaluatedRuns.toLocaleString(),
        caption: `${reportCoverage}% coverage`,
        detail: bestRun
          ? `当前最佳 run 是 ${bestRun.run.run_id}，F1 ${formatMetric(bestRun.f1)}。`
          : "还没有可用于全局质量判断的 F1 报告。",
        to: "/rank-board",
        tone: evaluatedRuns > 0 ? "good" : "warm"
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
        to: "/jobs",
        tone: runningJobs > 0 ? "live" : "idle"
      },
      {
        id: "queued",
        label: "Queued",
        value: queuedJobs.toLocaleString(),
        caption: "waiting",
        detail: "排队任务反映当前吞吐压力，必要时先扩容 runtime。",
        to: "/jobs",
        tone: queuedJobs > 0 ? "warm" : "idle"
      },
      {
        id: "failed",
        label: "Failed",
        value: failedJobs.toLocaleString(),
        caption: "needs review",
        detail: failedJobs > 0 ? "失败任务会阻断全局统计可信度，优先查看日志。" : "当前没有失败任务。",
        to: "/jobs",
        tone: failedJobs > 0 ? "danger" : "good"
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
      to: "/services",
      tone: schedulerLive && liveServices > 0 ? "good" : "warm"
    },
    {
      id: "queue",
      label: "Queue",
      value: activeQueue.toLocaleString(),
      caption: "running + queued",
      detail: activeQueue > 0 ? "队列仍在推进，观察运行日志和资源占用。" : "当前没有排队或运行中的任务。",
      to: "/jobs",
      tone: activeQueue > 0 ? "live" : "idle"
    }
  ];
}

export function trackPercent(value: number, total: number) {
  if (total <= 0 || value <= 0) {
    return 0;
  }
  return Math.max(5, Math.min(100, (value / total) * 100));
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
      icon: "alert"
    };
  }
  if (waitingEvaluation > 0) {
    return {
      label: "补齐评估报告",
      value: waitingEvaluation.toLocaleString(),
      detail: `${waitingEvaluation.toLocaleString()} 个 run 已有预测但缺少 report。`,
      to: "/runs",
      tone: "warm",
      icon: "report"
    };
  }
  if (activeQueue > 0) {
    return {
      label: "观察运行队列",
      value: activeQueue.toLocaleString(),
      detail: `${activeQueue.toLocaleString()} 个任务正在排队或运行。`,
      to: "/jobs",
      tone: "live",
      icon: "activity"
    };
  }
  if (evaluatedRuns > 0) {
    return {
      label: "查看最佳结果",
      value: evaluatedRuns.toLocaleString(),
      detail: "报告已生成，继续比较模型或 prompt。",
      to: "/rank-board",
      tone: "good",
      icon: "trophy"
    };
  }
  if (serviceCount > 0 && liveServices === 0) {
    return {
      label: "启动模型服务",
      value: `${liveServices}/${serviceCount}`,
      detail: "已有服务配置，发起任务前先确认 endpoint。",
      to: "/services",
      tone: "warm",
      icon: "server"
    };
  }
  return {
    label: "创建评测任务",
    value: "0",
    detail: "还没有评估报告，从 benchmark 和 job 开始。",
    to: "/jobs",
    tone: "idle",
    icon: "play"
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
