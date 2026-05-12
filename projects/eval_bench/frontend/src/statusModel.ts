import type { JobSummary, RunSummary, ServiceSummary } from "./api";

export type StatusDomain = "job" | "run" | "service" | "task" | "generic";
export type StatusTone = "success" | "danger" | "warning" | "info" | "neutral" | "muted";
export type LifecyclePhase = "pending" | "active" | "ready" | "terminal" | "inactive";

export type StatusDefinition = {
  label: string;
  tone: StatusTone;
  phase: LifecyclePhase;
  live?: boolean;
};

const GENERIC_STATUS_DEFINITIONS: Record<string, StatusDefinition> = {
  queued: { label: "排队中", tone: "warning", phase: "pending" },
  running: { label: "运行中", tone: "info", phase: "active", live: true },
  starting: { label: "启动中", tone: "info", phase: "active", live: true },
  succeeded: { label: "成功", tone: "success", phase: "terminal" },
  failed: { label: "失败", tone: "danger", phase: "terminal" },
  cancelled: { label: "已取消", tone: "muted", phase: "terminal" },
  stopped: { label: "已停止", tone: "muted", phase: "inactive" },
  registered: { label: "已登记", tone: "neutral", phase: "inactive" },
  imported: { label: "已导入", tone: "success", phase: "ready" },
  archived: { label: "已归档", tone: "muted", phase: "terminal" },
  detection: { label: "检测", tone: "neutral", phase: "ready" },
  keypoint: { label: "关键点", tone: "neutral", phase: "ready" }
};

const DOMAIN_OVERRIDES: Partial<Record<StatusDomain, Record<string, StatusDefinition>>> = {
  service: {
    running: { label: "服务就绪", tone: "success", phase: "ready", live: true },
    failed: { label: "服务异常", tone: "danger", phase: "terminal" }
  },
  run: {
    succeeded: { label: "已评估", tone: "success", phase: "ready" },
    imported: { label: "待评估", tone: "warning", phase: "pending" }
  }
};

export function statusInfo(value: string, domain: StatusDomain = "generic"): StatusDefinition {
  return (
    DOMAIN_OVERRIDES[domain]?.[value] ??
    GENERIC_STATUS_DEFINITIONS[value] ?? {
      label: value,
      tone: "neutral",
      phase: "inactive"
    }
  );
}

export function statusClassName(value: string, domain: StatusDomain = "generic") {
  const info = statusInfo(value, domain);
  return ["badge", info.tone, info.live ? "live" : ""].filter(Boolean).join(" ");
}

export function canCancelJob(job: JobSummary) {
  return job.status === "queued";
}

export function canDeleteJob(job: JobSummary) {
  return job.status !== "running";
}

export function canEvaluateRun(run: RunSummary) {
  return run.status !== "archived" && run.prediction_count > 0;
}

export function canArchiveRun(run: RunSummary) {
  return run.status !== "archived";
}

export function canDeleteRun(run: RunSummary) {
  return run.status !== "running";
}

export function canStartService(service: ServiceSummary) {
  return service.kind === "local_vllm" && ["registered", "stopped", "failed"].includes(service.status);
}

export function canStopService(service: ServiceSummary) {
  return ["starting", "running"].includes(service.status);
}

export function canDeleteService(service: ServiceSummary) {
  return !["starting", "running"].includes(service.status);
}

export function jobProgress(job: JobSummary) {
  const metadata = job.metadata ?? {};
  const done = metadataNumber(metadata.progress_done);
  const total = metadataNumber(metadata.progress_total);
  const phase = typeof metadata.progress_phase === "string" ? metadata.progress_phase : job.status;
  const message = typeof metadata.progress_message === "string" ? metadata.progress_message : "";
  const currentSample =
    typeof metadata.progress_current_sample === "string" ? metadata.progress_current_sample : "";
  const percent =
    total && total > 0 && done !== null
      ? Math.max(0, Math.min(100, Math.round((done / total) * 100)))
      : job.status === "succeeded"
        ? 100
        : null;
  const text =
    total && total > 0 && done !== null
      ? `${done}/${total} (${percent}%)`
      : progressPhaseText(phase);
  return { currentSample, done, message, percent, phase, text, total };
}

export function progressPhaseText(value: string) {
  const labels: Record<string, string> = {
    resolving: "解析配置",
    worker_starting: "启动后台 worker",
    starting_runtime: "启动模型服务",
    runtime_ready: "模型服务就绪",
    prepare_run: "准备 run",
    inference: "推理中",
    evaluating: "计算指标",
    succeeded: "完成",
    failed: "失败",
    running: "运行中",
    queued: "排队中"
  };
  return labels[value] ?? value;
}

function metadataNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}
