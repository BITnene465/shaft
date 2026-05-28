import type { FacetBuckets, RunSummary, ServiceSummary } from "./api";

export function formatDate(value: string | null) {
  if (!value) {
    return "-";
  }
  return value.replace("T", " ").replace("Z", "");
}

export function formatMetric(value: number | null) {
  if (value === null || Number.isNaN(value)) {
    return "-";
  }
  return value.toFixed(3);
}

export function f1Score(precision: number | null, recall: number | null) {
  if (precision === null || recall === null) {
    return null;
  }
  const denominator = precision + recall;
  if (denominator <= 0) {
    return 0;
  }
  return (2 * precision * recall) / denominator;
}

export function runF1Score(
  run: Pick<RunSummary, "f1_iou50" | "precision_iou50" | "recall_iou50">
) {
  if (typeof run.f1_iou50 === "number") {
    return run.f1_iou50;
  }
  return f1Score(run.precision_iou50, run.recall_iou50);
}

export function formatSignedMetric(value: number) {
  const prefix = value > 0 ? "+" : "";
  return `${prefix}${value.toFixed(3)}`;
}

export function formatCompactSignedMetric(value: number) {
  const prefix = value > 0 ? "+" : "";
  const absValue = Math.abs(value);
  const digits = absValue >= 100 ? 0 : absValue >= 10 ? 1 : 2;
  return `${prefix}${value.toFixed(digits)}`;
}

export function formatSignedInteger(value: number) {
  const prefix = value > 0 ? "+" : "";
  return `${prefix}${value.toLocaleString()}`;
}

export function jobTarget(payload: Record<string, unknown>) {
  const model = typeof payload.model_id === "string" ? payload.model_id : "model";
  const benchmark =
    typeof payload.benchmark_id === "string" ? payload.benchmark_id : "benchmark";
  const task = typeof payload.task === "string" ? payload.task : "task";
  return `${model} / ${benchmark} / ${task}`;
}

export function basename(path: string) {
  const parts = path.split("/");
  return parts[parts.length - 1] || path;
}

export function unique(values: string[]) {
  return Array.from(new Set(values)).sort((left, right) => left.localeCompare(right));
}

export function facetValues(
  facets: FacetBuckets | undefined,
  key: string,
  fallback: string[] = []
) {
  const values = facets?.[key]?.map((item) => item.value).filter(Boolean) ?? [];
  return values.length > 0 ? values : unique(fallback);
}

export function isTextInputTarget(target: EventTarget | null) {
  if (!(target instanceof HTMLElement)) {
    return false;
  }
  const tag = target.tagName.toLowerCase();
  return tag === "input" || tag === "textarea" || tag === "select" || target.isContentEditable;
}

export function formatRunOption(run: RunSummary) {
  const parts = [run.run_id];
  if (run.benchmark_id) {
    parts.push(`${run.benchmark_id}:${run.benchmark_split || "-"}`);
  }
  parts.push(run.model_id, `F1 ${formatMetric(runF1Score(run))}`);
  return parts.join(" / ");
}

export function runIdExists(runs: RunSummary[], runId: string) {
  return runs.some((run) => run.run_id === runId);
}

export function stringValue(value: unknown) {
  return typeof value === "string" ? value : "";
}

export function inferenceValue(inference: Record<string, unknown>, key: string) {
  const value = inference[key];
  if (value === null || value === undefined || value === "") {
    return "-";
  }
  return String(value);
}

export function serviceConfigValue(service: ServiceSummary, key: string) {
  const value = service.config[key];
  if (value === null || value === undefined || value === "") {
    return "-";
  }
  return String(value);
}

export function runtimeValue(service: ServiceSummary, key: string) {
  const value = service.runtime[key];
  if (value === null || value === undefined || value === "") {
    return "-";
  }
  return String(value);
}

export function serviceHealth(service: ServiceSummary) {
  const health = service.runtime.health;
  if (!health || typeof health !== "object" || Array.isArray(health)) {
    return {
      ok: false,
      status: "unchecked",
      message: "health has not been checked",
      checkedAt: "-"
    };
  }
  const payload = health as Record<string, unknown>;
  return {
    ok: payload.ok === true,
    status: stringValue(payload.status) || "unchecked",
    message: stringValue(payload.message) || "-",
    checkedAt: formatDate(stringValue(payload.checked_at) || null)
  };
}

export function serviceEndpointValue(service: ServiceSummary) {
  const configured = serviceConfigValue(service, "endpoint");
  if (configured !== "-") {
    return configured;
  }
  const runtime = runtimeValue(service, "endpoint");
  if (runtime !== "-") {
    return runtime;
  }
  const host = serviceConfigValue(service, "host");
  const port = serviceConfigValue(service, "port");
  return port === "-" ? "-" : `http://${host === "-" ? "127.0.0.1" : host}:${port}`;
}

export function pixelBudgetValue(inference: Record<string, unknown>) {
  const minPixels = inferenceValue(inference, "min_pixels");
  const maxPixels = inferenceValue(inference, "max_pixels");
  if (minPixels === "-" && maxPixels === "-") {
    return "-";
  }
  return `${minPixels} / ${maxPixels}`;
}

export function samplingValue(inference: Record<string, unknown>) {
  return `T ${inferenceValue(inference, "temperature")} / top_p ${inferenceValue(inference, "top_p")}`;
}

export function runSampleHref(runId: string, sampleIndex: number) {
  return `/runs/${encodeURIComponent(runId)}?sample=${sampleIndex}`;
}

export function comparisonSampleHref(
  baselineRunId: string,
  candidateRunId: string,
  sampleIndex: number,
  indexes: { baselineIndex?: number | null; candidateIndex?: number | null } = {}
) {
  const params = new URLSearchParams();
  if (indexes.baselineIndex !== undefined && indexes.baselineIndex !== null) {
    params.set("baseline", String(indexes.baselineIndex));
  }
  if (indexes.candidateIndex !== undefined && indexes.candidateIndex !== null) {
    params.set("candidate", String(indexes.candidateIndex));
  }
  const query = params.toString();
  const path = `/compare/${encodeURIComponent(baselineRunId)}/${encodeURIComponent(
    candidateRunId
  )}/${sampleIndex}`;
  return query ? `${path}?${query}` : path;
}
