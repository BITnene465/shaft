import type { RunSummary } from "./api";
import { inferLayerName, isRun } from "./compositeReportModel";
import type { LayerSlot } from "./compositeReportModel";

export type LayerFilter = "all" | "layout" | "arrow" | "shape" | "icon_image";

export type ReportGroup = {
  key: string;
  title: string;
  subtitle: string;
  slots: LayerSlot[];
};

export const LAYER_FILTERS: Array<{ value: LayerFilter; label: string }> = [
  { value: "all", label: "全部" },
  { value: "layout", label: "Layout" },
  { value: "arrow", label: "Arrow" },
  { value: "shape", label: "Shape" },
  { value: "icon_image", label: "Icon" }
];

export function filterReportRuns(runs: RunSummary[], query: string, layerFilter: LayerFilter) {
  const normalizedQuery = query.trim().toLocaleLowerCase();
  return runs.filter((run) => {
    const layer = inferLayerName(run);
    if (layerFilter !== "all" && layer !== layerFilter) {
      return false;
    }
    if (!normalizedQuery) {
      return true;
    }
    const haystack = [
      run.run_id,
      run.model_id,
      run.benchmark_id,
      run.benchmark_split,
      run.spec_task,
      run.prompt_id,
      ...run.target_labels,
      ...run.tasks
    ]
      .join(" ")
      .toLocaleLowerCase();
    return haystack.includes(normalizedQuery);
  });
}

export function groupSlots(slots: LayerSlot[], runById: Map<string, RunSummary>): ReportGroup[] {
  const groups = new Map<string, ReportGroup>();
  slots.forEach((slot) => {
    const run = runById.get(slot.runId);
    const suite = run?.suite_ids[0] ?? "ad-hoc";
    const task = run?.spec_task || run?.tasks[0] || "task";
    const benchmark = run?.benchmark_id || "benchmark";
    const key = `${suite}::${benchmark}::${task}`;
    const existing =
      groups.get(key) ??
      {
        key,
        title: `${task} / ${benchmark}`,
        subtitle: `${suite} · ${run?.benchmark_split || "split"}`,
        slots: []
      };
    existing.slots.push(slot);
    groups.set(key, existing);
  });
  return [...groups.values()];
}

export function pickLayerPreset(runs: RunSummary[], layers: string[]) {
  return layers
    .map((layer) => runs.find((run) => inferLayerName(run) === layer))
    .filter(isRun);
}

export function layerIndex(layer: string) {
  return Math.max(
    0,
    ["layout", "arrow", "shape", "icon_image"].findIndex((item) => item === layer)
  );
}

export function fallbackRun(runId: string): RunSummary {
  return {
    run_id: runId,
    status: "",
    benchmark_id: "",
    benchmark_split: "",
    tasks: [],
    spec_task: "",
    target_labels: [],
    model_id: "",
    model_path: "",
    prompt_id: "",
    prompt_path: null,
    prompt_hash: null,
    prompt_metadata: {},
    parser: "",
    metric_profile: "",
    visualization_profile: "",
    inference: {},
    created_at: null,
    prediction_count: 0,
    report_count: 0,
    manifest_path: "",
    report_path: null,
    benchmark_type: "",
    benchmark_official: false,
    integrity_status: "",
    integrity_reason: "",
    suite_ids: [],
    note: "",
    note_updated_at: null,
    note_max_length: 0,
    f1_iou50: null,
    precision_iou50: null,
    recall_iou50: null,
    mean_iou: null
  };
}
