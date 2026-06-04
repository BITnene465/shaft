import type { CompositeSampleLayer, RunSummary } from "./api";

export const SAMPLE_MAX_FALLBACK = 999;
export const DEFAULT_SLOT_COUNT = 2;

export const VIEW_MODES = [
  { value: "diff", label: "Diff" },
  { value: "prediction", label: "Pred" },
  { value: "gt", label: "GT" }
] as const;

export type ViewMode = (typeof VIEW_MODES)[number]["value"];

export type LayerSlot = {
  id: string;
  layer: string;
  runId: string;
  visible: boolean;
  showGt: boolean;
  showPred: boolean;
};

export type ActiveLayerConfig = LayerSlot & {
  key: string;
};

export function defaultLayerSlots(runs: RunSummary[]) {
  return runs.slice(0, DEFAULT_SLOT_COUNT).map((run, index) => ({
    id: `slot_${index}`,
    layer: inferLayerName(run),
    runId: run.run_id,
    visible: true,
    showGt: true,
    showPred: true
  }));
}

export function runOptionLabel(run: RunSummary) {
  const split = run.benchmark_split || run.spec_task || "run";
  return `${split} · ${run.model_id} · ${run.run_id}`;
}

export function inferLayerName(run: RunSummary) {
  const text = `${run.benchmark_split} ${run.target_labels.join(" ")} ${run.spec_task}`.toLowerCase();
  if (text.includes("layout")) {
    return "layout";
  }
  if (text.includes("arrow")) {
    return "arrow";
  }
  if (text.includes("shape")) {
    return "shape";
  }
  if (text.includes("icon")) {
    return "icon_image";
  }
  return sanitizeLayerName(run.benchmark_split || run.spec_task || run.run_id);
}

export function sanitizeLayerName(value: string) {
  return value.trim().replace(/[^a-zA-Z0-9_.-]+/g, "_") || "layer";
}

export function uniqueLayerKey(layer: string, existing: Record<string, string>, index: number) {
  const base = sanitizeLayerName(layer);
  if (!existing[base]) {
    return base;
  }
  return `${base}_${index + 1}`;
}

export function isRun(value: RunSummary | undefined): value is RunSummary {
  return Boolean(value);
}

export function combinedLayerInstances(
  layers: CompositeSampleLayer[],
  layerConfigs: ReadonlyArray<ActiveLayerConfig>
) {
  const configByLayer = new Map(layerConfigs.map((config) => [config.key, config]));
  return {
    gtInstances: layers.flatMap((layer) =>
      configByLayer.get(layer.layer)?.showGt === false
        ? []
        : layer.gt_instances.map((instance) => ({
            ...instance,
            label: `${layer.layer}:${instance.label}`
          }))
    ),
    predInstances: layers.flatMap((layer) =>
      configByLayer.get(layer.layer)?.showPred === false
        ? []
        : layer.pred_instances.map((instance) => ({
            ...instance,
            label: `${layer.layer}:${instance.label}`
          }))
    )
  };
}
