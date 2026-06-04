import type { RunSummary } from "./api";
import { LAYER_FILTERS } from "./compositeReportComposerModel";
import type { LayerFilter } from "./compositeReportComposerModel";
import { defaultLayerSlots } from "./compositeReportModel";
import type { LayerSlot } from "./compositeReportModel";

const COMPOSITE_REPORT_VIEW_STATE_KEY = "eval_bench_composite_report_view";
const DEFAULT_COMPOSITE_REPORT_VIEW_STATE: CompositeReportViewState = {
  slots: [],
  sampleIndex: 0,
  focusedLayerKey: null,
  query: "",
  layerFilter: "all",
  sidebarOpen: false
};

export type CompositeReportViewState = {
  slots: LayerSlot[];
  sampleIndex: number;
  focusedLayerKey: string | null;
  query: string;
  layerFilter: LayerFilter;
  sidebarOpen: boolean;
};

export function loadCompositeReportViewState(): CompositeReportViewState {
  try {
    const raw = localStorage.getItem(COMPOSITE_REPORT_VIEW_STATE_KEY);
    if (!raw) {
      return DEFAULT_COMPOSITE_REPORT_VIEW_STATE;
    }
    return normalizeCompositeReportViewState(JSON.parse(raw) as Partial<CompositeReportViewState>);
  } catch {
    return DEFAULT_COMPOSITE_REPORT_VIEW_STATE;
  }
}

export function saveCompositeReportViewState(state: CompositeReportViewState) {
  localStorage.setItem(
    COMPOSITE_REPORT_VIEW_STATE_KEY,
    JSON.stringify(normalizeCompositeReportViewState(state))
  );
}

export function normalizeCompositeReportViewState(
  value: Partial<CompositeReportViewState>
): CompositeReportViewState {
  return {
    slots: normalizeLayerSlots(value.slots),
    sampleIndex: normalizeSampleIndex(value.sampleIndex),
    focusedLayerKey: normalizeNullableString(value.focusedLayerKey),
    query: normalizeText(value.query, 180),
    layerFilter: normalizeLayerFilter(value.layerFilter),
    sidebarOpen: false
  };
}

export function reconcileCompositeReportSlots(slots: LayerSlot[], runs: RunSummary[]) {
  if (runs.length === 0) {
    return slots;
  }
  const runIds = new Set(runs.map((run) => run.run_id));
  const validSlots = slots.filter((slot) => runIds.has(slot.runId));
  return validSlots.length > 0 ? validSlots : defaultLayerSlots(runs);
}

export function sameLayerSlots(left: LayerSlot[], right: LayerSlot[]) {
  if (left.length !== right.length) {
    return false;
  }
  return left.every((slot, index) => sameLayerSlot(slot, right[index]));
}

function normalizeLayerSlots(value: unknown) {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .map((item, index) => normalizeLayerSlot(item, index))
    .filter((slot): slot is LayerSlot => Boolean(slot));
}

function normalizeLayerSlot(value: unknown, index: number): LayerSlot | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  const record = value as Record<string, unknown>;
  const layer = normalizeText(record.layer, 80);
  const runId = normalizeText(record.runId, 240);
  if (!layer || !runId) {
    return null;
  }
  return {
    id: normalizeText(record.id, 120) || `saved_slot_${index}`,
    layer,
    runId,
    visible: record.visible !== false,
    showGt: record.showGt !== false,
    showPred: record.showPred !== false
  };
}

function normalizeSampleIndex(value: unknown) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? Math.max(0, Math.floor(numeric)) : 0;
}

function normalizeLayerFilter(value: unknown): LayerFilter {
  return LAYER_FILTERS.some((filter) => filter.value === value) ? (value as LayerFilter) : "all";
}

function normalizeNullableString(value: unknown) {
  const normalized = normalizeText(value, 120);
  return normalized || null;
}

function normalizeText(value: unknown, maxLength: number) {
  if (typeof value !== "string") {
    return "";
  }
  return value.trim().slice(0, maxLength);
}

function sameLayerSlot(left: LayerSlot, right: LayerSlot) {
  return (
    left.id === right.id &&
    left.layer === right.layer &&
    left.runId === right.runId &&
    left.visible === right.visible &&
    left.showGt === right.showGt &&
    left.showPred === right.showPred
  );
}
