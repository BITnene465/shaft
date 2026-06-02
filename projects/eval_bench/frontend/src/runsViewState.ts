export type RunsViewState = {
  searchText: string;
  statusFilter: string;
  taskFilter: string;
  benchmarkFilter: string;
  benchmarkSplitFilter: string;
  labelFilter: string;
  modelFilter: string;
  promptFilter: string;
  metricProfileFilter: string;
  pageOffset: number;
};

export const RUNS_VIEW_STATE_KEY = "eval_bench_runs_view_state";
export const RUNS_VIEW_STATE_RESET_EVENT = "eval-bench-runs-reset";

export const DEFAULT_RUNS_VIEW_STATE: RunsViewState = {
  searchText: "",
  statusFilter: "all",
  taskFilter: "all",
  benchmarkFilter: "all",
  benchmarkSplitFilter: "all",
  labelFilter: "all",
  modelFilter: "all",
  promptFilter: "all",
  metricProfileFilter: "all",
  pageOffset: 0
};

export function readRunsViewState(): RunsViewState {
  if (typeof window === "undefined") {
    return DEFAULT_RUNS_VIEW_STATE;
  }
  const rawValue = window.sessionStorage.getItem(RUNS_VIEW_STATE_KEY);
  if (!rawValue) {
    return DEFAULT_RUNS_VIEW_STATE;
  }
  try {
    const value = JSON.parse(rawValue) as Partial<RunsViewState>;
    return {
      ...DEFAULT_RUNS_VIEW_STATE,
      ...value,
      pageOffset: normalizePageOffset(value.pageOffset)
    };
  } catch {
    return DEFAULT_RUNS_VIEW_STATE;
  }
}

export function writeRunsViewState(state: RunsViewState) {
  if (typeof window === "undefined") {
    return;
  }
  window.sessionStorage.setItem(RUNS_VIEW_STATE_KEY, JSON.stringify(state));
}

export function resetRunsViewState() {
  if (typeof window === "undefined") {
    return;
  }
  window.sessionStorage.removeItem(RUNS_VIEW_STATE_KEY);
  window.dispatchEvent(new Event(RUNS_VIEW_STATE_RESET_EVENT));
}

function normalizePageOffset(value: unknown) {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    return 0;
  }
  return Math.floor(value);
}
