export type CompareViewState = {
  searchText: string;
  statusFilter: string;
  taskFilter: string;
  benchmarkFilter: string;
  benchmarkSplitFilter: string;
  labelFilter: string;
  modelFilter: string;
  promptFilter: string;
  historyBaselineFilter: string;
  historyCandidateFilter: string;
  pageOffset: number;
  historyOffset: number;
  baselineRunId: string;
  candidateRunId: string;
  activeLabel: string;
};

export const COMPARE_VIEW_STATE_KEY = "eval_bench_compare_view_state";
export const COMPARE_VIEW_STATE_RESET_EVENT = "eval-bench-compare-reset";

export const DEFAULT_COMPARE_VIEW_STATE: CompareViewState = {
  searchText: "",
  statusFilter: "all",
  taskFilter: "all",
  benchmarkFilter: "all",
  benchmarkSplitFilter: "all",
  labelFilter: "all",
  modelFilter: "all",
  promptFilter: "all",
  historyBaselineFilter: "",
  historyCandidateFilter: "",
  pageOffset: 0,
  historyOffset: 0,
  baselineRunId: "",
  candidateRunId: "",
  activeLabel: "all"
};

export function readCompareViewState(overrides: Partial<CompareViewState> = {}): CompareViewState {
  const storedState = readStoredCompareViewState();
  return {
    ...DEFAULT_COMPARE_VIEW_STATE,
    ...storedState,
    ...overrides,
    pageOffset: normalizePageOffset(overrides.pageOffset ?? storedState.pageOffset),
    historyOffset: normalizePageOffset(overrides.historyOffset ?? storedState.historyOffset)
  };
}

export function writeCompareViewState(state: CompareViewState) {
  if (typeof window === "undefined") {
    return;
  }
  window.sessionStorage.setItem(COMPARE_VIEW_STATE_KEY, JSON.stringify(state));
}

export function resetCompareViewState() {
  if (typeof window === "undefined") {
    return;
  }
  window.sessionStorage.removeItem(COMPARE_VIEW_STATE_KEY);
  window.dispatchEvent(new Event(COMPARE_VIEW_STATE_RESET_EVENT));
}

function readStoredCompareViewState(): Partial<CompareViewState> {
  if (typeof window === "undefined") {
    return {};
  }
  const rawValue = window.sessionStorage.getItem(COMPARE_VIEW_STATE_KEY);
  if (!rawValue) {
    return {};
  }
  try {
    return JSON.parse(rawValue) as Partial<CompareViewState>;
  } catch {
    return {};
  }
}

function normalizePageOffset(value: unknown) {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    return 0;
  }
  return Math.floor(value);
}
