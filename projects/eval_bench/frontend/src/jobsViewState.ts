export type JobsViewState = {
  selectedJobId: string;
  searchText: string;
  statusFilter: string;
  kindFilter: string;
  pageOffset: number;
};

export const JOBS_VIEW_STATE_KEY = "eval_bench_jobs_view_state";
export const JOBS_VIEW_STATE_RESET_EVENT = "eval-bench-jobs-reset";

export const DEFAULT_JOBS_VIEW_STATE: JobsViewState = {
  selectedJobId: "",
  searchText: "",
  statusFilter: "all",
  kindFilter: "all",
  pageOffset: 0
};

export function readJobsViewState(): JobsViewState {
  if (typeof window === "undefined") {
    return DEFAULT_JOBS_VIEW_STATE;
  }
  const rawValue = window.sessionStorage.getItem(JOBS_VIEW_STATE_KEY);
  if (!rawValue) {
    return DEFAULT_JOBS_VIEW_STATE;
  }
  try {
    const value = JSON.parse(rawValue) as Partial<JobsViewState>;
    return {
      ...DEFAULT_JOBS_VIEW_STATE,
      ...value,
      pageOffset: normalizePageOffset(value.pageOffset)
    };
  } catch {
    return DEFAULT_JOBS_VIEW_STATE;
  }
}

export function writeJobsViewState(state: JobsViewState) {
  if (typeof window === "undefined") {
    return;
  }
  window.sessionStorage.setItem(JOBS_VIEW_STATE_KEY, JSON.stringify(state));
}

export function resetJobsViewState() {
  if (typeof window === "undefined") {
    return;
  }
  window.sessionStorage.removeItem(JOBS_VIEW_STATE_KEY);
  window.dispatchEvent(new Event(JOBS_VIEW_STATE_RESET_EVENT));
}

function normalizePageOffset(value: unknown) {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    return 0;
  }
  return Math.floor(value);
}
