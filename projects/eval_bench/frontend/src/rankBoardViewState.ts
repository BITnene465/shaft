export type RankBoardViewState = {
  boardMode: "run" | "suite";
  searchText: string;
  taskFilter: string;
  benchmarkFilter: string;
  benchmarkSplitFilter: string;
  statusFilter: string;
  labelFilter: string;
  modelFilter: string;
  promptFilter: string;
  metricProfileFilter: string;
  minScoreFilter: string;
  sortBy: string;
  sortOrder: string;
  pageOffset: number;
};

export const RANK_BOARD_VIEW_STATE_KEY = "eval_bench_rank_board_view_state";
export const RANK_BOARD_VIEW_STATE_RESET_EVENT = "eval-bench-rank-board-reset";

export const DEFAULT_RANK_BOARD_VIEW_STATE: RankBoardViewState = {
  boardMode: "run",
  searchText: "",
  taskFilter: "all",
  benchmarkFilter: "all",
  benchmarkSplitFilter: "all",
  statusFilter: "all",
  labelFilter: "all",
  modelFilter: "all",
  promptFilter: "all",
  metricProfileFilter: "all",
  minScoreFilter: "",
  sortBy: "f1_iou50",
  sortOrder: "desc",
  pageOffset: 0
};

export function readRankBoardViewState(): RankBoardViewState {
  if (typeof window === "undefined") {
    return DEFAULT_RANK_BOARD_VIEW_STATE;
  }
  const rawValue = window.sessionStorage.getItem(RANK_BOARD_VIEW_STATE_KEY);
  if (!rawValue) {
    return DEFAULT_RANK_BOARD_VIEW_STATE;
  }
  try {
    const value = JSON.parse(rawValue) as Partial<RankBoardViewState>;
    return {
      ...DEFAULT_RANK_BOARD_VIEW_STATE,
      ...value,
      boardMode: value.boardMode === "suite" ? "suite" : "run",
      pageOffset: normalizePageOffset(value.pageOffset)
    };
  } catch {
    return DEFAULT_RANK_BOARD_VIEW_STATE;
  }
}

export function writeRankBoardViewState(state: RankBoardViewState) {
  if (typeof window === "undefined") {
    return;
  }
  window.sessionStorage.setItem(RANK_BOARD_VIEW_STATE_KEY, JSON.stringify(state));
}

export function resetRankBoardViewState() {
  if (typeof window === "undefined") {
    return;
  }
  window.sessionStorage.removeItem(RANK_BOARD_VIEW_STATE_KEY);
  window.dispatchEvent(new Event(RANK_BOARD_VIEW_STATE_RESET_EVENT));
}

function normalizePageOffset(value: unknown) {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    return 0;
  }
  return Math.floor(value);
}
