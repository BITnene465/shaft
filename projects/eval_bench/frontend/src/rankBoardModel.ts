export const RANK_PRIMARY_METRICS = [
  "f1_iou50",
  "precision_iou50",
  "recall_iou50",
  "mean_iou",
  "prediction_count"
];

export const RANK_AUXILIARY_SORTS = [
  "created_at",
  "run_id"
];

export const RANK_SORTABLE_FIELDS = [...RANK_PRIMARY_METRICS, ...RANK_AUXILIARY_SORTS];
export const RANK_PAGE_SIZE = 80;

export function defaultRankSortOrder(value: string) {
  return value === "run_id" ? "asc" : "desc";
}

export function defaultSuiteSortOrder(value: string) {
  return value === "model_id" ? "asc" : "desc";
}

export function toggleSortOrder(value: string) {
  return value === "desc" ? "asc" : "desc";
}
