import type { ColumnDef } from "@tanstack/react-table";

import type { ComparisonSummary, RunSummary } from "./api";
import { FormSelectControl } from "./controlPrimitives";
import {
  formatDate,
  formatMetric,
  formatRunOption,
  formatSignedMetric,
  runF1Score,
  runIdExists
} from "./formatters";
import { PagerControl } from "./samplePager";
import { Badge, DataTable } from "./ui";

const COMPARISON_HISTORY_COLUMNS: ColumnDef<ComparisonSummary>[] = [
  { header: "对比记录", accessorKey: "comparison_id", meta: { width: "id" } },
  { header: "任务", accessorKey: "task", meta: { width: "compact" } },
  { header: "基准集", accessorKey: "benchmark_id", meta: { width: "id" } },
  { header: "Split", accessorKey: "benchmark_split", meta: { width: "id" } },
  {
    header: "Label",
    meta: { width: "wide", wrap: "wrap" },
    cell: ({ row }) => row.original.target_labels?.join(", ") || "all"
  },
  {
    header: "风险",
    meta: { width: "compact" },
    cell: ({ row }) => {
      const warnings = row.original.warnings ?? [];
      return warnings.length ? (
        <span className="badge warning" title={warnings.join("\n")}>
          {warnings.length} warning
        </span>
      ) : (
        <span className="muted-text">-</span>
      );
    }
  },
  {
    header: "样本数",
    meta: { width: "number", align: "end" },
    cell: ({ row }) => row.original.sample_count.toLocaleString()
  },
  {
    header: "Delta R",
    meta: { width: "metric", align: "end" },
    cell: ({ row }) => formatSignedMetric(row.original.delta.recall_iou50)
  },
  {
    header: "提升",
    meta: { width: "number", align: "end" },
    cell: ({ row }) => row.original.summary.improved_samples.toLocaleString()
  },
  {
    header: "退化",
    meta: { width: "number", align: "end" },
    cell: ({ row }) => row.original.summary.regressed_samples.toLocaleString()
  },
  { header: "创建时间", meta: { width: "date" }, cell: ({ row }) => formatDate(row.original.created_at) }
];

export function RunSelectRail({
  title,
  value,
  runs,
  disabled,
  onChange
}: {
  title: string;
  value: string;
  runs: RunSummary[];
  disabled: boolean;
  onChange: (value: string) => void;
}) {
  const selected = disabled ? undefined : runs.find((run) => run.run_id === value);
  const runOptions = disabled
    ? [{ value: "", label: "需要两个报告", disabled: true }]
    : [
        ...(value && !runIdExists(runs, value)
          ? [{ value, label: `${value} · 已选择` }]
          : []),
        ...runs.map((run) => ({
          value: run.run_id,
          label: formatRunOption(run)
        }))
      ];
  return (
    <div className="compare-run-select">
      <FormSelectControl
        label={title}
        value={disabled ? "" : value}
        options={runOptions}
        disabled={disabled}
        onChange={onChange}
      />
      {selected ? (
        <div className="compare-run-card">
          <strong title={selected.run_id}>{selected.run_id}</strong>
          <span>{selected.model_id}</span>
          <div>
            <Badge value={selected.status} domain="run" />
            <em className="compare-run-primary-metric">F1 {formatMetric(runF1Score(selected))}</em>
            <em>P {formatMetric(selected.precision_iou50)}</em>
            <em>R {formatMetric(selected.recall_iou50)}</em>
          </div>
        </div>
      ) : value && !disabled ? (
        <div className="compare-run-card">
          <strong title={value}>{value}</strong>
          <span>已选择；当前页未加载该 run</span>
          <div>
            <em>翻页不会清空当前对比</em>
          </div>
        </div>
      ) : null}
    </div>
  );
}

export function ComparisonHistoryPanel({
  comparisons,
  total,
  offset,
  limit,
  active = false,
  refreshing = false,
  onPageChange
}: {
  comparisons: ComparisonSummary[];
  total?: number;
  offset: number;
  limit: number;
  active?: boolean;
  refreshing?: boolean;
  onPageChange: (offset: number) => void;
}) {
  if (comparisons.length === 0 && !active) {
    return null;
  }
  return (
    <div className="history-block">
      <div className="comparison-sample-title">
        历史对比
        {typeof total === "number" ? <span>{total.toLocaleString()} 条</span> : null}
      </div>
      <DataTable
        columns={COMPARISON_HISTORY_COLUMNS}
        data={comparisons}
        emptyText="暂无历史对比。"
        refreshing={refreshing}
        compact
      />
      <PagerControl
        className="rank-board-pager compare-history-pager"
        offset={offset}
        limit={limit}
        total={total ?? comparisons.length}
        meta={<>{comparisons.length.toLocaleString()} visible</>}
        onPageChange={onPageChange}
      />
    </div>
  );
}
