import { useCallback, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import type { ColumnDef } from "@tanstack/react-table";
import { Archive, Eye, FileText, FileX, GitCompare, RotateCw, Trash2 } from "lucide-react";

import type { BenchmarkSummary, RunSummary } from "./api";
import { archiveRun, deleteRun, evaluateRun } from "./api";
import { StandaloneCheckboxControl } from "./controlPrimitives";
import { AdvancedFilterBar } from "./filterControls";
import type { AdvancedFilterControl } from "./filterControls";
import { formatDate, formatMetric, runF1Score } from "./formatters";
import { canArchiveRun, canDeleteRun, canEvaluateRun } from "./statusModel";
import {
  Badge,
  IconActionButton,
  IconNavLink,
  InlineAnchor,
  InlineNavLink
} from "./ui";
import { DataTable } from "./uiDataTable";
import { DangerConfirmDialog } from "./uiDialog";

const BENCHMARK_TABLE_COLUMNS: ColumnDef<BenchmarkSummary>[] = [
  {
    header: "基准集",
    meta: { width: "id" },
    cell: ({ row }) => (
      <Link to="/benchmarks/$benchmarkId" params={{ benchmarkId: row.original.benchmark_id }}>
        {row.original.benchmark_id}
      </Link>
    )
  },
  {
    header: "任务",
    meta: { width: "compact" },
    cell: ({ row }) => row.original.tasks.join(", ") || "-"
  },
  {
    header: "标注层",
    meta: { width: "compact" },
    cell: ({ row }) => row.original.layers.join(", ") || "-"
  },
  { header: "Split", accessorKey: "split", meta: { width: "id" } },
  {
    header: "样本数",
    accessorKey: "sample_count",
    meta: { width: "number", align: "end" },
    cell: ({ row }) => row.original.sample_count.toLocaleString()
  },
  { header: "创建时间", meta: { width: "date" }, cell: ({ row }) => formatDate(row.original.created_at) },
  {
    header: "",
    id: "actions",
    meta: { width: "actions", wrap: "wrap" },
    cell: ({ row }) => (
      <InlineNavLink
        icon={<Eye size={13} />}
        to="/benchmarks/$benchmarkId"
        params={{ benchmarkId: row.original.benchmark_id }}
        title="检查基准集真值样本"
      >
        检查
      </InlineNavLink>
    )
  }
];

const benchmarkTableRowId = (benchmark: BenchmarkSummary) => benchmark.benchmark_id;
const runTableRowId = (run: RunSummary) => run.run_id;

export function BenchmarkTable({
  benchmarks,
  compact = false,
  refreshing = false
}: {
  benchmarks: BenchmarkSummary[];
  compact?: boolean;
  refreshing?: boolean;
}) {
  return (
    <DataTable
      columns={BENCHMARK_TABLE_COLUMNS}
      data={benchmarks}
      emptyText="还没有登记基准集。"
      compact={compact}
      refreshing={refreshing}
      getRowId={benchmarkTableRowId}
    />
  );
}

export function RunTable({
  runs,
  compact = false,
  filterControls,
  filterMeta,
  filterTitle = "结果高级检索",
  refreshing = false,
  footer
}: {
  runs: RunSummary[];
  compact?: boolean;
  filterControls?: AdvancedFilterControl[];
  filterMeta?: string;
  filterTitle?: string;
  refreshing?: boolean;
  footer?: ReactNode;
}) {
  const queryClient = useQueryClient();
  const [selectedRunIds, setSelectedRunIds] = useState<string[]>([]);
  const [deleteRunTarget, setDeleteRunTarget] = useState<RunSummary | null>(null);
  const refreshRunViews = () => {
    void queryClient.invalidateQueries({ queryKey: ["dashboard-state"] });
    void queryClient.invalidateQueries({ queryKey: ["runs"] });
    void queryClient.invalidateQueries({ queryKey: ["rank-board"] });
    void queryClient.invalidateQueries({ queryKey: ["comparisons"] });
  };
  const evaluateMutation = useMutation({ mutationFn: evaluateRun, onSuccess: refreshRunViews });
  const archiveMutation = useMutation({ mutationFn: archiveRun, onSuccess: refreshRunViews });
  const deleteMutation = useMutation({
    mutationFn: deleteRun,
    onSuccess: () => {
      setSelectedRunIds([]);
      setDeleteRunTarget(null);
      refreshRunViews();
    }
  });
  const { mutate: evaluateRunMutate, isPending: evaluatePending } = evaluateMutation;
  const { mutate: archiveRunMutate, isPending: archivePending } = archiveMutation;
  const { isPending: deletePending } = deleteMutation;
  const filteredRuns = runs;
  const comparableSelection = selectedRunIds.slice(0, 2);
  const compareHref =
    comparableSelection.length === 2
      ? `/compare?baseline=${encodeURIComponent(comparableSelection[0])}&candidate=${encodeURIComponent(
          comparableSelection[1]
        )}`
      : "/compare";
  const toggleRunSelection = useCallback((runId: string) => {
    setSelectedRunIds((current) => {
      if (current.includes(runId)) {
        return current.filter((item) => item !== runId);
      }
      return [...current, runId].slice(-2);
    });
  }, []);
  const columns = useMemo<ColumnDef<RunSummary>[]>(
    () => [
      ...(compact
        ? []
        : [
            {
              header: "",
              id: "select",
              meta: { width: "select", align: "center" },
              cell: ({ row }) => (
                <StandaloneCheckboxControl
                  className="row-select-checkbox"
                  label={`选择 ${row.original.run_id} 进行对比`}
                  checked={selectedRunIds.includes(row.original.run_id)}
                  onChange={() => toggleRunSelection(row.original.run_id)}
                />
              )
            } satisfies ColumnDef<RunSummary>
          ]),
      {
        header: "评测",
        meta: { width: "id", wrap: "wrap" },
        cell: ({ row }) => (
          <Link className="run-id-link" to="/runs/$runId" params={{ runId: row.original.run_id }}>
            {row.original.run_id}
          </Link>
        )
      },
      {
        header: "状态",
        meta: { width: "status" },
        cell: ({ row }) => <Badge value={row.original.status} domain="run" />
      },
      { header: "任务", accessorKey: "spec_task", meta: { width: "compact" } },
      { header: "基准集", accessorKey: "benchmark_id", meta: { width: "id" } },
      { header: "Split", accessorKey: "benchmark_split", meta: { width: "id" } },
      { header: "模型", accessorKey: "model_id", meta: { width: "id" } },
      ...(compact
        ? []
        : [
            {
              header: "备注",
              id: "note",
              meta: { width: "compact", align: "center" },
              cell: ({ row }) => {
                const hasNote = Boolean(row.original.note.trim());
                return (
                  <Link
                    to="/runs/$runId"
                    params={{ runId: row.original.run_id }}
                    hash="run-note"
                    className={hasNote ? "run-note-preview" : "run-note-preview empty"}
                    title={hasNote ? "有备注" : "无备注"}
                    aria-label={hasNote ? "有备注" : "无备注"}
                  >
                    {hasNote ? <FileText size={14} /> : <FileX size={14} />}
                  </Link>
                );
              }
            } satisfies ColumnDef<RunSummary>
          ]),
      {
        header: "预测数",
        accessorKey: "prediction_count",
        meta: { width: "number", align: "end" },
        cell: ({ row }) => row.original.prediction_count.toLocaleString()
      },
      {
        header: "F1@.50",
        meta: { width: "metric", align: "end" },
        cell: ({ row }) => formatMetric(runF1Score(row.original))
      },
      {
        header: "P@.50",
        meta: { width: "metric", align: "end" },
        cell: ({ row }) => formatMetric(row.original.precision_iou50)
      },
      {
        header: "R@.50",
        meta: { width: "metric", align: "end" },
        cell: ({ row }) => formatMetric(row.original.recall_iou50)
      },
      { header: "报告数", accessorKey: "report_count", meta: { width: "number", align: "end" } },
      { header: "创建时间", meta: { width: "date" }, cell: ({ row }) => formatDate(row.original.created_at) },
      {
        header: "",
        id: "actions",
        meta: { width: "actions", wrap: "wrap" },
        cell: ({ row }) => (
          <div className="row-actions">
            <IconNavLink
              icon={<Eye size={13} />}
              to="/runs/$runId"
              params={{ runId: row.original.run_id }}
              title="检查样本级预测"
            />
            <IconActionButton
              icon={<RotateCw size={13} />}
              onClick={() => evaluateRunMutate(row.original.run_id)}
              disabled={!canEvaluateRun(row.original) || evaluatePending}
              title="计算预测指标"
            />
            {!compact ? (
              <>
                <IconActionButton
                  icon={<Archive size={14} />}
                  onClick={() => archiveRunMutate(row.original.run_id)}
                  disabled={!canArchiveRun(row.original) || archivePending}
                  title="归档 run"
                />
                <IconActionButton
                  icon={<Trash2 size={14} />}
                  danger
                  onClick={() => setDeleteRunTarget(row.original)}
                  disabled={!canDeleteRun(row.original) || deletePending}
                  title="删除 run"
                />
              </>
            ) : null}
          </div>
        )
      }
    ],
    [
      archivePending,
      archiveRunMutate,
      compact,
      deletePending,
      evaluatePending,
      evaluateRunMutate,
      selectedRunIds,
      toggleRunSelection
    ]
  );
  return (
    <div className={compact ? "run-table-stack compact" : "run-table-stack"}>
      {!compact && filterControls ? (
        <AdvancedFilterBar
          title={filterTitle}
          meta={filterMeta ?? `${filteredRuns.length.toLocaleString()} 条 run`}
          controls={filterControls}
          actions={
            <InlineAnchor
              className={
                comparableSelection.length === 2 ? "compare-ready" : "disabled"
              }
              href={compareHref}
            >
              <GitCompare size={13} />
              对比 {comparableSelection.length}/2
            </InlineAnchor>
          }
        />
      ) : null}
      <DataTable
        columns={columns}
        data={filteredRuns}
        emptyText="还没有评测记录。"
        compact={compact}
        refreshing={refreshing}
        getRowId={runTableRowId}
      />
      {footer}
      <DangerConfirmDialog
        open={Boolean(deleteRunTarget)}
        title="删除 run"
        subject={deleteRunTarget?.run_id ?? ""}
        description="Run 目录会移入回收站，相关报告、预测快照和备注会从结果库列表移除。"
        confirmLabel="移入回收站"
        pending={deleteMutation.isPending}
        onCancel={() => setDeleteRunTarget(null)}
        onConfirm={() => {
          if (deleteRunTarget) {
            deleteMutation.mutate(deleteRunTarget.run_id);
          }
        }}
      />
    </div>
  );
}
