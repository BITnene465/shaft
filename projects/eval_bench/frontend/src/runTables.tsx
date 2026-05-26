import { useState } from "react";
import type { ReactNode } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import type { ColumnDef } from "@tanstack/react-table";
import { Archive, Eye, FileText, GitCompare, RotateCw, Trash2 } from "lucide-react";

import type { BenchmarkSummary, RunSummary } from "./api";
import { archiveRun, deleteRun, evaluateRun } from "./api";
import { StandaloneCheckboxControl } from "./controlPrimitives";
import { AdvancedFilterBar } from "./filterControls";
import type { AdvancedFilterControl } from "./filterControls";
import { formatDate, formatMetric, runF1Score, unique } from "./formatters";
import { canArchiveRun, canDeleteRun, canEvaluateRun } from "./statusModel";
import {
  Badge,
  DangerConfirmDialog,
  DataTable,
  IconActionButton,
  IconNavLink,
  InlineAnchor,
  InlineNavLink
} from "./ui";

export function BenchmarkTable({
  benchmarks,
  compact = false
}: {
  benchmarks: BenchmarkSummary[];
  compact?: boolean;
}) {
  const columns: ColumnDef<BenchmarkSummary>[] = [
    {
      header: "基准集",
      cell: ({ row }) => (
        <Link to="/benchmarks/$benchmarkId" params={{ benchmarkId: row.original.benchmark_id }}>
          {row.original.benchmark_id}
        </Link>
      )
    },
    { header: "任务", cell: ({ row }) => row.original.tasks.join(", ") || "-" },
    { header: "标注层", cell: ({ row }) => row.original.layers.join(", ") || "-" },
    { header: "Split", accessorKey: "split" },
    {
      header: "样本数",
      accessorKey: "sample_count",
      cell: ({ row }) => row.original.sample_count.toLocaleString()
    },
    { header: "创建时间", cell: ({ row }) => formatDate(row.original.created_at) },
    {
      header: "",
      id: "actions",
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
  return (
    <DataTable
      columns={columns}
      data={benchmarks}
      emptyText="还没有登记基准集。"
      compact={compact}
    />
  );
}

export function RunTable({
  runs,
  compact = false,
  filterControls,
  filterMeta,
  filterTitle = "结果高级检索",
  footer
}: {
  runs: RunSummary[];
  compact?: boolean;
  filterControls?: AdvancedFilterControl[];
  filterMeta?: string;
  filterTitle?: string;
  footer?: ReactNode;
}) {
  const queryClient = useQueryClient();
  const [searchText, setSearchText] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [taskFilter, setTaskFilter] = useState("all");
  const [benchmarkFilter, setBenchmarkFilter] = useState("all");
  const [labelFilter, setLabelFilter] = useState("all");
  const [modelFilter, setModelFilter] = useState("all");
  const [promptFilter, setPromptFilter] = useState("all");
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
  const usesExternalFilters = Boolean(filterControls);
  const statuses = unique(runs.map((run) => run.status).filter(Boolean));
  const tasks = unique(runs.map((run) => run.spec_task).filter(Boolean));
  const benchmarks = unique(runs.map((run) => run.benchmark_id).filter(Boolean));
  const labels = unique(runs.flatMap((run) => run.target_labels).filter(Boolean));
  const models = unique(runs.map((run) => run.model_id).filter(Boolean));
  const prompts = unique(runs.map((run) => run.prompt_id).filter(Boolean));
  const filteredRuns = compact || usesExternalFilters
    ? runs
    : runs
        .filter((run) => statusFilter === "all" || run.status === statusFilter)
        .filter((run) => taskFilter === "all" || run.spec_task === taskFilter)
        .filter((run) => benchmarkFilter === "all" || run.benchmark_id === benchmarkFilter)
        .filter((run) => labelFilter === "all" || run.target_labels.includes(labelFilter))
        .filter((run) => modelFilter === "all" || run.model_id === modelFilter)
        .filter((run) => promptFilter === "all" || run.prompt_id === promptFilter)
        .filter((run) => {
          const query = searchText.trim().toLowerCase();
          if (!query) {
            return true;
          }
          return [
            run.run_id,
            run.model_id,
            run.benchmark_id,
            run.spec_task,
            run.prompt_id,
            run.target_labels.join(" "),
            run.metric_profile,
            run.note
          ].some((value) => String(value).toLowerCase().includes(query));
        });
  const comparableSelection = selectedRunIds.slice(0, 2);
  const compareHref =
    comparableSelection.length === 2
      ? `/compare?baseline=${encodeURIComponent(comparableSelection[0])}&candidate=${encodeURIComponent(
          comparableSelection[1]
        )}`
      : "/compare";
  const columns: ColumnDef<RunSummary>[] = [
    ...(compact
      ? []
      : [
          {
            header: "",
            id: "select",
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
      header: "记录",
      cell: ({ row }) => (
        <Link to="/runs/$runId" params={{ runId: row.original.run_id }}>
          {row.original.run_id}
        </Link>
      )
    },
    { header: "状态", cell: ({ row }) => <Badge value={row.original.status} domain="run" /> },
    { header: "任务", accessorKey: "spec_task" },
    { header: "基准集", accessorKey: "benchmark_id" },
    { header: "模型", accessorKey: "model_id" },
    ...(compact
      ? []
      : [
          {
            header: "备注",
            id: "note",
            cell: ({ row }) => (
              <Link
                to="/runs/$runId"
                params={{ runId: row.original.run_id }}
                hash="run-note"
                className={row.original.note ? "run-note-preview" : "run-note-preview empty"}
                title={row.original.note || "未记录备注"}
              >
                <FileText size={13} />
                {row.original.note || "未记录"}
              </Link>
            )
          } satisfies ColumnDef<RunSummary>
        ]),
    {
      header: "预测数",
      accessorKey: "prediction_count",
      cell: ({ row }) => row.original.prediction_count.toLocaleString()
    },
    { header: "F1@.50", cell: ({ row }) => formatMetric(runF1Score(row.original)) },
    { header: "P@.50", cell: ({ row }) => formatMetric(row.original.precision_iou50) },
    { header: "R@.50", cell: ({ row }) => formatMetric(row.original.recall_iou50) },
    { header: "报告数", accessorKey: "report_count" },
    { header: "创建时间", cell: ({ row }) => formatDate(row.original.created_at) },
    {
      header: "",
      id: "actions",
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
            onClick={() => evaluateMutation.mutate(row.original.run_id)}
            disabled={!canEvaluateRun(row.original) || evaluateMutation.isPending}
            title="计算预测指标"
          />
          {!compact ? (
            <>
              <IconActionButton
                icon={<Archive size={14} />}
                onClick={() => archiveMutation.mutate(row.original.run_id)}
                disabled={!canArchiveRun(row.original) || archiveMutation.isPending}
                title="归档 run"
              />
              <IconActionButton
                icon={<Trash2 size={14} />}
                danger
                onClick={() => setDeleteRunTarget(row.original)}
                disabled={!canDeleteRun(row.original) || deleteMutation.isPending}
                title="删除 run"
              />
            </>
          ) : null}
        </div>
      )
    }
  ];
  function toggleRunSelection(runId: string) {
    setSelectedRunIds((current) => {
      if (current.includes(runId)) {
        return current.filter((item) => item !== runId);
      }
      return [...current, runId].slice(-2);
    });
  }
  return (
    <div className={compact ? "run-table-stack compact" : "run-table-stack"}>
      {!compact ? (
        <AdvancedFilterBar
          title={filterTitle}
          meta={filterMeta ?? `${filteredRuns.length.toLocaleString()} / ${runs.length.toLocaleString()} 条 run`}
          controls={
            filterControls ?? [
              {
                type: "search",
                id: "query",
                label: "全文检索",
                value: searchText,
                onChange: setSearchText,
                placeholder: "搜索 run、模型、基准集、备注"
              },
              {
                type: "select",
                id: "status",
                label: "状态",
                value: statusFilter,
                values: ["all", ...statuses],
                labels: { all: "全部" },
                onChange: setStatusFilter
              },
              {
                type: "select",
                id: "task",
                label: "任务",
                value: taskFilter,
                values: ["all", ...tasks],
                labels: { all: "全部" },
                onChange: setTaskFilter
              },
              {
                type: "select",
                id: "benchmark",
                label: "基准集",
                value: benchmarkFilter,
                values: ["all", ...benchmarks],
                labels: { all: "全部" },
                onChange: setBenchmarkFilter
              },
              {
                type: "select",
                id: "label",
                label: "标签",
                value: labelFilter,
                values: ["all", ...labels],
                labels: { all: "全部" },
                onChange: setLabelFilter
              },
              {
                type: "select",
                id: "model",
                label: "模型",
                value: modelFilter,
                values: ["all", ...models],
                labels: { all: "全部" },
                onChange: setModelFilter
              },
              {
                type: "select",
                id: "prompt",
                label: "Prompt",
                value: promptFilter,
                values: ["all", ...prompts],
                labels: { all: "全部" },
                onChange: setPromptFilter
              }
            ]
          }
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
