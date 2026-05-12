import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import type { ColumnDef } from "@tanstack/react-table";
import { Archive, Eye, GitCompare, RotateCw, Search, Trash2 } from "lucide-react";

import type { BenchmarkSummary, RunSummary } from "./api";
import { archiveRun, deleteRun, evaluateRun } from "./api";
import { FilterSelect } from "./filterControls";
import { formatDate, formatMetric, unique } from "./formatters";
import { canArchiveRun, canDeleteRun, canEvaluateRun } from "./statusModel";
import { Badge, DataTable } from "./ui";

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
        <Link
          className="mini-link"
          to="/benchmarks/$benchmarkId"
          params={{ benchmarkId: row.original.benchmark_id }}
          title="检查基准集真值样本"
        >
          <Eye size={13} />
          检查
        </Link>
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

export function RunTable({ runs, compact = false }: { runs: RunSummary[]; compact?: boolean }) {
  const queryClient = useQueryClient();
  const [searchText, setSearchText] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [taskFilter, setTaskFilter] = useState("all");
  const [benchmarkFilter, setBenchmarkFilter] = useState("all");
  const [selectedRunIds, setSelectedRunIds] = useState<string[]>([]);
  const evaluateMutation = useMutation({
    mutationFn: evaluateRun,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["dashboard-state"] });
    }
  });
  const archiveMutation = useMutation({
    mutationFn: archiveRun,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["dashboard-state"] });
    }
  });
  const deleteMutation = useMutation({
    mutationFn: deleteRun,
    onSuccess: () => {
      setSelectedRunIds([]);
      void queryClient.invalidateQueries({ queryKey: ["dashboard-state"] });
    }
  });
  const statuses = unique(runs.map((run) => run.status).filter(Boolean));
  const tasks = unique(runs.map((run) => run.spec_task).filter(Boolean));
  const benchmarks = unique(runs.map((run) => run.benchmark_id).filter(Boolean));
  const filteredRuns = compact
    ? runs
    : runs
        .filter((run) => statusFilter === "all" || run.status === statusFilter)
        .filter((run) => taskFilter === "all" || run.spec_task === taskFilter)
        .filter((run) => benchmarkFilter === "all" || run.benchmark_id === benchmarkFilter)
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
            run.prompt_id
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
              <input
                className="row-select-checkbox"
                aria-label={`选择 ${row.original.run_id} 进行对比`}
                type="checkbox"
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
    {
      header: "预测数",
      accessorKey: "prediction_count",
      cell: ({ row }) => row.original.prediction_count.toLocaleString()
    },
    { header: "P@.50", cell: ({ row }) => formatMetric(row.original.precision_iou50) },
    { header: "R@.50", cell: ({ row }) => formatMetric(row.original.recall_iou50) },
    { header: "报告数", accessorKey: "report_count" },
    { header: "创建时间", cell: ({ row }) => formatDate(row.original.created_at) },
    {
      header: "",
      id: "actions",
      cell: ({ row }) => (
        <div className="row-actions">
          <Link
            className="icon-button dense"
            to="/runs/$runId"
            params={{ runId: row.original.run_id }}
            title="检查样本级预测"
          >
            <Eye size={13} />
          </Link>
          <button
            className="icon-button dense"
            type="button"
            onClick={() => evaluateMutation.mutate(row.original.run_id)}
            disabled={!canEvaluateRun(row.original) || evaluateMutation.isPending}
            title="计算预测指标"
          >
            <RotateCw size={13} />
          </button>
          {!compact ? (
            <>
              <button
                className="icon-button dense"
                type="button"
                onClick={() => archiveMutation.mutate(row.original.run_id)}
                disabled={!canArchiveRun(row.original) || archiveMutation.isPending}
                title="归档 run"
              >
                <Archive size={14} />
              </button>
              <button
                className="icon-button dense danger"
                type="button"
                onClick={() => {
                  if (confirm(`将 run ${row.original.run_id} 移入回收站？`)) {
                    deleteMutation.mutate(row.original.run_id);
                  }
                }}
                disabled={!canDeleteRun(row.original) || deleteMutation.isPending}
                title="删除 run"
              >
                <Trash2 size={14} />
              </button>
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
        <div className="run-query-bar">
          <label className="search-box">
            <Search size={15} />
            <input
              value={searchText}
              onChange={(event) => setSearchText(event.target.value)}
              placeholder="搜索 run、模型、基准集"
            />
          </label>
          <FilterSelect
            label="状态"
            value={statusFilter}
            values={["all", ...statuses]}
            labels={{ all: "全部" }}
            onChange={setStatusFilter}
            compact
          />
          <FilterSelect
            label="任务"
            value={taskFilter}
            values={["all", ...tasks]}
            labels={{ all: "全部" }}
            onChange={setTaskFilter}
            compact
          />
          <FilterSelect
            label="基准集"
            value={benchmarkFilter}
            values={["all", ...benchmarks]}
            labels={{ all: "全部" }}
            onChange={setBenchmarkFilter}
            compact
          />
          <a
            className={
              comparableSelection.length === 2 ? "mini-link compare-ready" : "mini-link disabled"
            }
            href={compareHref}
          >
            <GitCompare size={13} />
            对比 {comparableSelection.length}/2
          </a>
        </div>
      ) : null}
      <DataTable
        columns={columns}
        data={filteredRuns}
        emptyText="还没有评测记录。"
        compact={compact}
      />
    </div>
  );
}

