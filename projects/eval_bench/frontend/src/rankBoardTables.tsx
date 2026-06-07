import { Link } from "@tanstack/react-router";
import type { ColumnDef } from "@tanstack/react-table";
import { ArrowDown, ArrowUp, ChevronsUpDown, FileText, FileX } from "lucide-react";
import { useMemo } from "react";

import type { RankBoardEntry, SuiteRankEntry } from "./api";
import { f1Score, formatDate, formatMetric } from "./formatters";
import { RANK_PRIMARY_METRICS } from "./rankBoardModel";
import { ActionButton, Badge, DataTable } from "./ui";

import "./rankBoardTables.css";

const RANK_METRIC_COLUMNS: Array<{
  id: string;
  header: string;
  value: (entry: RankBoardEntry) => number | null;
  format?: (value: number | null) => string;
}> = [
  {
    id: "f1_iou50",
    header: "F1@.50",
    value: (entry) => rankF1Score(entry)
  },
  {
    id: "precision_iou50",
    header: "P@.50",
    value: (entry) => entry.precision_iou50
  },
  {
    id: "recall_iou50",
    header: "R@.50",
    value: (entry) => entry.recall_iou50
  },
  {
    id: "mean_iou",
    header: "mIoU",
    value: (entry) => entry.mean_iou
  },
  {
    id: "prediction_count",
    header: "预测数",
    value: (entry) => entry.prediction_count,
    format: (value) => (value === null ? "-" : value.toLocaleString())
  }
];

export function SuiteRankBoardTable({
  entries,
  sortBy,
  sortOrder,
  onSortChange,
  refreshing
}: {
  entries: SuiteRankEntry[];
  sortBy: string;
  sortOrder: string;
  onSortChange: (value: string) => void;
  refreshing: boolean;
}) {
  const columns = useMemo<ColumnDef<SuiteRankEntry>[]>(() => {
    const suiteHeader = (label: string, value: string) => (
      <SortableHeader
        label={label}
        sortValue={value}
        active={sortBy === value}
        sortOrder={sortOrder}
        onSortChange={onSortChange}
      />
    );
    return [
      {
        id: "rank",
        header: "Rank",
        meta: { width: "compact", align: "center" },
        cell: ({ row }) => <span className="rank-index">#{row.original.rank}</span>
      },
      {
        id: "model_id",
        header: () => suiteHeader("Model", "model_id"),
        meta: {
          width: "id",
          wrap: "wrap",
          className: sortBy === "model_id" ? "rank-sort-active-cell" : undefined
        },
        cell: ({ row }) => (
          <span className="suite-rank-model" title={row.original.campaign_id}>
            {row.original.model_id}
          </span>
        )
      },
      {
        id: "aggregate_score",
        header: () => suiteHeader("Suite F1", "aggregate_score"),
        meta: {
          width: "metric",
          align: "end",
          className:
            sortBy === "aggregate_score" ? "rank-sort-active-cell rank-metric-cell" : "rank-metric-cell"
        },
        cell: ({ row }) => <span className="rank-primary-score">{formatMetric(row.original.aggregate_score)}</span>
      },
      {
        id: "leader_delta",
        header: "Δ leader",
        meta: { width: "metric", align: "end" },
        cell: ({ row }) => (
          <span className={rankDeltaClassName(row.original.score_delta)}>
            {formatScoreDelta(row.original.score_delta)}
          </span>
        )
      },
      {
        id: "checkpoint_family",
        header: "Checkpoint family",
        meta: { width: "wide", wrap: "wrap" },
        cell: ({ row }) => checkpointFamily(row.original.checkpoint)
      },
      {
        id: "task_splits",
        header: "Task splits",
        meta: { width: "wide", wrap: "wrap" },
        cell: ({ row }) => (
          <div className="suite-rank-splits">
            {row.original.task_splits.map((split) => (
              <span key={split}>
                {split}
                <strong>{formatMetric(splitScore(row.original, split))}</strong>
              </span>
            ))}
          </div>
        )
      },
      {
        id: "run_count",
        header: () => suiteHeader("Runs", "run_count"),
        accessorKey: "run_count",
        meta: {
          width: "number",
          align: "end",
          className: sortBy === "run_count" ? "rank-sort-active-cell" : undefined
        }
      },
      {
        id: "created_at",
        header: () => suiteHeader("更新时间", "created_at"),
        meta: {
          width: "date",
          className: sortBy === "created_at" ? "rank-sort-active-cell" : undefined
        },
        cell: ({ row }) => formatDate(row.original.created_at)
      }
    ];
  }, [onSortChange, sortBy, sortOrder]);
  return (
    <DataTable
      columns={columns}
      data={entries}
      emptyText="暂无 official suite aggregate 排名。"
      refreshing={refreshing}
      refreshLabel="Suite ranking 更新中"
    />
  );
}

export function RankBoardTable({
  entries,
  primaryMetric,
  sortBy,
  sortOrder,
  onSortChange,
  refreshing
}: {
  entries: RankBoardEntry[];
  primaryMetric: string;
  sortBy: string;
  sortOrder: string;
  onSortChange: (value: string) => void;
  refreshing: boolean;
}) {
  const columns = useMemo<ColumnDef<RankBoardEntry>[]>(() => {
    const rankHeader = RANK_PRIMARY_METRICS.includes(sortBy) ? "Rank" : "#";
    const metricColumns: ColumnDef<RankBoardEntry>[] = RANK_METRIC_COLUMNS.map((metric) => {
      const active = sortBy === metric.id;
      const metricFormatter = metric.format ?? formatMetric;
      return {
        id: `metric_${metric.id}`,
        header: () => (
          <SortableHeader
            label={metric.header}
            sortValue={metric.id}
            active={active}
            sortOrder={sortOrder}
            onSortChange={onSortChange}
          />
        ),
        meta: {
          width: "metric",
          align: "end",
          className: active ? "rank-sort-active-cell rank-metric-cell" : "rank-metric-cell"
        },
        cell: ({ row }) => {
          const value = metric.id === primaryMetric ? row.original.score : metric.value(row.original);
          return (
            <span className={active ? "rank-primary-score" : undefined}>
              {metricFormatter(value)}
            </span>
          );
        }
      };
    });
    const auxiliaryHeader = (label: string, value: string) => (
      <SortableHeader
        label={label}
        sortValue={value}
        active={sortBy === value}
        sortOrder={sortOrder}
        onSortChange={onSortChange}
      />
    );
    return [
      {
        id: "rank",
        header: rankHeader,
        meta: { width: "compact", align: "center" },
        cell: ({ row }) => <span className="rank-index">#{row.original.rank}</span>
      },
      {
        id: "run_id",
        header: () => auxiliaryHeader("Run", "run_id"),
        meta: {
          width: "id",
          wrap: "wrap",
          className: sortBy === "run_id" ? "rank-sort-active-cell" : undefined
        },
        cell: ({ row }) => (
          <Link className="run-id-link" to="/runs/$runId" params={{ runId: row.original.run_id }}>
            {row.original.run_id}
          </Link>
        )
      },
      ...metricColumns,
      {
        id: "leader_delta",
        header: RANK_PRIMARY_METRICS.includes(sortBy) ? "Δ leader" : "Δ first",
        meta: { width: "metric", align: "end" },
        cell: ({ row }) => (
          <span className={rankDeltaClassName(row.original.score_delta)}>
            {formatScoreDelta(row.original.score_delta)}
          </span>
        )
      },
      {
        id: "status",
        header: "状态",
        meta: { width: "status" },
        cell: ({ row }) => <Badge value={row.original.status} domain="run" />
      },
      { header: "任务", accessorKey: "task", meta: { width: "compact" } },
      {
        id: "target_labels",
        header: "标签",
        meta: { width: "wide", wrap: "wrap" },
        cell: ({ row }) => row.original.target_labels.join(", ") || "-"
      },
      { header: "基准集", accessorKey: "benchmark_id", meta: { width: "id" } },
      { header: "Split", accessorKey: "benchmark_split", meta: { width: "id" } },
      { header: "模型", accessorKey: "model_id", meta: { width: "id" } },
      { header: "Prompt", accessorKey: "prompt_id", meta: { width: "id" } },
      {
        id: "created_at",
        header: () => auxiliaryHeader("创建时间", "created_at"),
        meta: {
          width: "date",
          className: sortBy === "created_at" ? "rank-sort-active-cell" : undefined
        },
        cell: ({ row }) => formatDate(row.original.created_at)
      },
      {
        id: "note",
        header: "备注",
        meta: { width: "compact", align: "center" },
        cell: ({ row }) => {
          const hasNote = Boolean(row.original.note.trim());
          return (
            <span
              className={hasNote ? "run-note-preview" : "run-note-preview empty"}
              title={hasNote ? "有备注" : "无备注"}
              aria-label={hasNote ? "有备注" : "无备注"}
            >
              {hasNote ? <FileText size={14} /> : <FileX size={14} />}
            </span>
          );
        }
      }
    ];
  }, [onSortChange, primaryMetric, sortBy, sortOrder]);
  return (
    <DataTable
      columns={columns}
      data={entries}
      emptyText="没有符合高级检索条件的 run。"
      refreshing={refreshing}
    />
  );
}

function SortableHeader({
  label,
  sortValue,
  active,
  sortOrder,
  onSortChange
}: {
  label: string;
  sortValue: string;
  active: boolean;
  sortOrder: string;
  onSortChange: (value: string) => void;
}) {
  const Icon = active ? (sortOrder === "asc" ? ArrowUp : ArrowDown) : ChevronsUpDown;
  return (
    <ActionButton
      variant="mini"
      compact
      className={active ? "rank-sort-header active" : "rank-sort-header"}
      aria-sort={active ? (sortOrder === "asc" ? "ascending" : "descending") : "none"}
      title={active ? `${label} ${sortOrder === "asc" ? "升序" : "降序"}` : `按 ${label} 排序`}
      onClick={() => onSortChange(sortValue)}
    >
      <span>{label}</span>
      <Icon size={13} aria-hidden="true" />
    </ActionButton>
  );
}

function rankF1Score(entry: RankBoardEntry) {
  return entry.f1_iou50 ?? f1Score(entry.precision_iou50, entry.recall_iou50) ?? entry.score ?? null;
}

function rankDeltaClassName(value: number | null | undefined) {
  if (value === null || value === undefined || Math.abs(value) < 0.000_000_1) {
    return "rank-score-delta neutral";
  }
  return value > 0 ? "rank-score-delta positive" : "rank-score-delta negative";
}

function formatScoreDelta(value: number | null | undefined) {
  if (value === null || value === undefined) {
    return "-";
  }
  if (Math.abs(value) < 0.000_000_1) {
    return "leader";
  }
  const prefix = value > 0 ? "+" : "";
  return `${prefix}${formatMetric(value)}`;
}

function splitScore(entry: SuiteRankEntry, split: string) {
  const payload = entry.per_split[split];
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    return null;
  }
  const value = (payload as Record<string, unknown>).f1_iou50;
  return typeof value === "number" ? value : null;
}

function checkpointFamily(checkpoint: string) {
  const normalized = checkpoint.replace(/\/+$/, "");
  const parts = normalized.split("/");
  if (parts.length <= 1) {
    return normalized || "-";
  }
  return parts.slice(0, -1).join("/") || parts[0];
}
