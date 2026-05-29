import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import type { ColumnDef } from "@tanstack/react-table";
import { ArrowDown, ArrowUp, ChevronsUpDown, FileText, FileX } from "lucide-react";

import type { RankBoard, RankBoardEntry } from "./api";
import { fetchRankBoard } from "./api";
import { useDashboardState } from "./dashboardState";
import { AdvancedFilterBar } from "./filterControls";
import { errorMessage, f1Score, facetValues, formatDate, formatMetric } from "./formatters";
import { PagerControl, clampListPageOffset, updatePagedFilterValue } from "./samplePager";
import { ActionButton, Badge, DataTable, EmptyState, OptionChipButton } from "./ui";

const RANK_PRIMARY_METRICS = [
  "f1_iou50",
  "precision_iou50",
  "recall_iou50",
  "mean_iou",
  "prediction_count"
];
const RANK_AUXILIARY_SORTS = [
  "created_at",
  "run_id"
];
const RANK_SORTABLE_FIELDS = [...RANK_PRIMARY_METRICS, ...RANK_AUXILIARY_SORTS];
const RANK_PAGE_SIZE = 80;
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

export function RankBoardPage() {
  const dashboardQuery = useDashboardState();
  const runs = dashboardQuery.data?.runs ?? [];
  const [searchText, setSearchText] = useState("");
  const [taskFilter, setTaskFilter] = useState("all");
  const [benchmarkFilter, setBenchmarkFilter] = useState("all");
  const [benchmarkSplitFilter, setBenchmarkSplitFilter] = useState("all");
  const [statusFilter, setStatusFilter] = useState("all");
  const [labelFilter, setLabelFilter] = useState("all");
  const [modelFilter, setModelFilter] = useState("all");
  const [promptFilter, setPromptFilter] = useState("all");
  const [metricProfileFilter, setMetricProfileFilter] = useState("all");
  const [minScoreFilter, setMinScoreFilter] = useState("");
  const [sortBy, setSortBy] = useState("f1_iou50");
  const [sortOrder, setSortOrder] = useState("desc");
  const [pageOffset, setPageOffset] = useState(0);
  const boardQuery = useQuery({
    queryKey: [
      "rank-board",
      searchText,
      taskFilter,
      benchmarkFilter,
      benchmarkSplitFilter,
      statusFilter,
      labelFilter,
      modelFilter,
      promptFilter,
      metricProfileFilter,
      minScoreFilter,
      sortBy,
      sortOrder,
      pageOffset
    ],
    queryFn: () =>
      fetchRankBoard({
        offset: pageOffset,
        limit: RANK_PAGE_SIZE,
        query: searchText,
        task: taskFilter,
        benchmarkId: benchmarkFilter,
        benchmarkSplit: benchmarkSplitFilter,
        status: statusFilter,
        label: labelFilter,
        modelId: modelFilter,
        promptId: promptFilter,
        metricProfile: metricProfileFilter,
        minScore: minScoreFilter,
        sortBy,
        sortOrder
      }),
    placeholderData: (previousData) => previousData
  });
  const board = boardQuery.data;
  const tasks = facetValues(board?.facets, "tasks", runs.map((run) => run.spec_task));
  const benchmarks = facetValues(
    board?.facets,
    "benchmarks",
    runs.map((run) => run.benchmark_id)
  );
  const benchmarkSplits = facetValues(
    board?.facets,
    "splits",
    runs.map((run) => run.benchmark_split)
  );
  const statuses = facetValues(board?.facets, "statuses", runs.map((run) => run.status));
  const labels = facetValues(board?.facets, "labels", runs.flatMap((run) => run.target_labels));
  const models = facetValues(board?.facets, "models", runs.map((run) => run.model_id));
  const prompts = facetValues(board?.facets, "prompts", runs.map((run) => run.prompt_id));
  const metricProfiles = facetValues(
    board?.facets,
    "metric_profiles",
    runs.map((run) => run.metric_profile)
  );
  const entries = board?.entries ?? [];
  const handleSortChange = (value: string) => {
    if (!RANK_SORTABLE_FIELDS.includes(value)) {
      return;
    }
    const nextOrder = sortBy === value ? toggleSortOrder(sortOrder) : defaultRankSortOrder(value);
    if (sortBy !== value) {
      setSortBy(value);
    }
    if (sortOrder !== nextOrder) {
      setSortOrder(nextOrder);
    }
    setPageOffset(0);
  };
  useEffect(() => {
    if (!board) {
      return;
    }
    const nextOffset = clampListPageOffset(pageOffset, board.total, RANK_PAGE_SIZE);
    if (nextOffset !== pageOffset) {
      setPageOffset(nextOffset);
    }
  }, [board?.total, pageOffset]);

  if (dashboardQuery.isLoading || (boardQuery.isLoading && !board)) {
    return <EmptyState title="正在加载排行榜" />;
  }
  if (dashboardQuery.error || !board) {
    return (
      <EmptyState
        title={`排行榜加载失败：${errorMessage(dashboardQuery.error || boardQuery.error)}`}
        tone="danger"
      />
    );
  }
  const tableRefreshing = boardQuery.isPlaceholderData && Boolean(board);

  return (
    <section className="page-stack density-page rank-board-page">
      <RankBoardStatusBar
        board={board}
        runCount={runs.length}
      />
      <AdvancedFilterBar
        title="筛选"
        meta="任务、基准集、状态、标签、模型、Prompt、Metric 与最低分"
        controls={[
          {
            type: "search",
            id: "rank-query",
            label: "全文检索",
            value: searchText,
            onChange: (value) =>
              updatePagedFilterValue(searchText, value, setSearchText, setPageOffset),
            placeholder: "搜索 run、模型、prompt、备注"
          },
          {
            type: "select",
            id: "rank-task",
            label: "任务",
            value: taskFilter,
            values: ["all", ...tasks],
            labels: { all: "全部" },
            onChange: (value) =>
              updatePagedFilterValue(taskFilter, value, setTaskFilter, setPageOffset)
          },
          {
            type: "select",
            id: "rank-benchmark",
            label: "基准集",
            value: benchmarkFilter,
            values: ["all", ...benchmarks],
            labels: { all: "全部" },
            onChange: (value) =>
              updatePagedFilterValue(benchmarkFilter, value, setBenchmarkFilter, setPageOffset)
          },
          {
            type: "select",
            id: "rank-benchmark-split",
            label: "Split",
            value: benchmarkSplitFilter,
            values: ["all", ...benchmarkSplits],
            labels: { all: "全部" },
            onChange: (value) =>
              updatePagedFilterValue(
                benchmarkSplitFilter,
                value,
                setBenchmarkSplitFilter,
                setPageOffset
              )
          },
          {
            type: "select",
            id: "rank-status",
            label: "状态",
            value: statusFilter,
            values: ["all", ...statuses],
            labels: { all: "全部" },
            onChange: (value) =>
              updatePagedFilterValue(statusFilter, value, setStatusFilter, setPageOffset)
          },
          {
            type: "select",
            id: "rank-label",
            label: "标签",
            value: labelFilter,
            values: ["all", ...labels],
            labels: { all: "全部" },
            onChange: (value) =>
              updatePagedFilterValue(labelFilter, value, setLabelFilter, setPageOffset)
          },
          {
            type: "select",
            id: "rank-model",
            label: "模型",
            value: modelFilter,
            values: ["all", ...models],
            labels: { all: "全部" },
            onChange: (value) =>
              updatePagedFilterValue(modelFilter, value, setModelFilter, setPageOffset)
          },
          {
            type: "select",
            id: "rank-prompt",
            label: "Prompt",
            value: promptFilter,
            values: ["all", ...prompts],
            labels: { all: "全部" },
            onChange: (value) =>
              updatePagedFilterValue(promptFilter, value, setPromptFilter, setPageOffset)
          },
          {
            type: "select",
            id: "rank-metric",
            label: "Metric",
            value: metricProfileFilter,
            values: ["all", ...metricProfiles],
            labels: { all: "全部" },
            onChange: (value) =>
              updatePagedFilterValue(
                metricProfileFilter,
                value,
                setMetricProfileFilter,
                setPageOffset
              )
          },
          {
            type: "number",
            id: "rank-min-score",
            label: "最低分",
            value: minScoreFilter,
            min: 0,
            max: 1,
            step: 0.01,
            placeholder: "0.70",
            onChange: (value) =>
              updatePagedFilterValue(minScoreFilter, value, setMinScoreFilter, setPageOffset)
          }
        ]}
      />
      <div className="workspace-card fill rank-board-table-card">
        <div className="rank-board-table-toolbar">
          <PagerControl
            className="rank-board-pager"
            offset={board.offset}
            limit={board.limit}
            total={board.total}
            onPageChange={setPageOffset}
          />
        </div>
        <RankBoardTable
          entries={entries}
          primaryMetric={board.primary_metric}
          sortBy={sortBy}
          sortOrder={sortOrder}
          onSortChange={handleSortChange}
          refreshing={tableRefreshing}
        />
      </div>
      <RankFacetRail
        board={board}
        filters={{
          task: taskFilter,
          benchmark: benchmarkFilter,
          split: benchmarkSplitFilter,
          status: statusFilter,
          label: labelFilter,
          model: modelFilter,
          prompt: promptFilter,
          metricProfile: metricProfileFilter
        }}
        onFilterChange={{
          task: (value) => updatePagedFilterValue(taskFilter, value, setTaskFilter, setPageOffset),
          benchmark: (value) =>
            updatePagedFilterValue(benchmarkFilter, value, setBenchmarkFilter, setPageOffset),
          split: (value) =>
            updatePagedFilterValue(
              benchmarkSplitFilter,
              value,
              setBenchmarkSplitFilter,
              setPageOffset
            ),
          status: (value) =>
            updatePagedFilterValue(statusFilter, value, setStatusFilter, setPageOffset),
          label: (value) =>
            updatePagedFilterValue(labelFilter, value, setLabelFilter, setPageOffset),
          model: (value) =>
            updatePagedFilterValue(modelFilter, value, setModelFilter, setPageOffset),
          prompt: (value) =>
            updatePagedFilterValue(promptFilter, value, setPromptFilter, setPageOffset),
          metricProfile: (value) =>
            updatePagedFilterValue(
              metricProfileFilter,
              value,
              setMetricProfileFilter,
              setPageOffset
            )
        }}
      />
    </section>
  );
}

function RankBoardStatusBar({
  board,
  runCount
}: {
  board: RankBoard;
  runCount: number;
}) {
  return (
    <section className="rank-board-statusbar">
      <div className="rank-board-summary">
        <strong>Leaderboard</strong>
        <span>{board.total.toLocaleString()} runs</span>
        <span>{board.evaluated_count.toLocaleString()} evaluated</span>
        <span>{facetTotal(board, "benchmarks").toLocaleString()} benchmarks</span>
        <span>{runCount.toLocaleString()} total</span>
      </div>
    </section>
  );
}

function facetTotal(board: Pick<RankBoard, "facets">, key: string) {
  return board.facets[key]?.length ?? 0;
}

function RankFacetRail({
  board,
  filters,
  onFilterChange
}: {
  board: Pick<RankBoard, "facets">;
  filters: {
    task: string;
    benchmark: string;
    split: string;
    status: string;
    label: string;
    model: string;
    prompt: string;
    metricProfile: string;
  };
  onFilterChange: {
    task: (value: string) => void;
    benchmark: (value: string) => void;
    split: (value: string) => void;
    status: (value: string) => void;
    label: (value: string) => void;
    model: (value: string) => void;
    prompt: (value: string) => void;
    metricProfile: (value: string) => void;
  };
}) {
  const groups = [
    {
      title: "Tasks",
      items: board.facets.tasks ?? [],
      activeValue: filters.task,
      onSelect: onFilterChange.task
    },
    {
      title: "Benchmarks",
      items: board.facets.benchmarks ?? [],
      activeValue: filters.benchmark,
      onSelect: onFilterChange.benchmark
    },
    {
      title: "Splits",
      items: board.facets.splits ?? [],
      activeValue: filters.split,
      onSelect: onFilterChange.split
    },
    {
      title: "Status",
      items: board.facets.statuses ?? [],
      activeValue: filters.status,
      onSelect: onFilterChange.status
    },
    {
      title: "Labels",
      items: board.facets.labels ?? [],
      activeValue: filters.label,
      onSelect: onFilterChange.label
    },
    {
      title: "Models",
      items: board.facets.models ?? [],
      activeValue: filters.model,
      onSelect: onFilterChange.model
    },
    {
      title: "Prompts",
      items: board.facets.prompts ?? [],
      activeValue: filters.prompt,
      onSelect: onFilterChange.prompt
    },
    {
      title: "Metrics",
      items: board.facets.metric_profiles ?? [],
      activeValue: filters.metricProfile,
      onSelect: onFilterChange.metricProfile
    }
  ].filter((group) => group.items.length > 0);
  if (groups.length === 0) {
    return null;
  }
  return (
    <div className="rank-facet-rail">
      {groups.map((group) => (
        <RankFacetGroup
          key={group.title}
          title={group.title}
          items={group.items}
          activeValue={group.activeValue}
          onSelect={group.onSelect}
        />
      ))}
    </div>
  );
}

function RankFacetGroup({
  title,
  items,
  activeValue,
  onSelect
}: {
  title: string;
  items: Array<{ value: string; count: number }>;
  activeValue: string;
  onSelect: (value: string) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const visibleItems = expanded ? items : items.slice(0, 5);
  const hiddenCount = Math.max(0, items.length - visibleItems.length);
  return (
    <section className={expanded ? "rank-facet-group expanded" : "rank-facet-group"}>
      <span>{title}</span>
      <div>
        {visibleItems.map((item) => {
          const active = activeValue === item.value;
          return (
            <OptionChipButton
              className="rank-facet-button"
              active={active}
              key={item.value}
              title={`${title}: ${item.value}`}
              onClick={() => onSelect(active ? "all" : item.value)}
            >
              <span>{item.value}</span>
              <strong>{item.count.toLocaleString()}</strong>
            </OptionChipButton>
          );
        })}
        {items.length > 5 ? (
          <OptionChipButton
            className="rank-facet-toggle"
            active={expanded}
            title={expanded ? `${title}: 收起 facet` : `${title}: 展开全部 facet`}
            onClick={() => setExpanded((value) => !value)}
          >
            {expanded ? "收起" : `展开全部 +${hiddenCount}`}
          </OptionChipButton>
        ) : null}
        {items.length === 0 ? <em>无</em> : null}
      </div>
    </section>
  );
}

function RankBoardTable({
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
  const columns: ColumnDef<RankBoardEntry>[] = [
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
    }
  ];
  columns.push(
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
  );
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

function defaultRankSortOrder(value: string) {
  return value === "run_id" ? "asc" : "desc";
}

function toggleSortOrder(value: string) {
  return value === "desc" ? "asc" : "desc";
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
