import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import type { ColumnDef } from "@tanstack/react-table";
import { FileText, FileX } from "lucide-react";

import type { RankBoard, RankBoardEntry } from "./api";
import { fetchRankBoard } from "./api";
import { useDashboardState } from "./dashboardState";
import { StandaloneTextareaControl, ToggleButton } from "./controlPrimitives";
import { AdvancedFilterBar } from "./filterControls";
import { f1Score, facetValues, formatMetric } from "./formatters";
import { PagerControl, clampListPageOffset } from "./samplePager";
import {
  ActionButton,
  Badge,
  DataTable,
  DisclosurePanel,
  EmptyState,
  InlineNavLink,
  OptionChipButton
} from "./ui";

const RANK_SORT_LABELS: Record<string, string> = {
  f1_iou50: "F1@.50",
  precision_iou50: "P@.50",
  recall_iou50: "R@.50",
  mean_iou: "mIoU",
  prediction_count: "预测数",
  created_at: "创建时间",
  run_id: "Run ID",
  weighted_score: "Weighted"
};
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
const RANK_DIRECT_METRICS = [...RANK_PRIMARY_METRICS, ...RANK_AUXILIARY_SORTS];
const RANK_PAGE_SIZE = 80;

export function RankBoardPage() {
  const dashboardQuery = useDashboardState();
  const runs = dashboardQuery.data?.runs ?? [];
  const [searchText, setSearchText] = useState("");
  const [taskFilter, setTaskFilter] = useState("all");
  const [benchmarkFilter, setBenchmarkFilter] = useState("all");
  const [statusFilter, setStatusFilter] = useState("all");
  const [labelFilter, setLabelFilter] = useState("all");
  const [modelFilter, setModelFilter] = useState("all");
  const [promptFilter, setPromptFilter] = useState("all");
  const [metricProfileFilter, setMetricProfileFilter] = useState("all");
  const [minScoreFilter, setMinScoreFilter] = useState("");
  const [sortBy, setSortBy] = useState("f1_iou50");
  const [sortOrder, setSortOrder] = useState("desc");
  const [pageOffset, setPageOffset] = useState(0);
  const [rankSchemeEnabled, setRankSchemeEnabled] = useState(false);
  const [rankSchemeDraft, setRankSchemeDraft] = useState(defaultRankSchemeDraft());
  const [rankSchemeRequestError, setRankSchemeRequestError] = useState<string | null>(null);
  const rankSchemeError = rankSchemeEnabled ? validateRankSchemeDraft(rankSchemeDraft) : null;
  const rankSchemeParam = rankSchemeEnabled && !rankSchemeError ? rankSchemeDraft : undefined;
  const boardQuery = useQuery({
    queryKey: [
      "rank-board",
      searchText,
      taskFilter,
      benchmarkFilter,
      statusFilter,
      labelFilter,
      modelFilter,
      promptFilter,
      metricProfileFilter,
      minScoreFilter,
      sortBy,
      sortOrder,
      pageOffset,
      rankSchemeParam
    ],
    queryFn: async () => {
      try {
        const nextBoard = await fetchRankBoard({
          offset: pageOffset,
          limit: RANK_PAGE_SIZE,
          query: searchText,
          task: taskFilter,
          benchmarkId: benchmarkFilter,
          status: statusFilter,
          label: labelFilter,
          modelId: modelFilter,
          promptId: promptFilter,
          metricProfile: metricProfileFilter,
          minScore: minScoreFilter,
          sortBy,
          sortOrder,
          rankScheme: rankSchemeParam
        });
        setRankSchemeRequestError(null);
        return nextBoard;
      } catch (error) {
        if (rankSchemeParam) {
          setRankSchemeRequestError(error instanceof Error ? error.message : String(error));
        }
        throw error;
      }
    },
    placeholderData: (previousData) => previousData
  });
  const board = boardQuery.data;
  const tasks = facetValues(board?.facets, "tasks", runs.map((run) => run.spec_task));
  const benchmarks = facetValues(
    board?.facets,
    "benchmarks",
    runs.map((run) => run.benchmark_id)
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
  const best = entries[0] ?? null;
  useEffect(() => {
    setPageOffset(0);
  }, [
    searchText,
    taskFilter,
    benchmarkFilter,
    statusFilter,
    labelFilter,
    modelFilter,
    promptFilter,
    metricProfileFilter,
    minScoreFilter,
    sortBy,
    sortOrder,
    rankSchemeParam
  ]);
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
    return <EmptyState title="排行榜加载失败" tone="danger" />;
  }
  const rankSchemeApiError = rankSchemeEnabled && !rankSchemeError ? rankSchemeRequestError : null;

  return (
    <section className="page-stack density-page rank-board-page">
      <RankDecisionPanel
        board={board}
        best={best}
        runCount={runs.length}
        sortBy={sortBy}
        sortOrder={sortOrder}
        onSortByChange={setSortBy}
        onSortOrderChange={setSortOrder}
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
            onChange: setSearchText,
            placeholder: "搜索 run、模型、prompt、备注"
          },
          {
            type: "select",
            id: "rank-task",
            label: "任务",
            value: taskFilter,
            values: ["all", ...tasks],
            labels: { all: "全部" },
            onChange: setTaskFilter
          },
          {
            type: "select",
            id: "rank-benchmark",
            label: "基准集",
            value: benchmarkFilter,
            values: ["all", ...benchmarks],
            labels: { all: "全部" },
            onChange: setBenchmarkFilter
          },
          {
            type: "select",
            id: "rank-status",
            label: "状态",
            value: statusFilter,
            values: ["all", ...statuses],
            labels: { all: "全部" },
            onChange: setStatusFilter
          },
          {
            type: "select",
            id: "rank-label",
            label: "标签",
            value: labelFilter,
            values: ["all", ...labels],
            labels: { all: "全部" },
            onChange: setLabelFilter
          },
          {
            type: "select",
            id: "rank-model",
            label: "模型",
            value: modelFilter,
            values: ["all", ...models],
            labels: { all: "全部" },
            onChange: setModelFilter
          },
          {
            type: "select",
            id: "rank-prompt",
            label: "Prompt",
            value: promptFilter,
            values: ["all", ...prompts],
            labels: { all: "全部" },
            onChange: setPromptFilter
          },
          {
            type: "select",
            id: "rank-metric",
            label: "Metric",
            value: metricProfileFilter,
            values: ["all", ...metricProfiles],
            labels: { all: "全部" },
            onChange: setMetricProfileFilter
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
            onChange: setMinScoreFilter
          }
        ]}
        actions={
          <span className={board.rank_scheme ? "rank-formula-chip weighted" : "rank-formula-chip"}>
            <strong>
              {board.rank_scheme
                ? `Weighted ${board.primary_metric_label}`
                : `主指标 ${board.primary_metric_label}`}
            </strong>
            {board.sort_by !== board.primary_metric ? (
              <em>排序 {rankSortLabel(board.sort_by)}</em>
            ) : null}
          </span>
        }
      />
      <div className="workspace-card fill rank-board-table-card">
        <PagerControl
          className="rank-board-pager"
          offset={board.offset}
          limit={board.limit}
          total={board.total}
          onPageChange={setPageOffset}
        />
        <RankBoardTable
          entries={entries}
          primaryMetric={board.primary_metric}
          primaryMetricLabel={board.primary_metric_label}
          weighted={Boolean(board.rank_scheme)}
        />
      </div>
      <RankSchemePanel
        enabled={rankSchemeEnabled}
        draft={rankSchemeDraft}
        error={rankSchemeError}
        apiError={rankSchemeApiError}
        board={board}
        benchmarks={benchmarks}
        onEnabledChange={setRankSchemeEnabled}
        onDraftChange={setRankSchemeDraft}
      />
      <RankFacetRail
        board={board}
        filters={{
          task: taskFilter,
          benchmark: benchmarkFilter,
          status: statusFilter,
          label: labelFilter,
          model: modelFilter,
          prompt: promptFilter,
          metricProfile: metricProfileFilter
        }}
        onFilterChange={{
          task: setTaskFilter,
          benchmark: setBenchmarkFilter,
          status: setStatusFilter,
          label: setLabelFilter,
          model: setModelFilter,
          prompt: setPromptFilter,
          metricProfile: setMetricProfileFilter
        }}
      />
    </section>
  );
}

function RankDecisionPanel({
  board,
  best,
  runCount,
  sortBy,
  sortOrder,
  onSortByChange,
  onSortOrderChange
}: {
  board: RankBoard;
  best: RankBoardEntry | null;
  runCount: number;
  sortBy: string;
  sortOrder: string;
  onSortByChange: (value: string) => void;
  onSortOrderChange: (value: string) => void;
}) {
  return (
    <section className="rank-decision-panel rank-leaderboard-toolbar">
      <div className="rank-board-summary">
        <strong>Leaderboard</strong>
        <span>{board.total.toLocaleString()} runs</span>
        <span>{board.evaluated_count.toLocaleString()} evaluated</span>
        <span>{facetTotal(board, "benchmarks").toLocaleString()} benchmarks</span>
        <span>{runCount.toLocaleString()} total</span>
      </div>
      <div className="rank-board-leading">
        <span>{rankBoardOrderLabel(board)}</span>
        {best ? (
          <InlineNavLink to="/runs/$runId" params={{ runId: best.run_id }}>
            #{best.rank} {best.run_id} · {formatMetric(best.score)}
          </InlineNavLink>
        ) : null}
      </div>
      <div className="rank-toolbar-controls">
        <div className="rank-sort-section">
          <span>主指标</span>
          <div className="rank-sort-dial" role="group" aria-label="排行榜主指标">
            {RANK_PRIMARY_METRICS.map((metric) => (
              <OptionChipButton
                key={metric}
                active={sortBy === metric}
                className="rank-sort-chip primary"
                onClick={() => onSortByChange(metric)}
              >
                {rankSortLabel(metric)}
              </OptionChipButton>
            ))}
          </div>
        </div>
        <div className="rank-sort-section auxiliary">
          <span>排序</span>
          <div className="rank-sort-dial" role="group" aria-label="排行榜辅助排序字段">
            {RANK_AUXILIARY_SORTS.map((metric) => (
              <OptionChipButton
                key={metric}
                active={sortBy === metric}
                className="rank-sort-chip auxiliary"
                onClick={() => onSortByChange(metric)}
              >
                {rankSortLabel(metric)}
              </OptionChipButton>
            ))}
          </div>
        </div>
        <div className="rank-order-row">
          <OptionChipButton
            active={sortOrder === "desc"}
            className="rank-order-chip"
            onClick={() => onSortOrderChange("desc")}
          >
            降序
          </OptionChipButton>
          <OptionChipButton
            active={sortOrder === "asc"}
            className="rank-order-chip"
            onClick={() => onSortOrderChange("asc")}
          >
            升序
          </OptionChipButton>
        </div>
      </div>
      <span className="rank-score-formula">{board.score_formula}</span>
    </section>
  );
}

function rankSortLabel(value: string) {
  return RANK_SORT_LABELS[value] ?? value;
}

function rankBoardOrderLabel(
  board: Pick<RankBoard, "primary_metric" | "primary_metric_label" | "sort_by" | "sort_order">
) {
  const direction = board.sort_order === "asc" ? "升序" : "降序";
  if (board.sort_by !== board.primary_metric) {
    return `主指标 ${board.primary_metric_label}，按 ${rankSortLabel(board.sort_by)} ${direction}`;
  }
  return `按主指标 ${board.primary_metric_label} ${direction}`;
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
    status: string;
    label: string;
    model: string;
    prompt: string;
    metricProfile: string;
  };
  onFilterChange: {
    task: (value: string) => void;
    benchmark: (value: string) => void;
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
  primaryMetricLabel,
  weighted
}: {
  entries: RankBoardEntry[];
  primaryMetric: string;
  primaryMetricLabel: string;
  weighted: boolean;
}) {
  const columns: ColumnDef<RankBoardEntry>[] = [
    {
      header: "Rank",
      cell: ({ row }) => <span className="rank-index">#{row.original.rank}</span>
    },
    {
      header: "Run",
      cell: ({ row }) => (
        <Link to="/runs/$runId" params={{ runId: row.original.run_id }}>
          {row.original.run_id}
        </Link>
      )
    },
    {
      header: primaryMetricLabel,
      cell: ({ row }) => <span className="rank-primary-score">{formatMetric(row.original.score)}</span>
    },
    {
      header: "Δ leader",
      cell: ({ row }) => (
        <span className={rankDeltaClassName(row.original.score_delta)}>
          {formatScoreDelta(row.original.score_delta)}
        </span>
      )
    }
  ];
  if (weighted) {
    columns.push(
      {
        header: "Components",
        cell: ({ row }) => <RankScoreComponents components={row.original.score_components} />
      }
    );
  }
  if (primaryMetric !== "f1_iou50") {
    columns.push({ header: "F1@.50", cell: ({ row }) => formatMetric(rankF1Score(row.original)) });
  }
  columns.push(
    { header: "状态", cell: ({ row }) => <Badge value={row.original.status} domain="run" /> },
    { header: "任务", accessorKey: "task" },
    { header: "标签", cell: ({ row }) => row.original.target_labels.join(", ") || "-" },
    { header: "基准集", accessorKey: "benchmark_id" },
    { header: "模型", accessorKey: "model_id" },
    { header: "Prompt", accessorKey: "prompt_id" },
    { header: "P@.50", cell: ({ row }) => formatMetric(row.original.precision_iou50) },
    { header: "R@.50", cell: ({ row }) => formatMetric(row.original.recall_iou50) },
    { header: "mIoU", cell: ({ row }) => formatMetric(row.original.mean_iou) },
    {
      header: "备注",
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
  return <DataTable columns={columns} data={entries} emptyText="没有符合高级检索条件的 run。" />;
}

function RankSchemePanel({
  enabled,
  draft,
  error,
  apiError,
  board,
  benchmarks,
  onEnabledChange,
  onDraftChange
}: {
  enabled: boolean;
  draft: string;
  error: string | null;
  apiError: string | null;
  board: RankBoard;
  benchmarks: string[];
  onEnabledChange: (value: boolean) => void;
  onDraftChange: (value: string) => void;
}) {
  const applied = Boolean(board.rank_scheme);
  return (
    <DisclosurePanel
      className="rank-scheme-panel"
      summary={
        <>
          <div>
            <span>Weighted rank scheme</span>
            <strong>{applied ? board.primary_metric_label : "显式加权方案"}</strong>
          </div>
          <em>{applied ? "已应用" : enabled ? "待验证" : "默认 F1"}</em>
        </>
      }
    >
      <div className="rank-scheme-body">
        <ToggleButton label="启用加权排行" active={enabled} onChange={onEnabledChange} />
        <StandaloneTextareaControl
          label="Weighted rank scheme JSON"
          value={draft}
          onChange={onDraftChange}
          spellCheck={false}
        />
        <div className="rank-scheme-footer">
          <span className={error || apiError ? "rank-scheme-status error" : "rank-scheme-status"}>
            {error ?? apiError ?? (applied ? board.score_formula : "默认按 F1@.50 排序，不使用综合分。")}
          </span>
          <ActionButton
            variant="mini"
            onClick={() => onDraftChange(defaultRankSchemeDraft(benchmarks))}
          >
            载入示例
          </ActionButton>
        </div>
      </div>
    </DisclosurePanel>
  );
}

function RankScoreComponents({ components }: { components: Array<Record<string, unknown>> }) {
  if (components.length === 0) {
    return <span className="rank-components empty">-</span>;
  }
  return (
    <span className="rank-components">
      {components.slice(0, 3).map((component, index) => (
        <em key={`${component.benchmark_id ?? "bench"}-${component.metric ?? "metric"}-${index}`}>
          {String(component.benchmark_id ?? "-")}.{String(component.metric ?? "-")}{" "}
          <strong>{formatMetric(numericValue(component.value))}</strong>
        </em>
      ))}
    </span>
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

function defaultRankSchemeDraft(benchmarks: string[] = []) {
  const benchmarkId = benchmarks[0] ?? "multitask_val_v1";
  return JSON.stringify(
    {
      name: "weighted_quality",
      terms: [
        { benchmark_id: benchmarkId, metric: "f1_iou50", weight: 0.7, missing: "drop" },
        { benchmark_id: benchmarkId, metric: "mean_iou", weight: 0.3, missing: "zero" }
      ]
    },
    null,
    2
  );
}

function validateRankSchemeDraft(value: string) {
  let payload: unknown;
  try {
    payload = JSON.parse(value);
  } catch {
    return "rank_scheme 必须是 JSON object";
  }
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    return "rank_scheme 必须是 JSON object";
  }
  const terms = (payload as { terms?: unknown; weights?: unknown }).terms
    ?? (payload as { weights?: unknown }).weights;
  if (!Array.isArray(terms) || terms.length === 0) {
    return "terms 必须是非空数组";
  }
  for (const [index, term] of terms.entries()) {
    if (!term || typeof term !== "object" || Array.isArray(term)) {
      return `terms[${index}] 必须是 object`;
    }
    const item = term as Record<string, unknown>;
    if (typeof item.benchmark_id !== "string" || item.benchmark_id.trim() === "") {
      return `terms[${index}].benchmark_id 必填`;
    }
    if (typeof item.metric !== "string" || item.metric.trim() === "") {
      return `terms[${index}].metric 必填`;
    }
    const weight = Number(item.weight);
    if (!Number.isFinite(weight) || weight <= 0) {
      return `terms[${index}].weight 必须为正数`;
    }
    const missing = typeof item.missing === "string" ? item.missing : "drop";
    if (!["drop", "skip", "zero"].includes(missing)) {
      return `terms[${index}].missing 必须是 drop/skip/zero`;
    }
  }
  return null;
}

function numericValue(value: unknown) {
  return typeof value === "number" ? value : null;
}
