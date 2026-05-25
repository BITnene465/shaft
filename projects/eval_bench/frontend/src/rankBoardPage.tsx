import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import type { ColumnDef } from "@tanstack/react-table";

import type { RankBoard, RankBoardEntry } from "./api";
import { fetchRankBoard } from "./api";
import { useDashboardState } from "./dashboardState";
import { AdvancedFilterBar } from "./filterControls";
import { formatMetric, unique } from "./formatters";
import { AppIcon } from "./iconLibrary";
import { ActionButton, Badge, DataTable, EmptyState, MetricCard } from "./ui";

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
      rankSchemeParam
    ],
    queryFn: async () => {
      try {
        const nextBoard = await fetchRankBoard({
          offset: 0,
          limit: 200,
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
  const tasks = unique(runs.map((run) => run.spec_task).filter(Boolean));
  const benchmarks = unique(runs.map((run) => run.benchmark_id).filter(Boolean));
  const statuses = unique(runs.map((run) => run.status).filter(Boolean));
  const labels = unique(runs.flatMap((run) => run.target_labels).filter(Boolean));
  const models = unique(runs.map((run) => run.model_id).filter(Boolean));
  const prompts = unique(runs.map((run) => run.prompt_id).filter(Boolean));
  const metricProfiles = unique(runs.map((run) => run.metric_profile).filter(Boolean));
  const board = boardQuery.data;
  const entries = board?.entries ?? [];
  const best = entries[0] ?? null;

  if (dashboardQuery.isLoading || (boardQuery.isLoading && !board)) {
    return <EmptyState title="正在加载排行榜" />;
  }
  if (dashboardQuery.error || !board) {
    return <EmptyState title="排行榜加载失败" tone="danger" />;
  }
  const rankSchemeApiError = rankSchemeEnabled && !rankSchemeError ? rankSchemeRequestError : null;

  return (
    <section className="page-stack density-page rank-board-page">
      <div className="page-command-row">
        <div>
          <h2>独立排行榜</h2>
          <span>
            {board.total.toLocaleString()} 条 run，按{" "}
            {board.primary_metric_label || rankSortLabel(board.sort_by)}{" "}
            {board.sort_order === "asc" ? "升序" : "降序"}
          </span>
        </div>
        {best ? (
          <Link
            className="mini-link compare-ready"
            to="/runs/$runId"
            params={{ runId: best.run_id }}
          >
            <AppIcon name="rankBoard" size={13} />
            当前第一 {best.run_id}
          </Link>
        ) : null}
      </div>
      <AdvancedFilterBar
        title="排行榜高级检索"
        meta="按论文检索式筛 run：任务、基准集、状态、label、模型、prompt、metric、分数门槛与备注全文"
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
          },
          {
            type: "select",
            id: "rank-sort-by",
            label: "排序",
            value: sortBy,
            values: [
              "f1_iou50",
              "precision_iou50",
              "recall_iou50",
              "mean_iou",
              "prediction_count",
              "created_at",
              "run_id"
            ],
            labels: RANK_SORT_LABELS,
            onChange: setSortBy
          },
          {
            type: "select",
            id: "rank-sort-order",
            label: "方向",
            value: sortOrder,
            values: ["desc", "asc"],
            labels: { desc: "降序", asc: "升序" },
            onChange: setSortOrder
          }
        ]}
        actions={
          <span className={board.rank_scheme ? "rank-formula-chip weighted" : "rank-formula-chip"}>
            {board.rank_scheme
              ? `Weighted ${board.primary_metric_label}`
              : `主指标 ${rankSortLabel(board.sort_by)}`}
          </span>
        }
      />
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
      <div className="rank-metric-strip">
        <MetricCard icon={<AppIcon name="rankEntry" size={24} />} label="入榜" value={board.total} />
        <MetricCard
          icon={<AppIcon name="evaluatedRun" size={24} />}
          label="已评估"
          value={board.evaluated_count}
        />
        <MetricCard
          icon={<AppIcon name="benchmark" size={24} />}
          label="基准集"
          value={facetTotal(board, "benchmarks")}
        />
        <MetricCard
          icon={<AppIcon name="runResults" size={24} />}
          label="Run 总数"
          value={runs.length}
        />
      </div>
      <RankFacetRail board={board} />
      <div className="workspace-card fill">
        <RankBoardTable entries={entries} weighted={Boolean(board.rank_scheme)} />
      </div>
    </section>
  );
}

function rankSortLabel(value: string) {
  return RANK_SORT_LABELS[value] ?? value;
}

function facetTotal(board: Pick<RankBoard, "facets">, key: string) {
  return board.facets[key]?.length ?? 0;
}

function RankFacetRail({ board }: { board: Pick<RankBoard, "facets"> }) {
  return (
    <div className="rank-facet-rail">
      <RankFacetGroup title="Labels" items={board.facets.labels ?? []} />
      <RankFacetGroup title="Models" items={board.facets.models ?? []} />
      <RankFacetGroup title="Prompts" items={board.facets.prompts ?? []} />
      <RankFacetGroup title="Metrics" items={board.facets.metric_profiles ?? []} />
    </div>
  );
}

function RankFacetGroup({
  title,
  items
}: {
  title: string;
  items: Array<{ value: string; count: number }>;
}) {
  return (
    <section className="rank-facet-group">
      <span>{title}</span>
      <div>
        {items.slice(0, 5).map((item) => (
          <em key={item.value}>
            {item.value} <strong>{item.count.toLocaleString()}</strong>
          </em>
        ))}
        {items.length === 0 ? <em>无</em> : null}
      </div>
    </section>
  );
}

function RankBoardTable({ entries, weighted }: { entries: RankBoardEntry[]; weighted: boolean }) {
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
    }
  ];
  if (weighted) {
    columns.push(
      {
        header: "Weighted",
        cell: ({ row }) => formatMetric(row.original.score)
      },
      {
        header: "Components",
        cell: ({ row }) => <RankScoreComponents components={row.original.score_components} />
      }
    );
  }
  columns.push(
    { header: "F1@.50", cell: ({ row }) => formatMetric(rankF1Score(row.original)) },
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
      cell: ({ row }) => (
        <span className={row.original.note ? "run-note-preview" : "run-note-preview empty"}>
          {row.original.note || "-"}
        </span>
      )
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
    <details className="rank-scheme-panel">
      <summary>
        <div>
          <span>Weighted rank scheme</span>
          <strong>{applied ? board.primary_metric_label : "显式加权方案"}</strong>
        </div>
        <em>{applied ? "已应用" : enabled ? "待验证" : "默认 F1"}</em>
      </summary>
      <div className="rank-scheme-body">
        <label className={enabled ? "control-check active" : "control-check"}>
          <input
            type="checkbox"
            checked={enabled}
            onChange={(event) => onEnabledChange(event.target.checked)}
          />
          启用加权排行
        </label>
        <textarea
          value={draft}
          onChange={(event) => onDraftChange(event.target.value)}
          spellCheck={false}
          aria-label="Weighted rank scheme JSON"
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
    </details>
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
  return entry.f1_iou50 ?? entry.score ?? null;
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
