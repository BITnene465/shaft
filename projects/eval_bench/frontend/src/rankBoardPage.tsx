import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import type { ColumnDef } from "@tanstack/react-table";
import { Trophy } from "lucide-react";

import type { RankBoard, RankBoardEntry } from "./api";
import { fetchRankBoard } from "./api";
import { useDashboardState } from "./dashboardState";
import { AdvancedFilterBar } from "./filterControls";
import { formatMetric, unique } from "./formatters";
import { AppIcon } from "./iconLibrary";
import { Badge, DataTable, EmptyState, MetricCard } from "./ui";

const RANK_SORT_LABELS: Record<string, string> = {
  score: "综合分",
  precision_iou50: "P@.50",
  recall_iou50: "R@.50",
  mean_iou: "mIoU",
  prediction_count: "预测数",
  created_at: "创建时间",
  run_id: "Run ID"
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
  const [sortBy, setSortBy] = useState("score");
  const [sortOrder, setSortOrder] = useState("desc");
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
      sortOrder
    ],
    queryFn: () =>
      fetchRankBoard({
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
        sortOrder
      })
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
  const best = entries.find((entry) => entry.score !== null) ?? null;

  if (dashboardQuery.isLoading || boardQuery.isLoading) {
    return <EmptyState title="正在加载排行榜" />;
  }
  if (dashboardQuery.error || boardQuery.error || !board) {
    return <EmptyState title="排行榜加载失败" tone="danger" />;
  }

  return (
    <section className="page-stack density-page rank-board-page">
      <div className="page-command-row">
        <div>
          <h2>独立排行榜</h2>
          <span>
            {board.total.toLocaleString()} 条 run，按 {rankSortLabel(board.sort_by)}{" "}
            {board.sort_order === "asc" ? "升序" : "降序"}
          </span>
        </div>
        {best ? (
          <Link className="mini-link compare-ready" to="/runs/$runId" params={{ runId: best.run_id }}>
            <Trophy size={13} />
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
              "score",
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
        actions={<span className="rank-formula-chip">{board.score_formula}</span>}
      />
      <div className="rank-metric-strip">
        <MetricCard icon={<Trophy size={24} />} label="入榜" value={board.total} />
        <MetricCard icon={<AppIcon name="metrics" size={24} />} label="已评估" value={board.evaluated_count} />
        <MetricCard icon={<AppIcon name="benchmark" size={24} />} label="基准集" value={facetTotal(board, "benchmarks")} />
        <MetricCard icon={<AppIcon name="runResults" size={24} />} label="Run 总数" value={runs.length} />
      </div>
      <RankFacetRail board={board} />
      <div className="workspace-card fill">
        <RankBoardTable entries={entries} />
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

function RankBoardTable({ entries }: { entries: RankBoardEntry[] }) {
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
    { header: "综合分", cell: ({ row }) => formatMetric(row.original.score) },
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
  ];
  return <DataTable columns={columns} data={entries} emptyText="没有符合高级检索条件的 run。" />;
}
