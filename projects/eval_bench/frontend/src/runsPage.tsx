import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { fetchRuns } from "./api";
import { useDashboardState } from "./dashboardState";
import { errorMessage, facetValues } from "./formatters";
import { AppIcon } from "./iconLibrary";
import { ImportPredictionsPanel } from "./runsImportPanel";
import {
  DEFAULT_RUNS_VIEW_STATE,
  RUNS_VIEW_STATE_RESET_EVENT,
  readRunsViewState,
  writeRunsViewState
} from "./runsViewState";
import { RunTable } from "./runTables";
import {
  PagerControl,
  clampListPageOffset,
  updatePagedFilterValue
} from "./samplePager";
import { CommandButton, EmptyState, WorkspaceDialog } from "./ui";

import "./runsPage.css";

export { RunDetailPage } from "./runDetailPage";

const RUN_PAGE_SIZE = 80;

export function RunsPage() {
  const dashboardQuery = useDashboardState();
  const [importOpen, setImportOpen] = useState(false);
  const [initialViewState] = useState(readRunsViewState);
  const [searchText, setSearchText] = useState(initialViewState.searchText);
  const [statusFilter, setStatusFilter] = useState(initialViewState.statusFilter);
  const [taskFilter, setTaskFilter] = useState(initialViewState.taskFilter);
  const [benchmarkFilter, setBenchmarkFilter] = useState(initialViewState.benchmarkFilter);
  const [benchmarkSplitFilter, setBenchmarkSplitFilter] = useState(
    initialViewState.benchmarkSplitFilter
  );
  const [labelFilter, setLabelFilter] = useState(initialViewState.labelFilter);
  const [modelFilter, setModelFilter] = useState(initialViewState.modelFilter);
  const [promptFilter, setPromptFilter] = useState(initialViewState.promptFilter);
  const [metricProfileFilter, setMetricProfileFilter] = useState(
    initialViewState.metricProfileFilter
  );
  const [pageOffset, setPageOffset] = useState(initialViewState.pageOffset);
  function resetViewState() {
    setImportOpen(false);
    setSearchText(DEFAULT_RUNS_VIEW_STATE.searchText);
    setStatusFilter(DEFAULT_RUNS_VIEW_STATE.statusFilter);
    setTaskFilter(DEFAULT_RUNS_VIEW_STATE.taskFilter);
    setBenchmarkFilter(DEFAULT_RUNS_VIEW_STATE.benchmarkFilter);
    setBenchmarkSplitFilter(DEFAULT_RUNS_VIEW_STATE.benchmarkSplitFilter);
    setLabelFilter(DEFAULT_RUNS_VIEW_STATE.labelFilter);
    setModelFilter(DEFAULT_RUNS_VIEW_STATE.modelFilter);
    setPromptFilter(DEFAULT_RUNS_VIEW_STATE.promptFilter);
    setMetricProfileFilter(DEFAULT_RUNS_VIEW_STATE.metricProfileFilter);
    setPageOffset(DEFAULT_RUNS_VIEW_STATE.pageOffset);
  }
  const runFilters = useMemo(
    () => ({
      offset: pageOffset,
      limit: RUN_PAGE_SIZE,
      status: statusFilter !== "all" ? statusFilter : undefined,
      task: taskFilter !== "all" ? taskFilter : undefined,
      benchmarkId: benchmarkFilter !== "all" ? benchmarkFilter : undefined,
      benchmarkSplit: benchmarkSplitFilter !== "all" ? benchmarkSplitFilter : undefined,
      label: labelFilter !== "all" ? labelFilter : undefined,
      modelId: modelFilter !== "all" ? modelFilter : undefined,
      promptId: promptFilter !== "all" ? promptFilter : undefined,
      metricProfile: metricProfileFilter !== "all" ? metricProfileFilter : undefined,
      query: searchText.trim() || undefined
    }),
    [
      benchmarkFilter,
      benchmarkSplitFilter,
      labelFilter,
      metricProfileFilter,
      modelFilter,
      pageOffset,
      promptFilter,
      searchText,
      statusFilter,
      taskFilter
    ]
  );
  const runsQuery = useQuery({
    queryKey: ["runs", runFilters],
    queryFn: ({ signal }) => fetchRuns(runFilters, { signal }),
    placeholderData: (previousData) => previousData
  });
  const runs = runsQuery.data?.runs ?? [];
  const facets = runsQuery.data?.facets;
  const tasks = facetValues(facets, "tasks", runs.map((run) => run.spec_task));
  const benchmarks = facetValues(facets, "benchmarks", runs.map((run) => run.benchmark_id));
  const benchmarkSplits = facetValues(
    facets,
    "splits",
    runs.map((run) => run.benchmark_split)
  );
  const statuses = facetValues(facets, "statuses", runs.map((run) => run.status));
  const labels = facetValues(facets, "labels", runs.flatMap((run) => run.target_labels));
  const models = facetValues(facets, "models", runs.map((run) => run.model_id));
  const prompts = facetValues(facets, "prompts", runs.map((run) => run.prompt_id));
  const metricProfiles = facetValues(
    facets,
    "metric_profiles",
    runs.map((run) => run.metric_profile)
  );
  const totalRuns = runsQuery.data?.total ?? runs.length;
  useEffect(() => {
    const nextOffset = clampListPageOffset(pageOffset, totalRuns, RUN_PAGE_SIZE);
    if (nextOffset !== pageOffset) {
      setPageOffset(nextOffset);
    }
  }, [pageOffset, totalRuns]);
  useEffect(() => {
    writeRunsViewState({
      searchText,
      statusFilter,
      taskFilter,
      benchmarkFilter,
      benchmarkSplitFilter,
      labelFilter,
      modelFilter,
      promptFilter,
      metricProfileFilter,
      pageOffset
    });
  }, [
    searchText,
    statusFilter,
    taskFilter,
    benchmarkFilter,
    benchmarkSplitFilter,
    labelFilter,
    modelFilter,
    promptFilter,
    metricProfileFilter,
    pageOffset
  ]);
  useEffect(() => {
    window.addEventListener(RUNS_VIEW_STATE_RESET_EVENT, resetViewState);
    return () => window.removeEventListener(RUNS_VIEW_STATE_RESET_EVENT, resetViewState);
  }, []);
  if (runsQuery.isLoading || dashboardQuery.isLoading) {
    return <EmptyState title="正在加载评测记录" />;
  }
  if (runsQuery.error || !runsQuery.data) {
    return <EmptyState title={`评测记录加载失败：${errorMessage(runsQuery.error)}`} tone="danger" />;
  }
  const benchmarkOptions = dashboardQuery.data?.benchmarks ?? [];
  return (
    <section className="page-stack density-page">
      <div className="page-command-row">
        <div>
          <h2>评测记录库</h2>
          <span>{totalRuns.toLocaleString()} 条 run snapshot</span>
        </div>
        <CommandButton
          variant="secondary"
          icon={<AppIcon name="importPrediction" size={17} />}
          onClick={() => setImportOpen(true)}
        >
          导入预测
        </CommandButton>
      </div>
      <div className="workspace-card fill run-table-card">
        <RunTable
          runs={runs}
          refreshing={runsQuery.isPlaceholderData}
          filterMeta={`${runs.length.toLocaleString()} / ${totalRuns.toLocaleString()} 条 run`}
          filterControls={[
            {
              type: "search",
              id: "run-query",
              label: "全文检索",
              value: searchText,
              onChange: (value) =>
                updatePagedFilterValue(searchText, value, setSearchText, setPageOffset),
              placeholder: "搜索 run、模型、基准集、备注"
            },
            {
              type: "select",
              id: "run-status",
              label: "状态",
              value: statusFilter,
              values: ["all", ...statuses],
              labels: { all: "全部" },
              onChange: (value) =>
                updatePagedFilterValue(statusFilter, value, setStatusFilter, setPageOffset)
            },
            {
              type: "select",
              id: "run-task",
              label: "任务",
              value: taskFilter,
              values: ["all", ...tasks],
              labels: { all: "全部" },
              onChange: (value) =>
                updatePagedFilterValue(taskFilter, value, setTaskFilter, setPageOffset)
            },
            {
              type: "select",
              id: "run-benchmark",
              label: "基准集",
              value: benchmarkFilter,
              values: ["all", ...benchmarks],
              labels: { all: "全部" },
              onChange: (value) =>
                updatePagedFilterValue(benchmarkFilter, value, setBenchmarkFilter, setPageOffset)
            },
            {
              type: "select",
              id: "run-benchmark-split",
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
              id: "run-label",
              label: "标签",
              value: labelFilter,
              values: ["all", ...labels],
              labels: { all: "全部" },
              onChange: (value) =>
                updatePagedFilterValue(labelFilter, value, setLabelFilter, setPageOffset)
            },
            {
              type: "select",
              id: "run-model",
              label: "模型",
              value: modelFilter,
              values: ["all", ...models],
              labels: { all: "全部" },
              onChange: (value) =>
                updatePagedFilterValue(modelFilter, value, setModelFilter, setPageOffset)
            },
            {
              type: "select",
              id: "run-prompt",
              label: "Prompt",
              value: promptFilter,
              values: ["all", ...prompts],
              labels: { all: "全部" },
              onChange: (value) =>
                updatePagedFilterValue(promptFilter, value, setPromptFilter, setPageOffset)
            },
            {
              type: "select",
              id: "run-metric",
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
            }
          ]}
          footer={
            <PagerControl
              className="rank-board-pager run-list-pager"
              offset={runsQuery.data.offset ?? pageOffset}
              limit={runsQuery.data.limit ?? RUN_PAGE_SIZE}
              total={totalRuns}
              onPageChange={setPageOffset}
            />
          }
        />
      </div>
      <WorkspaceDialog
        open={importOpen}
        title="导入预测快照"
        meta="把外部预测目录导入为 run，并和 GT 对比"
        onClose={() => setImportOpen(false)}
      >
        <ImportPredictionsPanel benchmarks={benchmarkOptions} bare />
      </WorkspaceDialog>
    </section>
  );
}
