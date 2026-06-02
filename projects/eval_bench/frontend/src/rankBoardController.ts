import { useCallback, useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { fetchRankBoard, fetchSuiteRankBoard } from "./api";
import { useDashboardState } from "./dashboardState";
import { errorMessage, facetValues } from "./formatters";
import { clampListPageOffset, updatePagedFilterValue } from "./samplePager";
import {
  DEFAULT_RANK_BOARD_VIEW_STATE,
  RANK_BOARD_VIEW_STATE_RESET_EVENT,
  readRankBoardViewState,
  writeRankBoardViewState
} from "./rankBoardViewState";
import {
  RANK_PAGE_SIZE,
  RANK_SORTABLE_FIELDS,
  defaultRankSortOrder,
  defaultSuiteSortOrder,
  toggleSortOrder
} from "./rankBoardModel";
import type {
  RankBoardFilterOptions,
  RankBoardFilterSetters,
  RankBoardFilterValues
} from "./rankBoardFilters";
import type {
  RankFacetFilterHandlers,
  RankFacetFilterValues
} from "./rankBoardFacets";

export function useRankBoardController() {
  const dashboardQuery = useDashboardState();
  const runs = dashboardQuery.data?.runs ?? [];
  const suiteApiReady = typeof dashboardQuery.data?.suite_count === "number";
  const [initialViewState] = useState(readRankBoardViewState);
  const [boardMode, setBoardMode] = useState<"run" | "suite">(initialViewState.boardMode);
  const [searchText, setSearchText] = useState(initialViewState.searchText);
  const [taskFilter, setTaskFilter] = useState(initialViewState.taskFilter);
  const [benchmarkFilter, setBenchmarkFilter] = useState(initialViewState.benchmarkFilter);
  const [benchmarkSplitFilter, setBenchmarkSplitFilter] = useState(
    initialViewState.benchmarkSplitFilter
  );
  const [statusFilter, setStatusFilter] = useState(initialViewState.statusFilter);
  const [labelFilter, setLabelFilter] = useState(initialViewState.labelFilter);
  const [modelFilter, setModelFilter] = useState(initialViewState.modelFilter);
  const [promptFilter, setPromptFilter] = useState(initialViewState.promptFilter);
  const [metricProfileFilter, setMetricProfileFilter] = useState(
    initialViewState.metricProfileFilter
  );
  const [minScoreFilter, setMinScoreFilter] = useState(initialViewState.minScoreFilter);
  const [sortBy, setSortBy] = useState(
    RANK_SORTABLE_FIELDS.includes(initialViewState.sortBy)
      ? initialViewState.sortBy
      : "f1_iou50"
  );
  const [sortOrder, setSortOrder] = useState(initialViewState.sortOrder === "asc" ? "asc" : "desc");
  const [pageOffset, setPageOffset] = useState(initialViewState.pageOffset);
  const [suiteSortBy, setSuiteSortBy] = useState("aggregate_score");
  const [suiteSortOrder, setSuiteSortOrder] = useState<"asc" | "desc">("desc");

  const resetViewState = useCallback(() => {
    setBoardMode(DEFAULT_RANK_BOARD_VIEW_STATE.boardMode);
    setSearchText(DEFAULT_RANK_BOARD_VIEW_STATE.searchText);
    setTaskFilter(DEFAULT_RANK_BOARD_VIEW_STATE.taskFilter);
    setBenchmarkFilter(DEFAULT_RANK_BOARD_VIEW_STATE.benchmarkFilter);
    setBenchmarkSplitFilter(DEFAULT_RANK_BOARD_VIEW_STATE.benchmarkSplitFilter);
    setStatusFilter(DEFAULT_RANK_BOARD_VIEW_STATE.statusFilter);
    setLabelFilter(DEFAULT_RANK_BOARD_VIEW_STATE.labelFilter);
    setModelFilter(DEFAULT_RANK_BOARD_VIEW_STATE.modelFilter);
    setPromptFilter(DEFAULT_RANK_BOARD_VIEW_STATE.promptFilter);
    setMetricProfileFilter(DEFAULT_RANK_BOARD_VIEW_STATE.metricProfileFilter);
    setMinScoreFilter(DEFAULT_RANK_BOARD_VIEW_STATE.minScoreFilter);
    setSortBy(DEFAULT_RANK_BOARD_VIEW_STATE.sortBy);
    setSortOrder(DEFAULT_RANK_BOARD_VIEW_STATE.sortOrder);
    setPageOffset(DEFAULT_RANK_BOARD_VIEW_STATE.pageOffset);
    setSuiteSortBy("aggregate_score");
    setSuiteSortOrder("desc");
  }, []);

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
  const suiteRankQuery = useQuery({
    queryKey: ["suite-rank-board", suiteSortBy, suiteSortOrder, pageOffset],
    queryFn: () =>
      fetchSuiteRankBoard({
        offset: pageOffset,
        limit: RANK_PAGE_SIZE,
        sortBy: suiteSortBy,
        sortOrder: suiteSortOrder
      }),
    enabled: suiteApiReady,
    placeholderData: (previousData) => previousData
  });

  const board = boardQuery.data;
  const suiteBoard = suiteRankQuery.data;
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
  const filterValues = useMemo<RankBoardFilterValues>(
    () => ({
      searchText,
      taskFilter,
      benchmarkFilter,
      benchmarkSplitFilter,
      statusFilter,
      labelFilter,
      modelFilter,
      promptFilter,
      metricProfileFilter,
      minScoreFilter
    }),
    [
      benchmarkFilter,
      benchmarkSplitFilter,
      labelFilter,
      metricProfileFilter,
      minScoreFilter,
      modelFilter,
      promptFilter,
      searchText,
      statusFilter,
      taskFilter
    ]
  );
  const filterOptions = useMemo<RankBoardFilterOptions>(
    () => ({
      tasks,
      benchmarks,
      benchmarkSplits,
      statuses,
      labels,
      models,
      prompts,
      metricProfiles
    }),
    [benchmarkSplits, benchmarks, labels, metricProfiles, models, prompts, statuses, tasks]
  );
  const filterSetters = useMemo<RankBoardFilterSetters>(
    () => ({
      setSearchText,
      setTaskFilter,
      setBenchmarkFilter,
      setBenchmarkSplitFilter,
      setStatusFilter,
      setLabelFilter,
      setModelFilter,
      setPromptFilter,
      setMetricProfileFilter,
      setMinScoreFilter,
      setPageOffset
    }),
    []
  );
  const facetFilters = useMemo<RankFacetFilterValues>(
    () => ({
      task: taskFilter,
      benchmark: benchmarkFilter,
      split: benchmarkSplitFilter,
      status: statusFilter,
      label: labelFilter,
      model: modelFilter,
      prompt: promptFilter,
      metricProfile: metricProfileFilter
    }),
    [
      benchmarkFilter,
      benchmarkSplitFilter,
      labelFilter,
      metricProfileFilter,
      modelFilter,
      promptFilter,
      statusFilter,
      taskFilter
    ]
  );
  const facetHandlers = useMemo<RankFacetFilterHandlers>(
    () => ({
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
    }),
    [
      benchmarkFilter,
      benchmarkSplitFilter,
      labelFilter,
      metricProfileFilter,
      modelFilter,
      promptFilter,
      statusFilter,
      taskFilter
    ]
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
  const handleSuiteSortChange = (value: string) => {
    const nextOrder =
      suiteSortBy === value ? toggleSortOrder(suiteSortOrder) : defaultSuiteSortOrder(value);
    setSuiteSortBy(value);
    setSuiteSortOrder(nextOrder);
    setPageOffset(0);
  };

  useEffect(() => {
    const total = boardMode === "suite" ? suiteBoard?.total : board?.total;
    if (typeof total !== "number") {
      return;
    }
    const nextOffset = clampListPageOffset(pageOffset, total, RANK_PAGE_SIZE);
    if (nextOffset !== pageOffset) {
      setPageOffset(nextOffset);
    }
  }, [board?.total, boardMode, pageOffset, suiteBoard?.total]);
  useEffect(() => {
    writeRankBoardViewState({
      boardMode,
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
    });
  }, [
    boardMode,
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
  ]);
  useEffect(() => {
    window.addEventListener(RANK_BOARD_VIEW_STATE_RESET_EVENT, resetViewState);
    return () => window.removeEventListener(RANK_BOARD_VIEW_STATE_RESET_EVENT, resetViewState);
  }, [resetViewState]);

  const loading =
    dashboardQuery.isLoading ||
    (boardMode === "run" && boardQuery.isLoading && !board) ||
    (boardMode === "suite" && suiteRankQuery.isLoading && !suiteBoard);
  const errorTitle =
    dashboardQuery.error ||
    (boardMode === "run" && !board) ||
    (boardMode === "suite" && !suiteBoard)
      ? `排行榜加载失败：${
          boardMode === "suite"
            ? errorMessage(dashboardQuery.error || suiteRankQuery.error)
            : errorMessage(dashboardQuery.error || boardQuery.error)
        }`
      : null;

  return {
    boardMode,
    setRunMode: () => {
      setBoardMode("run");
      setPageOffset(0);
    },
    setSuiteMode: () => {
      setBoardMode("suite");
      setPageOffset(0);
    },
    runs,
    board,
    suiteBoard,
    entries,
    loading,
    errorTitle,
    filterValues,
    filterOptions,
    filterSetters,
    facetFilters,
    facetHandlers,
    sortBy,
    sortOrder,
    suiteSortBy,
    suiteSortOrder,
    tableRefreshing: boardQuery.isPlaceholderData && Boolean(board),
    suiteTableRefreshing: suiteRankQuery.isPlaceholderData && Boolean(suiteBoard),
    setPageOffset,
    handleSortChange,
    handleSuiteSortChange
  };
}

