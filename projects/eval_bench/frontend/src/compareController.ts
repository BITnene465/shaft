import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { fetchComparison, fetchComparisons, fetchRuns } from "./api";
import {
  COMPARE_VIEW_STATE_RESET_EVENT,
  DEFAULT_COMPARE_VIEW_STATE,
  readCompareViewState,
  writeCompareViewState
} from "./compareViewState";
import { errorMessage, facetValues } from "./formatters";
import { clampListPageOffset } from "./samplePager";
import type {
  CompareFilterOptions,
  CompareFilterSetters,
  CompareFilterValues
} from "./compareFilters";
import { useDebouncedValueState } from "./useDebouncedValue";

export const COMPARE_RUN_PAGE_SIZE = 80;
export const COMPARISON_HISTORY_PAGE_SIZE = 50;

export function useCompareController() {
  const [initialViewState] = useState(() => {
    const params = new URLSearchParams(window.location.search);
    const baseline = params.get("baseline");
    const candidate = params.get("candidate");
    return readCompareViewState({
      ...(baseline ? { baselineRunId: baseline } : {}),
      ...(candidate ? { candidateRunId: candidate } : {})
    });
  });
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
  const [historyBaselineFilter, setHistoryBaselineFilter] = useState(
    initialViewState.historyBaselineFilter
  );
  const [historyCandidateFilter, setHistoryCandidateFilter] = useState(
    initialViewState.historyCandidateFilter
  );
  const [pageOffset, setPageOffset] = useState(initialViewState.pageOffset);
  const [historyOffset, setHistoryOffset] = useState(initialViewState.historyOffset);
  const [baselineRunId, setBaselineRunId] = useState(initialViewState.baselineRunId);
  const [candidateRunId, setCandidateRunId] = useState(initialViewState.candidateRunId);
  const [activeLabel, setActiveLabel] = useState(initialViewState.activeLabel);
  const debouncedSearch = useDebouncedValueState(searchText);

  function resetViewState() {
    setSearchText(DEFAULT_COMPARE_VIEW_STATE.searchText);
    setStatusFilter(DEFAULT_COMPARE_VIEW_STATE.statusFilter);
    setTaskFilter(DEFAULT_COMPARE_VIEW_STATE.taskFilter);
    setBenchmarkFilter(DEFAULT_COMPARE_VIEW_STATE.benchmarkFilter);
    setBenchmarkSplitFilter(DEFAULT_COMPARE_VIEW_STATE.benchmarkSplitFilter);
    setLabelFilter(DEFAULT_COMPARE_VIEW_STATE.labelFilter);
    setModelFilter(DEFAULT_COMPARE_VIEW_STATE.modelFilter);
    setPromptFilter(DEFAULT_COMPARE_VIEW_STATE.promptFilter);
    setHistoryBaselineFilter(DEFAULT_COMPARE_VIEW_STATE.historyBaselineFilter);
    setHistoryCandidateFilter(DEFAULT_COMPARE_VIEW_STATE.historyCandidateFilter);
    setPageOffset(DEFAULT_COMPARE_VIEW_STATE.pageOffset);
    setHistoryOffset(DEFAULT_COMPARE_VIEW_STATE.historyOffset);
    setBaselineRunId(DEFAULT_COMPARE_VIEW_STATE.baselineRunId);
    setCandidateRunId(DEFAULT_COMPARE_VIEW_STATE.candidateRunId);
    setActiveLabel(DEFAULT_COMPARE_VIEW_STATE.activeLabel);
  }

  const comparisonFilters = useMemo(
    () => ({
      task: taskFilter === "all" ? undefined : taskFilter,
      benchmarkId: benchmarkFilter === "all" ? undefined : benchmarkFilter,
      benchmarkSplit: benchmarkSplitFilter === "all" ? undefined : benchmarkSplitFilter,
      baselineRunId: historyBaselineFilter.trim() || undefined,
      candidateRunId: historyCandidateFilter.trim() || undefined,
      label: labelFilter === "all" ? undefined : labelFilter,
      query: debouncedSearch.value.trim() || undefined,
      offset: historyOffset,
      limit: COMPARISON_HISTORY_PAGE_SIZE
    }),
    [
      benchmarkFilter,
      benchmarkSplitFilter,
      historyBaselineFilter,
      historyCandidateFilter,
      historyOffset,
      labelFilter,
      debouncedSearch.value,
      taskFilter
    ]
  );
  const hasComparisonHistoryFilters = Boolean(
    searchText.trim() ||
      taskFilter !== "all" ||
      benchmarkFilter !== "all" ||
      benchmarkSplitFilter !== "all" ||
      labelFilter !== "all" ||
      historyBaselineFilter.trim() ||
      historyCandidateFilter.trim()
  );
  const runFilters = useMemo(
    () => ({
      offset: pageOffset,
      limit: COMPARE_RUN_PAGE_SIZE,
      status: statusFilter !== "all" ? statusFilter : undefined,
      task: taskFilter !== "all" ? taskFilter : undefined,
      benchmarkId: benchmarkFilter !== "all" ? benchmarkFilter : undefined,
      benchmarkSplit: benchmarkSplitFilter !== "all" ? benchmarkSplitFilter : undefined,
      label: labelFilter !== "all" ? labelFilter : undefined,
      modelId: modelFilter !== "all" ? modelFilter : undefined,
      promptId: promptFilter !== "all" ? promptFilter : undefined,
      query: debouncedSearch.value.trim() || undefined
    }),
    [
      benchmarkFilter,
      benchmarkSplitFilter,
      labelFilter,
      modelFilter,
      pageOffset,
      promptFilter,
      debouncedSearch.value,
      statusFilter,
      taskFilter
    ]
  );
  const comparisonListQuery = useQuery({
    queryKey: ["comparisons", comparisonFilters],
    queryFn: ({ signal }) => fetchComparisons(comparisonFilters, { signal }),
    placeholderData: (previousData) => previousData
  });
  const runsQuery = useQuery({
    queryKey: ["runs", "compare", runFilters],
    queryFn: ({ signal }) => fetchRuns(runFilters, { signal }),
    placeholderData: (previousData) => previousData
  });
  const runs = useMemo(() => runsQuery.data?.runs ?? [], [runsQuery.data?.runs]);
  const facets = runsQuery.data?.facets;
  const statuses = useMemo(
    () => facetValues(facets, "statuses", runs.map((run) => run.status)),
    [facets, runs]
  );
  const tasks = useMemo(
    () => facetValues(facets, "tasks", runs.map((run) => run.spec_task)),
    [facets, runs]
  );
  const benchmarks = useMemo(
    () => facetValues(facets, "benchmarks", runs.map((run) => run.benchmark_id)),
    [facets, runs]
  );
  const benchmarkSplits = useMemo(
    () =>
      facetValues(
        facets,
        "splits",
        runs.map((run) => run.benchmark_split)
      ),
    [facets, runs]
  );
  const labels = useMemo(
    () => facetValues(facets, "labels", runs.flatMap((run) => run.target_labels)),
    [facets, runs]
  );
  const models = useMemo(
    () => facetValues(facets, "models", runs.map((run) => run.model_id)),
    [facets, runs]
  );
  const prompts = useMemo(
    () => facetValues(facets, "prompts", runs.map((run) => run.prompt_id)),
    [facets, runs]
  );
  const comparableRuns = useMemo(
    () => runs.filter((run) => run.report_path),
    [runs]
  );
  const filteredCount = runsQuery.data?.total ?? runs.length;
  const runPageOffset = runsQuery.data?.offset ?? pageOffset;
  const runPageLimit = runsQuery.data?.limit ?? COMPARE_RUN_PAGE_SIZE;
  const comparisonHistoryTotal = comparisonListQuery.data?.total ?? 0;
  const comparisonHistoryOffset = comparisonListQuery.data?.offset ?? historyOffset;
  const comparisonHistoryLimit = comparisonListQuery.data?.limit ?? COMPARISON_HISTORY_PAGE_SIZE;
  const fallbackCandidate = comparableRuns[0]?.run_id ?? "";
  const fallbackBaseline =
    comparableRuns.find((run) => run.run_id !== fallbackCandidate)?.run_id ?? "";
  const effectiveBaseline = baselineRunId || fallbackBaseline;
  const candidateFallback =
    comparableRuns.find((run) => run.run_id !== effectiveBaseline)?.run_id ?? "";
  const effectiveCandidate =
    candidateRunId && candidateRunId !== effectiveBaseline
      ? candidateRunId
      : candidateFallback;
  const comparisonQuery = useQuery({
    queryKey: ["comparison", effectiveBaseline, effectiveCandidate],
    queryFn: ({ signal }) => fetchComparison(effectiveBaseline, effectiveCandidate, { signal }),
    enabled: Boolean(effectiveBaseline && effectiveCandidate && effectiveBaseline !== effectiveCandidate),
    placeholderData: (previousData) => previousData
  });
  const comparisonReport = comparisonQuery.data;
  const comparisonReportRefreshing =
    comparisonQuery.isPlaceholderData && Boolean(comparisonReport);
  const filterValues = useMemo<CompareFilterValues>(
    () => ({
      searchText,
      statusFilter,
      taskFilter,
      benchmarkFilter,
      benchmarkSplitFilter,
      labelFilter,
      modelFilter,
      promptFilter,
      historyBaselineFilter,
      historyCandidateFilter
    }),
    [
      benchmarkFilter,
      benchmarkSplitFilter,
      historyBaselineFilter,
      historyCandidateFilter,
      labelFilter,
      modelFilter,
      promptFilter,
      searchText,
      statusFilter,
      taskFilter
    ]
  );
  const filterOptions = useMemo<CompareFilterOptions>(
    () => ({
      statuses,
      tasks,
      benchmarks,
      benchmarkSplits,
      labels,
      models,
      prompts
    }),
    [benchmarkSplits, benchmarks, labels, models, prompts, statuses, tasks]
  );
  const filterSetters = useMemo<CompareFilterSetters>(
    () => ({
      setSearchText,
      setStatusFilter,
      setTaskFilter,
      setBenchmarkFilter,
      setBenchmarkSplitFilter,
      setLabelFilter,
      setModelFilter,
      setPromptFilter,
      setHistoryBaselineFilter,
      setHistoryCandidateFilter,
      setPageOffset,
      setHistoryOffset
    }),
    []
  );

  useEffect(() => {
    if (comparisonQuery.data?.comparison_id) {
      void comparisonListQuery.refetch();
    }
  }, [comparisonListQuery.refetch, comparisonQuery.data?.comparison_id]);
  useEffect(() => {
    const nextOffset = clampListPageOffset(pageOffset, filteredCount, COMPARE_RUN_PAGE_SIZE);
    if (nextOffset !== pageOffset) {
      setPageOffset(nextOffset);
    }
  }, [filteredCount, pageOffset]);
  useEffect(() => {
    const nextOffset = clampListPageOffset(
      historyOffset,
      comparisonHistoryTotal,
      COMPARISON_HISTORY_PAGE_SIZE
    );
    if (nextOffset !== historyOffset) {
      setHistoryOffset(nextOffset);
    }
  }, [comparisonHistoryTotal, historyOffset]);
  useEffect(() => {
    writeCompareViewState({
      searchText,
      statusFilter,
      taskFilter,
      benchmarkFilter,
      benchmarkSplitFilter,
      labelFilter,
      modelFilter,
      promptFilter,
      historyBaselineFilter,
      historyCandidateFilter,
      pageOffset,
      historyOffset,
      baselineRunId,
      candidateRunId,
      activeLabel
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
    historyBaselineFilter,
    historyCandidateFilter,
    pageOffset,
    historyOffset,
    baselineRunId,
    candidateRunId,
    activeLabel
  ]);
  useEffect(() => {
    window.addEventListener(COMPARE_VIEW_STATE_RESET_EVENT, resetViewState);
    return () => window.removeEventListener(COMPARE_VIEW_STATE_RESET_EVENT, resetViewState);
  }, []);

  return {
    runs,
    comparableRuns,
    filteredCount,
    runPageOffset,
    runPageLimit,
    comparisonHistoryTotal,
    comparisonHistoryOffset,
    comparisonHistoryLimit,
    hasComparisonHistoryFilters,
    effectiveBaseline,
    effectiveCandidate,
    comparisonReport,
    comparisonReportRefreshing,
    comparisonList: comparisonListQuery.data?.comparisons ?? [],
    runsRefreshing: runsQuery.isPlaceholderData || debouncedSearch.pending,
    comparisonHistoryRefreshing:
      comparisonListQuery.isPlaceholderData || debouncedSearch.pending,
    runsLoading: runsQuery.isLoading,
    runsErrorTitle:
      runsQuery.error || !runsQuery.data
        ? `对比状态加载失败：${errorMessage(runsQuery.error)}`
        : null,
    comparisonLoading: comparisonQuery.isLoading,
    comparisonError: comparisonQuery.error,
    comparisonIsError: comparisonQuery.isError,
    filterValues,
    filterOptions,
    filterSetters,
    activeLabel,
    setActiveLabel,
    setBaselineRunId,
    setCandidateRunId,
    setPageOffset,
    setHistoryOffset
  };
}
