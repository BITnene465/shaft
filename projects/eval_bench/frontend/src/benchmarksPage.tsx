import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { fetchBenchmarks } from "./api";
import { BenchmarkCreatePanel } from "./benchmarkCreatePanel";
import { benchmarkSplitValues } from "./benchmarkModel";
import { AdvancedFilterBar } from "./filterControls";
import { errorMessage, facetValues } from "./formatters";
import { AppIcon } from "./iconLibrary";
import { BenchmarkTable } from "./runTables";
import {
  PagerControl,
  clampListPageOffset,
  updatePagedFilterValue
} from "./samplePager";
import {
  CommandButton,
  EmptyState
} from "./ui";
import { TableLoadingState } from "./uiDataTable";
import { WorkspaceDialog } from "./uiDialog";
import { useDebouncedValueState } from "./useDebouncedValue";

export { BenchmarkDetailPage } from "./benchmarkSampleInspector";

const BENCHMARK_PAGE_SIZE = 80;

export function BenchmarksPage() {
  const [createOpen, setCreateOpen] = useState(false);
  const [searchText, setSearchText] = useState("");
  const [taskFilter, setTaskFilter] = useState("all");
  const [layerFilter, setLayerFilter] = useState("all");
  const [splitFilter, setSplitFilter] = useState("all");
  const [pageOffset, setPageOffset] = useState(0);
  const debouncedSearch = useDebouncedValueState(searchText);
  const benchmarkFilters = useMemo(
    () => ({
      offset: pageOffset,
      limit: BENCHMARK_PAGE_SIZE,
      task: taskFilter !== "all" ? taskFilter : undefined,
      layer: layerFilter !== "all" ? layerFilter : undefined,
      split: splitFilter !== "all" ? splitFilter : undefined,
      query: debouncedSearch.value.trim() || undefined
    }),
    [debouncedSearch.value, layerFilter, pageOffset, splitFilter, taskFilter]
  );
  const benchmarksQuery = useQuery({
    queryKey: ["benchmarks", benchmarkFilters],
    queryFn: ({ signal }) => fetchBenchmarks(benchmarkFilters, { signal }),
    placeholderData: (previousData) => previousData
  });
  const benchmarks = benchmarksQuery.data?.benchmarks ?? [];
  const facets = benchmarksQuery.data?.facets;
  const tasks = facetValues(facets, "tasks", benchmarks.flatMap((benchmark) => benchmark.tasks));
  const layers = facetValues(facets, "layers", benchmarks.flatMap((benchmark) => benchmark.layers));
  const splits = facetValues(facets, "splits", benchmarks.flatMap(benchmarkSplitValues));
  const totalBenchmarks = benchmarksQuery.data?.total ?? benchmarks.length;
  useEffect(() => {
    const nextOffset = clampListPageOffset(pageOffset, totalBenchmarks, BENCHMARK_PAGE_SIZE);
    if (nextOffset !== pageOffset) {
      setPageOffset(nextOffset);
    }
  }, [pageOffset, totalBenchmarks]);
  if (benchmarksQuery.isLoading) {
    return (
      <section className="page-stack density-page">
        <div className="page-command-row">
          <div>
            <h2>基准集目录</h2>
            <span>正在同步 benchmark 索引</span>
          </div>
          <CommandButton
            icon={<AppIcon name="createBenchmark" size={17} />}
            onClick={() => setCreateOpen(true)}
          >
            创建副本
          </CommandButton>
        </div>
        <div className="workspace-card fill">
          <TableLoadingState label="正在加载基准集" columns={7} />
        </div>
        <WorkspaceDialog
          open={createOpen}
          title="创建 benchmark 副本"
          meta="从 raw_data split 复制不可变 test/val 集"
          onClose={() => setCreateOpen(false)}
        >
          <BenchmarkCreatePanel bare />
        </WorkspaceDialog>
      </section>
    );
  }
  if (benchmarksQuery.error || !benchmarksQuery.data) {
    return <EmptyState title={`基准集加载失败：${errorMessage(benchmarksQuery.error)}`} tone="danger" />;
  }
  return (
    <section className="page-stack density-page">
      <div className="page-command-row">
        <div>
          <h2>基准集目录</h2>
          <span>{totalBenchmarks.toLocaleString()} 个不可变副本</span>
        </div>
        <CommandButton
          icon={<AppIcon name="createBenchmark" size={17} />}
          onClick={() => setCreateOpen(true)}
        >
          创建副本
        </CommandButton>
      </div>
      <AdvancedFilterBar
        title="基准集高级检索"
        meta={`${benchmarks.length.toLocaleString()} / ${totalBenchmarks.toLocaleString()} 个 benchmark`}
        controls={[
          {
            type: "search",
            id: "benchmark-query",
            label: "全文检索",
            value: searchText,
            onChange: (value) =>
              updatePagedFilterValue(searchText, value, setSearchText, setPageOffset),
            placeholder: "搜索 benchmark、manifest、root、来源"
          },
          {
            type: "select",
            id: "benchmark-task",
            label: "任务",
            value: taskFilter,
            values: ["all", ...tasks],
            labels: { all: "全部" },
            onChange: (value) =>
              updatePagedFilterValue(taskFilter, value, setTaskFilter, setPageOffset)
          },
          {
            type: "select",
            id: "benchmark-layer",
            label: "标注层",
            value: layerFilter,
            values: ["all", ...layers],
            labels: { all: "全部" },
            onChange: (value) =>
              updatePagedFilterValue(layerFilter, value, setLayerFilter, setPageOffset)
          },
          {
            type: "select",
            id: "benchmark-split",
            label: "Split",
            value: splitFilter,
            values: ["all", ...splits],
            labels: { all: "全部" },
            onChange: (value) =>
              updatePagedFilterValue(splitFilter, value, setSplitFilter, setPageOffset)
          }
        ]}
      />
      <div className="workspace-card fill">
        <BenchmarkTable
          benchmarks={benchmarks}
          refreshing={benchmarksQuery.isPlaceholderData || debouncedSearch.pending}
        />
        <PagerControl
          className="rank-board-pager benchmark-list-pager"
          offset={benchmarksQuery.data.offset ?? pageOffset}
          limit={benchmarksQuery.data.limit ?? BENCHMARK_PAGE_SIZE}
          total={totalBenchmarks}
          onPageChange={setPageOffset}
        />
      </div>
      <WorkspaceDialog
        open={createOpen}
        title="创建 benchmark 副本"
        meta="从 raw_data split 复制不可变 test/val 集"
        onClose={() => setCreateOpen(false)}
      >
        <BenchmarkCreatePanel bare />
      </WorkspaceDialog>
    </section>
  );
}
