import { useEffect, useMemo, useState } from "react";
import type { FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useParams } from "@tanstack/react-router";

import type {
  BenchmarkSampleDetail,
  BenchmarkSampleSummary,
  BenchmarkSummary,
  CreateBenchmarkSlicePayload
} from "./api";
import {
  createBenchmark,
  fetchBenchmark,
  fetchBenchmarkSampleDetail,
  fetchBenchmarkSamples,
  fetchBenchmarks
} from "./api";
import { CheckboxFieldControl, TextareaControl, TextInputControl } from "./controlPrimitives";
import { AdvancedFilterBar } from "./filterControls";
import { basename, facetValues, isTextInputTarget, unique } from "./formatters";
import { AppIcon } from "./iconLibrary";
import { BenchmarkTable } from "./runTables";
import {
  SAMPLE_PAGE_SIZE,
  clampSamplePageOffset,
  sampleIndexFromLocation,
  samplePageOffsetFromLocation,
  updateSampleIndexInLocation
} from "./sampleNavigation";
import { PagerControl, SamplePager, clampListPageOffset } from "./samplePager";
import {
  ActionButton,
  CommandButton,
  EmptyState,
  SectionHeader,
  SelectableRowButton,
  WorkspaceDialog
} from "./ui";
import { CanvasStage } from "./viewerCanvas";
import { displayImageUrl, preloadSampleImages } from "./viewerGeometry";
import { InstanceStats } from "./viewerPanels";
import { ResizableSplit } from "./workspaceLayout";
import { useWorkspaceSettings, useWorkspaceShortcuts } from "./workspaceSettings";

const BENCHMARK_PAGE_SIZE = 80;

export function BenchmarksPage() {
  const [createOpen, setCreateOpen] = useState(false);
  const [searchText, setSearchText] = useState("");
  const [taskFilter, setTaskFilter] = useState("all");
  const [layerFilter, setLayerFilter] = useState("all");
  const [splitFilter, setSplitFilter] = useState("all");
  const [pageOffset, setPageOffset] = useState(0);
  const benchmarkFilters = useMemo(
    () => ({
      offset: pageOffset,
      limit: BENCHMARK_PAGE_SIZE,
      task: taskFilter !== "all" ? taskFilter : undefined,
      layer: layerFilter !== "all" ? layerFilter : undefined,
      split: splitFilter !== "all" ? splitFilter : undefined,
      query: searchText.trim() || undefined
    }),
    [layerFilter, pageOffset, searchText, splitFilter, taskFilter]
  );
  const benchmarksQuery = useQuery({
    queryKey: ["benchmarks", benchmarkFilters],
    queryFn: () => fetchBenchmarks(benchmarkFilters),
    placeholderData: (previousData) => previousData
  });
  const benchmarks = benchmarksQuery.data?.benchmarks ?? [];
  const facets = benchmarksQuery.data?.facets;
  const tasks = facetValues(facets, "tasks", benchmarks.flatMap((benchmark) => benchmark.tasks));
  const layers = facetValues(facets, "layers", benchmarks.flatMap((benchmark) => benchmark.layers));
  const splits = facetValues(facets, "splits", benchmarks.flatMap(benchmarkSplitValues));
  const totalBenchmarks = benchmarksQuery.data?.total ?? benchmarks.length;
  useEffect(() => {
    setPageOffset(0);
  }, [searchText, taskFilter, layerFilter, splitFilter]);
  useEffect(() => {
    const nextOffset = clampListPageOffset(pageOffset, totalBenchmarks, BENCHMARK_PAGE_SIZE);
    if (nextOffset !== pageOffset) {
      setPageOffset(nextOffset);
    }
  }, [pageOffset, totalBenchmarks]);
  if (benchmarksQuery.isLoading) {
    return <EmptyState title="正在加载基准集" />;
  }
  if (benchmarksQuery.error || !benchmarksQuery.data) {
    return <EmptyState title="基准集加载失败" tone="danger" />;
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
            onChange: setSearchText,
            placeholder: "搜索 benchmark、manifest、root、来源"
          },
          {
            type: "select",
            id: "benchmark-task",
            label: "任务",
            value: taskFilter,
            values: ["all", ...tasks],
            labels: { all: "全部" },
            onChange: setTaskFilter
          },
          {
            type: "select",
            id: "benchmark-layer",
            label: "标注层",
            value: layerFilter,
            values: ["all", ...layers],
            labels: { all: "全部" },
            onChange: setLayerFilter
          },
          {
            type: "select",
            id: "benchmark-split",
            label: "Split",
            value: splitFilter,
            values: ["all", ...splits],
            labels: { all: "全部" },
            onChange: setSplitFilter
          }
        ]}
      />
      <div className="workspace-card fill">
        <BenchmarkTable benchmarks={benchmarks} refreshing={benchmarksQuery.isPlaceholderData} />
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

function BenchmarkCreatePanel({ bare }: { bare?: boolean }) {
  const queryClient = useQueryClient();
  const [benchmarkId, setBenchmarkId] = useState("");
  const [sourceRoot, setSourceRoot] = useState("data/raw_data");
  const [sourceManifest, setSourceManifest] = useState("data/raw_data/splits/layout_val.txt");
  const [split, setSplit] = useState("val");
  const [suiteSlices, setSuiteSlices] = useState("");
  const [tasks, setTasks] = useState<string[]>(["detection", "keypoint"]);
  const [layers, setLayers] = useState("layout,arrow");
  const [overwrite, setOverwrite] = useState(false);
  const suiteSliceParse = useMemo(
    () => parseBenchmarkSlices(suiteSlices, tasks, layers),
    [suiteSlices, tasks, layers]
  );
  const mutation = useMutation({
    mutationFn: createBenchmark,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["dashboard-state"] });
      void queryClient.invalidateQueries({ queryKey: ["benchmarks"] });
    }
  });

  function toggleTask(task: string) {
    setTasks((current) => {
      if (current.includes(task)) {
        return current.filter((item) => item !== task);
      }
      return [...current, task];
    });
  }

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (suiteSliceParse.error) {
      return;
    }
    const slices = suiteSliceParse.slices;
    const suiteMode = slices.length > 0;
    const normalizedSplit = split.trim();
    const benchmarkSplit =
      suiteMode && (!normalizedSplit || normalizedSplit === "val")
        ? "suite"
        : normalizedSplit || "val";
    mutation.mutate({
      benchmark_id: benchmarkId.trim(),
      source_root: sourceRoot.trim(),
      source_manifest: suiteMode ? undefined : sourceManifest.trim(),
      split: benchmarkSplit,
      tasks,
      layers: layers
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean),
      slices: suiteMode ? slices : undefined,
      default_slice: slices[0]?.split,
      overwrite
    });
  }

  const content = (
    <form className="job-form benchmark-form" onSubmit={submit}>
      <TextInputControl
        label="基准集 ID"
        value={benchmarkId}
        onChange={setBenchmarkId}
        placeholder="grounding_layout_main"
        required
      />
      <TextInputControl
        className="wide-field"
        label="数据根目录"
        value={sourceRoot}
        onChange={setSourceRoot}
        required
      />
      <TextInputControl
        className="wide-field"
        label="Split 文件"
        value={sourceManifest}
        onChange={setSourceManifest}
        required={!suiteSlices.trim()}
      />
      <TextInputControl label="Split 名称" value={split} onChange={setSplit} required />
      <TextInputControl label="标注层" value={layers} onChange={setLayers} />
      <TextareaControl
        className="wide-field"
        label="Suite slices"
        value={suiteSlices}
        onChange={setSuiteSlices}
        rows={4}
        placeholder="grounding_arrow=data/raw_data/splits/grounding_arrow.txt | detection | arrow | arrow"
      />
      {suiteSliceParse.error ? (
        <div className="form-result error full-field">{suiteSliceParse.error}</div>
      ) : null}
      <CheckboxFieldControl
        label="检测"
        checked={tasks.includes("detection")}
        onChange={() => toggleTask("detection")}
      />
      <CheckboxFieldControl
        label="关键点"
        checked={tasks.includes("keypoint")}
        onChange={() => toggleTask("keypoint")}
      />
      <CheckboxFieldControl label="覆盖已有副本" checked={overwrite} onChange={setOverwrite} />
      <ActionButton
        type="submit"
        variant="primary"
        icon={<AppIcon name="submitCreate" size={16} />}
        disabled={
          mutation.isPending
          || (suiteSliceParse.slices.length === 0 && tasks.length === 0)
          || Boolean(suiteSliceParse.error)
        }
      >
        创建
      </ActionButton>
      {mutation.data ? (
        <div className="form-result full-field">
          已创建 {mutation.data.benchmark_id}，包含 {mutation.data.sample_count.toLocaleString()} 个样本。{" "}
          <Link to="/benchmarks/$benchmarkId" params={{ benchmarkId: mutation.data.benchmark_id }}>
            打开
          </Link>
        </div>
      ) : null}
      {mutation.error ? (
        <div className="form-result error full-field">{mutation.error.message}</div>
      ) : null}
    </form>
  );
  return bare ? content : <div className="workspace-card compact-form-card">{content}</div>;
}

function parseBenchmarkSlices(
  value: string,
  defaultTasks: string[],
  defaultLayers: string
): { slices: CreateBenchmarkSlicePayload[]; error: string | null } {
  const fallbackLayers = splitCompactList(defaultLayers);
  const lines = value
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith("#"));
  if (value.trim() && lines.length === 0) {
    return { slices: [], error: "Suite slices 至少需要一行 split=manifest" };
  }
  const slices: CreateBenchmarkSlicePayload[] = [];
  const seenSplits = new Set<string>();
  for (const [index, line] of lines.entries()) {
    const [head, tasksText, layersText, labelsText] = line.split("|").map((item) => item.trim());
    const separatorIndex = head.indexOf("=");
    if (separatorIndex <= 0 || separatorIndex >= head.length - 1) {
      return {
        slices: [],
        error: `Suite slices 第 ${index + 1} 行必须使用 split=manifest 格式`
      };
    }
    const split = head.slice(0, separatorIndex).trim();
    const sourceManifest = head.slice(separatorIndex + 1).trim();
    if (seenSplits.has(split)) {
      return { slices: [], error: `Suite slices split 重复: ${split}` };
    }
    seenSplits.add(split);
    const parsedTasks = splitCompactList(tasksText);
    const parsedLayers = splitCompactList(layersText);
    const tasks = parsedTasks.length > 0 ? parsedTasks : defaultTasks;
    const invalidTasks = tasks.filter((task) => task !== "detection" && task !== "keypoint");
    if (invalidTasks.length > 0) {
      return { slices: [], error: `Suite slices 不支持的任务: ${invalidTasks.join(", ")}` };
    }
    slices.push({
      split,
      source_manifest: sourceManifest,
      tasks,
      layers: parsedLayers.length > 0 ? parsedLayers : fallbackLayers,
      target_labels: splitCompactList(labelsText)
    });
  }
  return { slices, error: null };
}

function splitCompactList(value: string | undefined): string[] {
  return String(value ?? "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

export function BenchmarkDetailPage() {
  const { benchmarkId } = useParams({ from: "/benchmarks/$benchmarkId" });
  const queryClient = useQueryClient();
  const [selectedIndex, setSelectedIndex] = useState(() => sampleIndexFromLocation());
  const [pageOffset, setPageOffset] = useState(() => samplePageOffsetFromLocation(SAMPLE_PAGE_SIZE));
  const [labelFilter, setLabelFilter] = useState("all");
  const [splitFilter, setSplitFilter] = useState("all");
  const benchmarkQuery = useQuery({
    queryKey: ["benchmark", benchmarkId],
    queryFn: () => fetchBenchmark(benchmarkId),
    staleTime: 30_000
  });
  const benchmark = benchmarkQuery.data?.benchmark;
  const splitOptions = useMemo(() => benchmarkSplitValues(benchmark), [benchmark]);
  const samplesQuery = useQuery({
    queryKey: ["benchmark-samples", benchmarkId, pageOffset, labelFilter, splitFilter],
    queryFn: () =>
      fetchBenchmarkSamples(benchmarkId, {
        offset: pageOffset,
        limit: SAMPLE_PAGE_SIZE,
        label: labelFilter,
        split: splitFilter
      }),
    placeholderData: (previousData) => previousData
  });
  const page = samplesQuery.data;
  const samples = page?.samples ?? [];
  const labels = page?.labels ?? [];
  const activeSample = samples.find((sample) => sample.index === selectedIndex) ?? samples[0] ?? null;
  const activeIndex = activeSample?.index ?? selectedIndex;
  const hasActiveSampleFilter = labelFilter !== "all" || splitFilter !== "all";
  const { actionForEvent } = useWorkspaceShortcuts();
  const detailQuery = useQuery({
    queryKey: ["benchmark-sample-detail", benchmarkId, activeIndex, splitFilter],
    queryFn: () => fetchBenchmarkSampleDetail(benchmarkId, activeIndex, { split: splitFilter }),
    enabled: Boolean(activeSample),
    placeholderData: (previousData) =>
      previousData?.benchmark_id === benchmarkId ? previousData : undefined,
    staleTime: 30_000
  });

  function selectSample(index: number) {
    setSelectedIndex(index);
    updateSampleIndexInLocation(index);
  }

  function changeLabelFilter(value: string) {
    setLabelFilter(value);
    setPageOffset(0);
  }

  function changeSplitFilter(value: string) {
    setSplitFilter(value);
    setPageOffset(0);
    setSelectedIndex(0);
    updateSampleIndexInLocation(0);
  }

  function moveSample(delta: number) {
    if (samples.length === 0) {
      return;
    }
    const position = samples.findIndex((sample) => sample.index === activeIndex);
    const next = samples[position + delta];
    if (next) {
      selectSample(next.index);
      return;
    }
    const nextOffset = pageOffset + delta * SAMPLE_PAGE_SIZE;
    if (nextOffset >= 0 && page && nextOffset < page.total) {
      setPageOffset(nextOffset);
    }
  }

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (isTextInputTarget(event.target)) {
        return;
      }
      const actionId = actionForEvent(event);
      if (actionId === "sample.previous") {
        event.preventDefault();
        moveSample(-1);
      }
      if (actionId === "sample.next") {
        event.preventDefault();
        moveSample(1);
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [actionForEvent, activeIndex, page?.total, pageOffset, samples]);

  useEffect(() => {
    if (activeSample && activeSample.index !== selectedIndex) {
      selectSample(activeSample.index);
    }
  }, [activeSample, selectedIndex]);

  useEffect(() => {
    if (!page) {
      return;
    }
    const nextOffset = clampSamplePageOffset(pageOffset, page.total, SAMPLE_PAGE_SIZE);
    if (nextOffset !== pageOffset) {
      setPageOffset(nextOffset);
    }
  }, [page?.total, pageOffset]);

  useEffect(() => {
    return preloadSampleImages(samples, activeIndex);
  }, [activeIndex, samples]);

  useEffect(() => {
    if (samples.length === 0) {
      return;
    }
    const position = Math.max(0, samples.findIndex((sample) => sample.index === activeIndex));
    const preload = samples.slice(Math.max(0, position - 1), position + 2);
    preload.forEach((sample) => {
      void queryClient.prefetchQuery({
        queryKey: ["benchmark-sample-detail", benchmarkId, sample.index, splitFilter],
        queryFn: () => fetchBenchmarkSampleDetail(benchmarkId, sample.index, { split: splitFilter }),
        staleTime: 30_000
      });
    });
  }, [activeIndex, benchmarkId, queryClient, samples, splitFilter]);

  if (samplesQuery.isLoading) {
    return <EmptyState title="正在加载样本" />;
  }
  if (samplesQuery.error) {
    return <EmptyState title="样本加载失败" tone="danger" />;
  }

  return (
    <section className="page-stack visual-inspector-page">
      <SectionHeader title="基准集检查" subtitle={`${benchmarkId} 的真值样本浏览器。`} />
      {page?.total === 0 && !hasActiveSampleFilter ? (
        <EmptyState title="这个基准集没有样本。" />
      ) : (
        <ResizableSplit
          className="inspector-grid"
          storageKey="eval_bench_benchmark_sidebar_width"
          defaultSize={224}
          minSize={148}
          maxSize={520}
          first={
            <div className="inspector-sidebar">
              <BenchmarkSampleFilters
                labelFilter={labelFilter}
                labels={labels}
                splitFilter={splitFilter}
                splits={splitOptions}
                onLabelFilterChange={changeLabelFilter}
                onSplitFilterChange={changeSplitFilter}
              />
              <BenchmarkSampleList
                samples={samples}
                selectedIndex={activeIndex}
                refreshing={samplesQuery.isPlaceholderData}
                onSelect={selectSample}
                emptyText="没有符合过滤条件的样本。"
              />
              {page ? (
                <SamplePager
                  offset={page.offset}
                  limit={page.limit}
                  total={page.total}
                  onPageChange={setPageOffset}
                />
              ) : null}
            </div>
          }
          second={
            <div className="viewer-panel">
              {samples.length === 0 ? (
                <div className="empty-panel">没有符合过滤条件的样本。</div>
              ) : detailQuery.error ? (
                <div className="empty-panel">样本详情加载失败</div>
              ) : detailQuery.isLoading || !detailQuery.data ? (
                <div className="empty-panel">正在加载样本详情</div>
              ) : (
                <>
                  {detailQuery.isFetching ? <div className="viewer-fetch-chip">正在刷新样本详情</div> : null}
                  <BenchmarkSampleViewer detail={detailQuery.data} />
                </>
              )}
            </div>
          }
        />
      )}
    </section>
  );
}

function BenchmarkSampleFilters({
  labelFilter,
  labels,
  splitFilter,
  splits,
  onLabelFilterChange,
  onSplitFilterChange
}: {
  labelFilter: string;
  labels: string[];
  splitFilter: string;
  splits: string[];
  onLabelFilterChange: (value: string) => void;
  onSplitFilterChange: (value: string) => void;
}) {
  return (
    <AdvancedFilterBar
      title="样本检索"
      meta={`${splits.length.toLocaleString()} splits / ${labels.length.toLocaleString()} labels`}
      controls={[
        {
          type: "select",
          id: "split",
          label: "分片",
          value: splitFilter,
          values: ["all", ...splits],
          labels: { all: "默认" },
          onChange: onSplitFilterChange
        },
        {
          type: "select",
          id: "label",
          label: "标签",
          value: labelFilter,
          values: ["all", ...labels],
          labels: { all: "全部" },
          onChange: onLabelFilterChange
        }
      ]}
    />
  );
}

function BenchmarkSampleList({
  samples,
  selectedIndex,
  refreshing = false,
  onSelect,
  emptyText
}: {
  samples: BenchmarkSampleSummary[];
  selectedIndex: number;
  refreshing?: boolean;
  onSelect: (index: number) => void;
  emptyText: string;
}) {
  if (samples.length === 0) {
    return <div className="sample-list empty">{emptyText}</div>;
  }
  return (
    <div className={refreshing ? "sample-list refreshing" : "sample-list"}>
      {refreshing ? (
        <span className="table-refresh-indicator" aria-live="polite">
          样本列表更新中
        </span>
      ) : null}
      {samples.map((sample) => (
        <SelectableRowButton
          key={sample.index}
          selected={sample.index === selectedIndex}
          onClick={() => onSelect(sample.index)}
        >
          <span className="sample-row-main">
            <strong>{sample.index + 1}</strong>
            <span title={sample.image}>{basename(sample.image)}</span>
          </span>
          <span className="sample-row-meta">
            真值 {sample.instance_count.toLocaleString()} / 标签 {sample.labels.join(", ") || "-"}
          </span>
        </SelectableRowButton>
      ))}
    </div>
  );
}

function BenchmarkSampleViewer({ detail }: { detail: BenchmarkSampleDetail }) {
  const width = detail.sample.image_width ?? 1000;
  const height = detail.sample.image_height ?? 1000;
  const labels = useMemo(
    () => unique(detail.gt_instances.map((instance) => instance.label)),
    [detail.gt_instances]
  );
  const { overlayColors, overlayStyle, labelColors, interactionSettings, overlayVars } =
    useWorkspaceSettings(labels);

  return (
    <div className="viewer-stack" style={overlayVars}>
      <div className="viewer-toolbar">
        <div>
          <h2>{basename(detail.sample.image)}</h2>
          <p>{detail.sample.image}</p>
        </div>
        <div className="legend-row">
          <span className="legend-item gt">真值</span>
        </div>
      </div>
      <div className="diagnostic-strip">
        <span>实例 {detail.sample.instance_count.toLocaleString()}</span>
        <span>标签 {detail.sample.labels.join(", ") || "-"}</span>
      </div>
      <CanvasStage
        width={width}
        height={height}
        imageUrl={displayImageUrl(detail.sample)}
        imageAlt={detail.sample.image}
        imageTileUrlTemplate={detail.sample.image_tile_url_template}
        imageTileSize={detail.sample.image_tile_size}
        gtInstances={detail.gt_instances}
        predInstances={[]}
        diagnostics={null}
        visibleLabels={new Set(labels)}
        showGt={true}
        showPred={false}
        showBoxes={true}
        showLines={true}
        showKeypoints={true}
        overlayColors={overlayColors}
        overlayStyle={overlayStyle}
        labelColors={labelColors}
        interactionSettings={interactionSettings}
      />
      <div className="instance-summary">
        <InstanceStats title="真值实例" instances={detail.gt_instances} />
      </div>
    </div>
  );
}

function benchmarkSplitValues(benchmark: BenchmarkSummary | null | undefined): string[] {
  if (!benchmark) {
    return [];
  }
  return unique([benchmark.split, ...Object.keys(benchmark.split_manifests ?? {})].filter(Boolean));
}
