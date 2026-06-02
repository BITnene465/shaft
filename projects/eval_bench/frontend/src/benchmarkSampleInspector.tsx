import { useEffect, useMemo, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useParams } from "@tanstack/react-router";

import type { BenchmarkSampleDetail, BenchmarkSampleSummary } from "./api";
import {
  fetchBenchmark,
  fetchBenchmarkSampleDetail,
  fetchBenchmarkSamples
} from "./api";
import { AdvancedFilterBar } from "./filterControls";
import { basename, errorMessage, isTextInputTarget, unique } from "./formatters";
import { benchmarkSplitValues } from "./benchmarkModel";
import {
  SAMPLE_PAGE_SIZE,
  clampSamplePageOffset,
  sampleIndexFromLocation,
  samplePageOffsetFromLocation,
  updateSampleIndexInLocation
} from "./sampleNavigation";
import { SamplePager, updatePagedFilterValue } from "./samplePager";
import { EmptyState, SectionHeader, SelectableRowButton } from "./ui";
import { CanvasStage } from "./viewerCanvas";
import { displayImageUrl, preloadSampleImages } from "./viewerGeometry";
import { InstanceStats } from "./viewerPanels";
import { ResizableSplit } from "./workspaceLayout";
import { useWorkspaceSettings, useWorkspaceShortcuts } from "./workspaceSettings";

import "./inspectorPage.css";

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
    updatePagedFilterValue(labelFilter, value, setLabelFilter, setPageOffset);
  }

  function changeSplitFilter(value: string) {
    if (Object.is(value, splitFilter)) {
      return;
    }
    updatePagedFilterValue(splitFilter, value, setSplitFilter, setPageOffset);
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
    return <EmptyState title={`样本加载失败：${errorMessage(samplesQuery.error)}`} tone="danger" />;
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
                <div className="empty-panel">样本详情加载失败：{errorMessage(detailQuery.error)}</div>
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

