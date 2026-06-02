import { useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useParams } from "@tanstack/react-router";

import { fetchRunSampleDetail, fetchRunSamples } from "./api";
import { useDashboardState } from "./dashboardState";
import { errorMessage, isTextInputTarget } from "./formatters";
import { RunConfigPanel, shouldOpenRunNotePanel } from "./runConfigPanel";
import { SampleFilters, SampleList } from "./runSampleSidebar";
import {
  SAMPLE_PAGE_SIZE,
  clampSamplePageOffset,
  sampleIndexFromLocation,
  samplePageOffsetFromLocation,
  updateSampleIndexInLocation
} from "./sampleNavigation";
import { SamplePager, updatePagedFilterValue } from "./samplePager";
import { SampleViewer } from "./sampleViewer";
import { EmptyState } from "./ui";
import { preloadSampleImages } from "./viewerGeometry";
import { ResizableSplit } from "./workspaceLayout";
import { useWorkspaceShortcuts } from "./workspaceSettings";

import "./inspectorPage.css";
import "./runsPage.css";

export function RunDetailPage() {
  const { runId } = useParams({ from: "/runs/$runId" });
  const queryClient = useQueryClient();
  const { data: dashboardState } = useDashboardState();
  const runSummary = dashboardState?.runs.find((run) => run.run_id === runId) ?? null;
  const [selectedIndex, setSelectedIndex] = useState(() => sampleIndexFromLocation());
  const [pageOffset, setPageOffset] = useState(() => samplePageOffsetFromLocation(SAMPLE_PAGE_SIZE));
  const [errorFilter, setErrorFilter] = useState("all");
  const [labelFilter, setLabelFilter] = useState("all");
  const samplesQuery = useQuery({
    queryKey: ["run-samples", runId, pageOffset, errorFilter, labelFilter],
    queryFn: () =>
      fetchRunSamples(runId, {
        offset: pageOffset,
        limit: SAMPLE_PAGE_SIZE,
        label: labelFilter,
        errorFilter
      }),
    placeholderData: (previousData) => previousData
  });
  const page = samplesQuery.data;
  const samples = page?.samples ?? [];
  const labels = page?.labels ?? [];
  const activeSample = samples.find((sample) => sample.index === selectedIndex) ?? samples[0] ?? null;
  const activeIndex = activeSample?.index ?? selectedIndex;
  const hasActiveSampleFilter = errorFilter !== "all" || labelFilter !== "all";
  const { actionForEvent } = useWorkspaceShortcuts();
  const detailQuery = useQuery({
    queryKey: ["run-sample-detail", runId, activeIndex],
    queryFn: () => fetchRunSampleDetail(runId, activeIndex),
    enabled: Boolean(activeSample),
    placeholderData: (previousData) => (previousData?.run_id === runId ? previousData : undefined),
    staleTime: 30_000
  });

  function selectSample(index: number) {
    setSelectedIndex(index);
    updateSampleIndexInLocation(index);
  }

  function changeErrorFilter(value: string) {
    updatePagedFilterValue(errorFilter, value, setErrorFilter, setPageOffset);
  }

  function changeLabelFilter(value: string) {
    updatePagedFilterValue(labelFilter, value, setLabelFilter, setPageOffset);
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
        queryKey: ["run-sample-detail", runId, sample.index],
        queryFn: () => fetchRunSampleDetail(runId, sample.index),
        staleTime: 30_000
      });
    });
  }, [activeIndex, queryClient, runId, samples]);

  if (samplesQuery.isLoading) {
    return <EmptyState title="正在加载评测样本" />;
  }
  if (samplesQuery.error) {
    return <EmptyState title={`评测样本加载失败：${errorMessage(samplesQuery.error)}`} tone="danger" />;
  }

  return (
    <section className="page-stack visual-inspector-page run-inspector-page">
      {runSummary ? <RunConfigPanel run={runSummary} defaultOpen={shouldOpenRunNotePanel()} /> : null}
      {page?.total === 0 && !hasActiveSampleFilter ? (
        <EmptyState title="这条评测记录没有基准集样本。" />
      ) : (
        <ResizableSplit
          className="inspector-grid"
          storageKey="eval_bench_run_sidebar_width"
          defaultSize={224}
          minSize={148}
          maxSize={520}
          first={
            <div className="inspector-sidebar">
              <SampleFilters
                errorFilter={errorFilter}
                labelFilter={labelFilter}
                labels={labels}
                onErrorFilterChange={changeErrorFilter}
                onLabelFilterChange={changeLabelFilter}
              />
              <SampleList
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
                  <SampleViewer detail={detailQuery.data} />
                </>
              )}
            </div>
          }
        />
      )}
    </section>
  );
}
