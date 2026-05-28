import { useQuery } from "@tanstack/react-query";
import { useParams } from "@tanstack/react-router";
import { Eye } from "lucide-react";

import type { ComparisonSampleDetail, RunSampleDetail } from "./api";
import { fetchComparisonSample } from "./api";
import { errorMessage, runSampleHref } from "./formatters";
import { SampleViewer } from "./sampleViewer";
import { EmptyState, InlineAnchor } from "./ui";
import { ResizableSplit } from "./workspaceLayout";

export function ComparisonSamplePage() {
  const { baselineRunId, candidateRunId, sampleIndex } = useParams({
    from: "/compare/$baselineRunId/$candidateRunId/$sampleIndex"
  });
  const numericIndex = Number(sampleIndex);
  const searchParams = new URLSearchParams(window.location.search);
  const baselineIndex = optionalSampleIndex(searchParams.get("baseline"));
  const candidateIndex = optionalSampleIndex(searchParams.get("candidate"));
  const validIndex = Number.isInteger(numericIndex) && numericIndex >= 0;
  const query = useQuery({
    queryKey: [
      "comparison-sample",
      baselineRunId,
      candidateRunId,
      numericIndex,
      baselineIndex,
      candidateIndex
    ],
    queryFn: () =>
      fetchComparisonSample(baselineRunId, candidateRunId, numericIndex, {
        baselineIndex,
        candidateIndex
      }),
    enabled: validIndex
  });

  if (!validIndex) {
    return <EmptyState title="样本序号无效" tone="danger" />;
  }
  if (query.isLoading) {
    return <EmptyState title="正在加载对比样本" />;
  }
  if (query.isError || !query.data) {
    return <EmptyState title={`对比样本加载失败：${errorMessage(query.error)}`} tone="danger" />;
  }

  return (
    <section className="page-stack comparison-sample-page">
      <div className="compare-topbar">
        <div className="compare-title">
          <span>样本对比</span>
          <strong>#{numericIndex + 1}</strong>
        </div>
        <div className="compare-chip-strip">
          <span className="sample-count-chip">{baselineRunId}</span>
          <span className="sample-count-chip">{candidateRunId}</span>
        </div>
      </div>
      <ComparisonSampleViewer detail={query.data} />
    </section>
  );
}

function optionalSampleIndex(value: string | null) {
  if (value === null || value.trim() === "") {
    return null;
  }
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed >= 0 ? parsed : null;
}

function ComparisonSampleViewer({ detail }: { detail: ComparisonSampleDetail }) {
  return (
    <ResizableSplit
      className="comparison-sample-detail"
      storageKey="eval_bench_comparison_sample_candidate_width"
      fixedPane="second"
      defaultSize={520}
      minSize={280}
      maxSize={1180}
      first={
        <ComparisonRunPanel
          title="基线"
          runId={detail.baseline_run_id}
          detail={detail.baseline}
        />
      }
      second={
        <ComparisonRunPanel
          title="候选"
          runId={detail.candidate_run_id}
          detail={detail.candidate}
        />
      }
    />
  );
}

function ComparisonRunPanel({
  title,
  runId,
  detail
}: {
  title: string;
  runId: string;
  detail: RunSampleDetail;
}) {
  return (
    <div className="comparison-run-panel">
      <div className="comparison-run-heading">
        <div>
          <div className="eyebrow">{title}</div>
          <h2>{runId}</h2>
        </div>
        <InlineAnchor icon={<Eye size={13} />} href={runSampleHref(runId, detail.sample.index)}>
          打开 run
        </InlineAnchor>
      </div>
      <SampleViewer detail={detail} />
    </div>
  );
}
