import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import type { ColumnDef } from "@tanstack/react-table";
import { Eye } from "lucide-react";

import type {
  ComparisonLabelDelta,
  ComparisonReport,
  ComparisonSample,
  ComparisonSummary,
  RunSummary
} from "./api";
import { fetchComparison, fetchComparisons, fetchRuns } from "./api";
import { FormSelectControl } from "./controlPrimitives";
import { AdvancedFilterBar } from "./filterControls";
import {
  basename,
  comparisonSampleHref,
  formatCompactSignedMetric,
  formatDate,
  formatMetric,
  formatRunOption,
  formatSignedInteger,
  formatSignedMetric,
  runIdExists,
  unique
} from "./formatters";
import { AppIcon } from "./iconLibrary";
import { PagerControl, clampListPageOffset } from "./samplePager";
import { Badge, DataTable, EmptyState, SelectableCardButton } from "./ui";
import { ResizableSplit } from "./workspaceLayout";

const COMPARE_RUN_PAGE_SIZE = 80;

export function ComparePage() {
  const [searchText, setSearchText] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [taskFilter, setTaskFilter] = useState("all");
  const [benchmarkFilter, setBenchmarkFilter] = useState("all");
  const [labelFilter, setLabelFilter] = useState("all");
  const [modelFilter, setModelFilter] = useState("all");
  const [promptFilter, setPromptFilter] = useState("all");
  const [pageOffset, setPageOffset] = useState(0);
  const [baselineRunId, setBaselineRunId] = useState(
    () => new URLSearchParams(window.location.search).get("baseline") ?? ""
  );
  const [candidateRunId, setCandidateRunId] = useState(
    () => new URLSearchParams(window.location.search).get("candidate") ?? ""
  );
  const comparisonFilters = useMemo(
    () => ({
      task: taskFilter === "all" ? undefined : taskFilter,
      label: labelFilter === "all" ? undefined : labelFilter,
      query: searchText.trim() || undefined,
      limit: 50
    }),
    [labelFilter, searchText, taskFilter]
  );
  const runFilters = useMemo(
    () => ({
      offset: pageOffset,
      limit: COMPARE_RUN_PAGE_SIZE,
      status: statusFilter !== "all" ? statusFilter : undefined,
      task: taskFilter !== "all" ? taskFilter : undefined,
      benchmarkId: benchmarkFilter !== "all" ? benchmarkFilter : undefined,
      label: labelFilter !== "all" ? labelFilter : undefined,
      modelId: modelFilter !== "all" ? modelFilter : undefined,
      promptId: promptFilter !== "all" ? promptFilter : undefined,
      query: searchText.trim() || undefined
    }),
    [
      benchmarkFilter,
      labelFilter,
      modelFilter,
      pageOffset,
      promptFilter,
      searchText,
      statusFilter,
      taskFilter
    ]
  );
  const comparisonListQuery = useQuery({
    queryKey: ["comparisons", comparisonFilters],
    queryFn: () => fetchComparisons(comparisonFilters)
  });
  const runsQuery = useQuery({
    queryKey: ["runs", "compare", runFilters],
    queryFn: () => fetchRuns(runFilters)
  });
  const runFacetsQuery = useQuery({
    queryKey: ["runs", "compare", "facets"],
    queryFn: () => fetchRuns({ limit: 500 })
  });
  const runs = runsQuery.data?.runs ?? [];
  const runFacets = runFacetsQuery.data?.runs ?? runs;
  const statuses = unique(runFacets.map((run) => run.status).filter(Boolean));
  const tasks = unique(runFacets.map((run) => run.spec_task).filter(Boolean));
  const benchmarks = unique(runFacets.map((run) => run.benchmark_id).filter(Boolean));
  const labels = unique(runFacets.flatMap((run) => run.target_labels).filter(Boolean));
  const models = unique(runFacets.map((run) => run.model_id).filter(Boolean));
  const prompts = unique(runFacets.map((run) => run.prompt_id).filter(Boolean));
  const comparableRuns = runs.filter((run) => run.report_path);
  const filteredCount = runsQuery.data?.total ?? runs.length;
  const runPageOffset = runsQuery.data?.offset ?? pageOffset;
  const runPageLimit = runsQuery.data?.limit ?? COMPARE_RUN_PAGE_SIZE;
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
    queryFn: () => fetchComparison(effectiveBaseline, effectiveCandidate),
    enabled: Boolean(effectiveBaseline && effectiveCandidate && effectiveBaseline !== effectiveCandidate)
  });

  useEffect(() => {
    if (comparisonQuery.data?.comparison_id) {
      void comparisonListQuery.refetch();
    }
  }, [comparisonListQuery.refetch, comparisonQuery.data?.comparison_id]);
  useEffect(() => {
    setPageOffset(0);
  }, [searchText, statusFilter, taskFilter, benchmarkFilter, labelFilter, modelFilter, promptFilter]);
  useEffect(() => {
    const nextOffset = clampListPageOffset(pageOffset, filteredCount, COMPARE_RUN_PAGE_SIZE);
    if (nextOffset !== pageOffset) {
      setPageOffset(nextOffset);
    }
  }, [filteredCount, pageOffset]);

  if (runsQuery.isLoading) {
    return <EmptyState title="正在加载对比状态" />;
  }
  if (runsQuery.error || !runsQuery.data) {
    return <EmptyState title="对比状态加载失败" tone="danger" />;
  }

  return (
    <section className="page-stack compare-page">
      <div className="compare-topbar">
        <div className="compare-title">
          <span>对比工作区</span>
          <strong>{filteredCount.toLocaleString()} 条 run</strong>
        </div>
        <Link className="mini-link compare-ready" to="/rank-board">
          <AppIcon name="rankBoard" size={13} />
          排行榜
        </Link>
      </div>
      <AdvancedFilterBar
        title="对比高级检索"
        meta="筛选候选 run：状态、任务、基准集、label、模型、prompt 和备注全文"
        controls={[
          {
            type: "search",
            id: "compare-query",
            label: "全文检索",
            value: searchText,
            onChange: setSearchText,
            placeholder: "搜索 run、模型、prompt、备注"
          },
          {
            type: "select",
            id: "compare-status",
            label: "状态",
            value: statusFilter,
            values: ["all", ...statuses],
            labels: { all: "全部" },
            onChange: setStatusFilter
          },
          {
            type: "select",
            id: "compare-task",
            label: "任务",
            value: taskFilter,
            values: ["all", ...tasks],
            labels: { all: "全部" },
            onChange: setTaskFilter
          },
          {
            type: "select",
            id: "compare-benchmark",
            label: "基准集",
            value: benchmarkFilter,
            values: ["all", ...benchmarks],
            labels: { all: "全部" },
            onChange: setBenchmarkFilter
          },
          {
            type: "select",
            id: "compare-label",
            label: "标签",
            value: labelFilter,
            values: ["all", ...labels],
            labels: { all: "全部" },
            onChange: setLabelFilter
          },
          {
            type: "select",
            id: "compare-model",
            label: "模型",
            value: modelFilter,
            values: ["all", ...models],
            labels: { all: "全部" },
            onChange: setModelFilter
          },
          {
            type: "select",
            id: "compare-prompt",
            label: "Prompt",
            value: promptFilter,
            values: ["all", ...prompts],
            labels: { all: "全部" },
            onChange: setPromptFilter
          }
        ]}
      />
      <ResizableSplit
        className="compare-workspace"
        storageKey="eval_bench_compare_rail_width"
        defaultSize={292}
        minSize={180}
        maxSize={680}
        first={
          <aside className="compare-run-rail">
            <RunSelectRail
              title="基线"
              value={effectiveBaseline}
              runs={comparableRuns}
              disabled={filteredCount < 2 && !effectiveBaseline}
              onChange={setBaselineRunId}
            />
            <RunSelectRail
              title="候选"
              value={effectiveCandidate}
              runs={comparableRuns}
              disabled={filteredCount < 2 && !effectiveCandidate}
              onChange={setCandidateRunId}
            />
            <PagerControl
              className="rank-board-pager compare-run-pager"
              offset={runPageOffset}
              limit={runPageLimit}
              total={filteredCount}
              meta={
                <>
                  {" · "}
                  {runs.length.toLocaleString()} visible / {comparableRuns.length.toLocaleString()} reports
                </>
              }
              onPageChange={setPageOffset}
            />
            <ComparisonHistoryPanel
              comparisons={comparisonListQuery.data?.comparisons ?? []}
              total={comparisonListQuery.data?.total}
            />
          </aside>
        }
        second={
          <ResizableSplit
            className="compare-main-split"
            storageKey="eval_bench_compare_context_width"
            fixedPane="second"
            defaultSize={340}
            minSize={260}
            maxSize={620}
            first={
              <main className="compare-report-pane">
                {!effectiveBaseline || !effectiveCandidate ? (
                  <div className="empty-panel">至少需要两个已完成评测的 run 才能对比。</div>
                ) : effectiveBaseline === effectiveCandidate ? (
                  <div className="empty-panel">请选择两个不同的 run。</div>
                ) : comparisonQuery.isLoading ? (
                  <div className="empty-panel">正在加载对比报告</div>
                ) : comparisonQuery.isError || !comparisonQuery.data ? (
                  <div className="empty-panel danger-text">对比报告加载失败。</div>
                ) : (
                  <ComparisonPanel report={comparisonQuery.data} />
                )}
              </main>
            }
            second={
              <aside className="compare-context-pane">
                <div className="comparison-sample-title">对比上下文</div>
                <CompareContextPanel
                  filteredCount={filteredCount}
                  comparableCount={comparableRuns.length}
                  baselineRunId={effectiveBaseline}
                  candidateRunId={effectiveCandidate}
                />
              </aside>
            }
          />
        }
      />
    </section>
  );
}

function RunSelectRail({
  title,
  value,
  runs,
  disabled,
  onChange
}: {
  title: string;
  value: string;
  runs: RunSummary[];
  disabled: boolean;
  onChange: (value: string) => void;
}) {
  const selected = disabled ? undefined : runs.find((run) => run.run_id === value);
  const runOptions = disabled
    ? [{ value: "", label: "需要两个报告", disabled: true }]
    : [
        ...(value && !runIdExists(runs, value)
          ? [{ value, label: `${value} · 已选择` }]
          : []),
        ...runs.map((run) => ({
          value: run.run_id,
          label: formatRunOption(run)
        }))
      ];
  return (
    <div className="compare-run-select">
      <FormSelectControl
        label={title}
        value={disabled ? "" : value}
        options={runOptions}
        disabled={disabled}
        onChange={onChange}
      />
      {selected ? (
        <div className="compare-run-card">
          <strong title={selected.run_id}>{selected.run_id}</strong>
          <span>{selected.model_id}</span>
          <div>
            <Badge value={selected.status} domain="run" />
            <em>R {formatMetric(selected.recall_iou50)}</em>
            <em>P {formatMetric(selected.precision_iou50)}</em>
          </div>
        </div>
      ) : value && !disabled ? (
        <div className="compare-run-card">
          <strong title={value}>{value}</strong>
          <span>已选择；当前页未加载该 run</span>
          <div>
            <em>翻页不会清空当前对比</em>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function ComparisonPanel({ report }: { report: ComparisonReport }) {
  const [activeLabel, setActiveLabel] = useState("all");
  const labelDeltas = report.labels ?? [];
  const labelValues = labelDeltas.map((item) => item.label);
  useEffect(() => {
    if (activeLabel !== "all" && !labelValues.includes(activeLabel)) {
      setActiveLabel("all");
    }
  }, [activeLabel, labelValues.join("|")]);
  const filteredImprovements = useMemo(
    () => filterComparisonSamplesByLabel(report.top_improvements, activeLabel),
    [activeLabel, report.top_improvements]
  );
  const filteredRegressions = useMemo(
    () => filterComparisonSamplesByLabel(report.top_regressions, activeLabel),
    [activeLabel, report.top_regressions]
  );
  const firstImprovement = firstComparableSample(filteredImprovements);
  const firstRegression = firstComparableSample(filteredRegressions);
  const showsEndpointMetric =
    report.metric_profile === "keypoint_endpoint_v1" ||
    report.delta.keypoint_pair_count !== 0 ||
    report.delta.mean_keypoint_distance !== 0;
  return (
    <div className="comparison-panel">
      <div className="comparison-title-row">
        <div>
          <div className="eyebrow">双模型对比报告</div>
          <h2>
            {report.baseline_run_id} vs {report.candidate_run_id}
          </h2>
        </div>
        <div className="compare-title-meta">
          <div className="sample-count-chip">{report.sample_count.toLocaleString()} 个样本</div>
          {report.target_labels?.length ? (
            <div className="sample-count-chip subtle">{report.target_labels.join(" / ")}</div>
          ) : null}
        </div>
      </div>
      {report.warnings?.length ? (
        <div className="comparison-warning-strip">
          {report.warnings.map((warning) => (
            <span key={warning}>{warning}</span>
          ))}
        </div>
      ) : null}
      <div className="comparison-delta-grid">
        <DeltaCard label="P@.50" value={report.delta.precision_iou50} />
        <DeltaCard label="R@.50" value={report.delta.recall_iou50} />
        <DeltaCard label="平均 IoU" value={report.delta.mean_iou} />
        {showsEndpointMetric ? (
          <DeltaCard label="端点距离" value={report.delta.mean_keypoint_distance} inverted />
        ) : null}
        <DeltaCard label="匹配数" value={report.delta.matched_count} integer />
        {showsEndpointMetric ? (
          <DeltaCard label="端点对" value={report.delta.keypoint_pair_count} integer />
        ) : null}
        <DeltaCard label="误检" value={report.delta.false_positive_count} integer inverted />
        <DeltaCard label="漏检" value={report.delta.false_negative_count} integer inverted />
      </div>
      <div className="comparison-summary-row">
        <SummaryPill label="提升" value={report.summary.improved_samples} tone="positive" />
        <SummaryPill label="退化" value={report.summary.regressed_samples} tone="negative" />
        <SummaryPill label="变化" value={report.summary.changed_samples} />
        <SummaryPill label="不变" value={report.summary.unchanged_samples} />
      </div>
      <ComparisonQuickActions
        baselineRunId={report.baseline_run_id}
        candidateRunId={report.candidate_run_id}
        firstImprovement={firstImprovement}
        firstRegression={firstRegression}
      />
      <ComparisonLabelDeltaStrip
        labels={labelDeltas}
        activeLabel={activeLabel}
        onChange={setActiveLabel}
      />
      <div className="comparison-columns">
        <ComparisonSampleTable
          title="提升最多"
          samples={filteredImprovements}
          baselineRunId={report.baseline_run_id}
          candidateRunId={report.candidate_run_id}
          tone="positive"
        />
        <ComparisonSampleTable
          title="退化最多"
          samples={filteredRegressions}
          baselineRunId={report.baseline_run_id}
          candidateRunId={report.candidate_run_id}
          tone="negative"
        />
      </div>
    </div>
  );
}

function ComparisonQuickActions({
  baselineRunId,
  candidateRunId,
  firstImprovement,
  firstRegression
}: {
  baselineRunId: string;
  candidateRunId: string;
  firstImprovement: ComparisonSample | null;
  firstRegression: ComparisonSample | null;
}) {
  return (
    <div className="comparison-quick-actions">
      {firstRegression ? (
        <a
          className="mini-link compare-alert"
          href={comparisonSampleHref(
            baselineRunId,
            candidateRunId,
            firstRegression.candidate_index ?? firstRegression.sample_index ?? 0
          )}
        >
          <Eye size={13} />
          看首个退化样本
        </a>
      ) : null}
      {firstImprovement ? (
        <a
          className="mini-link compare-ready"
          href={comparisonSampleHref(
            baselineRunId,
            candidateRunId,
            firstImprovement.candidate_index ?? firstImprovement.sample_index ?? 0
          )}
        >
          <Eye size={13} />
          看首个提升样本
        </a>
      ) : null}
    </div>
  );
}

function DeltaCard({
  label,
  value,
  integer,
  inverted
}: {
  label: string;
  value: number;
  integer?: boolean;
  inverted?: boolean;
}) {
  const positive = value > 0;
  const negative = value < 0;
  const good = inverted ? negative : positive;
  const bad = inverted ? positive : negative;
  const className = good ? "delta-card positive" : bad ? "delta-card negative" : "delta-card";
  return (
    <div className={className}>
      <span>{label}</span>
      <strong>{integer ? formatSignedInteger(value) : formatSignedMetric(value)}</strong>
    </div>
  );
}

function SummaryPill({
  label,
  value,
  tone
}: {
  label: string;
  value: number;
  tone?: "positive" | "negative";
}) {
  return (
    <div className={tone ? `summary-pill ${tone}` : "summary-pill"}>
      <span>{label}</span>
      <strong>{value.toLocaleString()}</strong>
    </div>
  );
}

function ComparisonLabelDeltaStrip({
  labels,
  activeLabel,
  onChange
}: {
  labels: ComparisonLabelDelta[];
  activeLabel: string;
  onChange: (label: string) => void;
}) {
  const visible = labels.slice(0, 8);
  if (visible.length === 0) {
    return null;
  }
  return (
    <div className="comparison-label-strip">
      <SelectableCardButton
        active={activeLabel === "all"}
        className="label-delta-card"
        onClick={() => onChange("all")}
      >
        <span>全部标签</span>
        <strong>All</strong>
        <em>查看全量变化样本</em>
      </SelectableCardButton>
      {visible.map((item) => {
        const tone =
          item.delta_score > 0 ? "positive" : item.delta_score < 0 ? "negative" : "neutral";
        return (
          <SelectableCardButton
            active={activeLabel === item.label}
            className={`label-delta-card ${tone}`}
            onClick={() => onChange(item.label)}
            key={item.label}
          >
            <span>{item.label}</span>
            <strong>
              {item.delta.keypoint_pair_count !== 0 || item.delta.mean_keypoint_distance !== 0
                ? `D ${formatSignedMetric(item.delta.mean_keypoint_distance)}`
                : `R ${formatSignedMetric(item.delta.recall_iou50)}`}
            </strong>
            <em>
              TP {formatSignedInteger(item.delta.matched_count)} · FP{" "}
              {formatSignedInteger(item.delta.false_positive_count)} · FN{" "}
              {formatSignedInteger(item.delta.false_negative_count)}
            </em>
          </SelectableCardButton>
        );
      })}
    </div>
  );
}

function filterComparisonSamplesByLabel(samples: ComparisonSample[], label: string) {
  if (label === "all") {
    return samples;
  }
  return samples.filter((sample) => Boolean(sample.labels?.[label]));
}

function firstComparableSample(samples: ComparisonSample[]) {
  return (
    samples.find((sample) => sample.candidate_index !== null || sample.sample_index !== null) ?? null
  );
}

function ComparisonSampleTable({
  title,
  samples,
  baselineRunId,
  candidateRunId,
  tone
}: {
  title: string;
  samples: ComparisonSample[];
  baselineRunId: string;
  candidateRunId: string;
  tone: "positive" | "negative";
}) {
  return (
    <div className={`comparison-sample-block ${tone}`}>
      <div className="comparison-sample-title">{title}</div>
      {samples.length === 0 ? (
        <div className="comparison-sample-empty">没有变化样本。</div>
      ) : (
        <div className="comparison-sample-list">
          {samples.map((sample) => {
            const index = sample.candidate_index ?? sample.sample_index;
            const name = basename(sample.image ?? sample.key);
            const sampleLabels = Object.keys(sample.labels ?? {}).slice(0, 4);
            const content = (
              <>
                <span className="comparison-sample-row-head">
                  <strong title={sample.image ?? sample.key}>{name}</strong>
                  <em>{index === null ? "未对齐" : `#${index + 1}`}</em>
                  <span>{sample.status}</span>
                </span>
                {sampleLabels.length > 0 ? (
                  <span className="comparison-sample-labels">
                    {sampleLabels.map((label) => (
                      <em key={label}>{label}</em>
                    ))}
                  </span>
                ) : null}
                <span className="comparison-sample-metrics">
                  <MetricDelta label="Score" value={sample.delta_score} />
                  <MetricDelta label="TP" value={sample.delta.matched_count} integer />
                  <MetricDelta
                    label="FP"
                    value={sample.delta.false_positive_count}
                    integer
                    inverted
                  />
                  <MetricDelta
                    label="FN"
                    value={sample.delta.false_negative_count}
                    integer
                    inverted
                  />
                  <MetricDelta label="IoU" value={sample.delta.mean_iou} />
                  {sample.delta.keypoint_pair_count !== 0 ||
                  sample.delta.mean_keypoint_distance !== 0 ? (
                    <>
                      <MetricDelta
                        label="D"
                        value={sample.delta.mean_keypoint_distance}
                        inverted
                      />
                      <MetricDelta label="Pts" value={sample.delta.keypoint_pair_count} integer />
                    </>
                  ) : null}
                </span>
              </>
            );
            if (index === null) {
              return (
                <div className="comparison-sample-row disabled" key={sample.key}>
                  {content}
                </div>
              );
            }
            return (
              <a
                className="comparison-sample-row"
                href={comparisonSampleHref(baselineRunId, candidateRunId, index)}
                key={sample.key}
              >
                {content}
                <Eye size={14} />
              </a>
            );
          })}
        </div>
      )}
    </div>
  );
}

function MetricDelta({
  label,
  value,
  integer,
  inverted
}: {
  label: string;
  value: number;
  integer?: boolean;
  inverted?: boolean;
}) {
  const positive = value > 0;
  const negative = value < 0;
  const good = inverted ? negative : positive;
  const bad = inverted ? positive : negative;
  const className = good ? "metric-delta positive" : bad ? "metric-delta negative" : "metric-delta";
  return (
    <span className={className}>
      <em>{label}</em>
      <strong>{integer ? formatSignedInteger(value) : formatCompactSignedMetric(value)}</strong>
    </span>
  );
}
function ComparisonHistoryPanel({
  comparisons,
  total
}: {
  comparisons: ComparisonSummary[];
  total?: number;
}) {
  if (comparisons.length === 0) {
    return null;
  }
  const columns: ColumnDef<ComparisonSummary>[] = [
    { header: "对比记录", accessorKey: "comparison_id" },
    { header: "任务", accessorKey: "task" },
    { header: "Label", cell: ({ row }) => row.original.target_labels?.join(", ") || "all" },
    { header: "样本数", cell: ({ row }) => row.original.sample_count.toLocaleString() },
    { header: "Delta R", cell: ({ row }) => formatSignedMetric(row.original.delta.recall_iou50) },
    { header: "提升", cell: ({ row }) => row.original.summary.improved_samples.toLocaleString() },
    { header: "退化", cell: ({ row }) => row.original.summary.regressed_samples.toLocaleString() },
    { header: "创建时间", cell: ({ row }) => formatDate(row.original.created_at) }
  ];
  return (
    <div className="history-block">
      <div className="comparison-sample-title">
        历史对比
        {typeof total === "number" ? <span>{total.toLocaleString()} 条</span> : null}
      </div>
      <DataTable
        columns={columns}
        data={comparisons}
        emptyText="暂无历史对比。"
        compact
      />
    </div>
  );
}

function CompareContextPanel({
  filteredCount,
  comparableCount,
  baselineRunId,
  candidateRunId
}: {
  filteredCount: number;
  comparableCount: number;
  baselineRunId: string;
  candidateRunId: string;
}) {
  return (
    <div className="compare-context-stack">
      <div className="compare-context-grid">
        <div className="compare-context-card">
          <span>过滤后</span>
          <strong>{filteredCount.toLocaleString()}</strong>
          <em>run</em>
        </div>
        <div className="compare-context-card">
          <span>可对比</span>
          <strong>{comparableCount.toLocaleString()}</strong>
          <em>report</em>
        </div>
      </div>
      <Link className="mini-link compare-ready compare-rank-link" to="/rank-board">
        <AppIcon name="rankBoard" size={13} />
        打开独立排行榜
      </Link>
      <div className="compare-context-links">
        {baselineRunId ? (
          <Link to="/runs/$runId" params={{ runId: baselineRunId }}>
            基线 {baselineRunId}
          </Link>
        ) : null}
        {candidateRunId ? (
          <Link to="/runs/$runId" params={{ runId: candidateRunId }}>
            候选 {candidateRunId}
          </Link>
        ) : null}
      </div>
    </div>
  );
}
