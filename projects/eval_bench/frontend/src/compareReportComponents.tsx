import { useEffect, useMemo } from "react";

import type { ComparisonReport } from "./api";
import {
  ComparisonMetricTable,
  ComparisonOutcomeBand,
  ComparisonReportTabs
} from "./compareReportMetrics";
import {
  ComparisonLabelDeltaStrip,
  ComparisonQuickActions,
  ComparisonSampleTable,
  filterComparisonSamplesByLabel,
  firstComparableSample
} from "./compareReportSamples";

export function ComparisonPanel({
  report,
  activeLabel,
  onActiveLabelChange
}: {
  report: ComparisonReport;
  activeLabel: string;
  onActiveLabelChange: (label: string) => void;
}) {
  const labelDeltas = report.labels ?? [];
  const labelValues = labelDeltas.map((item) => item.label);
  useEffect(() => {
    if (activeLabel !== "all" && !labelValues.includes(activeLabel)) {
      onActiveLabelChange("all");
    }
  }, [activeLabel, labelValues.join("|"), onActiveLabelChange]);
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
      <ComparisonReportTabs />
      <ComparisonMetricTable report={report} showsEndpointMetric={showsEndpointMetric} />
      <ComparisonOutcomeBand summary={report.summary} />
      <ComparisonQuickActions
        baselineRunId={report.baseline_run_id}
        candidateRunId={report.candidate_run_id}
        firstImprovement={firstImprovement}
        firstRegression={firstRegression}
      />
      <ComparisonLabelDeltaStrip
        labels={labelDeltas}
        activeLabel={activeLabel}
        onChange={onActiveLabelChange}
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
