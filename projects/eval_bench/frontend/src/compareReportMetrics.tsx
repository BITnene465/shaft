import type { ComparisonReport, ComparisonRunMetrics } from "./api";
import { formatMetric, formatSignedInteger, formatSignedMetric } from "./formatters";

export function ComparisonReportTabs() {
  return (
    <div className="comparison-report-tabs" aria-label="对比报告视图">
      <span className="active">Overview</span>
    </div>
  );
}

type ComparisonMetricRow = {
  key: keyof ComparisonRunMetrics | "false_positive_count" | "false_negative_count";
  label: string;
  baseline?: number;
  candidate?: number;
  delta: number;
  integer?: boolean;
  inverted?: boolean;
};

export function ComparisonMetricTable({
  report,
  showsEndpointMetric
}: {
  report: ComparisonReport;
  showsEndpointMetric: boolean;
}) {
  const rows: ComparisonMetricRow[] = [
    {
      key: "precision_iou50",
      label: "Precision @ .50",
      baseline: report.baseline.precision_iou50,
      candidate: report.candidate.precision_iou50,
      delta: report.delta.precision_iou50
    },
    {
      key: "recall_iou50",
      label: "Recall @ .50",
      baseline: report.baseline.recall_iou50,
      candidate: report.candidate.recall_iou50,
      delta: report.delta.recall_iou50
    },
    {
      key: "mean_iou",
      label: "Mean IoU",
      baseline: report.baseline.mean_iou,
      candidate: report.candidate.mean_iou,
      delta: report.delta.mean_iou
    },
    ...(showsEndpointMetric
      ? [
          {
            key: "mean_keypoint_distance" as const,
            label: "Endpoint distance",
            baseline: report.baseline.mean_keypoint_distance,
            candidate: report.candidate.mean_keypoint_distance,
            delta: report.delta.mean_keypoint_distance,
            inverted: true
          }
        ]
      : []),
    {
      key: "matched_count",
      label: "Matched instances",
      baseline: report.baseline.matched_count,
      candidate: report.candidate.matched_count,
      delta: report.delta.matched_count,
      integer: true
    },
    ...(showsEndpointMetric
      ? [
          {
            key: "keypoint_pair_count" as const,
            label: "Endpoint pairs",
            baseline: report.baseline.keypoint_pair_count,
            candidate: report.candidate.keypoint_pair_count,
            delta: report.delta.keypoint_pair_count,
            integer: true
          }
        ]
      : []),
    {
      key: "gt_instance_count",
      label: "GT instances",
      baseline: report.baseline.gt_instance_count,
      candidate: report.candidate.gt_instance_count,
      delta: report.candidate.gt_instance_count - report.baseline.gt_instance_count,
      integer: true
    },
    {
      key: "pred_instance_count",
      label: "Pred instances",
      baseline: report.baseline.pred_instance_count,
      candidate: report.candidate.pred_instance_count,
      delta: report.candidate.pred_instance_count - report.baseline.pred_instance_count,
      integer: true
    },
    {
      key: "false_positive_count",
      label: "False positives",
      delta: report.delta.false_positive_count,
      integer: true,
      inverted: true
    },
    {
      key: "false_negative_count",
      label: "False negatives",
      delta: report.delta.false_negative_count,
      integer: true,
      inverted: true
    }
  ];
  return (
    <section className="comparison-metric-table" aria-label="Overall Summary">
      <div className="comparison-section-title">
        <strong>Metrics</strong>
        <span>{rows.length.toLocaleString()}</span>
      </div>
      <table>
        <thead>
          <tr>
            <th>Metric</th>
            <th>Baseline</th>
            <th>Candidate</th>
            <th>Delta</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.key}>
              <td>{row.label}</td>
              <td>{formatComparisonMetricValue(row.baseline, row.integer)}</td>
              <td>{formatComparisonMetricValue(row.candidate, row.integer)}</td>
              <td className={comparisonDeltaClassName(row.delta, row.inverted)}>
                {row.integer ? formatSignedInteger(row.delta) : formatSignedMetric(row.delta)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

export function ComparisonOutcomeBand({ summary }: { summary: ComparisonReport["summary"] }) {
  const items = [
    { label: "Up", value: summary.improved_samples, tone: "positive" },
    { label: "Down", value: summary.regressed_samples, tone: "negative" },
    { label: "Changed", value: summary.changed_samples },
    { label: "Same", value: summary.unchanged_samples },
    { label: "Base -", value: summary.missing_in_baseline },
    { label: "Cand -", value: summary.missing_in_candidate }
  ];
  return (
    <div className="comparison-outcome-band">
      {items.map((item) => (
        <span className={item.tone ?? ""} key={item.label}>
          <em>{item.label}</em>
          <strong>{item.value.toLocaleString()}</strong>
        </span>
      ))}
    </div>
  );
}

function formatComparisonMetricValue(value: number | undefined, integer?: boolean) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "-";
  }
  return integer ? value.toLocaleString() : formatMetric(value);
}

function comparisonDeltaClassName(value: number, inverted?: boolean) {
  const positive = value > 0;
  const negative = value < 0;
  const good = inverted ? negative : positive;
  const bad = inverted ? positive : negative;
  return good ? "positive" : bad ? "negative" : "";
}
