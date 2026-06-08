import { Eye } from "lucide-react";

import type { ComparisonLabelDelta, ComparisonSample } from "./api";
import {
  basename,
  comparisonSampleHref,
  formatCompactSignedMetric,
  formatSignedInteger,
  formatSignedMetric
} from "./formatters";
import {
  InlineAnchor,
  NavigationCardAnchor,
  NavigationCardFrame,
  SelectableCardButton
} from "./ui";

export function ComparisonQuickActions({
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
        <InlineAnchor
          className="compare-alert"
          icon={<Eye size={13} />}
          href={comparisonSampleHref(
            baselineRunId,
            candidateRunId,
            firstRegression.candidate_index ?? firstRegression.sample_index ?? 0,
            {
              baselineIndex: firstRegression.baseline_index,
              candidateIndex: firstRegression.candidate_index
            }
          )}
      >
          首个退化
        </InlineAnchor>
      ) : null}
      {firstImprovement ? (
        <InlineAnchor
          className="compare-ready"
          icon={<Eye size={13} />}
          href={comparisonSampleHref(
            baselineRunId,
            candidateRunId,
            firstImprovement.candidate_index ?? firstImprovement.sample_index ?? 0,
            {
              baselineIndex: firstImprovement.baseline_index,
              candidateIndex: firstImprovement.candidate_index
            }
          )}
      >
          首个提升
        </InlineAnchor>
      ) : null}
    </div>
  );
}

export function ComparisonLabelDeltaStrip({
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
        <em>全量</em>
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
              TP {formatSignedInteger(item.delta.matched_count)} / FP{" "}
              {formatSignedInteger(item.delta.false_positive_count)} / FN{" "}
              {formatSignedInteger(item.delta.false_negative_count)}
            </em>
          </SelectableCardButton>
        );
      })}
    </div>
  );
}

export function filterComparisonSamplesByLabel(samples: ComparisonSample[], label: string) {
  if (label === "all") {
    return samples;
  }
  return samples.filter((sample) => Boolean(sample.labels?.[label]));
}

export function firstComparableSample(samples: ComparisonSample[]) {
  return (
    samples.find((sample) => sample.baseline_index !== null && sample.candidate_index !== null) ??
    null
  );
}

export function ComparisonSampleTable({
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
        <div className="comparison-sample-empty">无变化</div>
      ) : (
        <div className="comparison-sample-list">
          {samples.map((sample) => (
            <ComparisonSampleRow
              baselineRunId={baselineRunId}
              candidateRunId={candidateRunId}
              sample={sample}
              key={sample.key}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function ComparisonSampleRow({
  baselineRunId,
  candidateRunId,
  sample
}: {
  baselineRunId: string;
  candidateRunId: string;
  sample: ComparisonSample;
}) {
  const index =
    sample.baseline_index !== null && sample.candidate_index !== null
      ? sample.candidate_index
      : null;
  const name = basename(sample.image ?? sample.key);
  const sampleLabels = Object.keys(sample.labels ?? {}).slice(0, 2);
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
        <MetricDelta label="FP" value={sample.delta.false_positive_count} integer inverted />
        <MetricDelta label="FN" value={sample.delta.false_negative_count} integer inverted />
      </span>
    </>
  );
  if (index === null) {
    return (
      <NavigationCardFrame className="comparison-sample-row disabled">
        {content}
      </NavigationCardFrame>
    );
  }
  return (
    <NavigationCardAnchor
      className="comparison-sample-row"
      href={comparisonSampleHref(baselineRunId, candidateRunId, index, {
        baselineIndex: sample.baseline_index,
        candidateIndex: sample.candidate_index
      })}
    >
      {content}
      <Eye size={14} />
    </NavigationCardAnchor>
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
