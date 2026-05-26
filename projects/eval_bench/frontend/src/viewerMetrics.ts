import type { EvalInstance, RunSampleDetail } from "./api";

export type ObjectKind = "gt" | "pred";

export type ObjectStatus = "match" | "fn" | "fp" | "neutral";

export type ObjectRow = {
  id: string;
  kind: ObjectKind;
  index: number;
  label: string;
  status: ObjectStatus;
  bbox: number[];
  matchIndex: number | null;
  iou: number | null;
};

export type VisibleMetrics = {
  gtCount: number;
  predCount: number;
  matchedCount: number;
  falsePositiveCount: number;
  falseNegativeCount: number;
  meanIou: number;
};

type Diagnostics = RunSampleDetail["diagnostics"];

export function buildObjectRows({
  gtInstances,
  predInstances,
  labels,
  diagnostics
}: {
  gtInstances: EvalInstance[];
  predInstances: EvalInstance[];
  labels: Set<string>;
  diagnostics: Diagnostics;
}): ObjectRow[] {
  const rows: ObjectRow[] = [];
  const gtMatches = new Map((diagnostics?.matches ?? []).map((match) => [match.gt_index, match]));
  const predMatches = new Map(
    (diagnostics?.matches ?? []).map((match) => [match.pred_index, match])
  );
  for (const [index, instance] of gtInstances.entries()) {
    if (!labels.has(instance.label)) {
      continue;
    }
    const match = gtMatches.get(index);
    rows.push({
      id: `gt:${index}`,
      kind: "gt",
      index,
      label: instance.label,
      status: gtObjectStatus(index, diagnostics),
      bbox: instance.bbox,
      matchIndex: match ? match.pred_index : null,
      iou: match ? match.iou : null
    });
  }
  for (const [index, instance] of predInstances.entries()) {
    if (!labels.has(instance.label)) {
      continue;
    }
    const match = predMatches.get(index);
    rows.push({
      id: `pred:${index}`,
      kind: "pred",
      index,
      label: instance.label,
      status: predObjectStatus(index, diagnostics),
      bbox: instance.bbox,
      matchIndex: match ? match.gt_index : null,
      iou: match ? match.iou : null
    });
  }
  return rows.sort((left, right) => {
    if (left.label !== right.label) {
      return left.label.localeCompare(right.label);
    }
    if (left.kind !== right.kind) {
      return left.kind.localeCompare(right.kind);
    }
    return left.index - right.index;
  });
}

export function gtObjectStatus(index: number, diagnostics: Diagnostics): ObjectStatus {
  if (!diagnostics) {
    return "neutral";
  }
  if (diagnostics.false_negatives.some((item) => item.index === index)) {
    return "fn";
  }
  if (diagnostics.matches.some((item) => item.gt_index === index)) {
    return "match";
  }
  return "neutral";
}

export function predObjectStatus(index: number, diagnostics: Diagnostics): ObjectStatus {
  if (!diagnostics) {
    return "neutral";
  }
  if (diagnostics.false_positives.some((item) => item.index === index)) {
    return "fp";
  }
  if (diagnostics.matches.some((item) => item.pred_index === index)) {
    return "match";
  }
  return "neutral";
}

export function visibleSampleMetrics(
  detail: Pick<RunSampleDetail, "gt_instances" | "pred_instances" | "diagnostics">,
  labels: Set<string>
): VisibleMetrics {
  const gtCount = detail.gt_instances.filter((instance) => labels.has(instance.label)).length;
  const predCount = detail.pred_instances.filter((instance) => labels.has(instance.label)).length;
  const diagnostics = detail.diagnostics;
  if (!diagnostics) {
    return {
      gtCount,
      predCount,
      matchedCount: 0,
      falsePositiveCount: 0,
      falseNegativeCount: 0,
      meanIou: 0
    };
  }
  let matchedCount = 0;
  let falsePositiveCount = 0;
  let falseNegativeCount = 0;
  let weightedIou = 0;
  for (const [label, metrics] of Object.entries(diagnostics.labels)) {
    if (!labels.has(label)) {
      continue;
    }
    matchedCount += metrics.matched_count;
    falsePositiveCount += metrics.false_positive_count;
    falseNegativeCount += metrics.false_negative_count;
    weightedIou += metrics.mean_iou * metrics.matched_count;
  }
  return {
    gtCount,
    predCount,
    matchedCount,
    falsePositiveCount,
    falseNegativeCount,
    meanIou: matchedCount > 0 ? weightedIou / matchedCount : 0
  };
}

export function countInstancesByLabel(instances: EvalInstance[]) {
  return instances.reduce<Record<string, number>>((accumulator, instance) => {
    accumulator[instance.label] = (accumulator[instance.label] ?? 0) + 1;
    return accumulator;
  }, {});
}

export function objectStatusLabel(status: ObjectStatus) {
  if (status === "match") {
    return "TP";
  }
  if (status === "fn") {
    return "FN";
  }
  if (status === "fp") {
    return "FP";
  }
  return "unchecked";
}

export function objectMetricText(object: ObjectRow, formatMetric: (value: number | null) => string) {
  if (object.status === "match" && object.matchIndex !== null) {
    const peer = object.kind === "gt" ? "pred" : "gt";
    return `${peer} #${object.matchIndex + 1} / IoU ${formatMetric(object.iou)}`;
  }
  if (object.status === "fn") {
    return "missed GT";
  }
  if (object.status === "fp") {
    return "false positive";
  }
  return "not scored";
}

export function formatBbox(bbox: number[]) {
  if (bbox.length < 4) {
    return "bbox -";
  }
  const [x1, y1, x2, y2] = bbox;
  return `bbox ${formatCoord(x1)}, ${formatCoord(y1)}, ${formatCoord(x2)}, ${formatCoord(y2)}`;
}

function formatCoord(value: number) {
  return Number.isInteger(value) ? String(value) : value.toFixed(1);
}
