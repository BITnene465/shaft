import type { CompositeSampleLayer } from "./api";

export type CompositeObjectKind = "gt" | "pred";

export type CompositeObjectStatus = "match" | "fp" | "fn" | "neutral";

export type CompositeObjectRef = {
  key: string;
  layer: string;
  kind: CompositeObjectKind;
  index: number;
  status: CompositeObjectStatus;
  label: string;
  overlayObjectId: string;
};

export function compositeObjectKey(layer: string, kind: CompositeObjectKind, index: number) {
  return `${layer}::${kind}:${index}`;
}

export function parseCompositeObjectKey(key: string) {
  const match = key.match(/^(.*)::(gt|pred):(\d+)$/);
  if (!match) {
    return null;
  }
  return {
    layer: match[1],
    kind: match[2] as CompositeObjectKind,
    index: Number.parseInt(match[3], 10)
  };
}

export function localCanvasObjectIdToKey(layer: string, objectId: string) {
  const [kind, rawIndex] = objectId.split(":");
  const index = Number.parseInt(rawIndex, 10);
  if ((kind !== "gt" && kind !== "pred") || !Number.isFinite(index)) {
    return null;
  }
  return compositeObjectKey(layer, kind, index);
}

export function objectDiagnosticStatus(
  layer: CompositeSampleLayer,
  kind: CompositeObjectKind,
  index: number
): CompositeObjectStatus {
  const diagnostics = layer.diagnostics;
  if (!diagnostics) {
    return "neutral";
  }
  if (kind === "gt") {
    if (diagnostics.false_negatives.some((item) => item.index === index)) {
      return "fn";
    }
    if (diagnostics.matches.some((item) => item.gt_index === index)) {
      return "match";
    }
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

export function objectStatusWeight(status: CompositeObjectStatus) {
  if (status === "fn") {
    return 0;
  }
  if (status === "fp") {
    return 1;
  }
  if (status === "match") {
    return 2;
  }
  return 3;
}

export function normalizeObjectLabel(label: string) {
  return label.trim().toLocaleLowerCase();
}
