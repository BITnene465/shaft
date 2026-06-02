import type { CompositeSampleLayer, EvalInstance } from "./api";
import {
  compositeObjectKey,
  normalizeObjectLabel,
  objectDiagnosticStatus,
  objectStatusWeight,
  parseCompositeObjectKey
} from "./compositeObjectModel";
import type { CompositeObjectRef } from "./compositeObjectModel";
import type { ActiveLayerConfig } from "./compositeReportModel";

export type {
  CompositeObjectKind,
  CompositeObjectRef,
  CompositeObjectStatus
} from "./compositeObjectModel";

export function buildOverlayObjects(
  layers: CompositeSampleLayer[],
  layerConfigs: ReadonlyArray<ActiveLayerConfig>
) {
  const configByLayer = new Map(layerConfigs.map((config) => [config.key, config]));
  const objectRefs: CompositeObjectRef[] = [];
  const gtInstances: EvalInstance[] = [];
  const predInstances: EvalInstance[] = [];
  layers.forEach((layer) => {
    const config = configByLayer.get(layer.layer);
    if (config?.showGt !== false) {
      layer.gt_instances.forEach((instance, index) => {
        const overlayIndex = gtInstances.length;
        gtInstances.push({ ...instance, label: `${layer.layer}:${instance.label}` });
        objectRefs.push({
          key: compositeObjectKey(layer.layer, "gt", index),
          layer: layer.layer,
          kind: "gt",
          index,
          status: objectDiagnosticStatus(layer, "gt", index),
          label: instance.label,
          overlayObjectId: `gt:${overlayIndex}`
        });
      });
    }
    if (config?.showPred !== false) {
      layer.pred_instances.forEach((instance, index) => {
        const overlayIndex = predInstances.length;
        predInstances.push({ ...instance, label: `${layer.layer}:${instance.label}` });
        objectRefs.push({
          key: compositeObjectKey(layer.layer, "pred", index),
          layer: layer.layer,
          kind: "pred",
          index,
          status: objectDiagnosticStatus(layer, "pred", index),
          label: instance.label,
          overlayObjectId: `pred:${overlayIndex}`
        });
      });
    }
  });
  return { gtInstances, predInstances, objectRefs };
}

export function buildLayerObjectRefs(layer: CompositeSampleLayer) {
  return buildLayerObjectRefsForScope(layer)
    .filter((object) => object.status === "fp" || object.status === "fn")
    .sort((left, right) => objectStatusWeight(left.status) - objectStatusWeight(right.status));
}

export function buildLayerObjectRefsForScope(layer: CompositeSampleLayer) {
  return [
    ...layer.gt_instances.map((instance, index) => ({
      key: compositeObjectKey(layer.layer, "gt", index),
      layer: layer.layer,
      kind: "gt" as const,
      index,
      status: objectDiagnosticStatus(layer, "gt", index),
      label: instance.label,
      overlayObjectId: `gt:${index}`
    })),
    ...layer.pred_instances.map((instance, index) => ({
      key: compositeObjectKey(layer.layer, "pred", index),
      layer: layer.layer,
      kind: "pred" as const,
      index,
      status: objectDiagnosticStatus(layer, "pred", index),
      label: instance.label,
      overlayObjectId: `pred:${index}`
    }))
  ];
}

export function allCompositeObjectRefs(layers: CompositeSampleLayer[]) {
  return layers.flatMap((layer) => buildLayerObjectRefsForScope(layer));
}

export function nextCompositeObjectKey(
  layers: CompositeSampleLayer[],
  currentKey: string | null,
  direction: -1 | 1
) {
  const objects = allCompositeObjectRefs(layers);
  if (objects.length === 0) {
    return null;
  }
  const currentIndex = objects.findIndex((object) => object.key === currentKey);
  if (currentIndex < 0) {
    return direction > 0 ? objects[0].key : objects[objects.length - 1].key;
  }
  return objects[(currentIndex + direction + objects.length) % objects.length].key;
}

export function relatedCompositeObjectKeys(layers: CompositeSampleLayer[], key: string | null) {
  const activeObject = resolveCompositeObjectRef(layers, key);
  if (!activeObject) {
    return new Set<string>();
  }
  const normalizedLabel = normalizeObjectLabel(activeObject.label);
  if (!normalizedLabel) {
    return new Set<string>();
  }
  return new Set(
    layers
      .flatMap((layer) => buildLayerObjectRefsForScope(layer))
      .filter(
        (object) =>
          object.key !== activeObject.key &&
          object.kind === activeObject.kind &&
          normalizeObjectLabel(object.label) === normalizedLabel
      )
      .map((object) => object.key)
  );
}

export function resolveCompositeObjectRef(layers: CompositeSampleLayer[], key: string | null) {
  if (!key) {
    return null;
  }
  const parsed = parseCompositeObjectKey(key);
  if (!parsed) {
    return null;
  }
  const layer = layers.find((item) => item.layer === parsed.layer);
  if (!layer) {
    return null;
  }
  const instances = parsed.kind === "gt" ? layer.gt_instances : layer.pred_instances;
  const instance = instances[parsed.index];
  if (!instance) {
    return null;
  }
  return {
    key,
    layer: parsed.layer,
    kind: parsed.kind,
    index: parsed.index,
    status: objectDiagnosticStatus(layer, parsed.kind, parsed.index),
    label: instance.label,
    overlayObjectId: `${parsed.kind}:${parsed.index}`
  } satisfies CompositeObjectRef;
}
