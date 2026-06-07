import { useMemo } from "react";
import type { WheelEvent } from "react";

import type { CompositeSampleLayer } from "./api";
import {
  localObjectIdForKey,
  objectKeyForLocalObject,
  relatedLocalObjectIds
} from "./compositeCanvasObjectMapping";
import { CompositeLayerCanvas } from "./compositeLayerCanvas";
import type { ActiveLayerConfig } from "./compositeReportModel";
import { unique } from "./formatters";

export function CompositeSplitLayerCanvas({
  layer,
  config,
  viewportSyncKey,
  activeObjectKey,
  relatedObjectKeys,
  onObjectHover,
  onObjectLock,
  onObjectInspect,
  onObjectWheel,
  onObjectContextMenu
}: {
  layer: CompositeSampleLayer;
  config?: ActiveLayerConfig;
  viewportSyncKey?: string | null;
  activeObjectKey: string | null;
  relatedObjectKeys: Set<string>;
  onObjectHover: (objectKey: string | null) => void;
  onObjectLock: (objectKey: string | null) => void;
  onObjectInspect: (objectKey: string | null) => void;
  onObjectWheel: (event: WheelEvent<HTMLElement>) => void;
  onObjectContextMenu: (request: { objectKey: string; clientX: number; clientY: number }) => void;
}) {
  const labels = useMemo(
    () => unique([...layer.gt_instances, ...layer.pred_instances].map((instance) => instance.label)),
    [layer.gt_instances, layer.pred_instances]
  );
  const activeLocalObjectId = localObjectIdForKey(layer.layer, activeObjectKey);
  const relatedLocalIds = useMemo(
    () => relatedLocalObjectIds(layer.layer, relatedObjectKeys),
    [layer.layer, relatedObjectKeys]
  );

  function resolveLocalObjectKey(objectId: string | null) {
    return objectKeyForLocalObject(layer.layer, objectId);
  }

  return (
    <CompositeLayerCanvas
      className="small"
      sample={layer.sample}
      gtInstances={config?.showGt === false ? [] : layer.gt_instances}
      predInstances={config?.showPred === false ? [] : layer.pred_instances}
      diagnostics={layer.diagnostics}
      labels={labels}
      viewportSyncKey={viewportSyncKey}
      activeObjectId={activeLocalObjectId}
      relatedObjectIds={relatedLocalIds}
      onHover={(objectId) => onObjectHover(resolveLocalObjectKey(objectId))}
      onLock={(objectId) => onObjectLock(resolveLocalObjectKey(objectId))}
      onInspect={(objectId) => onObjectInspect(resolveLocalObjectKey(objectId))}
      onObjectWheel={onObjectWheel}
      onObjectContextMenu={(request) => {
        const objectKey = resolveLocalObjectKey(request.objectId);
        if (objectKey) {
          onObjectContextMenu({
            objectKey,
            clientX: request.clientX,
            clientY: request.clientY
          });
        }
      }}
    />
  );
}
