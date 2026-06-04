import { useMemo } from "react";
import type { ReactNode } from "react";
import type { WheelEvent } from "react";

import type { CompositeLayerStatus, CompositeSampleLayer } from "./api";
import {
  objectKeyForOverlayObject,
  overlayObjectIdForKey,
  relatedOverlayObjectIds
} from "./compositeCanvasObjectMapping";
import { layerColor } from "./compositeLayerPalette";
import { CompositeLayerCanvas } from "./compositeLayerCanvas";
import { buildOverlayObjects } from "./compositeObjectInteraction";
import type { ActiveLayerConfig } from "./compositeReportModel";
import { unique } from "./formatters";

import "./compositeOverlayStage.css";

export function OverlayStage({
  layers,
  statuses,
  layerConfigs,
  viewportSyncKey,
  activeObjectKey,
  relatedObjectKeys,
  onObjectHover,
  onObjectLock,
  onObjectInspect,
  onObjectWheel,
  onObjectContextMenu,
  statusBar,
  navigator
}: {
  layers: CompositeSampleLayer[];
  statuses: CompositeLayerStatus[];
  layerConfigs: ActiveLayerConfig[];
  viewportSyncKey?: string | null;
  activeObjectKey: string | null;
  relatedObjectKeys: Set<string>;
  onObjectHover: (objectKey: string | null) => void;
  onObjectLock: (objectKey: string | null) => void;
  onObjectInspect: (objectKey: string | null) => void;
  onObjectWheel: (event: WheelEvent<HTMLElement>) => void;
  onObjectContextMenu: (request: { objectKey: string; clientX: number; clientY: number }) => void;
  statusBar?: ReactNode;
  navigator?: ReactNode;
}) {
  const baseLayer = layers[0];
  const sample = baseLayer?.sample;
  const overlay = useMemo(
    () => buildOverlayObjects(layers, layerConfigs),
    [layers, layerConfigs]
  );
  const combined = {
    gtInstances: overlay.gtInstances,
    predInstances: overlay.predInstances
  };
  const labels = useMemo(
    () => unique([...combined.gtInstances, ...combined.predInstances].map((instance) => instance.label)),
    [combined.gtInstances, combined.predInstances]
  );
  const activeOverlayObjectId = overlayObjectIdForKey(overlay.objectRefs, activeObjectKey);
  const relatedOverlayIds = useMemo(
    () => relatedOverlayObjectIds(overlay.objectRefs, relatedObjectKeys),
    [overlay.objectRefs, relatedObjectKeys]
  );
  function resolveOverlayObjectKey(objectId: string | null) {
    return objectKeyForOverlayObject(overlay.objectRefs, objectId);
  }
  return (
    <div className="composite-overlay-stage">
      {statusBar}
      {sample ? (
        <CompositeLayerCanvas
          sample={sample}
          gtInstances={combined.gtInstances}
          predInstances={combined.predInstances}
          diagnostics={null}
          labels={labels}
          viewportSyncKey={viewportSyncKey}
          activeObjectId={activeOverlayObjectId}
          relatedObjectIds={relatedOverlayIds}
          onHover={(objectId) => onObjectHover(resolveOverlayObjectKey(objectId))}
          onLock={(objectId) => onObjectLock(resolveOverlayObjectKey(objectId))}
          onInspect={(objectId) => onObjectInspect(resolveOverlayObjectKey(objectId))}
          onObjectWheel={onObjectWheel}
          onObjectContextMenu={(request) => {
            const objectKey = resolveOverlayObjectKey(request.objectId);
            if (objectKey) {
              onObjectContextMenu({
                objectKey,
                clientX: request.clientX,
                clientY: request.clientY
              });
            }
          }}
        />
      ) : null}
      {navigator}
      <LayerLegend layers={layers} statuses={statuses} />
    </div>
  );
}

function LayerLegend({
  layers,
  statuses
}: {
  layers: CompositeSampleLayer[];
  statuses: CompositeLayerStatus[];
}) {
  const ready = new Set(layers.map((layer) => layer.layer));
  return (
    <div className="composite-layer-legend">
      {statuses.map((status, index) => (
        <span className={ready.has(status.layer) ? "" : "missing"} key={status.layer}>
          <i style={{ background: layerColor(index) }} />
          {status.layer}
          <strong>{status.status === "ready" ? status.benchmark_split : status.status}</strong>
        </span>
      ))}
    </div>
  );
}
