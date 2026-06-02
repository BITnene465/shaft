import type { CompositeLayerStatus, CompositeSampleLayer } from "./api";
import type { WheelEvent } from "react";
import { CompositeSplitPane } from "./compositeSplitPane";
import type { ActiveLayerConfig } from "./compositeReportModel";

import "./compositeSplitStage.css";

export function SplitStage({
  layers,
  statuses,
  layerConfigs,
  viewportSyncKey,
  focusedLayerKey,
  onFocusedLayerChange,
  activeObjectKey,
  relatedObjectKeys,
  onObjectHover,
  onObjectLock,
  onObjectInspect,
  onObjectWheel,
  onObjectContextMenu
}: {
  layers: CompositeSampleLayer[];
  statuses: CompositeLayerStatus[];
  layerConfigs: ActiveLayerConfig[];
  viewportSyncKey?: string | null;
  focusedLayerKey: string | null;
  onFocusedLayerChange: (layer: string | null) => void;
  activeObjectKey: string | null;
  relatedObjectKeys: Set<string>;
  onObjectHover: (objectKey: string | null) => void;
  onObjectLock: (objectKey: string | null) => void;
  onObjectInspect: (objectKey: string | null) => void;
  onObjectWheel: (event: WheelEvent<HTMLElement>) => void;
  onObjectContextMenu: (request: { objectKey: string; clientX: number; clientY: number }) => void;
}) {
  const layersByName = new Map(layers.map((layer) => [layer.layer, layer]));
  const configByLayer = new Map(layerConfigs.map((config) => [config.key, config]));
  return (
    <div className="composite-split-stage">
      {statuses.map((status) => {
        const layer = layersByName.get(status.layer);
        const config = configByLayer.get(status.layer);
        return (
          <CompositeSplitPane
            key={status.layer}
            status={status}
            layer={layer}
            config={config}
            viewportSyncKey={viewportSyncKey}
            focused={focusedLayerKey === status.layer}
            activeObjectKey={activeObjectKey}
            relatedObjectKeys={relatedObjectKeys}
            onFocusedLayerChange={onFocusedLayerChange}
            onObjectHover={onObjectHover}
            onObjectLock={onObjectLock}
            onObjectInspect={onObjectInspect}
            onObjectWheel={onObjectWheel}
            onObjectContextMenu={onObjectContextMenu}
          />
        );
      })}
    </div>
  );
}
