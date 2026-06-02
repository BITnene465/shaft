import { useMemo } from "react";
import type { WheelEvent } from "react";

import type { CompositeLayerStatus, CompositeSampleLayer } from "./api";
import { layerAvailabilityColor } from "./compositeLayerPalette";
import { LayerObjectStrip } from "./compositeLayerObjectStrip";
import { buildLayerObjectRefs } from "./compositeObjectInteraction";
import type { CompositeObjectRef } from "./compositeObjectModel";
import type { ActiveLayerConfig } from "./compositeReportModel";

import "./compositeLayerInspector.css";

export function CompositeInspector({
  layers,
  statuses,
  layerConfigs,
  focusedLayerKey,
  onFocusedLayerChange,
  activeObjectKey,
  relatedObjectKeys,
  lockedObjectKey,
  onObjectHover,
  onObjectLock,
  onObjectInspect,
  onObjectWheel
}: {
  layers: CompositeSampleLayer[];
  statuses: CompositeLayerStatus[];
  layerConfigs: ActiveLayerConfig[];
  focusedLayerKey: string | null;
  onFocusedLayerChange: (layer: string | null) => void;
  activeObjectKey: string | null;
  relatedObjectKeys: Set<string>;
  lockedObjectKey: string | null;
  onObjectHover: (objectKey: string | null) => void;
  onObjectLock: (objectKey: string | null) => void;
  onObjectInspect: (objectKey: string | null) => void;
  onObjectWheel: (event: WheelEvent<HTMLElement>) => void;
}) {
  const configByLayer = new Map(layerConfigs.map((config) => [config.key, config]));
  const objectsByLayer = useMemo(() => {
    const values = new Map<string, CompositeObjectRef[]>();
    layers.forEach((layer) => {
      values.set(layer.layer, buildLayerObjectRefs(layer));
    });
    return values;
  }, [layers]);
  const readyCount = statuses.filter((status) => status.available).length;
  const missingCount = statuses.length - readyCount;
  return (
    <aside className="composite-inspector-panel">
      <div className="composite-inspector-head">
        <span>Layer Report</span>
        <strong>
          {readyCount} ready / {missingCount} missing
        </strong>
      </div>
      <div className="composite-inspector-list">
        {statuses.map((status, index) => {
          const config = configByLayer.get(status.layer);
          return (
            <article
              className={[
                "layer-report-row",
                status.available ? "" : "missing",
                focusedLayerKey === status.layer ? "focused" : ""
              ]
                .filter(Boolean)
                .join(" ")}
              key={status.layer}
              onClick={() =>
                status.available &&
                onFocusedLayerChange(focusedLayerKey === status.layer ? null : status.layer)
              }
            >
              <i style={{ background: layerAvailabilityColor(index, status.available) }} />
              <div>
                <strong>{status.layer}</strong>
                <span title={status.run_id}>{status.run_id}</span>
              </div>
              <div className="layer-report-flags">
                <em className={config?.showGt === false ? "off" : ""}>GT</em>
                <em className={config?.showPred === false ? "off" : ""}>Pred</em>
              </div>
              <dl>
                <div>
                  <dt>M</dt>
                  <dd>{status.diagnostic_summary.matched_count}</dd>
                </div>
                <div>
                  <dt>FP</dt>
                  <dd>{status.diagnostic_summary.false_positive_count}</dd>
                </div>
                <div>
                  <dt>FN</dt>
                  <dd>{status.diagnostic_summary.false_negative_count}</dd>
                </div>
              </dl>
              <LayerObjectStrip
                objects={objectsByLayer.get(status.layer) ?? []}
                activeObjectKey={activeObjectKey}
                relatedObjectKeys={relatedObjectKeys}
                lockedObjectKey={lockedObjectKey}
                onObjectHover={onObjectHover}
                onObjectLock={onObjectLock}
                onObjectInspect={onObjectInspect}
                onObjectWheel={onObjectWheel}
              />
            </article>
          );
        })}
      </div>
    </aside>
  );
}
