import type { CompositeSampleView } from "./api";
import { layerAvailabilityColor } from "./compositeLayerPalette";

import "./compositeComposerDockPreview.css";

export function ReportComposerDockPreview({
  activeSlots,
  readyLayerCount,
  missingLayerCount,
  statuses
}: {
  activeSlots: number;
  readyLayerCount: number;
  missingLayerCount: number;
  statuses?: CompositeSampleView["layer_statuses"];
}) {
  const visibleStatuses = (statuses ?? []).slice(0, 6);
  return (
    <div className="composer-dock-preview" aria-hidden="true">
      <div className="composer-dock-preview-head">
        <span>Composition</span>
        <strong>
          {readyLayerCount}/{activeSlots || readyLayerCount + missingLayerCount} ready
        </strong>
      </div>
      {visibleStatuses.length ? (
        <div className="composer-dock-preview-list">
          {visibleStatuses.map((status, index) => (
            <span
              className={status.available ? "ready" : "missing"}
              key={`${status.layer}_${status.run_id}`}
            >
              <i style={{ background: layerAvailabilityColor(index, status.available) }} />
              <strong title={status.layer}>{status.layer}</strong>
              <em>{status.status}</em>
            </span>
          ))}
        </div>
      ) : (
        <div className="composer-dock-preview-empty">打开编排器添加 layout / arrow 图层。</div>
      )}
    </div>
  );
}
