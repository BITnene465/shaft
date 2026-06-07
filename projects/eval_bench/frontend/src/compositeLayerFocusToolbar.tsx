import type { WheelEvent as ReactWheelEvent } from "react";

import type { CompositeLayerStatus } from "./api";
import { layerAvailabilityColor } from "./compositeLayerPalette";
import { OptionChipButton } from "./ui";

import "./compositeLayerFocusToolbar.css";

export function CompositeLayerFocusToolbar({
  statuses,
  focusedLayerKey,
  onFocusedLayerChange
}: {
  statuses: CompositeLayerStatus[];
  focusedLayerKey: string | null;
  onFocusedLayerChange: (layer: string | null) => void;
}) {
  const availableLayers = statuses.filter((status) => status.available).map((status) => status.layer);

  function cycleFocusFromWheel(event: ReactWheelEvent<HTMLDivElement>) {
    if (availableLayers.length === 0 || Math.abs(event.deltaY) <= Math.abs(event.deltaX)) {
      return;
    }
    const direction = event.deltaY > 0 ? 1 : -1;
    const currentIndex = focusedLayerKey ? availableLayers.indexOf(focusedLayerKey) : -1;
    const nextIndex =
      currentIndex < 0
        ? direction > 0
          ? 0
          : availableLayers.length - 1
        : (currentIndex + direction + availableLayers.length) % availableLayers.length;
    onFocusedLayerChange(availableLayers[nextIndex] ?? null);
  }

  return (
    <div className="composite-workbench-toolbar">
      <div>
        <span>Layer Focus</span>
        <strong>{focusedLayerKey ?? "全部图层"}</strong>
      </div>
      <div
        className="composite-layer-focus-strip"
        aria-label="聚焦图层"
        onWheelCapture={cycleFocusFromWheel}
      >
        <OptionChipButton active={!focusedLayerKey} onClick={() => onFocusedLayerChange(null)}>
          全部
        </OptionChipButton>
        {statuses.map((status, index) => (
          <OptionChipButton
            active={focusedLayerKey === status.layer}
            className={status.available ? "" : "muted"}
            disabled={!status.available}
            key={status.layer}
            onClick={() => onFocusedLayerChange(status.layer)}
            title={status.available ? "点击聚焦，滚轮切换图层" : "该图层在当前图片不可用"}
          >
            <i style={{ background: layerAvailabilityColor(index, status.available) }} />
            {status.layer}
          </OptionChipButton>
        ))}
      </div>
    </div>
  );
}
