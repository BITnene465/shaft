import type { CompositeLayerStatus, CompositeSampleLayer } from "./api";
import type { WheelEvent } from "react";
import type { ActiveLayerConfig } from "./compositeReportModel";
import { CompositeSplitLayerCanvas } from "./compositeSplitLayerCanvas";
import { OptionChipButton } from "./ui";

import "./compositeSplitPane.css";

export function CompositeSplitPane({
  status,
  layer,
  config,
  viewportSyncKey,
  focused,
  activeObjectKey,
  relatedObjectKeys,
  onFocusedLayerChange,
  onObjectHover,
  onObjectLock,
  onObjectInspect,
  onObjectWheel,
  onObjectContextMenu
}: {
  status: CompositeLayerStatus;
  layer?: CompositeSampleLayer;
  config?: ActiveLayerConfig;
  viewportSyncKey?: string | null;
  focused: boolean;
  activeObjectKey: string | null;
  relatedObjectKeys: Set<string>;
  onFocusedLayerChange: (layer: string | null) => void;
  onObjectHover: (objectKey: string | null) => void;
  onObjectLock: (objectKey: string | null) => void;
  onObjectInspect: (objectKey: string | null) => void;
  onObjectWheel: (event: WheelEvent<HTMLElement>) => void;
  onObjectContextMenu: (request: { objectKey: string; clientX: number; clientY: number }) => void;
}) {
  return (
    <article
      className={["composite-split-pane", layer ? "" : "missing", focused ? "focused" : ""]
        .filter(Boolean)
        .join(" ")}
      onDoubleClick={() => {
        if (layer) {
          onFocusedLayerChange(focused ? null : status.layer);
        }
      }}
    >
      <SplitPaneHead
        status={status}
        focused={focused}
        canFocus={Boolean(layer)}
        onFocusedLayerChange={onFocusedLayerChange}
      />
      {layer ? (
        <CompositeSplitLayerCanvas
          layer={layer}
          config={config}
          viewportSyncKey={viewportSyncKey}
          activeObjectKey={activeObjectKey}
          relatedObjectKeys={relatedObjectKeys}
          onObjectHover={onObjectHover}
          onObjectLock={onObjectLock}
          onObjectInspect={onObjectInspect}
          onObjectWheel={onObjectWheel}
          onObjectContextMenu={onObjectContextMenu}
        />
      ) : (
        <MissingLayerPane status={status} />
      )}
    </article>
  );
}

function SplitPaneHead({
  status,
  focused,
  canFocus,
  onFocusedLayerChange
}: {
  status: CompositeLayerStatus;
  focused: boolean;
  canFocus: boolean;
  onFocusedLayerChange: (layer: string | null) => void;
}) {
  return (
    <div className="composite-pane-head">
      <div>
        <strong>{status.layer}</strong>
        <span>{status.run_id}</span>
      </div>
      <div className="composite-diagnostic-chips">
        {canFocus ? (
          <OptionChipButton
            active={focused}
            className="composite-focus-chip"
            onClick={() => onFocusedLayerChange(focused ? null : status.layer)}
          >
            Focus
          </OptionChipButton>
        ) : null}
        <em>{status.status}</em>
        <em>M {status.diagnostic_summary.matched_count}</em>
        <em>FP {status.diagnostic_summary.false_positive_count}</em>
        <em>FN {status.diagnostic_summary.false_negative_count}</em>
      </div>
    </div>
  );
}

function MissingLayerPane({ status }: { status: CompositeLayerStatus }) {
  return (
    <div className="missing-layer-pane">
      <strong>{status.layer}</strong>
      <span>{status.status === "image_missing" ? "当前图片不在该 run 的集合中" : "该图片没有预测结果"}</span>
    </div>
  );
}
