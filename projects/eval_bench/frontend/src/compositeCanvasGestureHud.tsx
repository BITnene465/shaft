import { CircleDot, Crosshair, Hand, MousePointer2, RotateCw } from "lucide-react";

import { CompositeCanvasOverlayChip, CompositeCanvasOverlayPanel } from "./compositeCanvasOverlay";

import "./compositeCanvasGestureHud.css";

export function CompositeCanvasGestureHud({
  activeObjectId,
  relatedObjectCount,
  wheelCruise,
  surfacePan,
  contextMenu
}: {
  activeObjectId?: string | null;
  relatedObjectCount: number;
  wheelCruise: boolean;
  surfacePan: boolean;
  contextMenu: boolean;
}) {
  return (
    <CompositeCanvasOverlayPanel
      active={Boolean(activeObjectId)}
      className="composite-canvas-gesture-hud"
      aria-label="组合画布鼠标交互状态"
    >
      <CompositeCanvasOverlayChip
        state={activeObjectId ? "active" : undefined}
        title={activeObjectId ? activeObjectId : "No active object"}
      >
        <MousePointer2 size={13} />
        <strong>{activeObjectId ? "1" : "0"}</strong>
      </CompositeCanvasOverlayChip>
      <CompositeCanvasOverlayChip
        state={relatedObjectCount > 0 ? "active" : undefined}
        title="Related objects"
      >
        <Crosshair size={13} />
        <strong>{relatedObjectCount.toLocaleString()}</strong>
      </CompositeCanvasOverlayChip>
      <CompositeCanvasOverlayChip state={surfacePan ? "ready" : undefined} title="Surface pan">
        <Hand size={13} />
      </CompositeCanvasOverlayChip>
      <CompositeCanvasOverlayChip state={wheelCruise ? "ready" : undefined} title="Object wheel cruise">
        <RotateCw size={13} />
      </CompositeCanvasOverlayChip>
      <CompositeCanvasOverlayChip state={contextMenu ? "ready" : undefined} title="Object context menu">
        <CircleDot size={13} />
      </CompositeCanvasOverlayChip>
    </CompositeCanvasOverlayPanel>
  );
}
