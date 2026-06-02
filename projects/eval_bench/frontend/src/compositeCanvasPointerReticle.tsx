import {
  CompositeCanvasCoordinateTag,
  CompositeCanvasOverlayPanel
} from "./compositeCanvasOverlay";
import type { CompositeCanvasPointerState } from "./compositeCanvasPointerTracker";

import "./compositeCanvasPointerReticle.css";

export function CompositeCanvasPointerReticle({
  pointer
}: {
  pointer: CompositeCanvasPointerState | null;
}) {
  if (!pointer) {
    return null;
  }
  return (
    <CompositeCanvasOverlayPanel
      anchor="full"
      className="composite-canvas-pointer-reticle"
      aria-hidden="true"
    >
      <i className="axis-x" />
      <i className="axis-y" />
      <CompositeCanvasCoordinateTag>
        {Math.round(pointer.percentX).toString().padStart(2, "0")} /{" "}
        {Math.round(pointer.percentY).toString().padStart(2, "0")}
      </CompositeCanvasCoordinateTag>
    </CompositeCanvasOverlayPanel>
  );
}
