import {
  CompositeCanvasCoordinateTag,
  CompositeCanvasOverlayPanel
} from "./compositeCanvasOverlay";
import type { RefObject } from "react";

import "./compositeCanvasPointerReticle.css";

export function CompositeCanvasPointerReticle({
  coordinateRef
}: {
  coordinateRef: RefObject<HTMLSpanElement>;
}) {
  return (
    <CompositeCanvasOverlayPanel
      anchor="full"
      className="composite-canvas-pointer-reticle"
      aria-hidden="true"
    >
      <i className="axis-x" />
      <i className="axis-y" />
      <CompositeCanvasCoordinateTag>
        <span ref={coordinateRef} />
      </CompositeCanvasCoordinateTag>
    </CompositeCanvasOverlayPanel>
  );
}
