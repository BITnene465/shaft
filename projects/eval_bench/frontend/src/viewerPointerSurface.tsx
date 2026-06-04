import type { ReactNode } from "react";

import { CompositeCanvasPointerReticle } from "./compositeCanvasPointerReticle";
import { useCompositeCanvasPointerTracker } from "./compositeCanvasPointerTracker";

export function ViewerPointerSurface({
  children,
  className = ""
}: {
  children: ReactNode;
  className?: string;
}) {
  const pointer = useCompositeCanvasPointerTracker();

  return (
    <div
      className={["viewer-pointer-surface", className].filter(Boolean).join(" ")}
      {...pointer.pointerHandlers}
    >
      {children}
      <CompositeCanvasPointerReticle coordinateRef={pointer.coordinateRef} />
    </div>
  );
}
