import type { HTMLAttributes, ReactNode } from "react";

import { joinClassNames } from "./ui";

import "./compositeCanvasOverlay.css";

export function CompositeCanvasOverlayPanel({
  active,
  anchor = "bottom-right",
  className,
  children,
  ...props
}: HTMLAttributes<HTMLDivElement> & {
  active?: boolean;
  anchor?: "bottom-right" | "full";
  children: ReactNode;
}) {
  return (
    <div
      {...props}
      className={joinClassNames(
        "composite-canvas-overlay-panel",
        `anchor-${anchor}`,
        active && "active",
        className
      )}
    >
      {children}
    </div>
  );
}

export function CompositeCanvasOverlayChip({
  state,
  className,
  children,
  ...props
}: HTMLAttributes<HTMLSpanElement> & {
  state?: "active" | "ready";
  children: ReactNode;
}) {
  return (
    <span {...props} className={joinClassNames("composite-canvas-overlay-chip", state, className)}>
      {children}
    </span>
  );
}

export function CompositeCanvasCoordinateTag({
  className,
  children,
  ...props
}: HTMLAttributes<HTMLSpanElement> & {
  children: ReactNode;
}) {
  return (
    <span {...props} className={joinClassNames("composite-canvas-coordinate-tag", className)}>
      {children}
    </span>
  );
}
