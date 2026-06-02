import type { ReactNode } from "react";

import "./compositePanelPrimitives.css";

export function CompositePanelHeader({
  title,
  eyebrow,
  meta,
  action,
  className = "",
  actionClassName = "",
  density = "regular",
  framed = false
}: {
  title: ReactNode;
  eyebrow?: ReactNode;
  meta?: ReactNode;
  action?: ReactNode;
  className?: string;
  actionClassName?: string;
  density?: "compact" | "regular";
  framed?: boolean;
}) {
  return (
    <div
      className={[
        "composite-panel-head",
        `density-${density}`,
        framed ? "framed" : "",
        className
      ]
        .filter(Boolean)
        .join(" ")}
    >
      <div>
        {eyebrow ? <span>{eyebrow}</span> : null}
        <strong>{title}</strong>
        {meta ? <span>{meta}</span> : null}
      </div>
      {action ? (
        <div className={["composite-panel-action", actionClassName].filter(Boolean).join(" ")}>
          {action}
        </div>
      ) : null}
    </div>
  );
}

export function CompositePanelEmptyState({
  children,
  className = ""
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={["composite-panel-empty", className].filter(Boolean).join(" ")}>
      {children}
    </div>
  );
}
