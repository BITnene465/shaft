import type { ReactNode } from "react";

import "./visualStatusBar.css";

export type VisualStatusItem = {
  label: string;
  value: string;
  tone?: "default" | "good" | "warn" | "bad";
  title?: string;
};

export function VisualStatusBar({
  title,
  subtitle,
  items,
  actions,
  refreshing = false,
  className = ""
}: {
  title: string;
  subtitle?: string;
  items: VisualStatusItem[];
  actions?: ReactNode;
  refreshing?: boolean;
  className?: string;
}) {
  return (
    <div
      className={["visual-status-bar", className].filter(Boolean).join(" ")}
      data-refreshing={refreshing ? "true" : "false"}
    >
      <div className="visual-status-identity">
        <strong title={title}>{title}</strong>
        {subtitle ? <span title={subtitle}>{subtitle}</span> : null}
      </div>
      <div className="visual-status-items">
        {items.map((item) => (
          <span className={item.tone ?? "default"} key={`${item.label}:${item.value}`} title={item.title}>
            <em>{item.label}</em>
            <b>{item.value}</b>
          </span>
        ))}
      </div>
      {actions ? <div className="visual-status-actions">{actions}</div> : null}
    </div>
  );
}
