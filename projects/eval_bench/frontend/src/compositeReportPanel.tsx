import type { ReactNode } from "react";

import { CompositePanelEmptyState, CompositePanelHeader } from "./compositePanelPrimitives";
import "./compositeReportPanel.css";

export function CompositeReportPanelHeader({
  eyebrow,
  title,
  action
}: {
  eyebrow: string;
  title: string;
  action?: ReactNode;
}) {
  return (
    <CompositePanelHeader
      eyebrow={eyebrow}
      title={title}
      action={action}
      className="report-panel-head"
      actionClassName="report-panel-actions"
    />
  );
}

export function CompositeReportEmptyState({ children }: { children: ReactNode }) {
  return <CompositePanelEmptyState className="report-empty-state">{children}</CompositePanelEmptyState>;
}
