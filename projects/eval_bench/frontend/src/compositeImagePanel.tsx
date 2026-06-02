import type { ReactNode } from "react";

import { CompositePanelHeader } from "./compositePanelPrimitives";
import "./compositeImagePanel.css";

export function CompositeImagePanelHeader({
  title,
  meta,
  action,
  className = ""
}: {
  title: ReactNode;
  meta?: ReactNode;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <CompositePanelHeader
      title={title}
      meta={meta}
      action={action}
      className={["image-panel-head", className].filter(Boolean).join(" ")}
      actionClassName="image-panel-head-action"
      density="compact"
      framed
    />
  );
}
