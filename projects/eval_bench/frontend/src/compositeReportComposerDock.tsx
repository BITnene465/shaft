import { PanelLeftClose, PanelLeftOpen } from "lucide-react";

import { ActionButton, IconActionButton } from "./ui";

import "./compositeComposerDock.css";

export function ReportComposerDock({
  open,
  activeSlots,
  readyLayerCount,
  missingLayerCount,
  onOpenChange
}: {
  open: boolean;
  activeSlots: number;
  readyLayerCount: number;
  missingLayerCount: number;
  onOpenChange: (value: boolean) => void;
}) {
  const dockLabel = open ? "OPEN" : "PLAN";
  const dockTitle = `${activeSlots.toLocaleString()} slots, ${readyLayerCount.toLocaleString()} ready, ${missingLayerCount.toLocaleString()} missing`;
  return (
    <aside
      className={open ? "composite-composer-dock open" : "composite-composer-dock collapsed"}
      data-state={open ? "open" : "collapsed"}
      aria-label="组合报告编排器快捷栏"
      aria-expanded={open}
      onDoubleClick={() => onOpenChange(!open)}
    >
      <IconActionButton
        title={open ? "折叠报告编排器" : "展开报告编排器"}
        dense={false}
        icon={open ? <PanelLeftClose size={16} /> : <PanelLeftOpen size={16} />}
        onClick={() => onOpenChange(!open)}
      />
      <ActionButton
        variant="mini"
        compact
        className="composer-dock-grip"
        aria-label={open ? "折叠报告编排器" : "展开报告编排器"}
        title={dockTitle}
        onClick={() => onOpenChange(!open)}
      >
        <span>{dockLabel}</span>
        <strong>{activeSlots.toLocaleString()}</strong>
      </ActionButton>
    </aside>
  );
}
