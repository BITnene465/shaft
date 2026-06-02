import { PanelLeftClose, PanelLeftOpen } from "lucide-react";

import type { CompositeSampleView } from "./api";
import { layerAvailabilityColor } from "./compositeLayerPalette";
import { CompositeMicroMeter } from "./compositeMicroMeter";
import { ReportComposerDockPreview } from "./compositeReportComposerDockPreview";
import { ActionButton, IconActionButton } from "./ui";

import "./compositeComposerDock.css";

export function ReportComposerDock({
  open,
  activeSlots,
  readyLayerCount,
  missingLayerCount,
  statuses,
  onOpenChange
}: {
  open: boolean;
  activeSlots: number;
  readyLayerCount: number;
  missingLayerCount: number;
  statuses?: CompositeSampleView["layer_statuses"];
  onOpenChange: (value: boolean) => void;
}) {
  const readyTotal = readyLayerCount + missingLayerCount;
  const activeSlotProgress = activeSlots > 0 ? 1 : 0;
  const readyProgress = readyTotal > 0 ? readyLayerCount / readyTotal : 0;
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
        title={open ? "折叠报告编排器" : "展开报告编排器"}
        onClick={() => onOpenChange(!open)}
      >
        <span>{open ? "OPEN" : "PLAN"}</span>
      </ActionButton>
      <CompositeMicroMeter
        className="composer-dock-meter"
        label="layers"
        value={activeSlots.toLocaleString()}
        meta="slots"
        progress={activeSlotProgress}
        idle={activeSlots <= 0}
        ariaLabel={`${activeSlots.toLocaleString()} layers`}
      />
      <CompositeMicroMeter
        className="composer-dock-meter"
        label="ready"
        value={readyLayerCount.toLocaleString()}
        meta={`${missingLayerCount.toLocaleString()} miss`}
        progress={readyProgress}
        idle={readyTotal <= 0}
        ariaLabel={`${readyLayerCount.toLocaleString()} ready / ${missingLayerCount.toLocaleString()} missing`}
      />
      <div className="composer-dock-dots" aria-label="当前图层状态">
        {(statuses ?? []).slice(0, 8).map((status, index) => (
          <i
            className={status.available ? "ready" : "missing"}
            key={`${status.layer}_${status.run_id}`}
            style={{ background: layerAvailabilityColor(index, status.available) }}
            title={`${status.layer}: ${status.status}`}
          />
        ))}
      </div>
      <ReportComposerDockPreview
        activeSlots={activeSlots}
        readyLayerCount={readyLayerCount}
        missingLayerCount={missingLayerCount}
        statuses={statuses}
      />
    </aside>
  );
}
