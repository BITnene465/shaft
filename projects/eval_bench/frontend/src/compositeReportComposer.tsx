import { PanelLeftClose } from "lucide-react";

import type { RunSummary } from "./api";
import type { LayerFilter, ReportGroup } from "./compositeReportComposerModel";
import { ReportLayerPlan } from "./compositeReportLayerPlan";
import { ReportRunPool } from "./compositeReportRunPool";
import type { LayerSlot } from "./compositeReportModel";
import { IconActionButton } from "./ui";

import "./compositeComposerDrawer.css";

export { ReportComposerDock } from "./compositeReportComposerDock";

export function ReportComposerDrawer({
  runs,
  total,
  groups,
  runById,
  selectedRunIds,
  query,
  layerFilter,
  onClose,
  onQueryChange,
  onLayerFilterChange,
  onAddRun,
  onReset,
  onApplyLayoutArrowPreset,
  onUpdateSlot,
  onRemoveSlot
}: {
  runs: RunSummary[];
  total: number;
  groups: ReportGroup[];
  runById: Map<string, RunSummary>;
  selectedRunIds: Set<string>;
  query: string;
  layerFilter: LayerFilter;
  onClose: () => void;
  onQueryChange: (value: string) => void;
  onLayerFilterChange: (value: LayerFilter) => void;
  onAddRun: (run: RunSummary) => void;
  onReset: () => void;
  onApplyLayoutArrowPreset: () => void;
  onUpdateSlot: (id: string, patch: Partial<LayerSlot>) => void;
  onRemoveSlot: (id: string) => void;
}) {
  return (
    <aside className="composite-sidebar-drawer" aria-label="组合报告编排器">
      <div className="composite-sidebar-head">
        <div>
          <span>Composer</span>
          <strong>报告编排器</strong>
        </div>
        <IconActionButton
          title="折叠报告编排器"
          icon={<PanelLeftClose size={15} />}
          onClick={onClose}
        />
      </div>
      <div className="composite-sidebar-grid">
        <ReportRunPool
          runs={runs}
          total={total}
          selectedRunIds={selectedRunIds}
          query={query}
          layerFilter={layerFilter}
          onQueryChange={onQueryChange}
          onLayerFilterChange={onLayerFilterChange}
          onAddRun={onAddRun}
        />
        <ReportLayerPlan
          groups={groups}
          runById={runById}
          onReset={onReset}
          onApplyLayoutArrowPreset={onApplyLayoutArrowPreset}
          onUpdateSlot={onUpdateSlot}
          onRemoveSlot={onRemoveSlot}
        />
      </div>
    </aside>
  );
}
