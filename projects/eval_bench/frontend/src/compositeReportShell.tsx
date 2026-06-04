import { useEffect } from "react";

import {
  ReportComposerDock,
  ReportComposerDrawer
} from "./compositeReportComposer";
import type { CompositeReportController } from "./compositeReportController";
import { CompositeStage } from "./compositeReportStage";
import { ActionButton } from "./ui";

export function CompositeReportShell({ report }: { report: CompositeReportController }) {
  useCompositeSidebarDismiss({
    open: report.sidebarOpen,
    onClose: () => report.setSidebarOpen(false)
  });

  const sidebarState = report.sidebarOpen ? "open" : "collapsed";

  return (
    <div
      className={
        report.sidebarOpen
          ? "composite-report-shell sidebar-open"
          : "composite-report-shell sidebar-collapsed"
      }
      data-sidebar={sidebarState}
    >
      <ReportComposerDock
        open={report.sidebarOpen}
        activeSlots={report.activeSlots.length}
        readyLayerCount={report.readyLayerCount}
        missingLayerCount={report.missingLayerCount}
        onOpenChange={report.setSidebarOpen}
      />
      {report.sidebarOpen ? (
        <>
          <ActionButton
            variant="secondary"
            className="composite-sidebar-backdrop"
            aria-label="关闭报告编排器"
            onClick={() => report.setSidebarOpen(false)}
          />
          <ReportComposerDrawer
            runs={report.filteredRuns}
            total={report.reportRuns.length}
            groups={report.groups}
            runById={report.runById}
            selectedRunIds={report.selectedRunIds}
            query={report.query}
            layerFilter={report.layerFilter}
            onClose={() => report.setSidebarOpen(false)}
            onQueryChange={report.setQuery}
            onLayerFilterChange={report.setLayerFilter}
            onAddRun={report.addRun}
            onReset={report.resetComposition}
            onApplyLayoutArrowPreset={report.applyLayoutArrowPreset}
            onUpdateSlot={report.updateSlot}
            onRemoveSlot={report.removeSlot}
          />
        </>
      ) : null}
      <main className="composite-stage-region">
        <CompositeStage
          composite={report.compositeQuery.data}
          layerConfigs={report.activeLayerConfigs}
          loading={report.compositeQuery.isLoading && report.compositeEnabled}
          refreshing={report.compositeQuery.isFetching && !report.compositeQuery.isLoading}
          error={report.compositeQuery.error}
          enabled={report.compositeEnabled}
          onImageIndexChange={report.setSampleIndex}
          activeSlotCount={report.activeSlots.length}
          readyLayerCount={report.readyLayerCount}
          missingLayerCount={report.missingLayerCount}
          focusedLayerKey={report.focusedLayerKey}
          onFocusedLayerChange={report.setFocusedLayerKey}
        />
      </main>
    </div>
  );
}

function useCompositeSidebarDismiss({
  open,
  onClose
}: {
  open: boolean;
  onClose: () => void;
}) {
  useEffect(() => {
    if (!open) {
      return undefined;
    }

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        onClose();
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose, open]);
}
