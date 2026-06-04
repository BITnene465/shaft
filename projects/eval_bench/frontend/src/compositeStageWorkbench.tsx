import { useEffect, useState } from "react";

import { CompositeInspector } from "./compositeLayerInspector";
import { CompositeObjectContextMenu } from "./compositeObjectContextMenu";
import { OverlayStage } from "./compositeOverlayStage";
import type { CompositeSampleView } from "./api";
import type { ReactNode } from "react";
import type { ActiveLayerConfig } from "./compositeReportModel";
import type { CompositeReportStageState } from "./compositeReportStageController";
import { basename, formatMetric } from "./formatters";
import { ResizableSplit } from "./workspaceLayout";
import { VisualStatusBar } from "./visualStatusBar";
import type { VisualStatusItem } from "./visualStatusBar";

import "./compositeStageWorkbench.css";

export function CompositeStageWorkbench({
  stage,
  layerConfigs,
  composite,
  activeSlotCount,
  readyLayerCount,
  missingLayerCount,
  refreshing,
  onFocusedLayerChange,
  navigator
}: {
  stage: CompositeReportStageState;
  layerConfigs: ActiveLayerConfig[];
  composite: CompositeSampleView;
  activeSlotCount: number;
  readyLayerCount: number;
  missingLayerCount: number;
  refreshing: boolean;
  onFocusedLayerChange: (layer: string | null) => void;
  navigator: ReactNode;
}) {
  const activeObject = stage.objectInteraction.activeObject;
  const aggregate = aggregateCompositeDiagnostics(stage.layers);
  const statusItems: VisualStatusItem[] = [
    {
      label: "Sample",
      value: `${composite.image_index + 1}/${composite.image_count}`,
      title: composite.image_key
    },
    {
      label: "Layers",
      value: `${readyLayerCount}/${activeSlotCount}`,
      tone: missingLayerCount > 0 ? "warn" : "good"
    },
    {
      label: "Match",
      value: aggregate.matched.toLocaleString(),
      tone: "good"
    },
    {
      label: "FN/FP",
      value: `${aggregate.fn.toLocaleString()}/${aggregate.fp.toLocaleString()}`,
      tone: aggregate.fn + aggregate.fp > 0 ? "warn" : "good"
    },
    {
      label: "mIoU",
      value: formatMetric(aggregate.meanIou),
      tone: aggregate.meanIou === null ? "default" : aggregate.meanIou >= 0.7 ? "good" : "warn"
    },
    {
      label: activeObject ? "Object" : "Objects",
      value: activeObject
        ? `${activeObject.layer}:${activeObject.kind.toUpperCase()}#${activeObject.index + 1}`
        : stage.objectInteraction.objectCount.toLocaleString(),
      title: activeObject?.label
    }
  ];
  return (
    <>
      <div className="composite-report-workbench">
        <CompositeOverlayWorkbenchPane
          stage={stage}
          layerConfigs={layerConfigs}
          statusBar={
            <VisualStatusBar
              title={basename(composite.image)}
              subtitle={composite.image_key}
              items={statusItems}
              refreshing={refreshing}
              className="composite-visual-status"
            />
          }
          navigator={navigator}
          onFocusedLayerChange={onFocusedLayerChange}
        />
      </div>
      <CompositeObjectContextMenu
        request={stage.objectInteraction.contextMenu}
        object={stage.objectInteraction.contextMenuObject}
        locked={Boolean(
          stage.objectInteraction.contextMenuObject &&
            stage.objectInteraction.lockedObjectKey === stage.objectInteraction.contextMenuObject.key
        )}
        onLock={stage.objectInteraction.toggleObjectLock}
        onInspect={stage.objectInteraction.inspectObject}
        onFocusLayer={onFocusedLayerChange}
        onClear={stage.objectInteraction.clearObjectInteraction}
        onClose={stage.objectInteraction.closeContextMenu}
      />
    </>
  );
}

function CompositeOverlayWorkbenchPane({
  stage,
  layerConfigs,
  statusBar,
  navigator,
  onFocusedLayerChange
}: {
  stage: CompositeReportStageState;
  layerConfigs: ActiveLayerConfig[];
  statusBar: ReactNode;
  navigator: ReactNode;
  onFocusedLayerChange: (layer: string | null) => void;
}) {
  const compact = useCompositeWorkbenchCompactMode();
  const overlayStage = (
    <OverlayStage
      layers={stage.focusedLayers}
      statuses={stage.focusedStatuses}
      layerConfigs={layerConfigs}
      viewportSyncKey={stage.viewportSyncKey}
      activeObjectKey={stage.objectInteraction.activeObjectKey}
      relatedObjectKeys={stage.objectInteraction.relatedObjectKeys}
      onObjectHover={stage.objectInteraction.setHoveredObjectKey}
      onObjectLock={stage.objectInteraction.toggleObjectLock}
      onObjectInspect={stage.objectInteraction.inspectObject}
      onObjectWheel={stage.objectInteraction.onObjectWheel}
      onObjectContextMenu={stage.objectInteraction.openObjectContextMenu}
      statusBar={statusBar}
      navigator={navigator}
    />
  );
  const inspector = (
    <CompositeInspector
      layers={stage.layers}
      statuses={stage.statuses}
      layerConfigs={layerConfigs}
      focusedLayerKey={stage.activeFocusedLayerKey}
      activeObjectKey={stage.objectInteraction.activeObjectKey}
      relatedObjectKeys={stage.objectInteraction.relatedObjectKeys}
      lockedObjectKey={stage.objectInteraction.lockedObjectKey}
      onObjectHover={stage.objectInteraction.setHoveredObjectKey}
      onObjectLock={stage.objectInteraction.toggleObjectLock}
      onObjectInspect={stage.objectInteraction.inspectObject}
      onObjectWheel={stage.objectInteraction.onObjectWheel}
    />
  );
  if (compact) {
    return (
      <div className="composite-report-focus compact">
        {overlayStage}
        {inspector}
      </div>
    );
  }
  return (
    <ResizableSplit
      className="composite-report-focus"
      storageKey="eval_bench_composite_inspector_width"
      fixedPane="second"
      defaultSize={260}
      minSize={220}
      maxSize={520}
      first={overlayStage}
      second={inspector}
    />
  );
}

function useCompositeWorkbenchCompactMode() {
  const [compact, setCompact] = useState(() =>
    typeof window === "undefined" ? false : window.matchMedia("(max-width: 980px)").matches
  );
  useEffect(() => {
    const query = window.matchMedia("(max-width: 980px)");
    function updateCompactMode() {
      setCompact(query.matches);
    }
    updateCompactMode();
    query.addEventListener("change", updateCompactMode);
    return () => query.removeEventListener("change", updateCompactMode);
  }, []);
  return compact;
}

function aggregateCompositeDiagnostics(layers: CompositeReportStageState["layers"]) {
  const totals = layers.reduce(
    (current, layer) => {
      const diagnostics = layer.diagnostics;
      return {
        matched: current.matched + (diagnostics?.matched_count ?? 0),
        fn: current.fn + (diagnostics?.false_negative_count ?? 0),
        fp: current.fp + (diagnostics?.false_positive_count ?? 0),
        iouSum: current.iouSum + (diagnostics?.mean_iou ?? 0),
        iouCount: current.iouCount + (diagnostics ? 1 : 0)
      };
    },
    { matched: 0, fn: 0, fp: 0, iouSum: 0, iouCount: 0 }
  );
  return {
    matched: totals.matched,
    fn: totals.fn,
    fp: totals.fp,
    meanIou: totals.iouCount > 0 ? totals.iouSum / totals.iouCount : null
  };
}
