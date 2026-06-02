import { CompositeInspector } from "./compositeLayerInspector";
import { CompositeLayerFocusToolbar } from "./compositeLayerFocusToolbar";
import { CompositeObjectContextMenu } from "./compositeObjectContextMenu";
import { CompositeObjectHud } from "./compositeObjectHud";
import { OverlayStage } from "./compositeOverlayStage";
import type { ActiveLayerConfig, StageMode } from "./compositeReportModel";
import type { CompositeReportStageState } from "./compositeReportStageController";
import { SplitStage } from "./compositeSplitStage";

import "./compositeStageWorkbench.css";

export function CompositeStageWorkbench({
  stage,
  layerConfigs,
  mode,
  onFocusedLayerChange
}: {
  stage: CompositeReportStageState;
  layerConfigs: ActiveLayerConfig[];
  mode: StageMode;
  onFocusedLayerChange: (layer: string | null) => void;
}) {
  return (
    <>
      <CompositeLayerFocusToolbar
        statuses={stage.statuses}
        focusedLayerKey={stage.activeFocusedLayerKey}
        onFocusedLayerChange={onFocusedLayerChange}
      />
      <CompositeObjectHud
        activeObject={stage.objectInteraction.activeObject}
        activeObjectIndex={stage.objectInteraction.activeObjectIndex}
        objectCount={stage.objectInteraction.objectCount}
        relatedObjectCount={stage.objectInteraction.relatedObjectKeys.size}
        locked={Boolean(stage.objectInteraction.lockedObjectKey)}
        onClear={stage.objectInteraction.clearObjectInteraction}
        onFocusedLayerChange={onFocusedLayerChange}
      />
      <div className={`composite-report-workbench mode-${mode}`}>
        {mode === "both" || mode === "overlay" ? (
          <CompositeOverlayWorkbenchPane
            stage={stage}
            layerConfigs={layerConfigs}
            onFocusedLayerChange={onFocusedLayerChange}
          />
        ) : null}
        {mode === "both" || mode === "split" ? (
          <SplitStage
            layers={stage.focusedLayers}
            statuses={stage.focusedStatuses}
            layerConfigs={layerConfigs}
            viewportSyncKey={stage.viewportSyncKey}
            focusedLayerKey={stage.activeFocusedLayerKey}
            onFocusedLayerChange={onFocusedLayerChange}
            activeObjectKey={stage.objectInteraction.activeObjectKey}
            relatedObjectKeys={stage.objectInteraction.relatedObjectKeys}
            onObjectHover={stage.objectInteraction.setHoveredObjectKey}
            onObjectLock={stage.objectInteraction.toggleObjectLock}
            onObjectInspect={stage.objectInteraction.inspectObject}
            onObjectWheel={stage.objectInteraction.onObjectWheel}
            onObjectContextMenu={stage.objectInteraction.openObjectContextMenu}
          />
        ) : null}
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
  onFocusedLayerChange
}: {
  stage: CompositeReportStageState;
  layerConfigs: ActiveLayerConfig[];
  onFocusedLayerChange: (layer: string | null) => void;
}) {
  return (
    <div className="composite-report-focus">
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
      />
      <CompositeInspector
        layers={stage.layers}
        statuses={stage.statuses}
        layerConfigs={layerConfigs}
        focusedLayerKey={stage.activeFocusedLayerKey}
        onFocusedLayerChange={onFocusedLayerChange}
        activeObjectKey={stage.objectInteraction.activeObjectKey}
        relatedObjectKeys={stage.objectInteraction.relatedObjectKeys}
        lockedObjectKey={stage.objectInteraction.lockedObjectKey}
        onObjectHover={stage.objectInteraction.setHoveredObjectKey}
        onObjectLock={stage.objectInteraction.toggleObjectLock}
        onObjectInspect={stage.objectInteraction.inspectObject}
        onObjectWheel={stage.objectInteraction.onObjectWheel}
      />
    </div>
  );
}
