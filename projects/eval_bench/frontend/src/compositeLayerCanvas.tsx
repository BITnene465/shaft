import type { EvalInstance, RunSampleSummary, SampleDiagnostics } from "./api";
import type { WheelEvent } from "react";
import { CompositeCanvasGestureHud } from "./compositeCanvasGestureHud";
import { CompositeCanvasPointerReticle } from "./compositeCanvasPointerReticle";
import { useCompositeCanvasPointerTracker } from "./compositeCanvasPointerTracker";
import { useCompositeLayerCanvasController } from "./compositeLayerCanvasController";
import { CanvasStage } from "./viewerCanvas";
import type { CanvasObjectContextMenuRequest } from "./viewerInstanceLayer";

import "./compositeLayerCanvas.css";

export function CompositeLayerCanvas({
  className = "",
  sample,
  gtInstances,
  predInstances,
  diagnostics,
  labels,
  viewportSyncKey,
  activeObjectId,
  relatedObjectIds,
  onHover,
  onLock,
  onInspect,
  onObjectWheel,
  onObjectContextMenu
}: {
  className?: string;
  sample: RunSampleSummary;
  gtInstances: EvalInstance[];
  predInstances: EvalInstance[];
  diagnostics: SampleDiagnostics | null;
  labels: string[];
  viewportSyncKey?: string | null;
  activeObjectId?: string | null;
  relatedObjectIds?: Set<string>;
  onHover?: (objectId: string | null) => void;
  onLock?: (objectId: string | null) => void;
  onInspect?: (objectId: string | null) => void;
  onObjectWheel?: (event: WheelEvent<HTMLElement>) => void;
  onObjectContextMenu?: (request: CanvasObjectContextMenuRequest) => void;
}) {
  const canvas = useCompositeLayerCanvasController({
    labels,
    activeObjectId,
    onHover,
    onLock
  });
  const pointer = useCompositeCanvasPointerTracker();

  return (
    <div
      className={["composite-workbench-canvas", className].filter(Boolean).join(" ")}
      data-object-wheel-cruise={onObjectWheel ? "modified" : undefined}
      data-overlay-surface-pan="modified"
      onWheelCapture={onObjectWheel}
      style={canvas.overlayVars}
      {...pointer.pointerHandlers}
    >
      <CanvasStage
        width={sample.image_width ?? 1000}
        height={sample.image_height ?? 1000}
        imageUrl={sample.image_preview_url ?? sample.image_url}
        imageAlt={sample.image}
        imageTileUrlTemplate={sample.image_tile_url_template}
        imageTileSize={sample.image_tile_size}
        gtInstances={gtInstances}
        predInstances={predInstances}
        diagnostics={diagnostics}
        visibleLabels={new Set(labels)}
        showGt
        showPred
        showBoxes
        showLines
        showKeypoints
        activeObjectId={canvas.resolvedActiveObjectId}
        relatedObjectIds={relatedObjectIds}
        overlayColors={canvas.overlayColors}
        overlayStyle={canvas.overlayStyle}
        labelColors={canvas.labelColors}
        interactionSettings={canvas.interactionSettings}
        allowOverlaySurfacePan
        viewportSyncKey={viewportSyncKey}
        onHover={canvas.handleHover}
        onLock={canvas.handleLock}
        onInspect={onInspect}
        onObjectContextMenu={onObjectContextMenu}
      />
      <CompositeCanvasGestureHud
        activeObjectId={canvas.resolvedActiveObjectId}
        relatedObjectCount={relatedObjectIds?.size ?? 0}
        wheelCruise={Boolean(onObjectWheel)}
        surfacePan
        contextMenu={Boolean(onObjectContextMenu)}
      />
      <CompositeCanvasPointerReticle coordinateRef={pointer.coordinateRef} />
    </div>
  );
}
