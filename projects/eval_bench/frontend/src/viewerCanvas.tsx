import { useMemo } from "react";
import type { EvalInstance, RunSampleDetail } from "./api";
import { DEFAULT_INTERACTION_SETTINGS } from "./workspaceSettings";
import { MemoizedInstanceLayer } from "./viewerInstanceLayer";
import type { CanvasObjectContextMenuRequest } from "./viewerInstanceLayer";
import { recordViewerRenderMetric } from "./viewerRenderMetrics";
import { PyramidTileLayer } from "./viewerTileLayer";
import { useViewerViewportController } from "./viewerViewportController";
import type {
  InteractionSettings,
  LabelColors,
  OverlayColors,
  OverlayStyle
} from "./workspaceSettings";
import { ActionButton } from "./ui";

import "./viewerOverlayCanvas.css";
import "./viewerCanvas.css";

const MIN_SCREEN_LABEL_SIZE = 6;
const MAX_SCREEN_LABEL_SIZE = 12;

function clampScreenLabelSize(value: number) {
  return Math.max(MIN_SCREEN_LABEL_SIZE, Math.min(MAX_SCREEN_LABEL_SIZE, value));
}

export function CanvasStage({
  width,
  height,
  imageUrl,
  imageAlt,
  gtInstances,
  predInstances,
  diagnostics,
  visibleLabels,
  showGt,
  showPred,
  showBoxes,
  showLines,
  showKeypoints,
  overlayColors,
  overlayStyle,
  labelColors,
  imageTileUrlTemplate,
  imageTileSize = 512,
  interactionSettings = DEFAULT_INTERACTION_SETTINGS,
  allowOverlaySurfacePan = false,
  viewportSyncKey = null,
  activeObjectId = null,
  relatedObjectIds,
  onHover,
  onLock,
  onInspect,
  onObjectContextMenu
}: {
  width: number;
  height: number;
  imageUrl: string;
  imageAlt: string;
  gtInstances: EvalInstance[];
  predInstances: EvalInstance[];
  diagnostics: RunSampleDetail["diagnostics"];
  visibleLabels?: Set<string>;
  showGt: boolean;
  showPred: boolean;
  showBoxes: boolean;
  showLines: boolean;
  showKeypoints: boolean;
  overlayColors: OverlayColors;
  overlayStyle: OverlayStyle;
  labelColors: LabelColors;
  imageTileUrlTemplate?: string | null;
  imageTileSize?: number | null;
  interactionSettings?: InteractionSettings;
  allowOverlaySurfacePan?: boolean;
  viewportSyncKey?: string | null;
  activeObjectId?: string | null;
  relatedObjectIds?: Set<string>;
  onHover?: (objectId: string | null) => void;
  onLock?: (objectId: string | null) => void;
  onInspect?: (objectId: string | null) => void;
  onObjectContextMenu?: (request: CanvasObjectContextMenuRequest) => void;
}) {
  recordViewerRenderMetric("canvasStage");
  const overlayInteractive = Boolean(onHover || onLock || onInspect || onObjectContextMenu);
  const overlayClassName = [
    "overlay-svg",
    overlayInteractive ? "interactive" : "",
    activeObjectId ? "has-active" : ""
  ]
    .filter(Boolean)
    .join(" ");
  const {
    contentRef,
    contentTransform,
    endPan,
    fitSize,
    handlePointerDown,
    handlePointerMove,
    isPanning,
    resetViewport,
    stageRef,
    tileLevel,
    viewportDirty,
    zoomLabelRef
  } = useViewerViewportController({
    height,
    imageTileSize,
    imageTileUrlTemplate,
    imageUrl,
    interactionSettings,
    allowOverlaySurfacePan,
    viewportSyncKey,
    width
  });
  const adaptiveOverlayStyle = useMemo(() => {
    const fitWidth = Math.max(1, fitSize.width);
    const imageWidth = Math.max(1, width);
    const screenLabelSize = clampScreenLabelSize(overlayStyle.labelFontSize);
    return {
      ...overlayStyle,
      labelFontSize: (screenLabelSize * imageWidth) / fitWidth
    };
  }, [fitSize.width, overlayStyle, width]);

  return (
    <div
      ref={stageRef}
      className={isPanning ? "image-stage panning" : "image-stage pannable"}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={endPan}
      onPointerCancel={endPan}
    >
      <div
        ref={contentRef}
        className="image-zoom-layer"
        style={{
          width: `${fitSize.width}px`,
          height: `${fitSize.height}px`,
          transform: contentTransform
        }}
      >
        <img
          className="base-image"
          src={imageUrl}
          alt={imageAlt}
          draggable={false}
          loading="eager"
          decoding="async"
        />
        {imageTileUrlTemplate && imageTileSize && tileLevel !== null ? (
          <PyramidTileLayer
            width={width}
            height={height}
            tileSize={imageTileSize}
            level={tileLevel}
            urlTemplate={imageTileUrlTemplate}
          />
        ) : null}
        <svg
          className={overlayClassName}
          viewBox={`0 0 ${width} ${height}`}
          preserveAspectRatio="none"
          onClick={(event) => {
            if (event.target !== event.currentTarget || !activeObjectId) {
              return;
            }
            event.stopPropagation();
            onLock?.(activeObjectId);
          }}
          onDoubleClick={(event) => {
            if (event.target !== event.currentTarget) {
              return;
            }
            event.stopPropagation();
            resetViewport();
          }}
          onContextMenu={(event) => {
            if (event.target !== event.currentTarget || !activeObjectId) {
              return;
            }
            event.preventDefault();
            event.stopPropagation();
            onObjectContextMenu?.({
              objectId: activeObjectId,
              clientX: event.clientX,
              clientY: event.clientY
            });
          }}
        >
          {showGt ? (
            <MemoizedInstanceLayer
              instances={gtInstances}
              kind="gt"
              diagnostics={diagnostics}
              visibleLabels={visibleLabels}
              showBoxes={showBoxes}
              showLines={showLines}
              showKeypoints={showKeypoints}
              activeObjectId={activeObjectId}
              relatedObjectIds={relatedObjectIds}
              overlayColors={overlayColors}
              overlayStyle={adaptiveOverlayStyle}
              labelColors={labelColors}
              onHover={onHover}
              onLock={onLock}
              onInspect={onInspect}
              onObjectContextMenu={onObjectContextMenu}
            />
          ) : null}
          {showPred ? (
            <MemoizedInstanceLayer
              instances={predInstances}
              kind="pred"
              diagnostics={diagnostics}
              visibleLabels={visibleLabels}
              showBoxes={showBoxes}
              showLines={showLines}
              showKeypoints={showKeypoints}
              activeObjectId={activeObjectId}
              relatedObjectIds={relatedObjectIds}
              overlayColors={overlayColors}
              overlayStyle={adaptiveOverlayStyle}
              labelColors={labelColors}
              onHover={onHover}
              onLock={onLock}
              onInspect={onInspect}
              onObjectContextMenu={onObjectContextMenu}
            />
          ) : null}
        </svg>
      </div>
      <div className="canvas-hud">
        <span ref={zoomLabelRef}>100%</span>
        {viewportDirty ? (
          <ActionButton variant="mini" className="canvas-reset-button" onClick={resetViewport}>
            复位
          </ActionButton>
        ) : null}
      </div>
    </div>
  );
}
