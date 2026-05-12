import React, { useEffect, useMemo, useState } from "react";

import type { EvalInstance, RunSampleDetail } from "./api";
import { isTextInputTarget } from "./formatters";
import type { ObjectKind } from "./viewerMetrics";
import {
  arrowHeadPoints,
  boundsFromPoints,
  clampNumber,
  clampPan,
  computeFitSize,
  normalizeBbox,
  normalizePointList,
  normalizedWheelDelta,
  resolveInstanceColor
} from "./viewerGeometry";
import {
  DEFAULT_INTERACTION_SETTINGS,
  DEFAULT_OVERLAY_STYLE,
  useWorkspaceShortcuts
} from "./workspaceSettings";
import type {
  InteractionSettings,
  LabelColors,
  OverlayColors,
  OverlayStyle
} from "./workspaceSettings";

function recordViewerRenderMetric(name: string) {
  if (typeof window === "undefined" || !window.location.search.includes("perf=1")) {
    return;
  }
  const target = window as Window & { __evalBenchRenderMetrics?: Record<string, number> };
  target.__evalBenchRenderMetrics ??= {};
  target.__evalBenchRenderMetrics[name] = (target.__evalBenchRenderMetrics[name] ?? 0) + 1;
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
  interactionSettings = DEFAULT_INTERACTION_SETTINGS,
  activeObjectId = null,
  onHover,
  onLock
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
  interactionSettings?: InteractionSettings;
  activeObjectId?: string | null;
  onHover?: (objectId: string | null) => void;
  onLock?: (objectId: string | null) => void;
}) {
  recordViewerRenderMetric("canvasStage");
  const stageRef = React.useRef<HTMLDivElement | null>(null);
  const contentRef = React.useRef<HTMLDivElement | null>(null);
  const panBoundsRef = React.useRef({
    viewportWidth: 1,
    viewportHeight: 1,
    contentWidth: 1,
    contentHeight: 1
  });
  const dragRef = React.useRef<{
    pointerId: number;
    startX: number;
    startY: number;
    startPan: { x: number; y: number };
  } | null>(null);
  const viewportRef = React.useRef({ zoom: 1, pan: { x: 0, y: 0 } });
  const pendingPanRef = React.useRef<{ x: number; y: number } | null>(null);
  const frameRef = React.useRef<number | null>(null);
  const transformingTimeoutRef = React.useRef<number | null>(null);
  const zoomLabelRef = React.useRef<HTMLSpanElement | null>(null);
  const viewportDirtyRef = React.useRef(false);
  const [viewportDirty, setViewportDirty] = useState(false);
  const [stageSize, setStageSize] = useState({ width: 1, height: 1 });
  const [isPanning, setIsPanning] = useState(false);
  const fitSize = useMemo(
    () => computeFitSize(width, height, stageSize),
    [height, stageSize, width]
  );
  const overlayInteractive = Boolean(onHover || onLock);
  const { actionForEvent } = useWorkspaceShortcuts();

  useEffect(() => {
    viewportRef.current = { zoom: 1, pan: { x: 0, y: 0 } };
    pendingPanRef.current = null;
    if (frameRef.current !== null) {
      cancelAnimationFrame(frameRef.current);
      frameRef.current = null;
    }
    applyViewportToDom();
    updateViewportDirty(false);
    dragRef.current = null;
    setIsPanning(false);
  }, [imageUrl]);

  useEffect(() => {
    const node = stageRef.current;
    if (!node) {
      return undefined;
    }
    const stageNode = node;
    function updateStageSize() {
      const rect = stageNode.getBoundingClientRect();
      const nextSize = {
        width: Math.max(1, rect.width),
        height: Math.max(1, rect.height)
      };
      panBoundsRef.current = {
        ...panBoundsRef.current,
        viewportWidth: nextSize.width,
        viewportHeight: nextSize.height
      };
      setStageSize(nextSize);
    }
    updateStageSize();
    const observer = new ResizeObserver(updateStageSize);
    observer.observe(stageNode);
    return () => observer.disconnect();
  }, []);

  React.useLayoutEffect(() => {
    panBoundsRef.current = {
      ...panBoundsRef.current,
      contentWidth: fitSize.width,
      contentHeight: fitSize.height
    };
    const viewport = viewportRef.current;
    viewportRef.current = {
      zoom: viewport.zoom,
      pan: clampStagePan(viewport.pan, viewport.zoom)
    };
    applyViewportToDom();
  }, [fitSize.width, fitSize.height]);

  useEffect(() => {
    return () => {
      if (frameRef.current !== null) {
        cancelAnimationFrame(frameRef.current);
      }
      if (transformingTimeoutRef.current !== null) {
        window.clearTimeout(transformingTimeoutRef.current);
      }
    };
  }, []);

  function clampStagePan(
    nextPan: { x: number; y: number },
    nextZoom = viewportRef.current.zoom
  ) {
    return clampPan(nextPan, nextZoom, panBoundsRef.current);
  }

  function updateViewportDirty(nextDirty: boolean) {
    if (viewportDirtyRef.current === nextDirty) {
      return;
    }
    viewportDirtyRef.current = nextDirty;
    setViewportDirty(nextDirty);
  }

  function applyViewportToDom() {
    const content = contentRef.current;
    const zoomLabel = zoomLabelRef.current;
    const viewport = viewportRef.current;
    if (content) {
      content.style.transform = `translate(-50%, -50%) translate(${viewport.pan.x}px, ${viewport.pan.y}px) scale(${viewport.zoom})`;
    }
    if (zoomLabel) {
      zoomLabel.textContent = `${Math.round(viewport.zoom * 100)}%`;
    }
  }

  function markViewportTransforming() {
    const node = stageRef.current;
    if (!node) {
      return;
    }
    node.classList.add("transforming");
    if (transformingTimeoutRef.current !== null) {
      window.clearTimeout(transformingTimeoutRef.current);
    }
    transformingTimeoutRef.current = window.setTimeout(() => {
      node.classList.remove("transforming");
      transformingTimeoutRef.current = null;
    }, 140);
  }

  function scheduleViewportUpdate() {
    if (frameRef.current !== null) {
      return;
    }
    frameRef.current = requestAnimationFrame(() => {
      frameRef.current = null;
      const pendingPan = pendingPanRef.current;
      if (pendingPan) {
        const viewport = viewportRef.current;
        viewportRef.current = {
          zoom: viewport.zoom,
          pan: clampStagePan(pendingPan, viewport.zoom)
        };
        pendingPanRef.current = null;
      }
      applyViewportToDom();
    });
  }

  function flushViewportUpdate() {
    if (frameRef.current !== null) {
      cancelAnimationFrame(frameRef.current);
      frameRef.current = null;
    }
    const pendingPan = pendingPanRef.current;
    if (pendingPan) {
      const viewport = viewportRef.current;
      viewportRef.current = {
        zoom: viewport.zoom,
        pan: clampStagePan(pendingPan, viewport.zoom)
      };
      pendingPanRef.current = null;
    }
    applyViewportToDom();
  }

  function applyZoom(nextZoom: number, anchor?: { x: number; y: number }) {
    const currentViewport = viewportRef.current;
    const clampedZoom = clampNumber(
      nextZoom,
      interactionSettings.minZoom,
      interactionSettings.maxZoom
    );
    if (Math.abs(clampedZoom - currentViewport.zoom) < 0.001) {
      return;
    }
    let nextPan = currentViewport.pan;
    if (anchor) {
      const stage = stageRef.current;
      const center = stage
        ? { x: stage.clientWidth / 2, y: stage.clientHeight / 2 }
        : { x: 0, y: 0 };
      const scale = clampedZoom / currentViewport.zoom;
      const relativeAnchor = {
        x: anchor.x - center.x,
        y: anchor.y - center.y
      };
      nextPan = {
        x: relativeAnchor.x - (relativeAnchor.x - currentViewport.pan.x) * scale,
        y: relativeAnchor.y - (relativeAnchor.y - currentViewport.pan.y) * scale
      };
    }
    viewportRef.current = {
      zoom: clampedZoom,
      pan: clampStagePan(nextPan, clampedZoom)
    };
    updateViewportDirty(true);
    markViewportTransforming();
    scheduleViewportUpdate();
  }

  function resetViewport() {
    viewportRef.current = { zoom: 1, pan: { x: 0, y: 0 } };
    pendingPanRef.current = null;
    if (frameRef.current !== null) {
      cancelAnimationFrame(frameRef.current);
      frameRef.current = null;
    }
    if (transformingTimeoutRef.current !== null) {
      window.clearTimeout(transformingTimeoutRef.current);
      transformingTimeoutRef.current = null;
    }
    stageRef.current?.classList.remove("transforming");
    applyViewportToDom();
    updateViewportDirty(false);
    dragRef.current = null;
    setIsPanning(false);
  }

  function handleWheel(event: WheelEvent) {
    event.preventDefault();
    const node = stageRef.current;
    if (!node) {
      return;
    }
    const rect = node.getBoundingClientRect();
    const anchor = {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top
    };
    applyZoom(
      viewportRef.current.zoom *
        Math.exp(-normalizedWheelDelta(event) * interactionSettings.wheelZoomSensitivity),
      anchor
    );
  }

  useEffect(() => {
    const node = stageRef.current;
    if (!node) {
      return undefined;
    }
    node.addEventListener("wheel", handleWheel, { passive: false });
    return () => node.removeEventListener("wheel", handleWheel);
  }, [
    interactionSettings.maxZoom,
    interactionSettings.minZoom,
    interactionSettings.wheelZoomSensitivity
  ]);

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (isTextInputTarget(event.target)) {
        return;
      }
      if (actionForEvent(event) === "viewer.resetViewport") {
        event.preventDefault();
        resetViewport();
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [actionForEvent]);

  function handlePointerDown(event: React.PointerEvent<HTMLDivElement>) {
    if (event.button !== 0) {
      return;
    }
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      startPan: viewportRef.current.pan
    };
    setIsPanning(true);
  }

  function handlePointerMove(event: React.PointerEvent<HTMLDivElement>) {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) {
      return;
    }
    pendingPanRef.current = {
      x: drag.startPan.x + (event.clientX - drag.startX) * interactionSettings.panSensitivity,
      y: drag.startPan.y + (event.clientY - drag.startY) * interactionSettings.panSensitivity
    };
    updateViewportDirty(true);
    scheduleViewportUpdate();
  }

  function endPan(event: React.PointerEvent<HTMLDivElement>) {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) {
      return;
    }
    dragRef.current = null;
    flushViewportUpdate();
    setIsPanning(false);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

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
          transform: `translate(-50%, -50%) translate(${viewportRef.current.pan.x}px, ${viewportRef.current.pan.y}px) scale(${viewportRef.current.zoom})`
        }}
      >
        <img
          src={imageUrl}
          alt={imageAlt}
          draggable={false}
          loading="eager"
          decoding="async"
        />
        <svg
          className={overlayInteractive ? "overlay-svg interactive" : "overlay-svg"}
          viewBox={`0 0 ${width} ${height}`}
          preserveAspectRatio="none"
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
              overlayColors={overlayColors}
              overlayStyle={overlayStyle}
              labelColors={labelColors}
              onHover={onHover}
              onLock={onLock}
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
              overlayColors={overlayColors}
              overlayStyle={overlayStyle}
              labelColors={labelColors}
              onHover={onHover}
              onLock={onLock}
            />
          ) : null}
        </svg>
      </div>
      <div className="canvas-hud">
        <span ref={zoomLabelRef}>100%</span>
        {viewportDirty ? (
          <button type="button" onClick={resetViewport}>
            复位
          </button>
        ) : null}
      </div>
    </div>
  );
}

function InstanceLayer({
  instances,
  kind,
  diagnostics,
  visibleLabels,
  showBoxes = true,
  showLines = true,
  showKeypoints = true,
  activeObjectId = null,
  overlayColors,
  overlayStyle = DEFAULT_OVERLAY_STYLE,
  labelColors,
  onHover,
  onLock
}: {
  instances: EvalInstance[];
  kind: ObjectKind;
  diagnostics: RunSampleDetail["diagnostics"];
  visibleLabels?: Set<string>;
  showBoxes?: boolean;
  showLines?: boolean;
  showKeypoints?: boolean;
  activeObjectId?: string | null;
  overlayColors: OverlayColors;
  overlayStyle?: OverlayStyle;
  labelColors: LabelColors;
  onHover?: (objectId: string | null) => void;
  onLock?: (objectId: string | null) => void;
}) {
  recordViewerRenderMetric(`instanceLayer:${kind}`);
  const matched = new Set(
    (diagnostics?.matches ?? []).map((match) => (kind === "gt" ? match.gt_index : match.pred_index))
  );
  const errorItems =
    kind === "gt" ? diagnostics?.false_negatives ?? [] : diagnostics?.false_positives ?? [];
  const errors = new Set(errorItems.map((item) => item.index));
  return (
    <>
      {instances.map((instance, index) => {
        if (visibleLabels && !visibleLabels.has(instance.label)) {
          return null;
        }
        const objectId = `${kind}:${index}`;
        const bbox = normalizeBbox((instance as { bbox?: unknown }).bbox);
        const linePoints = normalizePointList(
          (instance as { linestrip?: unknown; line_strip?: unknown; points?: unknown }).linestrip ??
            (instance as { line_strip?: unknown }).line_strip
        );
        const keypoints = normalizePointList((instance as { keypoints?: unknown }).keypoints);
        const anchorBox = bbox ?? boundsFromPoints(linePoints ?? keypoints);
        if (!bbox && (!linePoints || linePoints.length === 0) && (!keypoints || keypoints.length === 0)) {
          return null;
        }
        const status = errors.has(index)
          ? kind === "gt"
            ? "fn"
            : "fp"
          : matched.has(index)
            ? "match"
            : "neutral";
        const color = resolveInstanceColor(instance.label, status, kind, overlayColors, labelColors);
        const directionHead =
          linePoints && linePoints.length >= 2
            ? arrowHeadPoints(
                linePoints,
                overlayStyle.lineStrokeWidth,
                overlayStyle.directionHeadScale
              )
            : null;
        const lineRadius = Math.max(overlayStyle.pointRadius, overlayStyle.lineStrokeWidth * 0.75);
        const labelX = anchorBox ? anchorBox[0] + 3 : 0;
        const labelY = anchorBox ? Math.max(12, anchorBox[1] - 4) : 0;
        const labelWidth = Math.max(
          28,
          instance.label.length * overlayStyle.labelFontSize * 0.62 + 10
        );
        const labelHeight = overlayStyle.labelFontSize + 6;
        return (
          <g
            key={objectId}
            className={
              objectId === activeObjectId
                ? `overlay-instance ${kind} ${status} active`
                : `overlay-instance ${kind} ${status}`
            }
            style={{ "--instance-color": color } as React.CSSProperties}
            onPointerEnter={() => onHover?.(objectId)}
            onPointerLeave={() => onHover?.(null)}
            onClick={(event) => {
              event.stopPropagation();
              onLock?.(objectId);
            }}
          >
            {showBoxes && bbox ? (
              <rect x={bbox[0]} y={bbox[1]} width={bbox[2] - bbox[0]} height={bbox[3] - bbox[1]} />
            ) : null}
            {showBoxes && anchorBox ? (
              <g className="overlay-label">
                <rect
                  className="label-backplate"
                  x={labelX - 3}
                  y={labelY - overlayStyle.labelFontSize - 3}
                  width={labelWidth}
                  height={labelHeight}
                  rx={2}
                />
                <text x={labelX} y={labelY}>
                  {instance.label}
                </text>
              </g>
            ) : null}
            {showLines && linePoints && linePoints.length >= 2 ? (
              <>
                <polyline points={linePoints.map((point) => `${point[0]},${point[1]}`).join(" ")} />
                <circle
                  className="line-endpoint start"
                  cx={linePoints[0][0]}
                  cy={linePoints[0][1]}
                  r={lineRadius}
                />
                <circle
                  className="line-endpoint end"
                  cx={linePoints[linePoints.length - 1][0]}
                  cy={linePoints[linePoints.length - 1][1]}
                  r={lineRadius}
                />
                {directionHead ? (
                  <polygon
                    className="direction-head"
                    points={directionHead.map((point) => `${point[0]},${point[1]}`).join(" ")}
                  />
                ) : null}
              </>
            ) : null}
            {showKeypoints && keypoints && keypoints.length > 0 ? (
              keypoints.map((point, pointIndex) => (
                <circle
                  key={`${objectId}-point-${pointIndex}`}
                  cx={point[0]}
                  cy={point[1]}
                  r={overlayStyle.pointRadius}
                />
              ))
            ) : null}
          </g>
        );
      })}
    </>
  );
}

const MemoizedInstanceLayer = React.memo(InstanceLayer);
