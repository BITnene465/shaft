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
import { ActionButton } from "./ui";

const TILE_LOAD_IDLE_DELAY_MS = 700;
const TILE_ZOOM_THRESHOLD = 2.1;
const MAX_RENDERED_TILES = 24;

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
  imageTileUrlTemplate,
  imageTileSize = 512,
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
  imageTileUrlTemplate?: string | null;
  imageTileSize?: number | null;
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
  const tileLoadTimeoutRef = React.useRef<number | null>(null);
  const zoomLabelRef = React.useRef<HTMLSpanElement | null>(null);
  const viewportDirtyRef = React.useRef(false);
  const [viewportDirty, setViewportDirty] = useState(false);
  const [tileLevel, setTileLevel] = useState<number | null>(null);
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
    setTileLevel(null);
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
      if (tileLoadTimeoutRef.current !== null) {
        window.clearTimeout(tileLoadTimeoutRef.current);
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

  function scheduleTileLevelUpdate(nextZoom: number) {
    if (!imageTileUrlTemplate || !imageTileSize) {
      setTileLevel(null);
      return;
    }
    if (tileLoadTimeoutRef.current !== null) {
      window.clearTimeout(tileLoadTimeoutRef.current);
    }
    const nextTileLevel = tileLevelForZoom({
      width,
      height,
      zoom: nextZoom,
      tileSize: imageTileSize
    });
    tileLoadTimeoutRef.current = window.setTimeout(() => {
      setTileLevel(nextTileLevel);
      tileLoadTimeoutRef.current = null;
    }, TILE_LOAD_IDLE_DELAY_MS);
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
    scheduleTileLevelUpdate(clampedZoom);
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
    if (tileLoadTimeoutRef.current !== null) {
      window.clearTimeout(tileLoadTimeoutRef.current);
      tileLoadTimeoutRef.current = null;
    }
    stageRef.current?.classList.remove("transforming");
    applyViewportToDom();
    updateViewportDirty(false);
    setTileLevel(null);
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
          <ActionButton variant="mini" className="canvas-reset-button" onClick={resetViewport}>
            复位
          </ActionButton>
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

function PyramidTileLayer({
  width,
  height,
  tileSize,
  level,
  urlTemplate
}: {
  width: number;
  height: number;
  tileSize: number;
  level: number;
  urlTemplate: string;
}) {
  const scale = 2 ** level;
  const levelWidth = Math.ceil(width / scale);
  const levelHeight = Math.ceil(height / scale);
  const columns = Math.ceil(levelWidth / tileSize);
  const rows = Math.ceil(levelHeight / tileSize);
  if (columns * rows > MAX_RENDERED_TILES) {
    return null;
  }
  const tiles = [];
  for (let y = 0; y < rows; y += 1) {
    for (let x = 0; x < columns; x += 1) {
      const originalLeft = x * tileSize * scale;
      const originalTop = y * tileSize * scale;
      const originalRight = Math.min(width, (x + 1) * tileSize * scale);
      const originalBottom = Math.min(height, (y + 1) * tileSize * scale);
      tiles.push({
        key: `${level}-${x}-${y}`,
        src: urlTemplate
          .replace("{level}", String(level))
          .replace("{x}", String(x))
          .replace("{y}", String(y)),
        left: `${(originalLeft / width) * 100}%`,
        top: `${(originalTop / height) * 100}%`,
        width: `${((originalRight - originalLeft) / width) * 100}%`,
        height: `${((originalBottom - originalTop) / height) * 100}%`
      });
    }
  }
  return (
    <div className="pyramid-tile-layer" aria-hidden="true">
      {tiles.map((tile) => (
        <img
          key={tile.key}
          className="pyramid-tile"
          src={tile.src}
          alt=""
          draggable={false}
          loading="lazy"
          decoding="async"
          style={{
            left: tile.left,
            top: tile.top,
            width: tile.width,
            height: tile.height
          }}
        />
      ))}
    </div>
  );
}

function tileLevelForZoom({
  width,
  height,
  zoom,
  tileSize
}: {
  width: number;
  height: number;
  zoom: number;
  tileSize: number;
}) {
  if (zoom < TILE_ZOOM_THRESHOLD || Math.max(width, height) <= IMAGE_PREVIEW_MAX_DISPLAY_SIDE) {
    return null;
  }
  const preferredLevel = zoom >= 3.5 ? 0 : 1;
  for (let level = preferredLevel; level <= 8; level += 1) {
    const scale = 2 ** level;
    const columns = Math.ceil(Math.ceil(width / scale) / tileSize);
    const rows = Math.ceil(Math.ceil(height / scale) / tileSize);
    if (columns * rows <= MAX_RENDERED_TILES) {
      return level;
    }
  }
  return null;
}

const IMAGE_PREVIEW_MAX_DISPLAY_SIDE = 1800;
