import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";

import { isTextInputTarget } from "./formatters";
import {
  clampNumber,
  clampPan,
  computeFitSize
} from "./viewerGeometry";
import { useViewerViewportPointerInteraction } from "./viewerViewportPointerInteraction";
import { useViewerViewportTileLevel } from "./viewerViewportTileLevel";
import { useViewerViewportWheelZoom } from "./viewerViewportWheelZoom";
import {
  currentSyncedViewport,
  publishSyncedViewport,
  subscribeSyncedViewport
} from "./viewerViewportSync";
import {
  VIEWPORT_RESET_COMMAND,
  viewportResetCommandDetail
} from "./viewerViewportCommands";
import type { SyncedViewportSnapshot } from "./viewerViewportSync";
import { useWorkspaceShortcuts } from "./workspaceSettings";
import type { InteractionSettings } from "./workspaceSettings";

type ViewportControllerOptions = {
  allowOverlaySurfacePan?: boolean;
  width: number;
  height: number;
  imageUrl: string;
  imageTileUrlTemplate?: string | null;
  imageTileSize?: number | null;
  interactionSettings: InteractionSettings;
  viewportSyncKey?: string | null;
};

export function useViewerViewportController({
  allowOverlaySurfacePan = false,
  width,
  height,
  imageUrl,
  imageTileUrlTemplate,
  imageTileSize,
  interactionSettings,
  viewportSyncKey
}: ViewportControllerOptions) {
  const controllerIdRef = useRef(`viewport_${Math.random().toString(36).slice(2)}`);
  const stageRef = useRef<HTMLDivElement | null>(null);
  const contentRef = useRef<HTMLDivElement | null>(null);
  const zoomLabelRef = useRef<HTMLSpanElement | null>(null);
  const panBoundsRef = useRef({
    viewportWidth: 1,
    viewportHeight: 1,
    contentWidth: 1,
    contentHeight: 1
  });
  const viewportRef = useRef({ zoom: 1, pan: { x: 0, y: 0 } });
  const pendingPanRef = useRef<{ x: number; y: number } | null>(null);
  const frameRef = useRef<number | null>(null);
  const transformingTimeoutRef = useRef<number | null>(null);
  const viewportDirtyRef = useRef(false);
  const [viewportDirty, setViewportDirty] = useState(false);
  const [stageSize, setStageSize] = useState({ width: 1, height: 1 });
  const {
    resetTileLevel,
    scheduleTileLevelUpdate,
    tileLevel
  } = useViewerViewportTileLevel({
    width,
    height,
    imageTileUrlTemplate,
    imageTileSize
  });
  const fitSize = useMemo(
    () => computeFitSize(width, height, stageSize),
    [height, stageSize, width]
  );
  const { actionForEvent } = useWorkspaceShortcuts();

  const updateViewportDirty = useCallback((nextDirty: boolean) => {
    if (viewportDirtyRef.current === nextDirty) {
      return;
    }
    viewportDirtyRef.current = nextDirty;
    setViewportDirty(nextDirty);
  }, []);

  const scheduleViewportUpdate = useCallback(() => {
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
      publishCurrentViewport(true);
    });
  }, []);

  const flushViewportUpdate = useCallback(() => {
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
    publishCurrentViewport(true);
  }, []);

  const {
    endPan,
    handlePointerDown,
    handlePointerMove,
    isPanning,
    resetPointerInteraction
  } = useViewerViewportPointerInteraction({
    allowOverlaySurfacePan,
    interactionSettings,
    pendingPanRef,
    scheduleViewportUpdate,
    updateViewportDirty,
    viewportRef,
    flushViewportUpdate
  });
  useViewerViewportWheelZoom({
    interactionSettings,
    stageRef,
    viewportRef,
    onWheelZoom: applyZoom
  });

  useEffect(() => {
    viewportRef.current = { zoom: 1, pan: { x: 0, y: 0 } };
    pendingPanRef.current = null;
    if (frameRef.current !== null) {
      cancelAnimationFrame(frameRef.current);
      frameRef.current = null;
    }
    applyViewportToDom();
    updateViewportDirty(false);
    resetTileLevel();
    resetPointerInteraction();
  }, [imageUrl, resetPointerInteraction, resetTileLevel]);

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

  useLayoutEffect(() => {
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
    if (!viewportSyncKey) {
      return undefined;
    }
    const currentSnapshot = currentSyncedViewport(viewportSyncKey);
    if (currentSnapshot) {
      applySyncedViewport(currentSnapshot);
    } else {
      publishCurrentViewport(false);
    }
    return subscribeSyncedViewport(viewportSyncKey, (snapshot) => {
      if (snapshot.sourceId === controllerIdRef.current) {
        return;
      }
      applySyncedViewport(snapshot);
    });
  }, [
    fitSize.height,
    fitSize.width,
    interactionSettings.maxZoom,
    interactionSettings.minZoom,
    viewportSyncKey
  ]);

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

  useEffect(() => {
    function handleViewportReset(event: Event) {
      const detail = viewportResetCommandDetail(event);
      if (!detail) {
        return;
      }
      if (detail.viewportSyncKey && detail.viewportSyncKey !== viewportSyncKey) {
        return;
      }
      resetViewport();
    }
    window.addEventListener(VIEWPORT_RESET_COMMAND, handleViewportReset);
    return () => window.removeEventListener(VIEWPORT_RESET_COMMAND, handleViewportReset);
  }, [viewportSyncKey]);

  function clampStagePan(
    nextPan: { x: number; y: number },
    nextZoom = viewportRef.current.zoom
  ) {
    return clampPan(nextPan, nextZoom, panBoundsRef.current);
  }

  function viewportTransform() {
    const viewport = viewportRef.current;
    return `translate(-50%, -50%) translate(${viewport.pan.x}px, ${viewport.pan.y}px) scale(${viewport.zoom})`;
  }

  function applyViewportToDom() {
    const content = contentRef.current;
    const zoomLabel = zoomLabelRef.current;
    const viewport = viewportRef.current;
    if (content) {
      content.style.transform = viewportTransform();
    }
    if (zoomLabel) {
      zoomLabel.textContent = `${Math.round(viewport.zoom * 100)}%`;
    }
  }

  function currentSyncedSnapshot(dirty: boolean): SyncedViewportSnapshot {
    const viewport = viewportRef.current;
    const bounds = panBoundsRef.current;
    return {
      zoom: viewport.zoom,
      panRatio: {
        x: viewport.pan.x / Math.max(1, bounds.contentWidth),
        y: viewport.pan.y / Math.max(1, bounds.contentHeight)
      },
      dirty,
      sourceId: controllerIdRef.current
    };
  }

  function publishCurrentViewport(dirty = viewportDirtyRef.current) {
    if (!viewportSyncKey) {
      return;
    }
    publishSyncedViewport(viewportSyncKey, currentSyncedSnapshot(dirty));
  }

  function applySyncedViewport(snapshot: SyncedViewportSnapshot) {
    const bounds = panBoundsRef.current;
    const zoom = clampNumber(snapshot.zoom, interactionSettings.minZoom, interactionSettings.maxZoom);
    viewportRef.current = {
      zoom,
      pan: clampStagePan(
        {
          x: snapshot.panRatio.x * bounds.contentWidth,
          y: snapshot.panRatio.y * bounds.contentHeight
        },
        zoom
      )
    };
    pendingPanRef.current = null;
    scheduleTileLevelUpdate(zoom);
    markViewportTransforming();
    applyViewportToDom();
    updateViewportDirty(snapshot.dirty);
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
    stageRef.current?.classList.remove("transforming");
    applyViewportToDom();
    updateViewportDirty(false);
    publishCurrentViewport(false);
    resetTileLevel();
    resetPointerInteraction();
  }

  return {
    contentRef,
    contentTransform: viewportTransform(),
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
  };
}
