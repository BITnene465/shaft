import { useCallback, useRef, useState } from "react";
import type { MutableRefObject, PointerEvent as ReactPointerEvent } from "react";

import type { InteractionSettings } from "./workspaceSettings";

type ViewportState = {
  zoom: number;
  pan: { x: number; y: number };
};

type PointerInteractionOptions = {
  allowOverlaySurfacePan?: boolean;
  interactionSettings: InteractionSettings;
  pendingPanRef: MutableRefObject<{ x: number; y: number } | null>;
  scheduleViewportUpdate: () => void;
  updateViewportDirty: (nextDirty: boolean) => void;
  viewportRef: MutableRefObject<ViewportState>;
  flushViewportUpdate: () => void;
};

export function useViewerViewportPointerInteraction({
  allowOverlaySurfacePan = false,
  interactionSettings,
  pendingPanRef,
  scheduleViewportUpdate,
  updateViewportDirty,
  viewportRef,
  flushViewportUpdate
}: PointerInteractionOptions) {
  const dragRef = useRef<{
    pointerId: number;
    startX: number;
    startY: number;
    startPan: { x: number; y: number };
  } | null>(null);
  const [isPanning, setIsPanning] = useState(false);

  const resetPointerInteraction = useCallback(() => {
    dragRef.current = null;
    setIsPanning(false);
  }, []);

  const handlePointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      if (event.button !== 0 && event.button !== 1) {
        return;
      }
      if (
        isOverlayInteractionTarget(event.target) &&
        !canPanOverlaySurface(event, allowOverlaySurfacePan)
      ) {
        return;
      }
      event.preventDefault();
      event.stopPropagation();
      event.currentTarget.setPointerCapture(event.pointerId);
      dragRef.current = {
        pointerId: event.pointerId,
        startX: event.clientX,
        startY: event.clientY,
        startPan: viewportRef.current.pan
      };
      setIsPanning(true);
    },
    [allowOverlaySurfacePan, viewportRef]
  );

  const handlePointerMove = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
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
    },
    [
      interactionSettings.panSensitivity,
      pendingPanRef,
      scheduleViewportUpdate,
      updateViewportDirty
    ]
  );

  const endPan = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
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
    },
    [flushViewportUpdate]
  );

  return {
    endPan,
    handlePointerDown,
    handlePointerMove,
    isPanning,
    resetPointerInteraction
  };
}

function isOverlayInteractionTarget(target: EventTarget | null) {
  if (!(target instanceof Element)) {
    return false;
  }
  return Boolean(target.closest(".overlay-instance, .canvas-hud"));
}

function canPanOverlaySurface(
  event: ReactPointerEvent<HTMLDivElement>,
  allowOverlaySurfacePan: boolean
) {
  return allowOverlaySurfacePan && (event.button === 1 || event.altKey || event.shiftKey);
}
