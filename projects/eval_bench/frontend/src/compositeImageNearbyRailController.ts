import { useRef, useState } from "react";
import type { WheelEvent as ReactWheelEvent } from "react";

import { clampImageIndex } from "./compositeImageNavigationModel";
import { usePointerDrag } from "./compositePointerDrag";

const DRAG_THRESHOLD_PX = 6;
const DRAG_STEP_PX = 44;

type RailDragHint = "next" | "previous" | null;

export function useCompositeImageNearbyRailController({
  imageCount,
  imageIndex,
  onJump,
  onStep
}: {
  imageCount: number;
  imageIndex: number;
  onJump: (index: number) => void;
  onStep: (delta: -1 | 1) => void;
}) {
  const railRef = useRef<HTMLDivElement | null>(null);
  const dragStepXRef = useRef(0);
  const dragVirtualIndexRef = useRef(0);
  const dragStartScrollLeftRef = useRef(0);
  const [dragHint, setDragHint] = useState<RailDragHint>(null);

  function handleWheel(event: ReactWheelEvent<HTMLDivElement>) {
    const wheelDelta =
      Math.abs(event.deltaY) >= Math.abs(event.deltaX) ? event.deltaY : event.deltaX;
    if (Math.abs(wheelDelta) < 1) {
      return;
    }
    event.preventDefault();
    onStep(wheelDelta > 0 ? 1 : -1);
  }

  const drag = usePointerDrag<HTMLElement>({
    elementRef: railRef,
    thresholdPx: DRAG_THRESHOLD_PX,
    onStart: (event, _state, rail) => {
      dragStepXRef.current = event.clientX;
      dragVirtualIndexRef.current = imageIndex;
      dragStartScrollLeftRef.current = rail.scrollLeft;
      setDragHint(null);
    },
    onMove: (event, state, rail) => {
      const deltaX = state.deltaX;
      rail.scrollLeft = dragStartScrollLeftRef.current - deltaX;
      const stepDeltaX = event.clientX - dragStepXRef.current;
      const stepCount = Math.trunc(stepDeltaX / DRAG_STEP_PX);
      if (stepCount !== 0) {
        const direction = stepCount > 0 ? -1 : 1;
        Array.from({ length: Math.min(4, Math.abs(stepCount)) }).forEach(() => {
          dragVirtualIndexRef.current = clampImageIndex(
            dragVirtualIndexRef.current + direction,
            imageCount
          );
        });
        onJump(dragVirtualIndexRef.current);
        dragStepXRef.current += stepCount * DRAG_STEP_PX;
        setDragHint(direction > 0 ? "next" : "previous");
      } else if (Math.abs(deltaX) > DRAG_THRESHOLD_PX) {
        setDragHint(deltaX < 0 ? "next" : "previous");
      }
    },
    onEnd: () => {
      setDragHint(null);
    },
    onCancel: () => {
      setDragHint(null);
    }
  });

  return {
    dragging: drag.dragging,
    dragHint,
    railRef,
    onPointerDown: drag.pointerDragHandlers.onPointerDown,
    onPointerMove: drag.pointerDragHandlers.onPointerMove,
    onPointerUp: drag.pointerDragHandlers.onPointerUp,
    onPointerCancel: drag.pointerDragHandlers.onPointerCancel,
    onWheelCapture: handleWheel,
    shouldSuppressClick: drag.shouldSuppressClick
  };
}
