import { useRef, useState } from "react";
import type { PointerEvent, RefObject } from "react";

const DEFAULT_DRAG_THRESHOLD_PX = 4;

export type PointerDragState = {
  startX: number;
  startY: number;
  lastX: number;
  lastY: number;
  deltaX: number;
  deltaY: number;
  moved: boolean;
};

export function usePointerDrag<TElement extends HTMLElement>({
  elementRef,
  thresholdPx = DEFAULT_DRAG_THRESHOLD_PX,
  onStart,
  onMove,
  onEnd,
  onCancel
}: {
  elementRef?: RefObject<TElement | null>;
  thresholdPx?: number;
  onStart?: (event: PointerEvent<TElement>, state: PointerDragState, element: TElement) => void;
  onMove?: (event: PointerEvent<TElement>, state: PointerDragState, element: TElement) => void;
  onEnd?: (event: PointerEvent<TElement>, state: PointerDragState, element: TElement) => void;
  onCancel?: (event: PointerEvent<TElement>, state: PointerDragState, element: TElement) => void;
}) {
  const [dragging, setDragging] = useState(false);
  const draggingRef = useRef(false);
  const movedRef = useRef(false);
  const stateRef = useRef<PointerDragState>({
    startX: 0,
    startY: 0,
    lastX: 0,
    lastY: 0,
    deltaX: 0,
    deltaY: 0,
    moved: false
  });

  function dragElement(event: PointerEvent<TElement>) {
    return elementRef?.current ?? event.currentTarget;
  }

  function nextState(event: PointerEvent<TElement>, moved: boolean): PointerDragState {
    const current = stateRef.current;
    const deltaX = event.clientX - current.startX;
    const deltaY = event.clientY - current.startY;
    const nextMoved =
      moved || Math.hypot(deltaX, deltaY) > thresholdPx;
    const next = {
      startX: current.startX,
      startY: current.startY,
      lastX: event.clientX,
      lastY: event.clientY,
      deltaX,
      deltaY,
      moved: nextMoved
    };
    stateRef.current = next;
    movedRef.current = nextMoved;
    return next;
  }

  function releasePointer(event: PointerEvent<TElement>, element: TElement) {
    if (element.hasPointerCapture(event.pointerId)) {
      element.releasePointerCapture(event.pointerId);
    }
  }

  function shouldSuppressClick() {
    const suppress = movedRef.current;
    movedRef.current = false;
    return suppress;
  }

  const pointerDragHandlers = {
    onPointerDown(event: PointerEvent<TElement>) {
      const element = dragElement(event);
      const initialState = {
        startX: event.clientX,
        startY: event.clientY,
        lastX: event.clientX,
        lastY: event.clientY,
        deltaX: 0,
        deltaY: 0,
        moved: false
      };
      stateRef.current = initialState;
      movedRef.current = false;
      draggingRef.current = true;
      setDragging(true);
      onStart?.(event, initialState, element);
      element.setPointerCapture(event.pointerId);
    },
    onPointerMove(event: PointerEvent<TElement>) {
      if (!draggingRef.current) {
        return;
      }
      const element = dragElement(event);
      onMove?.(event, nextState(event, movedRef.current), element);
    },
    onPointerUp(event: PointerEvent<TElement>) {
      if (!draggingRef.current) {
        return;
      }
      const element = dragElement(event);
      const state = nextState(event, movedRef.current);
      releasePointer(event, element);
      draggingRef.current = false;
      setDragging(false);
      onEnd?.(event, state, element);
    },
    onPointerCancel(event: PointerEvent<TElement>) {
      const element = dragElement(event);
      const state = nextState(event, movedRef.current);
      releasePointer(event, element);
      draggingRef.current = false;
      movedRef.current = false;
      setDragging(false);
      onCancel?.(event, state, element);
    }
  };

  return {
    dragging,
    pointerDragHandlers,
    shouldSuppressClick
  };
}
