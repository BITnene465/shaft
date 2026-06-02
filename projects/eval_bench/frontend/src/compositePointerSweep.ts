import { useRef, useState } from "react";
import type { PointerEvent } from "react";

export function usePointerSweepSelection<T>({
  resolveValueFromPointer,
  onPreview,
  onCommit
}: {
  resolveValueFromPointer: (event: PointerEvent<HTMLElement>) => T | null;
  onPreview: (value: T) => void;
  onCommit: (value: T) => void;
}) {
  const [dragging, setDragging] = useState(false);
  const draggingRef = useRef(false);
  const activeValueRef = useRef<T | null>(null);
  const suppressClickRef = useRef(false);

  function previewValueAtPointer(event: PointerEvent<HTMLElement>) {
    const value = resolveValueFromPointer(event);
    if (value === null) {
      return;
    }
    activeValueRef.current = value;
    onPreview(value);
  }

  function releasePointerCapture(event: PointerEvent<HTMLElement>) {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  function shouldSuppressClick() {
    if (!suppressClickRef.current) {
      return false;
    }
    suppressClickRef.current = false;
    return true;
  }

  const pointerSweepHandlers = {
    onPointerDown(event: PointerEvent<HTMLElement>) {
      if (event.button !== 0) {
        return;
      }
      const value = resolveValueFromPointer(event);
      if (value === null) {
        return;
      }
      activeValueRef.current = value;
      suppressClickRef.current = true;
      draggingRef.current = true;
      setDragging(true);
      onPreview(value);
      event.currentTarget.setPointerCapture(event.pointerId);
    },
    onPointerMove(event: PointerEvent<HTMLElement>) {
      if (draggingRef.current) {
        previewValueAtPointer(event);
      }
    },
    onPointerUp(event: PointerEvent<HTMLElement>) {
      if (!draggingRef.current) {
        return;
      }
      previewValueAtPointer(event);
      releasePointerCapture(event);
      draggingRef.current = false;
      setDragging(false);
      const value = activeValueRef.current;
      activeValueRef.current = null;
      if (value !== null) {
        onCommit(value);
      }
    },
    onPointerCancel(event: PointerEvent<HTMLElement>) {
      releasePointerCapture(event);
      draggingRef.current = false;
      activeValueRef.current = null;
      suppressClickRef.current = false;
      setDragging(false);
    }
  };

  return {
    dragging,
    pointerSweepHandlers,
    shouldSuppressClick
  };
}
