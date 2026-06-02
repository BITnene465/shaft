import { useMemo, useState } from "react";
import type { CSSProperties, PointerEvent } from "react";

export type CompositeCanvasPointerState = {
  x: number;
  y: number;
  percentX: number;
  percentY: number;
};

export function useCompositeCanvasPointerTracker() {
  const [pointer, setPointer] = useState<CompositeCanvasPointerState | null>(null);
  const pointerVars = useMemo(() => pointerStyle(pointer), [pointer]);

  function handleCanvasPointerMove(event: PointerEvent<HTMLElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = clamp(event.clientX - rect.left, 0, rect.width);
    const y = clamp(event.clientY - rect.top, 0, rect.height);
    setPointer({
      x,
      y,
      percentX: rect.width > 0 ? (x / rect.width) * 100 : 0,
      percentY: rect.height > 0 ? (y / rect.height) * 100 : 0
    });
  }

  function handleCanvasPointerLeave() {
    setPointer(null);
  }

  return {
    pointer,
    pointerActive: Boolean(pointer),
    pointerVars,
    pointerHandlers: {
      onPointerMove: handleCanvasPointerMove,
      onPointerLeave: handleCanvasPointerLeave
    }
  };
}

function pointerStyle(pointer: CompositeCanvasPointerState | null) {
  if (!pointer) {
    return undefined;
  }
  return {
    "--composite-pointer-x": `${pointer.percentX.toFixed(2)}%`,
    "--composite-pointer-y": `${pointer.percentY.toFixed(2)}%`
  } as CSSProperties;
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}
