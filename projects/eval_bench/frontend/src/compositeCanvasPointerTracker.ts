import { useRef } from "react";
import type { PointerEvent, RefObject } from "react";

export function useCompositeCanvasPointerTracker() {
  const coordinateRef = useRef<HTMLSpanElement | null>(null);

  function handleCanvasPointerMove(event: PointerEvent<HTMLElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = clamp(event.clientX - rect.left, 0, rect.width);
    const y = clamp(event.clientY - rect.top, 0, rect.height);
    const percentX = rect.width > 0 ? (x / rect.width) * 100 : 0;
    const percentY = rect.height > 0 ? (y / rect.height) * 100 : 0;
    event.currentTarget.dataset.pointerReticle = "active";
    event.currentTarget.style.setProperty("--composite-pointer-x", `${percentX.toFixed(2)}%`);
    event.currentTarget.style.setProperty("--composite-pointer-y", `${percentY.toFixed(2)}%`);
    if (coordinateRef.current) {
      coordinateRef.current.textContent =
        `${Math.round(percentX).toString().padStart(2, "0")} / ` +
        Math.round(percentY).toString().padStart(2, "0");
    }
  }

  function handleCanvasPointerLeave(event: PointerEvent<HTMLElement>) {
    delete event.currentTarget.dataset.pointerReticle;
    if (coordinateRef.current) {
      coordinateRef.current.textContent = "";
    }
  }

  return {
    coordinateRef: coordinateRef as RefObject<HTMLSpanElement>,
    pointerHandlers: {
      onPointerMove: handleCanvasPointerMove,
      onPointerLeave: handleCanvasPointerLeave
    }
  };
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}
