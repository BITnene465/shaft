import { useEffect } from "react";
import type { RefObject } from "react";

export function imageSearchWheelStep(event: WheelEvent): -1 | 0 | 1 {
  const wheelDelta =
    Math.abs(event.deltaY) >= Math.abs(event.deltaX) ? event.deltaY : event.deltaX;
  if (Math.abs(wheelDelta) < 1) {
    return 0;
  }
  event.preventDefault();
  event.stopPropagation();
  return wheelDelta > 0 ? 1 : -1;
}

export function useImageSearchWheelCruise<T extends HTMLElement>({
  elementRef,
  onResultWheel
}: {
  elementRef: RefObject<T | null>;
  onResultWheel: (event: WheelEvent) => void;
}) {
  useEffect(() => {
    const node = elementRef.current;
    if (!node) {
      return undefined;
    }
    function handleNativeWheel(event: WheelEvent) {
      onResultWheel(event);
    }
    node.addEventListener("wheel", handleNativeWheel, { passive: false });
    return () => node.removeEventListener("wheel", handleNativeWheel);
  }, [elementRef, onResultWheel]);
}
