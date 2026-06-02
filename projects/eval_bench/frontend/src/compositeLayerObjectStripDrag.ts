import type { PointerEvent } from "react";

import { usePointerSweepSelection } from "./compositePointerSweep";

export function useLayerObjectStripDrag({
  onObjectHover,
  onObjectLock
}: {
  onObjectHover: (objectKey: string | null) => void;
  onObjectLock: (objectKey: string | null) => void;
}) {
  function objectKeyFromPointer(event: PointerEvent<HTMLElement>) {
    const target = document
      .elementFromPoint(event.clientX, event.clientY)
      ?.closest<HTMLElement>("[data-object-key]");
    if (!target || !event.currentTarget.contains(target)) {
      return null;
    }
    return target.dataset.objectKey || null;
  }

  const sweep = usePointerSweepSelection({
    resolveValueFromPointer: objectKeyFromPointer,
    onPreview: onObjectHover,
    onCommit: onObjectLock
  });

  return {
    dragging: sweep.dragging,
    objectStripDragHandlers: sweep.pointerSweepHandlers,
    shouldSuppressClick: sweep.shouldSuppressClick
  };
}
