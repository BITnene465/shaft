import { useEffect, useMemo, useState } from "react";
import type {
  MouseEvent as ReactMouseEvent,
  PointerEvent as ReactPointerEvent,
  WheelEvent as ReactWheelEvent
} from "react";

import type { CompositeSampleView } from "./api";
import {
  nearbyImageKeys,
  previewFromScrubPointer
} from "./compositeImageNavigationModel";
import type { ScrubPreview } from "./compositeImageNavigationModel";
import { usePointerDrag } from "./compositePointerDrag";

type ScrubHoverEvent =
  | ReactPointerEvent<HTMLDivElement>
  | ReactMouseEvent<HTMLDivElement>;

export function useCompositeImageTimelineController({
  composite,
  jumpTo,
  step
}: {
  composite: CompositeSampleView;
  jumpTo: (index: number) => void;
  step: (delta: -1 | 1) => void;
}) {
  const [scrubPreview, setScrubPreview] = useState<ScrubPreview | null>(null);
  const nearbyImages = useMemo(
    () => nearbyImageKeys(composite.image_keys, composite.image_index),
    [composite.image_index, composite.image_keys]
  );

  useEffect(() => {
    setScrubPreview(null);
  }, [composite.image_key]);

  function handleTimelineWheel(event: ReactWheelEvent<HTMLDivElement>) {
    if (Math.abs(event.deltaY) <= Math.abs(event.deltaX)) {
      return;
    }
    event.preventDefault();
    step(event.deltaY > 0 ? 1 : -1);
  }

  function jumpFromScrubPointer(event: ReactPointerEvent<HTMLDivElement>) {
    const preview = previewFromScrubPointer(event, composite.image_keys, composite.image_index);
    if (!preview) {
      return;
    }
    setScrubPreview(preview);
    jumpTo(preview.index);
  }

  function updateScrubPreview(event: ScrubHoverEvent) {
    setScrubPreview(
      previewFromScrubPointer(event, composite.image_keys, composite.image_index)
    );
  }

  const scrubDrag = usePointerDrag<HTMLDivElement>({
    onStart: (event) => jumpFromScrubPointer(event),
    onMove: (event) => jumpFromScrubPointer(event),
    onEnd: (event) => jumpFromScrubPointer(event)
  });

  function handleScrubPointerMove(event: ReactPointerEvent<HTMLDivElement>) {
    scrubDrag.pointerDragHandlers.onPointerMove(event);
    if (!scrubDrag.dragging) {
      updateScrubPreview(event);
    }
  }

  function handleScrubPointerEnd(event: ReactPointerEvent<HTMLDivElement>) {
    if (event.type === "pointercancel") {
      scrubDrag.pointerDragHandlers.onPointerCancel(event);
      return;
    }
    scrubDrag.pointerDragHandlers.onPointerUp(event);
  }

  function handleScrubPointerLeave() {
    if (!scrubDrag.dragging) {
      setScrubPreview(null);
    }
  }

  function handleScrubMouseMove(event: ReactMouseEvent<HTMLDivElement>) {
    if (!scrubDrag.dragging) {
      updateScrubPreview(event);
    }
  }

  function handleScrubMouseLeave() {
    if (!scrubDrag.dragging) {
      setScrubPreview(null);
    }
  }

  return {
    imageIndex: composite.image_index,
    imageCount: composite.image_count,
    nearbyImages,
    onStep: step,
    scrubbing: scrubDrag.dragging,
    scrubPreview,
    onJump: jumpTo,
    onTimelineWheel: handleTimelineWheel,
    onScrubPointerDown: scrubDrag.pointerDragHandlers.onPointerDown,
    onScrubPointerMove: handleScrubPointerMove,
    onScrubPointerEnd: handleScrubPointerEnd,
    onScrubPointerLeave: handleScrubPointerLeave,
    onScrubMouseMove: handleScrubMouseMove,
    onScrubMouseLeave: handleScrubMouseLeave
  };
}
