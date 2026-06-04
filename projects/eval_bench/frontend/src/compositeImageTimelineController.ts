import type { WheelEvent as ReactWheelEvent } from "react";

import type { CompositeSampleView } from "./api";

export function useCompositeImageTimelineController({
  composite,
  step
}: {
  composite: CompositeSampleView;
  step: (delta: -1 | 1) => void;
}) {
  function handleTimelineWheel(event: ReactWheelEvent<HTMLDivElement>) {
    if (Math.abs(event.deltaY) <= Math.abs(event.deltaX)) {
      return;
    }
    event.preventDefault();
    step(event.deltaY > 0 ? 1 : -1);
  }

  return {
    imageIndex: composite.image_index,
    imageCount: composite.image_count,
    onTimelineWheel: handleTimelineWheel
  };
}
