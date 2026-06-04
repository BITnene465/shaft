import type { WheelEvent as ReactWheelEvent } from "react";

import { CompositeImageIndexMeter } from "./compositeImageIndexMeter";

import "./compositeImageTimeline.css";

export function CompositeImageTimeline({
  imageIndex,
  imageCount,
  onTimelineWheel
}: {
  imageIndex: number;
  imageCount: number;
  onTimelineWheel: (event: ReactWheelEvent<HTMLDivElement>) => void;
}) {
  return (
    <div className="image-navigator-timeline" onWheelCapture={onTimelineWheel}>
      <CompositeImageIndexMeter imageIndex={imageIndex} imageCount={imageCount} />
    </div>
  );
}
