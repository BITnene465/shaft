import type { ComponentProps, WheelEvent as ReactWheelEvent } from "react";

import type { ImageJumpItem, ScrubPreview } from "./compositeImageNavigationModel";
import { imageProgressPercent } from "./compositeImageNavigationModel";
import { CompositeImageIndexMeter } from "./compositeImageIndexMeter";
import { CompositeImageNearbyRail } from "./compositeImageNearbyRail";
import { CompositeImageScrubTrack } from "./compositeImageScrubTrack";

import "./compositeImageTimeline.css";

export function CompositeImageTimeline({
  imageIndex,
  imageCount,
  nearbyImages,
  scrubbing,
  scrubPreview,
  onJump,
  onStep,
  onTimelineWheel,
  onScrubPointerDown,
  onScrubPointerMove,
  onScrubPointerEnd,
  onScrubPointerLeave,
  onScrubMouseMove,
  onScrubMouseLeave
}: {
  imageIndex: number;
  imageCount: number;
  nearbyImages: ImageJumpItem[];
  scrubbing: boolean;
  scrubPreview: ScrubPreview | null;
  onJump: (index: number) => void;
  onStep: (delta: -1 | 1) => void;
  onTimelineWheel: (event: ReactWheelEvent<HTMLDivElement>) => void;
  onScrubPointerDown: ComponentProps<typeof CompositeImageScrubTrack>["onScrubPointerDown"];
  onScrubPointerMove: ComponentProps<typeof CompositeImageScrubTrack>["onScrubPointerMove"];
  onScrubPointerEnd: ComponentProps<typeof CompositeImageScrubTrack>["onScrubPointerEnd"];
  onScrubPointerLeave: ComponentProps<typeof CompositeImageScrubTrack>["onScrubPointerLeave"];
  onScrubMouseMove: ComponentProps<typeof CompositeImageScrubTrack>["onScrubMouseMove"];
  onScrubMouseLeave: ComponentProps<typeof CompositeImageScrubTrack>["onScrubMouseLeave"];
}) {
  const progress = imageProgressPercent(imageIndex, imageCount);
  return (
    <div className="image-navigator-timeline" onWheelCapture={onTimelineWheel}>
      <CompositeImageIndexMeter imageIndex={imageIndex} imageCount={imageCount} />
      <CompositeImageScrubTrack
        progress={progress}
        scrubbing={scrubbing}
        scrubPreview={scrubPreview}
        onScrubPointerDown={onScrubPointerDown}
        onScrubPointerMove={onScrubPointerMove}
        onScrubPointerEnd={onScrubPointerEnd}
        onScrubPointerLeave={onScrubPointerLeave}
        onScrubMouseMove={onScrubMouseMove}
        onScrubMouseLeave={onScrubMouseLeave}
      />
      <CompositeImageNearbyRail
        imageCount={imageCount}
        imageIndex={imageIndex}
        nearbyImages={nearbyImages}
        onJump={onJump}
        onStep={onStep}
      />
    </div>
  );
}
