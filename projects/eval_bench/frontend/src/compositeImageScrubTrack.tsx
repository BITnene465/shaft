import type {
  MouseEvent as ReactMouseEvent,
  PointerEvent as ReactPointerEvent
} from "react";

import type { ScrubPreview } from "./compositeImageNavigationModel";
import { basename } from "./formatters";

import "./compositeImageScrubTrack.css";

export function CompositeImageScrubTrack({
  progress,
  scrubbing,
  scrubPreview,
  onScrubPointerDown,
  onScrubPointerMove,
  onScrubPointerEnd,
  onScrubPointerLeave,
  onScrubMouseMove,
  onScrubMouseLeave
}: {
  progress: number;
  scrubbing: boolean;
  scrubPreview: ScrubPreview | null;
  onScrubPointerDown: (event: ReactPointerEvent<HTMLDivElement>) => void;
  onScrubPointerMove: (event: ReactPointerEvent<HTMLDivElement>) => void;
  onScrubPointerEnd: (event: ReactPointerEvent<HTMLDivElement>) => void;
  onScrubPointerLeave: (event: ReactPointerEvent<HTMLDivElement>) => void;
  onScrubMouseMove: (event: ReactMouseEvent<HTMLDivElement>) => void;
  onScrubMouseLeave: (event: ReactMouseEvent<HTMLDivElement>) => void;
}) {
  return (
    <div
      className={scrubbing ? "image-scrub-track scrubbing" : "image-scrub-track"}
      aria-label="拖拽切换组合图片"
      onPointerDown={onScrubPointerDown}
      onPointerMove={onScrubPointerMove}
      onPointerUp={onScrubPointerEnd}
      onPointerCancel={onScrubPointerEnd}
      onPointerLeave={onScrubPointerLeave}
      onMouseMove={onScrubMouseMove}
      onMouseLeave={onScrubMouseLeave}
    >
      <span style={{ width: `${progress}%` }} />
      <i style={{ left: `${progress}%` }} />
      {scrubPreview ? (
        <output
          className="image-scrub-preview"
          style={{ left: `${scrubPreview.percent}%` }}
          aria-live="polite"
        >
          <span>
            {scrubPreview.index + 1}
            <em>{scrubDeltaLabel(scrubPreview.delta)}</em>
          </span>
          <strong>{basename(scrubPreview.image)}</strong>
        </output>
      ) : null}
    </div>
  );
}

function scrubDeltaLabel(delta: number) {
  if (delta === 0) {
    return "current";
  }
  return delta > 0 ? `+${delta}` : String(delta);
}
