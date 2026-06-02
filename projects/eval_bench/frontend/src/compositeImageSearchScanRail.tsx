import type { CSSProperties } from "react";

import type { ImageJumpItem } from "./compositeImageNavigationModel";

export function CompositeImageSearchScanRail({
  activeResult,
  activeResultIndex,
  imageIndex,
  resultCount,
  dragging
}: {
  activeResult: ImageJumpItem | null;
  activeResultIndex: number;
  imageIndex: number;
  resultCount: number;
  dragging: boolean;
}) {
  if (!activeResult || resultCount <= 0) {
    return null;
  }
  const progress = Math.min(1, Math.max(0, (activeResultIndex + 0.5) / resultCount));
  const delta = activeResult.index - imageIndex;
  const direction = delta === 0 ? "current" : delta > 0 ? "forward" : "backward";
  const style = {
    "--image-result-scan-progress": progress.toFixed(4),
    "--image-result-scan-top": `${(progress * 100).toFixed(2)}%`
  } as CSSProperties;

  return (
    <div
      className={`image-jump-scan-rail direction-${direction}`}
      data-dragging={dragging ? "true" : undefined}
      aria-hidden="true"
      style={style}
    >
      <i />
      <span>#{(activeResult.index + 1).toLocaleString()}</span>
    </div>
  );
}
