import type { ImageJumpItem, ImageMapBin } from "./compositeImageNavigationModel";
import { CompositeImageAtlasPanel } from "./compositeImageAtlasPanel";
import { CompositeImageSearchResults } from "./compositeImageSearchResults";

import "./compositeImageSearchPopover.css";

export function CompositeImageSearchPopover({
  placement,
  imageIndex,
  imageKey,
  filteredCount,
  imageCount,
  imageMapBins,
  visibleSearchResults,
  activeResultIndex,
  hiddenBeforeCount,
  hiddenAfterCount,
  hiddenCount,
  onJump,
  onClose,
  onResultWheel,
  onActiveResultIndexChange
}: {
  placement: "top" | "bottom";
  imageIndex: number;
  imageKey: string;
  filteredCount: number;
  imageCount: number;
  imageMapBins: ImageMapBin[];
  visibleSearchResults: ImageJumpItem[];
  activeResultIndex: number;
  hiddenBeforeCount: number;
  hiddenAfterCount: number;
  hiddenCount: number;
  onJump: (index: number) => void;
  onClose: () => void;
  onResultWheel: (event: WheelEvent) => void;
  onActiveResultIndexChange: (index: number) => void;
}) {
  return (
    <div className="image-jump-popover" data-placement={placement}>
      <div className="image-jump-popover-body">
        <CompositeImageAtlasPanel
          imageIndex={imageIndex}
          imageKey={imageKey}
          filteredCount={filteredCount}
          imageMapBins={imageMapBins}
          onJump={onJump}
        />
        <CompositeImageSearchResults
          imageIndex={imageIndex}
          imageCount={imageCount}
          visibleSearchResults={visibleSearchResults}
          activeResultIndex={activeResultIndex}
          hiddenBeforeCount={hiddenBeforeCount}
          hiddenAfterCount={hiddenAfterCount}
          hiddenCount={hiddenCount}
          onJump={onJump}
          onClose={onClose}
          onResultWheel={onResultWheel}
          onActiveResultIndexChange={onActiveResultIndexChange}
        />
      </div>
    </div>
  );
}
