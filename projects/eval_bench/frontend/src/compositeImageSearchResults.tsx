import type { ImageJumpItem } from "./compositeImageNavigationModel";
import { CompositeImagePanelHeader } from "./compositeImagePanel";
import { useCompositeImageSearchResultDrag } from "./compositeImageSearchResultDrag";
import { CompositeImageSearchResultList } from "./compositeImageSearchResultList";
import { CompositeImageSearchPreview } from "./compositeImageSearchPreview";
import { CompositeImageSearchMore, CompositeImageSearchStatus } from "./compositeImageSearchStatus";

import "./compositeImageSearchResults.css";

export function CompositeImageSearchResults({
  imageIndex,
  imageCount,
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
  imageIndex: number;
  imageCount: number;
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
  const activeResult = visibleSearchResults[activeResultIndex] ?? visibleSearchResults[0] ?? null;
  const resultDrag = useCompositeImageSearchResultDrag({
    visibleSearchResults,
    onJump,
    onClose,
    onActiveResultIndexChange
  });
  const dragTargetLabel = resultDragTargetLabel(resultDrag.dragging, activeResult);

  return (
    <section className="image-jump-results-panel" aria-label="图片跳转结果">
      <CompositeImagePanelHeader
        title="Matches"
        meta={`${visibleSearchResults.length.toLocaleString()} visible${
          hiddenCount > 0 ? ` · ${hiddenCount.toLocaleString()} clipped` : ""
        }`}
        action={<kbd>↑↓ / Wheel / Enter</kbd>}
      />
      <CompositeImageSearchPreview
        activeResult={activeResult}
        onJump={onJump}
        onClose={onClose}
      />
      <CompositeImageSearchStatus
        hiddenBeforeCount={hiddenBeforeCount}
        hiddenAfterCount={hiddenAfterCount}
        hiddenCount={hiddenCount}
        dragging={resultDrag.dragging}
        dragTargetLabel={dragTargetLabel}
      />
      <CompositeImageSearchResultList
        imageIndex={imageIndex}
        imageCount={imageCount}
        visibleSearchResults={visibleSearchResults}
        activeResultIndex={activeResultIndex}
        dragging={resultDrag.dragging}
        resultDragHandlers={resultDrag.resultDragHandlers}
        shouldSuppressClick={resultDrag.shouldSuppressClick}
        onJump={onJump}
        onClose={onClose}
        onResultWheel={onResultWheel}
        onActiveResultIndexChange={onActiveResultIndexChange}
      />
      <CompositeImageSearchMore hiddenCount={hiddenCount} />
    </section>
  );
}

function resultDragTargetLabel(dragging: boolean, item: ImageJumpItem | null) {
  if (!dragging || !item) {
    return "";
  }
  return `#${(item.index + 1).toLocaleString()}`;
}
