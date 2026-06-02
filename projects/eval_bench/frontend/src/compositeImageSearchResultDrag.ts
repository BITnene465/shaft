import type { PointerEvent } from "react";

import type { ImageJumpItem } from "./compositeImageNavigationModel";
import { usePointerSweepSelection } from "./compositePointerSweep";

type SearchResultSweepValue = {
  item: ImageJumpItem;
  windowIndex: number;
};

export function useCompositeImageSearchResultDrag({
  visibleSearchResults,
  onJump,
  onClose,
  onActiveResultIndexChange
}: {
  visibleSearchResults: ImageJumpItem[];
  onJump: (index: number) => void;
  onClose: () => void;
  onActiveResultIndexChange: (index: number) => void;
}) {
  function resultValueFromPointer(event: PointerEvent<HTMLElement>): SearchResultSweepValue | null {
    const target = document
      .elementFromPoint(event.clientX, event.clientY)
      ?.closest<HTMLElement>("[data-result-window-index]");
    if (!target || !event.currentTarget.contains(target)) {
      return null;
    }
    const index = Number(target.dataset.resultWindowIndex);
    if (!Number.isInteger(index) || index < 0) {
      return null;
    }
    const item = visibleSearchResults[index];
    return item ? { item, windowIndex: index } : null;
  }

  const sweep = usePointerSweepSelection({
    resolveValueFromPointer: resultValueFromPointer,
    onPreview: (value) => onActiveResultIndexChange(value.windowIndex),
    onCommit: (value) => {
      onJump(value.item.index);
      onClose();
    }
  });

  return {
    dragging: sweep.dragging,
    resultDragHandlers: sweep.pointerSweepHandlers,
    shouldSuppressClick: sweep.shouldSuppressClick
  };
}
