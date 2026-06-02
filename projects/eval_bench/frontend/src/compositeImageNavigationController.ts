import { useCallback, useEffect, useRef, useState } from "react";

import type { CompositeSampleView } from "./api";
import { useCompositeImageKeyboard } from "./compositeImageNavigatorKeyboard";
import { clampImageIndex } from "./compositeImageNavigationModel";
import { useCompositeImageSearchController } from "./compositeImageSearchController";
import { useCompositeImageTimelineController } from "./compositeImageTimelineController";

export function useCompositeImageNavigationController({
  composite,
  onImageIndexChange
}: {
  composite: CompositeSampleView;
  onImageIndexChange: (index: number) => void;
}) {
  const rootRef = useRef<HTMLDivElement | null>(null);
  const [jumpDraft, setJumpDraft] = useState(String(composite.image_index + 1));
  const canPrevious = composite.image_index > 0;
  const canNext = composite.image_index < composite.image_count - 1;

  useEffect(() => {
    setJumpDraft(String(composite.image_index + 1));
  }, [composite.image_index]);

  const jumpTo = useCallback((index: number) => {
    onImageIndexChange(clampImageIndex(index, composite.image_count));
  }, [composite.image_count, onImageIndexChange]);

  const searchProps = useCompositeImageSearchController({
    composite,
    rootRef,
    jumpTo
  });

  useCompositeImageKeyboard({
    rootRef,
    imageCount: composite.image_count,
    imageIndex: composite.image_index,
    setSearchOpen: searchProps.onSearchOpenChange,
    jumpTo
  });

  function submitJump() {
    const nextIndex = Number.parseInt(jumpDraft, 10);
    if (Number.isFinite(nextIndex)) {
      jumpTo(nextIndex - 1);
    } else {
      setJumpDraft(String(composite.image_index + 1));
    }
  }

  function step(delta: -1 | 1) {
    jumpTo(composite.image_index + delta);
  }

  const timelineProps = useCompositeImageTimelineController({
    composite,
    jumpTo,
    step
  });

  return {
    primaryProps: {
      composite,
      jumpDraft,
      canPrevious,
      canNext,
      onJumpDraftChange: setJumpDraft,
      onSubmitJump: submitJump,
      onJump: jumpTo,
      onStep: step
    },
    rootRef,
    searchProps,
    timelineProps
  };
}
