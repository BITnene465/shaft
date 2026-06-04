import { useEffect, useState } from "react";
import type { RefObject } from "react";

import type { CompositeSampleView } from "./api";
import { CompositeInteractionPalette } from "./compositeInteractionPalette";
import { useCompositeImageNavigationController } from "./compositeImageNavigationController";
import { focusCompositeImageSearchInput } from "./compositeImageNavigatorKeyboard";
import { CompositeImageNavigatorPrimary } from "./compositeImageNavigatorPrimary";
import { CompositeImageSearchBar } from "./compositeImageSearchBar";
import { CompositeImageTimeline } from "./compositeImageTimeline";
import { requestViewportReset } from "./viewerViewportCommands";

import "./compositeImageNavigator.css";

export function CompositeImageNavigator({
  composite,
  onImageIndexChange
}: {
  composite: CompositeSampleView;
  onImageIndexChange: (index: number) => void;
}) {
  const navigation = useCompositeImageNavigationController({
    composite,
    onImageIndexChange
  });
  const density = useCompositeNavigatorDensity(navigation.rootRef);
  return (
    <div
      ref={navigation.rootRef}
      className="composite-image-navigator"
      data-density={density}
      aria-label="组合报告图片导航"
    >
      <CompositeImageNavigatorPrimary {...navigation.primaryProps} />
      <CompositeImageTimeline {...navigation.timelineProps} />
      <CompositeImageSearchBar {...navigation.searchProps} />
      <CompositeInteractionPalette
        canPrevious={navigation.primaryProps.canPrevious}
        canNext={navigation.primaryProps.canNext}
        onPrevious={() => navigation.primaryProps.onStep(-1)}
        onNext={() => navigation.primaryProps.onStep(1)}
        onSearch={() => {
          navigation.searchProps.onSearchOpenChange(true);
          focusCompositeImageSearchInput(navigation.rootRef.current);
        }}
        onResetViewport={() => requestViewportReset(`composite:${composite.image_key}`)}
      />
    </div>
  );
}

function useCompositeNavigatorDensity(rootRef: RefObject<HTMLDivElement | null>) {
  const [density, setDensity] = useState<"full" | "controls" | "compact">("full");
  useEffect(() => {
    const node = rootRef.current;
    if (!node) {
      return undefined;
    }
    const currentNode = node;
    function updateDensity() {
      const width = currentNode.getBoundingClientRect().width;
      setDensity(width >= 1100 ? "full" : width >= 720 ? "controls" : "compact");
    }
    updateDensity();
    const observer = new ResizeObserver(updateDensity);
    observer.observe(currentNode);
    return () => observer.disconnect();
  }, [rootRef]);
  return density;
}
