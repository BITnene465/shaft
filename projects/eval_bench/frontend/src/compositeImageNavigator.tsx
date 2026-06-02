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
  return (
    <div ref={navigation.rootRef} className="composite-image-navigator" aria-label="组合报告图片导航">
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
