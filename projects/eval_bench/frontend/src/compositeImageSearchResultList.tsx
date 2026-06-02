import { useRef } from "react";
import type { PointerEventHandler } from "react";

import type { ImageJumpItem } from "./compositeImageNavigationModel";
import { useImageSearchActiveScroll } from "./compositeImageSearchActiveScroll";
import { CompositeImageSearchResultItem } from "./compositeImageSearchResultItem";
import { CompositeImageSearchScanRail } from "./compositeImageSearchScanRail";
import { useImageSearchWheelCruise } from "./compositeImageSearchWheel";

export function CompositeImageSearchResultList({
  imageIndex,
  imageCount,
  visibleSearchResults,
  activeResultIndex,
  dragging,
  resultDragHandlers,
  shouldSuppressClick,
  onJump,
  onClose,
  onResultWheel,
  onActiveResultIndexChange
}: {
  imageIndex: number;
  imageCount: number;
  visibleSearchResults: ImageJumpItem[];
  activeResultIndex: number;
  dragging: boolean;
  resultDragHandlers: {
    onPointerDown: PointerEventHandler<HTMLElement>;
    onPointerMove: PointerEventHandler<HTMLElement>;
    onPointerUp: PointerEventHandler<HTMLElement>;
    onPointerCancel: PointerEventHandler<HTMLElement>;
  };
  shouldSuppressClick: () => boolean;
  onJump: (index: number) => void;
  onClose: () => void;
  onResultWheel: (event: WheelEvent) => void;
  onActiveResultIndexChange: (index: number) => void;
}) {
  const resultListRef = useRef<HTMLDivElement | null>(null);

  useImageSearchWheelCruise({
    elementRef: resultListRef,
    onResultWheel
  });

  useImageSearchActiveScroll({
    elementRef: resultListRef,
    activeResultIndex,
    resultCount: visibleSearchResults.length
  });

  return (
    <div
      ref={resultListRef}
      className={dragging ? "image-jump-results dragging" : "image-jump-results"}
      data-wheel-cruise="native"
      role="listbox"
      aria-label="图片搜索结果"
      {...resultDragHandlers}
    >
      <CompositeImageSearchScanRail
        activeResult={visibleSearchResults[activeResultIndex] ?? null}
        activeResultIndex={activeResultIndex}
        imageIndex={imageIndex}
        resultCount={visibleSearchResults.length}
        dragging={dragging}
      />
      {visibleSearchResults.length === 0 ? (
        <div className="image-jump-empty">没有匹配图片</div>
      ) : (
        visibleSearchResults.map((item, index) => {
          const active = index === activeResultIndex;
          return (
            <CompositeImageSearchResultItem
              key={`${item.image}_${item.index}`}
              item={item}
              imageIndex={imageIndex}
              imageCount={imageCount}
              windowIndex={index}
              active={active}
              shouldSuppressClick={shouldSuppressClick}
              onPreview={onActiveResultIndexChange}
              onJump={onJump}
              onClose={onClose}
            />
          );
        })
      )}
    </div>
  );
}
