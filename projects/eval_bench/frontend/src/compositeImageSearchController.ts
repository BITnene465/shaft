import { useEffect, useMemo, useState } from "react";
import type { RefObject } from "react";

import type { CompositeSampleView } from "./api";
import {
  activeImageResultIndex,
  buildImageMapBins,
  filterImageKeys,
  imageResultWindow
} from "./compositeImageNavigationModel";
import { imageSearchWheelStep } from "./compositeImageSearchWheel";

export function useCompositeImageSearchController({
  composite,
  rootRef,
  jumpTo
}: {
  composite: CompositeSampleView;
  rootRef: RefObject<HTMLDivElement | null>;
  jumpTo: (index: number) => void;
}) {
  const [query, setQuery] = useState("");
  const [searchOpen, setSearchOpen] = useState(false);
  const [activeResultIndex, setActiveResultIndex] = useState(0);
  const filteredImages = useMemo(
    () => filterImageKeys(composite.image_keys, query),
    [composite.image_keys, query]
  );
  const resultWindow = useMemo(
    () => imageResultWindow(filteredImages, activeResultIndex),
    [activeResultIndex, filteredImages]
  );
  const imageMapBins = useMemo(
    () =>
      buildImageMapBins({
        imageCount: composite.image_count,
        activeIndex: composite.image_index,
        filteredImages
      }),
    [composite.image_count, composite.image_index, filteredImages]
  );

  useEffect(() => {
    setActiveResultIndex(activeImageResultIndex(filteredImages, composite.image_index));
  }, [composite.image_index, filteredImages, query]);

  useEffect(() => {
    if (!searchOpen) {
      return undefined;
    }
    function closeFromDocument(event: PointerEvent) {
      if (!rootRef.current?.contains(event.target as Node)) {
        setSearchOpen(false);
      }
    }
    document.addEventListener("pointerdown", closeFromDocument);
    return () => document.removeEventListener("pointerdown", closeFromDocument);
  }, [rootRef, searchOpen]);

  function moveActiveResult(delta: -1 | 1) {
    if (filteredImages.length === 0) {
      return;
    }
    setActiveResultIndex((current) =>
      (current + delta + filteredImages.length) % filteredImages.length
    );
  }

  function handleSearchResultWheel(event: WheelEvent) {
    if (filteredImages.length < 2) {
      return;
    }
    const step = imageSearchWheelStep(event);
    if (step === 0) {
      return;
    }
    moveActiveResult(step);
  }

  function selectActiveSearchResult() {
    const item = filteredImages[activeResultIndex];
    if (!item) {
      return;
    }
    jumpTo(item.index);
    setSearchOpen(false);
  }

  function focusCurrentImageSearch() {
    setSearchOpen(false);
    rootRef.current?.querySelector<HTMLInputElement>(".image-navigator-search input")?.focus();
  }

  return {
    query,
    searchOpen,
    imageIndex: composite.image_index,
    imageKey: composite.image_key,
    filteredCount: filteredImages.length,
    imageCount: composite.image_count,
    imageMapBins,
    visibleSearchResults: resultWindow.items,
    activeResultIndex: resultWindow.activeItemIndex,
    hiddenBeforeCount: resultWindow.hiddenBeforeCount,
    hiddenAfterCount: resultWindow.hiddenAfterCount,
    hiddenCount: resultWindow.hiddenCount,
    onQueryChange: setQuery,
    onSearchOpenChange: setSearchOpen,
    onMoveActiveResult: moveActiveResult,
    onSearchResultWheel: handleSearchResultWheel,
    onSelectActiveSearchResult: selectActiveSearchResult,
    onActiveResultIndexChange: (index: number) =>
      setActiveResultIndex(resultWindow.offset + index),
    onJump: jumpTo,
    onLocateActive: focusCurrentImageSearch
  };
}
