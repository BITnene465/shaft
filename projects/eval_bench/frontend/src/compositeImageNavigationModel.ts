export const IMAGE_RESULT_LIMIT = 18;
export const IMAGE_MAP_BIN_COUNT = 48;
export const NEIGHBOR_RADIUS = 3;

export type ImageJumpItem = {
  image: string;
  index: number;
};

export type ImageResultWindow = {
  items: ImageJumpItem[];
  offset: number;
  activeItemIndex: number;
  hiddenBeforeCount: number;
  hiddenAfterCount: number;
  hiddenCount: number;
};

export type ScrubPreview = ImageJumpItem & {
  delta: number;
  percent: number;
};

export type ImageMapBin = {
  key: string;
  start: number;
  end: number;
  midpoint: number;
  count: number;
  matchCount: number;
  active: boolean;
  intensity: number;
};

type ScrubPointerEvent = {
  clientX: number;
  currentTarget: {
    getBoundingClientRect: () => DOMRect;
  };
};

export function filterImageKeys(imageKeys: string[], query: string): ImageJumpItem[] {
  const normalized = query.trim().toLocaleLowerCase();
  return imageKeys
    .map((image, index) => ({ image, index }))
    .filter((item) => {
      if (!normalized) {
        return true;
      }
      return (
        item.image.toLocaleLowerCase().includes(normalized) ||
        String(item.index + 1).includes(normalized)
      );
    });
}

export function nearbyImageKeys(imageKeys: string[], activeIndex: number): ImageJumpItem[] {
  const start = Math.max(0, activeIndex - NEIGHBOR_RADIUS);
  const end = Math.min(imageKeys.length, activeIndex + NEIGHBOR_RADIUS + 1);
  return imageKeys.slice(start, end).map((image, offset) => ({
    image,
    index: start + offset
  }));
}

export function imageResultWindow(
  filteredImages: ImageJumpItem[],
  activeResultIndex: number,
  limit = IMAGE_RESULT_LIMIT
): ImageResultWindow {
  if (filteredImages.length === 0) {
    return {
      items: [],
      offset: 0,
      activeItemIndex: 0,
      hiddenBeforeCount: 0,
      hiddenAfterCount: 0,
      hiddenCount: 0
    };
  }
  const resolvedLimit = Math.max(1, limit);
  const activeIndex = Math.max(0, Math.min(filteredImages.length - 1, activeResultIndex));
  const maxOffset = Math.max(0, filteredImages.length - resolvedLimit);
  const offset = Math.max(0, Math.min(maxOffset, activeIndex - Math.floor(resolvedLimit / 2)));
  const items = filteredImages.slice(offset, offset + resolvedLimit);
  const hiddenBeforeCount = offset;
  const hiddenAfterCount = Math.max(0, filteredImages.length - offset - items.length);
  return {
    items,
    offset,
    activeItemIndex: activeIndex - offset,
    hiddenBeforeCount,
    hiddenAfterCount,
    hiddenCount: hiddenBeforeCount + hiddenAfterCount
  };
}

export function activeImageResultIndex(filteredImages: ImageJumpItem[], imageIndex: number) {
  const index = filteredImages.findIndex((item) => item.index === imageIndex);
  return index >= 0 ? index : 0;
}

export function buildImageMapBins({
  imageCount,
  activeIndex,
  filteredImages,
  binCount = IMAGE_MAP_BIN_COUNT
}: {
  imageCount: number;
  activeIndex: number;
  filteredImages: ImageJumpItem[];
  binCount?: number;
}): ImageMapBin[] {
  if (imageCount <= 0) {
    return [];
  }
  const resolvedBinCount = Math.max(1, Math.min(imageCount, binCount));
  const matchCounts = new Array(resolvedBinCount).fill(0);
  const matchedAll = filteredImages.length === imageCount;
  if (!matchedAll) {
    filteredImages.forEach((item) => {
      const binIndex = binIndexForImageIndex(item.index, imageCount, resolvedBinCount);
      matchCounts[binIndex] += 1;
    });
  }
  return Array.from({ length: resolvedBinCount }, (_, binIndex) => {
    const start = Math.floor((binIndex * imageCount) / resolvedBinCount);
    const end = Math.max(start, Math.ceil(((binIndex + 1) * imageCount) / resolvedBinCount) - 1);
    const count = Math.max(1, end - start + 1);
    const matchCount = matchedAll ? count : matchCounts[binIndex];
    return {
      key: `${start}_${end}`,
      start,
      end,
      midpoint: clampImageIndex(Math.round((start + end) / 2), imageCount),
      count,
      matchCount,
      active: activeIndex >= start && activeIndex <= end,
      intensity: count <= 0 ? 0 : matchCount / count
    };
  });
}

export function clampImageIndex(index: number, imageCount: number) {
  if (imageCount <= 0) {
    return 0;
  }
  return Math.max(0, Math.min(imageCount - 1, Math.floor(index)));
}

export function imageProgressPercent(index: number, imageCount: number) {
  if (imageCount <= 1) {
    return 0;
  }
  return Math.max(0, Math.min(100, (index / (imageCount - 1)) * 100));
}

export function indexFromScrubPointer(event: ScrubPointerEvent, imageCount: number) {
  const rect = event.currentTarget.getBoundingClientRect();
  const ratio = rect.width <= 0 ? 0 : (event.clientX - rect.left) / rect.width;
  return clampImageIndex(Math.round(ratio * Math.max(0, imageCount - 1)), imageCount);
}

function binIndexForImageIndex(index: number, imageCount: number, binCount: number) {
  if (imageCount <= 1 || binCount <= 1) {
    return 0;
  }
  return Math.max(0, Math.min(binCount - 1, Math.floor((index / imageCount) * binCount)));
}

export function previewFromScrubPointer(
  event: ScrubPointerEvent,
  imageKeys: string[],
  activeIndex = 0
): ScrubPreview | null {
  if (imageKeys.length === 0) {
    return null;
  }
  const index = indexFromScrubPointer(event, imageKeys.length);
  return {
    delta: index - clampImageIndex(activeIndex, imageKeys.length),
    image: imageKeys[index],
    index,
    percent: imageProgressPercent(index, imageKeys.length)
  };
}
