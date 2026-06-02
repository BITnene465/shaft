import { useMemo, useState } from "react";
import type {
  KeyboardEvent as ReactKeyboardEvent,
  PointerEvent as ReactPointerEvent
} from "react";

import type { ImageMapBin } from "./compositeImageNavigationModel";
import { usePointerSweepSelection } from "./compositePointerSweep";

export function useCompositeImageAtlasController({
  imageMapBins,
  onJump
}: {
  imageMapBins: ImageMapBin[];
  onJump: (index: number) => void;
}) {
  const [hoveredMapBin, setHoveredMapBin] = useState<ImageMapBin | null>(null);
  const binByKey = useMemo(
    () => new Map(imageMapBins.map((bin) => [bin.key, bin])),
    [imageMapBins]
  );

  function previewBin(bin: ImageMapBin) {
    setHoveredMapBin(bin);
  }

  function jumpToBin(bin: ImageMapBin) {
    onJump(bin.midpoint);
  }

  function binFromPointer(event: ReactPointerEvent<HTMLElement>) {
    const target = document
      .elementFromPoint(event.clientX, event.clientY)
      ?.closest<HTMLElement>("[data-image-map-bin-key]");
    if (!target || !event.currentTarget.contains(target)) {
      return null;
    }
    const key = target.dataset.imageMapBinKey;
    return key ? binByKey.get(key) ?? null : null;
  }

  const sweep = usePointerSweepSelection({
    resolveValueFromPointer: binFromPointer,
    onPreview: (bin) => {
      previewBin(bin);
      jumpToBin(bin);
    },
    onCommit: jumpToBin
  });

  function hoverAtlasBin(bin: ImageMapBin) {
    previewBin(bin);
  }

  function leaveAtlasBin() {
    if (!sweep.dragging) {
      setHoveredMapBin(null);
    }
  }

  function handleAtlasBinKeyDown(bin: ImageMapBin, event: ReactKeyboardEvent<HTMLElement>) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      jumpToBin(bin);
      return;
    }
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") {
      return;
    }
    event.preventDefault();
    const index = imageMapBins.findIndex((item) => item.key === bin.key);
    const delta = event.key === "ArrowRight" ? 1 : -1;
    const nextBin = imageMapBins[Math.max(0, Math.min(imageMapBins.length - 1, index + delta))];
    if (nextBin) {
      previewBin(nextBin);
      jumpToBin(nextBin);
    }
  }

  return {
    hoveredMapBin,
    atlasDragging: sweep.dragging,
    atlasSweepHandlers: sweep.pointerSweepHandlers,
    shouldSuppressAtlasClick: sweep.shouldSuppressClick,
    onAtlasBinPointerEnter: hoverAtlasBin,
    onAtlasBinPointerMove: hoverAtlasBin,
    onAtlasBinPointerLeave: leaveAtlasBin,
    onAtlasBinClick: jumpToBin,
    onAtlasBinKeyDown: handleAtlasBinKeyDown
  };
}
