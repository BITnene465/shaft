import { useCallback, useEffect, useRef, useState } from "react";

import { tileLevelForZoom } from "./viewerTileLayer";

const TILE_LOAD_IDLE_DELAY_MS = 700;

export function useViewerViewportTileLevel({
  width,
  height,
  imageTileUrlTemplate,
  imageTileSize
}: {
  width: number;
  height: number;
  imageTileUrlTemplate?: string | null;
  imageTileSize?: number | null;
}) {
  const tileLoadTimeoutRef = useRef<number | null>(null);
  const [tileLevel, setTileLevel] = useState<number | null>(null);

  const clearScheduledTileLevel = useCallback(() => {
    if (tileLoadTimeoutRef.current !== null) {
      window.clearTimeout(tileLoadTimeoutRef.current);
      tileLoadTimeoutRef.current = null;
    }
  }, []);

  useEffect(() => {
    return () => clearScheduledTileLevel();
  }, [clearScheduledTileLevel]);

  const resetTileLevel = useCallback(() => {
    clearScheduledTileLevel();
    setTileLevel(null);
  }, [clearScheduledTileLevel]);

  const scheduleTileLevelUpdate = useCallback((nextZoom: number) => {
    if (!imageTileUrlTemplate || !imageTileSize) {
      resetTileLevel();
      return;
    }
    clearScheduledTileLevel();
    const nextTileLevel = tileLevelForZoom({
      width,
      height,
      zoom: nextZoom,
      tileSize: imageTileSize
    });
    tileLoadTimeoutRef.current = window.setTimeout(() => {
      setTileLevel(nextTileLevel);
      tileLoadTimeoutRef.current = null;
    }, TILE_LOAD_IDLE_DELAY_MS);
  }, [height, imageTileSize, imageTileUrlTemplate, resetTileLevel, width]);

  return {
    resetTileLevel,
    scheduleTileLevelUpdate,
    tileLevel
  };
}
