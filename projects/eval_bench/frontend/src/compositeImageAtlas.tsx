import type {
  CSSProperties,
  KeyboardEvent as ReactKeyboardEvent,
  PointerEvent as ReactPointerEvent
} from "react";

import type { ImageMapBin } from "./compositeImageNavigationModel";
import { basename } from "./formatters";

import "./compositeImageAtlas.css";

export function CompositeImageAtlas({
  imageIndex,
  imageKey,
  filteredCount,
  imageMapBins,
  hoveredMapBin,
  atlasDragging,
  atlasSweepHandlers,
  shouldSuppressAtlasClick,
  onAtlasBinPointerEnter,
  onAtlasBinPointerMove,
  onAtlasBinPointerLeave,
  onAtlasBinClick,
  onAtlasBinKeyDown
}: {
  imageIndex: number;
  imageKey: string;
  filteredCount: number;
  imageMapBins: ImageMapBin[];
  hoveredMapBin: ImageMapBin | null;
  atlasDragging: boolean;
  atlasSweepHandlers: {
    onPointerDown: (event: ReactPointerEvent<HTMLElement>) => void;
    onPointerMove: (event: ReactPointerEvent<HTMLElement>) => void;
    onPointerUp: (event: ReactPointerEvent<HTMLElement>) => void;
    onPointerCancel: (event: ReactPointerEvent<HTMLElement>) => void;
  };
  shouldSuppressAtlasClick: () => boolean;
  onAtlasBinPointerEnter: (bin: ImageMapBin) => void;
  onAtlasBinPointerMove: (bin: ImageMapBin) => void;
  onAtlasBinPointerLeave: () => void;
  onAtlasBinClick: (bin: ImageMapBin) => void;
  onAtlasBinKeyDown: (bin: ImageMapBin, event: ReactKeyboardEvent<HTMLElement>) => void;
}) {
  return (
    <div
      className={atlasDragging ? "image-jump-atlas dragging" : "image-jump-atlas"}
      aria-label="图片空间索引"
    >
      <div className="image-jump-atlas-current">
        <span>{(imageIndex + 1).toLocaleString()}</span>
        <strong title={imageKey}>{basename(imageKey)}</strong>
        <em>
          {hoveredMapBin
            ? `${hoveredMapBin.start + 1}-${hoveredMapBin.end + 1} · ${hoveredMapBin.matchCount}/${hoveredMapBin.count}`
            : `${filteredCount.toLocaleString()} matches`}
        </em>
      </div>
      <div
        className="image-jump-map"
        role="listbox"
        aria-label="图片热点地图"
        onPointerDown={atlasSweepHandlers.onPointerDown}
        onPointerMove={atlasSweepHandlers.onPointerMove}
        onPointerUp={atlasSweepHandlers.onPointerUp}
        onPointerCancel={atlasSweepHandlers.onPointerCancel}
      >
        {imageMapBins.map((bin) => (
          <div
            className={[
              "image-map-bin",
              bin.active ? "active" : "",
              hoveredMapBin?.key === bin.key ? "hovered" : "",
              bin.matchCount > 0 ? "matched" : "empty"
            ]
              .filter(Boolean)
              .join(" ")}
            key={bin.key}
            role="option"
            aria-selected={bin.active}
            tabIndex={0}
            data-image-map-bin-key={bin.key}
            title={`${bin.start + 1}-${bin.end + 1} · ${bin.matchCount}/${bin.count}`}
            style={{ "--match-density": bin.intensity } as CSSProperties}
            onPointerEnter={() => onAtlasBinPointerEnter(bin)}
            onPointerMove={() => onAtlasBinPointerMove(bin)}
            onPointerLeave={onAtlasBinPointerLeave}
            onClick={(event) => {
              if (shouldSuppressAtlasClick()) {
                event.preventDefault();
                return;
              }
              onAtlasBinClick(bin);
            }}
            onKeyDown={(event) => onAtlasBinKeyDown(bin, event)}
          >
            <span />
          </div>
        ))}
      </div>
    </div>
  );
}
