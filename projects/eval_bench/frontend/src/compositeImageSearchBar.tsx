import { LocateFixed, Search, X } from "lucide-react";

import type { ImageJumpItem, ImageMapBin } from "./compositeImageNavigationModel";
import { CompositeImageSearchPopover } from "./compositeImageSearchPopover";
import { SearchInputControl } from "./controlPrimitives";
import { IconActionButton } from "./ui";

import "./compositeImageSearchBar.css";

export function CompositeImageSearchBar({
  query,
  searchOpen,
  imageIndex,
  imageKey,
  filteredCount,
  imageCount,
  imageMapBins,
  visibleSearchResults,
  activeResultIndex,
  hiddenBeforeCount,
  hiddenAfterCount,
  hiddenCount,
  onQueryChange,
  onSearchOpenChange,
  onMoveActiveResult,
  onSearchResultWheel,
  onSelectActiveSearchResult,
  onActiveResultIndexChange,
  onJump,
  onLocateActive
}: {
  query: string;
  searchOpen: boolean;
  imageIndex: number;
  imageKey: string;
  filteredCount: number;
  imageCount: number;
  imageMapBins: ImageMapBin[];
  visibleSearchResults: ImageJumpItem[];
  activeResultIndex: number;
  hiddenBeforeCount: number;
  hiddenAfterCount: number;
  hiddenCount: number;
  onQueryChange: (query: string) => void;
  onSearchOpenChange: (open: boolean) => void;
  onMoveActiveResult: (delta: -1 | 1) => void;
  onSearchResultWheel: (event: WheelEvent) => void;
  onSelectActiveSearchResult: () => void;
  onActiveResultIndexChange: (index: number) => void;
  onJump: (index: number) => void;
  onLocateActive: () => void;
}) {
  return (
    <div className="image-navigator-search-row">
      <SearchInputControl
        className="image-navigator-search"
        icon={<Search size={14} />}
        label="搜索图片"
        value={query}
        placeholder="搜索图片或输入序号"
        onFocus={() => onSearchOpenChange(true)}
        onKeyDown={(event) => {
          if (!searchOpen) {
            return;
          }
          if (event.key === "ArrowDown") {
            event.preventDefault();
            onMoveActiveResult(1);
            return;
          }
          if (event.key === "ArrowUp") {
            event.preventDefault();
            onMoveActiveResult(-1);
            return;
          }
          if (event.key === "Enter") {
            event.preventDefault();
            onSelectActiveSearchResult();
          }
        }}
        onChange={(value) => {
          onQueryChange(value);
          onSearchOpenChange(Boolean(value.trim()));
        }}
        action={
          query ? (
            <IconActionButton
              title="清空图片搜索"
              icon={<X size={13} />}
              onClick={() => {
                onQueryChange("");
                onSearchOpenChange(false);
              }}
            />
          ) : null
        }
      />
      <IconActionButton title="定位当前图片" icon={<LocateFixed size={14} />} onClick={onLocateActive} />
      <span className="image-navigator-count">
        {filteredCount.toLocaleString()} / {imageCount.toLocaleString()}
      </span>
      {searchOpen ? (
        <CompositeImageSearchPopover
          imageIndex={imageIndex}
          imageKey={imageKey}
          filteredCount={filteredCount}
          imageCount={imageCount}
          imageMapBins={imageMapBins}
          visibleSearchResults={visibleSearchResults}
          activeResultIndex={activeResultIndex}
          hiddenBeforeCount={hiddenBeforeCount}
          hiddenAfterCount={hiddenAfterCount}
          hiddenCount={hiddenCount}
          onJump={onJump}
          onClose={() => onSearchOpenChange(false)}
          onResultWheel={onSearchResultWheel}
          onActiveResultIndexChange={onActiveResultIndexChange}
        />
      ) : null}
    </div>
  );
}
