import type { MouseEventHandler } from "react";

import type { ImageJumpItem } from "./compositeImageNavigationModel";
import { CompositeImageJumpSummary } from "./compositeImageJumpItem";
import { ActionButton } from "./ui";

import "./compositeImageSearchResultItem.css";

export function CompositeImageSearchResultItem({
  item,
  imageIndex,
  imageCount,
  windowIndex,
  active,
  shouldSuppressClick,
  onPreview,
  onJump,
  onClose
}: {
  item: ImageJumpItem;
  imageIndex: number;
  imageCount: number;
  windowIndex: number;
  active: boolean;
  shouldSuppressClick: () => boolean;
  onPreview: (index: number) => void;
  onJump: (imageIndex: number) => void;
  onClose: () => void;
}) {
  const current = item.index === imageIndex;
  const delta = item.index - imageIndex;
  const direction = current ? "current" : delta > 0 ? "forward" : "backward";
  const handleClick: MouseEventHandler<HTMLButtonElement> = () => {
    if (shouldSuppressClick()) {
      return;
    }
    onJump(item.index);
    onClose();
  };

  return (
    <ActionButton
      variant="mini"
      compact
      role="option"
      aria-selected={active}
      className={[
        "image-jump-result",
        `direction-${direction}`,
        current ? "current" : "",
        active ? "active" : ""
      ]
        .filter(Boolean)
        .join(" ")}
      key={`${item.image}_${item.index}`}
      data-result-window-index={windowIndex}
      data-result-direction={direction}
      title={`${item.index + 1}. ${item.image}`}
      onMouseEnter={() => onPreview(windowIndex)}
      onFocus={() => onPreview(windowIndex)}
      onClick={handleClick}
    >
      <CompositeImageJumpSummary
        item={item}
        imageCount={imageCount}
        currentIndex={imageIndex}
        showPosition
        showDelta
      />
    </ActionButton>
  );
}
