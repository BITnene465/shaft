import type { ImageJumpItem } from "./compositeImageNavigationModel";
import { useCompositeImageNearbyRailController } from "./compositeImageNearbyRailController";
import { basename } from "./formatters";
import { ActionButton } from "./ui";

import "./compositeImageNearbyRail.css";

export function CompositeImageNearbyRail({
  imageCount,
  imageIndex,
  nearbyImages,
  onJump,
  onStep
}: {
  imageCount: number;
  imageIndex: number;
  nearbyImages: ImageJumpItem[];
  onJump: (index: number) => void;
  onStep: (delta: -1 | 1) => void;
}) {
  const rail = useCompositeImageNearbyRailController({
    imageCount,
    imageIndex,
    onJump,
    onStep
  });
  return (
    <div
      ref={rail.railRef}
      className={rail.dragging ? "image-nearby-rail dragging" : "image-nearby-rail"}
      role="listbox"
      aria-label="当前图片附近的索引轨道"
      onPointerDown={rail.onPointerDown}
      onPointerMove={rail.onPointerMove}
      onPointerUp={rail.onPointerUp}
      onPointerCancel={rail.onPointerCancel}
      onWheelCapture={rail.onWheelCapture}
    >
      {rail.dragHint ? (
        <output className={`image-nearby-drag-hint ${rail.dragHint}`} aria-live="polite">
          {rail.dragHint === "next" ? "Next image" : "Previous image"}
        </output>
      ) : null}
      <span className="image-nearby-axis" aria-hidden="true" />
      {nearbyImages.map((item) => {
        const active = item.index === imageIndex;
        const distance = Math.abs(item.index - imageIndex);
        return (
          <ActionButton
            variant="mini"
            compact
            className={[
              "image-nearby-card",
              active ? "active" : "",
              distance === 1 ? "adjacent" : "",
              distance > 1 ? "context" : ""
            ]
              .filter(Boolean)
              .join(" ")}
            key={`${item.image}_${item.index}`}
            role="option"
            aria-selected={active}
            title={`${item.index + 1}. ${item.image}`}
            onPointerDown={(event) => {
              rail.onPointerDown(event);
              event.stopPropagation();
            }}
            onClick={(event) => {
              if (rail.shouldSuppressClick()) {
                event.preventDefault();
                return;
              }
              onJump(item.index);
            }}
          >
            <span>{item.index + 1}</span>
            <strong>{basename(item.image)}</strong>
          </ActionButton>
        );
      })}
    </div>
  );
}
