import type { CSSProperties, ReactNode } from "react";

import { imageProgressPercent, type ImageJumpItem } from "./compositeImageNavigationModel";
import { basename } from "./formatters";

import "./compositeImageJumpItem.css";

export function CompositeImageJumpIdentity({
  item,
  badge,
  compact
}: {
  item: ImageJumpItem;
  badge?: ReactNode;
  compact?: boolean;
}) {
  return (
    <span className={compact ? "image-jump-identity compact" : "image-jump-identity"}>
      {badge ? <span className="image-jump-identity-badge">{badge}</span> : null}
      <span className="image-jump-identity-index">{item.index + 1}</span>
      <span className="image-jump-identity-main">
        <strong>{basename(item.image)}</strong>
        <em>{item.image}</em>
      </span>
    </span>
  );
}

export function CompositeImageJumpSummary({
  item,
  imageCount,
  currentIndex,
  badge,
  compact = false,
  showPosition = false,
  showDelta = false
}: {
  item: ImageJumpItem;
  imageCount?: number;
  currentIndex?: number;
  badge?: ReactNode;
  compact?: boolean;
  showPosition?: boolean;
  showDelta?: boolean;
}) {
  const canShowPosition = showPosition && typeof imageCount === "number";
  const canShowDelta = showDelta && typeof currentIndex === "number";
  const delta = canShowDelta ? item.index - currentIndex : 0;
  return (
    <span
      className={[
        "image-jump-summary",
        compact ? "compact" : "",
        canShowPosition ? "with-position" : "",
        canShowDelta ? "with-delta" : ""
      ]
        .filter(Boolean)
        .join(" ")}
    >
      <CompositeImageJumpIdentity item={item} badge={badge} compact={compact} />
      {canShowPosition ? <CompositeImageJumpPosition item={item} imageCount={imageCount} /> : null}
      {canShowDelta ? (
        <CompositeImageJumpDelta current={item.index === currentIndex} delta={delta} />
      ) : null}
    </span>
  );
}

export function CompositeImageJumpPosition({
  item,
  imageCount
}: {
  item: ImageJumpItem;
  imageCount: number;
}) {
  return (
    <span
      className="image-jump-position"
      aria-hidden="true"
      style={
        {
          "--image-result-position": `${imageProgressPercent(item.index, imageCount)}%`
        } as CSSProperties
      }
    >
      <b />
    </span>
  );
}

export function CompositeImageJumpDelta({
  current,
  delta
}: {
  current: boolean;
  delta: number;
}) {
  return (
    <i className="image-jump-delta">
      {imageJumpDeltaLabel(current, delta)}
    </i>
  );
}

export function imageJumpDeltaLabel(current: boolean, delta: number) {
  if (current) {
    return "Current";
  }
  return delta > 0 ? `+${delta}` : String(delta);
}
