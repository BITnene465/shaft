import React from "react";

import type { EvalInstance, RunSampleDetail } from "./api";
import type { ObjectKind } from "./viewerMetrics";
import {
  arrowHeadPoints,
  boundsFromPoints,
  normalizeBbox,
  normalizePointList,
  resolveInstanceColor
} from "./viewerGeometry";
import { recordViewerRenderMetric } from "./viewerRenderMetrics";
import { DEFAULT_OVERLAY_STYLE } from "./workspaceSettings";
import type { LabelColors, OverlayColors, OverlayStyle } from "./workspaceSettings";

const LABEL_MIN_WIDTH = 18;
const LABEL_RADIUS = 3;
const LABEL_OFFSET_X = 2;
const LABEL_OFFSET_Y = 4;
const LABEL_PAD_X = 4;
const LABEL_PAD_Y = 2;
const LABEL_CHAR_WIDTH_RATIO = 0.52;
const LABEL_MIN_FONT_SIZE = 6;
const LABEL_MAX_BOX_RATIO = 0.86;
const LABEL_MAX_WIDTH_FACTOR = 14;
const LABEL_MIN_WIDTH_FACTOR = 4.5;
type OverlayObjectStatus = "match" | "neutral" | "fn" | "fp";

export type CanvasObjectContextMenuRequest = {
  objectId: string;
  clientX: number;
  clientY: number;
};

export function compactOverlayLabel(label: string) {
  const normalized = label.trim();
  return normalized;
}

export function overlayLabelBounds({
  anchorBox,
  label,
  fontSize
}: {
  anchorBox: [number, number, number, number];
  label: string;
  fontSize: number;
}) {
  const fitted = fitOverlayLabel({
    anchorBox,
    fontSize,
    label
  });
  const width = Math.max(
    LABEL_MIN_WIDTH,
    fitted.label.length * fitted.fontSize * LABEL_CHAR_WIDTH_RATIO + LABEL_PAD_X * 2
  );
  const height = fitted.fontSize + LABEL_PAD_Y * 2;
  const textX = anchorBox[0] + LABEL_OFFSET_X + LABEL_PAD_X;
  const textY = Math.max(
    fitted.fontSize + LABEL_PAD_Y + LABEL_OFFSET_Y,
    anchorBox[1] - LABEL_OFFSET_Y
  );
  return {
    fontSize: fitted.fontSize,
    height,
    label: fitted.label,
    radius: LABEL_RADIUS,
    textX,
    textY,
    width,
    x: textX - LABEL_PAD_X,
    y: textY - fontSize - LABEL_PAD_Y
  };
}

function fitOverlayLabel({
  anchorBox,
  fontSize,
  label
}: {
  anchorBox: [number, number, number, number];
  fontSize: number;
  label: string;
}) {
  const normalized = compactOverlayLabel(label);
  const boxWidth = Math.max(1, anchorBox[2] - anchorBox[0]);
  const maxWidth = Math.max(
    fontSize * LABEL_MIN_WIDTH_FACTOR,
    Math.min(boxWidth * LABEL_MAX_BOX_RATIO, fontSize * LABEL_MAX_WIDTH_FACTOR)
  );
  const fullWidth = normalized.length * fontSize * LABEL_CHAR_WIDTH_RATIO + LABEL_PAD_X * 2;
  const fittedFontSize =
    fullWidth <= maxWidth
      ? fontSize
      : Math.max(
          LABEL_MIN_FONT_SIZE,
          Math.min(fontSize, (maxWidth - LABEL_PAD_X * 2) / Math.max(1, normalized.length * LABEL_CHAR_WIDTH_RATIO))
        );
  const characterBudget = Math.floor(
    (maxWidth - LABEL_PAD_X * 2) / Math.max(1, fittedFontSize * LABEL_CHAR_WIDTH_RATIO)
  );
  if (normalized.length <= characterBudget) {
    return {
      fontSize: fittedFontSize,
      label: normalized
    };
  }
  if (characterBudget <= 1) {
    return {
      fontSize: fittedFontSize,
      label: normalized.slice(0, 1)
    };
  }
  return {
    fontSize: fittedFontSize,
    label: `${normalized.slice(0, characterBudget - 1)}…`
  };
}

function InstanceLayer({
  instances,
  kind,
  diagnostics,
  visibleLabels,
  showBoxes = true,
  showLines = true,
  showKeypoints = true,
  activeObjectId = null,
  overlayColors,
  overlayStyle = DEFAULT_OVERLAY_STYLE,
  labelColors,
  onHover,
  onLock,
  onInspect,
  onObjectContextMenu
}: {
  instances: EvalInstance[];
  kind: ObjectKind;
  diagnostics: RunSampleDetail["diagnostics"];
  visibleLabels?: Set<string>;
  showBoxes?: boolean;
  showLines?: boolean;
  showKeypoints?: boolean;
  activeObjectId?: string | null;
  relatedObjectIds?: Set<string>;
  overlayColors: OverlayColors;
  overlayStyle?: OverlayStyle;
  labelColors: LabelColors;
  onHover?: (objectId: string | null) => void;
  onLock?: (objectId: string | null) => void;
  onInspect?: (objectId: string | null) => void;
  onObjectContextMenu?: (request: CanvasObjectContextMenuRequest) => void;
}) {
  recordViewerRenderMetric(`instanceLayer:${kind}`);
  const matched = new Set(
    (diagnostics?.matches ?? []).map((match) => (kind === "gt" ? match.gt_index : match.pred_index))
  );
  const errorItems =
    kind === "gt" ? diagnostics?.false_negatives ?? [] : diagnostics?.false_positives ?? [];
  const errors = new Set(errorItems.map((item) => item.index));
  const renderItems = instances
    .map((instance, index) => {
      if (visibleLabels && !visibleLabels.has(instance.label)) {
        return null;
      }
      const bbox = normalizeBbox((instance as { bbox?: unknown }).bbox);
      const linePoints = normalizePointList(
        (instance as { linestrip?: unknown; line_strip?: unknown; points?: unknown }).linestrip ??
          (instance as { line_strip?: unknown }).line_strip
      );
      const keypoints = normalizePointList((instance as { keypoints?: unknown }).keypoints);
      const anchorBox = bbox ?? boundsFromPoints(linePoints ?? keypoints);
      if (
        !anchorBox ||
        (!bbox && (!linePoints || linePoints.length === 0) && (!keypoints || keypoints.length === 0))
      ) {
        return null;
      }
      const area =
        Math.max(1, anchorBox[2] - anchorBox[0]) * Math.max(1, anchorBox[3] - anchorBox[1]);
      return {
        anchorBox,
        area,
        bbox,
        index,
        instance,
        keypoints,
        linePoints,
        objectId: `${kind}:${index}`
      };
    })
    .filter((item): item is NonNullable<typeof item> => item !== null)
    .sort((left, right) => right.area - left.area || left.index - right.index);
  return (
    <>
      {renderItems.map(({
        anchorBox,
        bbox,
        index,
        instance,
        keypoints,
        linePoints,
        objectId
      }) => {
        const status: OverlayObjectStatus = errors.has(index)
          ? kind === "gt"
            ? "fn"
            : "fp"
          : matched.has(index)
            ? "match"
            : "neutral";
        const color = resolveInstanceColor(instance.label, status, kind, overlayColors, labelColors);
        const directionHead =
          linePoints && linePoints.length >= 2
            ? arrowHeadPoints(
                linePoints,
                overlayStyle.lineStrokeWidth,
                overlayStyle.directionHeadScale
              )
            : null;
        const lineRadius = Math.max(overlayStyle.pointRadius, overlayStyle.lineStrokeWidth * 0.75);
        const labelBounds = anchorBox
          ? overlayLabelBounds({
              anchorBox,
              label: instance.label,
              fontSize: overlayStyle.labelFontSize
            })
          : null;
        const active = objectId === activeObjectId;
        const className = [
          "overlay-instance",
          kind,
          status,
          active ? "active" : ""
        ]
          .filter(Boolean)
          .join(" ");
        return (
          <g
            key={objectId}
            className={className}
            style={{ "--instance-color": color } as React.CSSProperties}
            onClick={(event) => {
              event.stopPropagation();
              onLock?.(objectId);
            }}
            onDoubleClick={(event) => {
              event.stopPropagation();
              onInspect?.(objectId);
            }}
            onContextMenu={(event) => {
              event.preventDefault();
              event.stopPropagation();
              onObjectContextMenu?.({
                objectId,
                clientX: event.clientX,
                clientY: event.clientY
              });
            }}
          >
            {anchorBox ? (
              <rect
                className="overlay-hitbox"
                x={anchorBox[0]}
                y={anchorBox[1]}
                width={anchorBox[2] - anchorBox[0]}
                height={anchorBox[3] - anchorBox[1]}
              />
            ) : null}
            {linePoints && linePoints.length >= 2 ? (
              <polyline
                className="overlay-hitline"
                points={linePoints.map((point) => `${point[0]},${point[1]}`).join(" ")}
              />
            ) : null}
            {showBoxes && bbox ? (
              <rect
                className="overlay-box"
                x={bbox[0]}
                y={bbox[1]}
                width={bbox[2] - bbox[0]}
                height={bbox[3] - bbox[1]}
              />
            ) : null}
            {showBoxes && labelBounds ? (
              <g
                className="overlay-label"
                onPointerEnter={() => onHover?.(objectId)}
                onPointerLeave={() => onHover?.(null)}
              >
                <title>{instance.label}</title>
                <rect
                  className="label-backplate"
                  x={labelBounds.x}
                  y={labelBounds.y}
                  width={labelBounds.width}
                  height={labelBounds.height}
                  rx={labelBounds.radius}
                />
                <text
                  x={labelBounds.textX}
                  y={labelBounds.textY}
                  fontSize={labelBounds.fontSize}
                >
                  {labelBounds.label}
                </text>
              </g>
            ) : null}
            {showLines && linePoints && linePoints.length >= 2 ? (
              <>
                <polyline points={linePoints.map((point) => `${point[0]},${point[1]}`).join(" ")} />
                <circle
                  className="line-endpoint start"
                  cx={linePoints[0][0]}
                  cy={linePoints[0][1]}
                  r={lineRadius}
                />
                <circle
                  className="line-endpoint end"
                  cx={linePoints[linePoints.length - 1][0]}
                  cy={linePoints[linePoints.length - 1][1]}
                  r={lineRadius}
                />
                {directionHead ? (
                  <polygon
                    className="direction-head"
                    points={directionHead.map((point) => `${point[0]},${point[1]}`).join(" ")}
                  />
                ) : null}
              </>
            ) : null}
            {showKeypoints && keypoints && keypoints.length > 0 ? (
              keypoints.map((point, pointIndex) => (
                <circle
                  key={`${objectId}-point-${pointIndex}`}
                  cx={point[0]}
                  cy={point[1]}
                  r={overlayStyle.pointRadius}
                />
              ))
            ) : null}
          </g>
        );
      })}
    </>
  );
}

export const MemoizedInstanceLayer = React.memo(InstanceLayer);
