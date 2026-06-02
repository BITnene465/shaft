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

const LABEL_MAX_CHARS = 24;
const LABEL_MIN_WIDTH = 18;
const LABEL_RADIUS = 3;
const LABEL_OFFSET_X = 2;
const LABEL_OFFSET_Y = 4;
const LABEL_PAD_X = 4;
const LABEL_PAD_Y = 2;
const LABEL_CHAR_WIDTH_RATIO = 0.52;

export type CanvasObjectContextMenuRequest = {
  objectId: string;
  clientX: number;
  clientY: number;
};

export function compactOverlayLabel(label: string) {
  const normalized = label.trim();
  if (normalized.length <= LABEL_MAX_CHARS) {
    return normalized;
  }
  return `${normalized.slice(0, LABEL_MAX_CHARS - 1)}…`;
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
  const width = Math.max(
    LABEL_MIN_WIDTH,
    label.length * fontSize * LABEL_CHAR_WIDTH_RATIO + LABEL_PAD_X * 2
  );
  const height = fontSize + LABEL_PAD_Y * 2;
  const textX = anchorBox[0] + LABEL_OFFSET_X + LABEL_PAD_X;
  const textY = Math.max(fontSize + LABEL_PAD_Y + LABEL_OFFSET_Y, anchorBox[1] - LABEL_OFFSET_Y);
  return {
    height,
    radius: LABEL_RADIUS,
    textX,
    textY,
    width,
    x: textX - LABEL_PAD_X,
    y: textY - fontSize - LABEL_PAD_Y
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
  relatedObjectIds,
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
  return (
    <>
      {instances.map((instance, index) => {
        if (visibleLabels && !visibleLabels.has(instance.label)) {
          return null;
        }
        const objectId = `${kind}:${index}`;
        const bbox = normalizeBbox((instance as { bbox?: unknown }).bbox);
        const linePoints = normalizePointList(
          (instance as { linestrip?: unknown; line_strip?: unknown; points?: unknown }).linestrip ??
            (instance as { line_strip?: unknown }).line_strip
        );
        const keypoints = normalizePointList((instance as { keypoints?: unknown }).keypoints);
        const anchorBox = bbox ?? boundsFromPoints(linePoints ?? keypoints);
        if (!bbox && (!linePoints || linePoints.length === 0) && (!keypoints || keypoints.length === 0)) {
          return null;
        }
        const status = errors.has(index)
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
        const label = compactOverlayLabel(instance.label);
        const labelBounds = anchorBox
          ? overlayLabelBounds({
              anchorBox,
              label,
              fontSize: overlayStyle.labelFontSize
            })
          : null;
        const active = objectId === activeObjectId;
        const related = !active && Boolean(relatedObjectIds?.has(objectId));
        const className = [
          "overlay-instance",
          kind,
          status,
          active ? "active" : "",
          related ? "related" : ""
        ]
          .filter(Boolean)
          .join(" ");
        return (
          <g
            key={objectId}
            className={className}
            style={{ "--instance-color": color } as React.CSSProperties}
            onPointerEnter={() => onHover?.(objectId)}
            onPointerLeave={() => onHover?.(null)}
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
              <rect x={bbox[0]} y={bbox[1]} width={bbox[2] - bbox[0]} height={bbox[3] - bbox[1]} />
            ) : null}
            {showBoxes && labelBounds ? (
              <g className="overlay-label">
                <title>{instance.label}</title>
                <rect
                  className="label-backplate"
                  x={labelBounds.x}
                  y={labelBounds.y}
                  width={labelBounds.width}
                  height={labelBounds.height}
                  rx={labelBounds.radius}
                />
                <text x={labelBounds.textX} y={labelBounds.textY}>
                  {label}
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
