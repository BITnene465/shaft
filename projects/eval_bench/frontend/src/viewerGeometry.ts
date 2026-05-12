import type { BenchmarkSampleSummary, RunSampleSummary } from "./api";
import type { ObjectKind } from "./viewerMetrics";
import { explicitLabelColor } from "./workspaceSettings";
import type {
  InstanceColorRole,
  LabelColors,
  OverlayColors
} from "./workspaceSettings";

const CANVAS_FIT_PADDING = 18;
const PRELOAD_RADIUS = 1;
const MAX_PRELOADED_IMAGE_URLS = 96;
const preloadedImageUrls = new Set<string>();

type IdleWindow = Window & {
  requestIdleCallback?: (callback: () => void, options?: { timeout: number }) => number;
  cancelIdleCallback?: (handle: number) => void;
};

export function resolveInstanceColor(
  label: string,
  status: "match" | "neutral" | "fn" | "fp",
  kind: ObjectKind,
  overlayColors: OverlayColors,
  labelColors: LabelColors
) {
  const role = instanceColorRole(kind, status);
  return explicitLabelColor(labelColors, label, role) ?? overlayColors[role];
}

function instanceColorRole(
  kind: ObjectKind,
  status: "match" | "neutral" | "fn" | "fp"
): InstanceColorRole {
  if (status === "fn" || status === "fp") {
    return status;
  }
  return kind;
}

export function arrowHeadPoints(points: number[][], lineWidth: number, scale = 1): number[][] | null {
  if (points.length < 2) {
    return null;
  }
  const segments = points
    .slice(0, -1)
    .map((start, index) => {
      const end = points[index + 1];
      return { start, end, length: Math.hypot(end[0] - start[0], end[1] - start[1]) };
    })
    .filter((segment) => segment.length > 1);
  if (segments.length === 0) {
    return null;
  }
  const totalLength = segments.reduce((total, segment) => total + segment.length, 0);
  const target = totalLength * 0.5;
  let accumulated = 0;
  let selected = segments[Math.floor(segments.length / 2)];
  for (const segment of segments) {
    if (accumulated + segment.length >= target) {
      selected = segment;
      break;
    }
    accumulated += segment.length;
  }
  const [x1, y1] = selected.start;
  const [x2, y2] = selected.end;
  const dx = x2 - x1;
  const dy = y2 - y1;
  const length = selected.length;
  const baseSize = Math.max(6, Math.min(18, lineWidth * 2.4)) * scale;
  const size = Math.min(baseSize, length * 0.22);
  if (length < size * 1.8) {
    return null;
  }
  const segmentOffset = Math.max(0, target - accumulated);
  const localRatio = clampNumber(segmentOffset / length, 0.32, 0.68);
  const unitX = dx / length;
  const unitY = dy / length;
  const tipX = x1 + dx * localRatio;
  const tipY = y1 + dy * localRatio;
  const baseX = tipX - unitX * size;
  const baseY = tipY - unitY * size;
  const wing = size * 0.45;
  return [
    [tipX, tipY],
    [baseX - unitY * wing, baseY + unitX * wing],
    [baseX + unitY * wing, baseY - unitX * wing]
  ];
}

export function preloadSampleImages(
  samples: Array<
    | Pick<RunSampleSummary, "index" | "image_url" | "image_preview_url">
    | Pick<BenchmarkSampleSummary, "index" | "image_url" | "image_preview_url">
  >,
  selectedIndex: number
) {
  const nearby = samples.filter(
    (sample) =>
      sample.index !== selectedIndex &&
      Math.abs(sample.index - selectedIndex) <= PRELOAD_RADIUS &&
      displayImageUrl(sample) &&
      !preloadedImageUrls.has(displayImageUrl(sample))
  );
  if (nearby.length === 0) {
    return () => undefined;
  }
  let cancelled = false;
  const loadNearby = () => {
    if (cancelled) {
      return;
    }
    for (const sample of nearby) {
      const imageUrl = displayImageUrl(sample);
      if (cancelled || preloadedImageUrls.has(imageUrl)) {
        continue;
      }
      preloadedImageUrls.add(imageUrl);
      if (preloadedImageUrls.size > MAX_PRELOADED_IMAGE_URLS) {
        const oldestUrl = preloadedImageUrls.values().next().value;
        if (oldestUrl) {
          preloadedImageUrls.delete(oldestUrl);
        }
      }
      const image = new Image();
      image.decoding = "async";
      image.src = imageUrl;
    }
  };
  const idleWindow = window as IdleWindow;
  const idleHandle =
    idleWindow.requestIdleCallback?.(loadNearby, { timeout: 1_200 }) ?? window.setTimeout(loadNearby, 250);
  return () => {
    cancelled = true;
    if (idleWindow.cancelIdleCallback) {
      idleWindow.cancelIdleCallback(idleHandle);
    } else {
      window.clearTimeout(idleHandle);
    }
  };
}

export function displayImageUrl(sample: { image_url: string; image_preview_url?: string | null }) {
  return sample.image_preview_url || sample.image_url;
}

export function computeFitSize(
  width: number,
  height: number,
  stageSize: { width: number; height: number }
) {
  const safeWidth = Math.max(1, width);
  const safeHeight = Math.max(1, height);
  const availableWidth = Math.max(1, stageSize.width - CANVAS_FIT_PADDING * 2);
  const availableHeight = Math.max(1, stageSize.height - CANVAS_FIT_PADDING * 2);
  const scale = Math.min(availableWidth / safeWidth, availableHeight / safeHeight);
  return {
    width: Math.max(1, Math.floor(safeWidth * scale)),
    height: Math.max(1, Math.floor(safeHeight * scale))
  };
}

export function normalizeBbox(value: unknown): [number, number, number, number] | null {
  if (Array.isArray(value) && value.length >= 4 && value.slice(0, 4).every(isFiniteNumber)) {
    const [x1, y1, x2, y2] = value.slice(0, 4) as number[];
    return normalizeBoxNumbers(x1, y1, x2, y2);
  }
  if (
    Array.isArray(value) &&
    value.length >= 2 &&
    Array.isArray(value[0]) &&
    Array.isArray(value[1]) &&
    value[0].length >= 2 &&
    value[1].length >= 2 &&
    [value[0][0], value[0][1], value[1][0], value[1][1]].every(isFiniteNumber)
  ) {
    return normalizeBoxNumbers(value[0][0], value[0][1], value[1][0], value[1][1]);
  }
  return null;
}

function normalizeBoxNumbers(
  x1: number,
  y1: number,
  x2: number,
  y2: number
): [number, number, number, number] | null {
  const left = Math.min(x1, x2);
  const top = Math.min(y1, y2);
  const right = Math.max(x1, x2);
  const bottom = Math.max(y1, y2);
  if (right <= left || bottom <= top) {
    return null;
  }
  return [left, top, right, bottom];
}

export function normalizePointList(value: unknown): number[][] | null {
  if (!Array.isArray(value)) {
    return null;
  }
  const points = value
    .filter((point): point is [number, number] => {
      return (
        Array.isArray(point) &&
        point.length >= 2 &&
        isFiniteNumber(point[0]) &&
        isFiniteNumber(point[1])
      );
    })
    .map((point) => [point[0], point[1]]);
  return points.length > 0 ? points : null;
}

export function boundsFromPoints(points: number[][] | null): [number, number, number, number] | null {
  if (!points || points.length === 0) {
    return null;
  }
  const xs = points.map((point) => point[0]);
  const ys = points.map((point) => point[1]);
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

export function clampNumber(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

export function normalizedWheelDelta(event: WheelEvent) {
  const lineHeightPx = 16;
  const pageHeightPx = 800;
  const pixelDelta =
    event.deltaMode === WheelEvent.DOM_DELTA_LINE
      ? event.deltaY * lineHeightPx
      : event.deltaMode === WheelEvent.DOM_DELTA_PAGE
        ? event.deltaY * pageHeightPx
        : event.deltaY;
  return clampNumber(pixelDelta, -180, 180);
}

export function clampPan(
  pan: { x: number; y: number },
  zoom: number,
  bounds: {
    viewportWidth: number;
    viewportHeight: number;
    contentWidth: number;
    contentHeight: number;
  }
) {
  const viewportWidth = bounds.viewportWidth;
  const viewportHeight = bounds.viewportHeight;
  const contentWidth = bounds.contentWidth * zoom;
  const contentHeight = bounds.contentHeight * zoom;
  const maxX = Math.max(0, Math.abs(contentWidth - viewportWidth) / 2);
  const maxY = Math.max(0, Math.abs(contentHeight - viewportHeight) / 2);
  return {
    x: clampNumber(pan.x, -maxX, maxX),
    y: clampNumber(pan.y, -maxY, maxY)
  };
}
