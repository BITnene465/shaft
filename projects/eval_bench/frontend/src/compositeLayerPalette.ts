export const LAYER_COLORS = [
  "var(--composite-layer-blue)",
  "var(--composite-layer-red)",
  "var(--composite-layer-green)",
  "var(--composite-layer-amber)",
  "var(--composite-layer-violet)",
  "var(--composite-layer-cyan)"
];
export const LAYER_UNAVAILABLE_COLOR = "var(--composite-layer-unavailable)";

export function layerColor(index: number) {
  return LAYER_COLORS[index % LAYER_COLORS.length];
}

export function layerAvailabilityColor(index: number, available: boolean) {
  return available ? layerColor(index) : LAYER_UNAVAILABLE_COLOR;
}
