export const DEFAULT_OVERLAY_COLORS = {
  gt: "#23c483",
  pred: "#f2a51f",
  fn: "#ff4d57",
  fp: "#d6409f",
  active: "#ffffff"
};

export type OverlayColorKey = keyof typeof DEFAULT_OVERLAY_COLORS;
export type OverlayColors = Record<OverlayColorKey, string>;
export type InstanceColorRole = Exclude<OverlayColorKey, "active">;
export type LabelColors = Record<string, Partial<Record<InstanceColorRole, string>>>;

export type OverlayStyle = {
  boxStrokeWidth: number;
  lineStrokeWidth: number;
  pointRadius: number;
  labelFontSize: number;
  labelStrokeWidth: number;
  labelBackgroundOpacity: number;
  boxFillOpacity: number;
  activeStrokeWidth: number;
  opacity: number;
  directionHeadScale: number;
  predLineStyle: "solid" | "dashed";
};

export type OverlayStyleKey = keyof OverlayStyle;
export type OverlayStyleNumberKey = Exclude<OverlayStyleKey, "predLineStyle">;

export type NumberSettingControl<Key extends string> = {
  key: Key;
  label: string;
  min: number;
  max: number;
  step: number;
  scale?: number;
};

export const DEFAULT_OVERLAY_STYLE: OverlayStyle = {
  boxStrokeWidth: 3,
  lineStrokeWidth: 4,
  pointRadius: 4,
  labelFontSize: 10,
  labelStrokeWidth: 0.45,
  labelBackgroundOpacity: 0.82,
  boxFillOpacity: 0.06,
  activeStrokeWidth: 6,
  opacity: 1,
  directionHeadScale: 1,
  predLineStyle: "dashed"
};

export const OVERLAY_STYLE_CONTROLS = [
  { key: "boxStrokeWidth", label: "框线宽", min: 1, max: 10, step: 0.5 },
  { key: "lineStrokeWidth", label: "骨架线宽", min: 1, max: 12, step: 0.5 },
  { key: "pointRadius", label: "点半径", min: 1, max: 12, step: 0.5 },
  { key: "labelFontSize", label: "标签字号", min: 7, max: 18, step: 1 },
  { key: "labelStrokeWidth", label: "标签描边", min: 0, max: 2, step: 0.05 },
  { key: "boxFillOpacity", label: "框填充", min: 0, max: 0.5, step: 0.02 },
  { key: "labelBackgroundOpacity", label: "标签底色", min: 0, max: 1, step: 0.05 },
  { key: "directionHeadScale", label: "箭头大小", min: 0.5, max: 2.5, step: 0.05 },
  { key: "opacity", label: "整体透明度", min: 0.2, max: 1, step: 0.05 },
  { key: "activeStrokeWidth", label: "高亮线宽", min: 2, max: 16, step: 0.5 }
] as const satisfies readonly NumberSettingControl<OverlayStyleNumberKey>[];

export const PRED_LINE_STYLE_OPTIONS = [
  { value: "dashed", label: "虚线" },
  { value: "solid", label: "实线" }
] as const;

export const INSTANCE_COLOR_ROLES = [
  { key: "gt", label: "GT" },
  { key: "pred", label: "Pred" },
  { key: "fn", label: "FN" },
  { key: "fp", label: "FP" }
] as const satisfies readonly { key: InstanceColorRole; label: string }[];

export type InteractionSettings = {
  minZoom: number;
  maxZoom: number;
  wheelZoomSensitivity: number;
  panSensitivity: number;
};

export type InteractionSettingKey = keyof InteractionSettings;

export const DEFAULT_INTERACTION_SETTINGS: InteractionSettings = {
  minZoom: 0.25,
  maxZoom: 8,
  wheelZoomSensitivity: 0.00032,
  panSensitivity: 1
};

export const INTERACTION_SETTING_CONTROLS = [
  {
    key: "wheelZoomSensitivity",
    label: "滚轮缩放灵敏度",
    min: 0.00005,
    max: 0.003,
    step: 0.00001,
    scale: 100000
  },
  { key: "panSensitivity", label: "拖拽平移灵敏度", min: 0.2, max: 3, step: 0.05 },
  { key: "minZoom", label: "最小缩放", min: 0.1, max: 1, step: 0.05 },
  { key: "maxZoom", label: "最大缩放", min: 1, max: 20, step: 0.25 }
] as const satisfies readonly NumberSettingControl<InteractionSettingKey>[];

export const SHORTCUT_ACTIONS = [
  { id: "viewer.resetViewport", group: "画布", label: "复位画布", defaultBinding: "F" },
  { id: "sample.previous", group: "样本", label: "上一个样本", defaultBinding: "[" },
  { id: "sample.next", group: "样本", label: "下一个样本", defaultBinding: "]" },
  { id: "selection.clear", group: "选择", label: "清除选中对象", defaultBinding: "Escape" },
  { id: "layer.toggleGt", group: "图层", label: "切换真值图层", defaultBinding: "G" },
  { id: "layer.togglePred", group: "图层", label: "切换预测图层", defaultBinding: "P" },
  { id: "geometry.toggleBoxes", group: "几何", label: "切换框", defaultBinding: "B" },
  { id: "geometry.toggleLines", group: "几何", label: "切换线", defaultBinding: "L" },
  { id: "geometry.toggleKeypoints", group: "几何", label: "切换点", defaultBinding: "K" }
] as const;

export type ShortcutActionId = (typeof SHORTCUT_ACTIONS)[number]["id"];
export type ShortcutBindings = Record<ShortcutActionId, string>;
export type ShortcutKeyboardEvent = Pick<
  KeyboardEvent,
  "altKey" | "ctrlKey" | "key" | "metaKey" | "shiftKey"
>;

export const DEFAULT_LABEL_COLORS: LabelColors = {};
export const FALLBACK_LABEL_PALETTE = [
  "#2563eb",
  "#16a34a",
  "#f97316",
  "#7c3aed",
  "#0891b2",
  "#db2777",
  "#65a30d",
  "#9333ea"
];

export const STORAGE_KEYS = {
  overlayStyle: "eval_bench_overlay_style",
  labelColors: "eval_bench_label_colors",
  interaction: "eval_bench_interaction_settings",
  shortcuts: "eval_bench_shortcuts",
  sidebarCollapsed: "eval_bench_sidebar_collapsed",
  viewerActiveLabels: "evalBench.viewer.layers.activeLabels",
  viewerShowGt: "evalBench.viewer.layers.showGt",
  viewerShowPred: "evalBench.viewer.layers.showPred",
  viewerShowBoxes: "evalBench.viewer.layers.showBoxes",
  viewerShowLines: "evalBench.viewer.layers.showLines",
  viewerShowKeypoints: "evalBench.viewer.layers.showKeypoints"
};

export const DEFAULT_SHORTCUT_BINDINGS = Object.fromEntries(
  SHORTCUT_ACTIONS.map((action) => [action.id, action.defaultBinding])
) as ShortcutBindings;

export const SHORTCUTS_CHANGED_EVENT = "eval-bench-shortcuts-changed";
