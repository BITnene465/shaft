import { useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties, Dispatch, SetStateAction } from "react";

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
  labelFontSize: 14,
  labelStrokeWidth: 4,
  labelBackgroundOpacity: 0.86,
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
  { key: "labelFontSize", label: "标签字号", min: 9, max: 28, step: 1 },
  { key: "labelStrokeWidth", label: "标签描边", min: 0, max: 8, step: 0.5 },
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
type ShortcutKeyboardEvent = Pick<
  KeyboardEvent,
  "altKey" | "ctrlKey" | "key" | "metaKey" | "shiftKey"
>;

const DEFAULT_LABEL_COLORS: LabelColors = {};
const FALLBACK_LABEL_PALETTE = [
  "#2563eb",
  "#16a34a",
  "#f97316",
  "#7c3aed",
  "#0891b2",
  "#db2777",
  "#65a30d",
  "#9333ea"
];

const STORAGE_KEYS = {
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

const OVERLAY_STYLE_CONTROL_MAP = Object.fromEntries(
  OVERLAY_STYLE_CONTROLS.map((control) => [control.key, control])
) as Record<OverlayStyleNumberKey, NumberSettingControl<OverlayStyleNumberKey>>;
const INTERACTION_SETTING_CONTROL_MAP = Object.fromEntries(
  INTERACTION_SETTING_CONTROLS.map((control) => [control.key, control])
) as Record<InteractionSettingKey, NumberSettingControl<InteractionSettingKey>>;
const DEFAULT_SHORTCUT_BINDINGS = Object.fromEntries(
  SHORTCUT_ACTIONS.map((action) => [action.id, action.defaultBinding])
) as ShortcutBindings;
const SHORTCUTS_CHANGED_EVENT = "eval-bench-shortcuts-changed";

export function useWorkspaceSettings(labels: string[] = []) {
  const [overlayStyle, setOverlayStyle] = useState<OverlayStyle>(() => loadOverlayStyle());
  const [labelColors, setLabelColors] = useState<LabelColors>(() => loadLabelColors());
  const [interactionSettings, setInteractionSettings] = useState<InteractionSettings>(() =>
    loadInteractionSettings()
  );
  const overlayColors = DEFAULT_OVERLAY_COLORS;
  const labelsKey = labels.join("|");
  const normalizedLabels = useMemo(
    () => uniqueValues([...labels, ...Object.keys(labelColors)]),
    [labelsKey, labelColors]
  );
  const overlayVars = useMemo(
    () =>
      ({
        "--overlay-active": overlayColors.active,
        "--overlay-gt": overlayColors.gt,
        "--overlay-pred": overlayColors.pred,
        "--overlay-fn": overlayColors.fn,
        "--overlay-fp": overlayColors.fp,
        "--overlay-box-width": overlayStyle.boxStrokeWidth,
        "--overlay-line-width": overlayStyle.lineStrokeWidth,
        "--overlay-label-size": `${overlayStyle.labelFontSize}px`,
        "--overlay-label-stroke": `${overlayStyle.labelStrokeWidth}px`,
        "--overlay-label-bg-opacity": overlayStyle.labelBackgroundOpacity,
        "--overlay-box-fill-opacity": overlayStyle.boxFillOpacity,
        "--overlay-active-width": overlayStyle.activeStrokeWidth,
        "--overlay-opacity": overlayStyle.opacity,
        "--overlay-pred-dash": overlayStyle.predLineStyle === "solid" ? "none" : "8 5"
      }) as CSSProperties,
    [overlayColors, overlayStyle]
  );

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.overlayStyle, JSON.stringify(overlayStyle));
  }, [overlayStyle]);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.labelColors, JSON.stringify(labelColors));
  }, [labelColors]);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.interaction, JSON.stringify(interactionSettings));
  }, [interactionSettings]);

  function updateOverlayStyle(key: OverlayStyleKey, value: number | string) {
    setOverlayStyle((current) => normalizeOverlayStyle({ ...current, [key]: value }));
  }

  function updateInteractionSetting(key: InteractionSettingKey, value: number) {
    setInteractionSettings((current) => normalizeInteractionSettings({ ...current, [key]: value }));
  }

  function updateLabelColor(label: string, role: InstanceColorRole, value: string) {
    const key = labelColorKey(label);
    if (!key) {
      return;
    }
    setLabelColors((current) => ({
      ...current,
      [key]: {
        ...(current[key] ?? {}),
        [role]: value
      }
    }));
  }

  function removeLabelColor(label: string, role?: InstanceColorRole) {
    setLabelColors((current) => {
      const next = { ...current };
      const key = labelColorKey(label);
      if (!role) {
        delete next[key];
        return next;
      }
      const nextRoles = { ...(next[key] ?? {}) };
      delete nextRoles[role];
      if (Object.keys(nextRoles).length === 0) {
        delete next[key];
      } else {
        next[key] = nextRoles;
      }
      return next;
    });
  }

  return {
    labels: normalizedLabels,
    overlayColors,
    overlayStyle,
    labelColors,
    interactionSettings,
    overlayVars,
    updateOverlayStyle,
    updateInteractionSetting,
    updateLabelColor,
    removeLabelColor,
    resetOverlayStyle: () => setOverlayStyle(DEFAULT_OVERLAY_STYLE),
    resetInteractionSettings: () => setInteractionSettings(DEFAULT_INTERACTION_SETTINGS),
    resetLabelColors: () => setLabelColors(DEFAULT_LABEL_COLORS)
  };
}

export function useSidebarPreference() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => loadSidebarCollapsed());

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.sidebarCollapsed, sidebarCollapsed ? "1" : "0");
  }, [sidebarCollapsed]);

  return { sidebarCollapsed, setSidebarCollapsed };
}

export function useViewerLayerPreferences(labels: string[]) {
  const labelsKey = labels.join("|");
  const previousLabelsRef = useRef(labels);
  const hasStoredLabelPreferenceRef = useRef(hasStoredActiveLabelPreference());
  const [preferredLabels, setPreferredLabels] = useState<string[]>(() =>
    loadActiveLabelPreference(labels)
  );
  const [showGt, setShowGt] = useState(() => loadBooleanPreference(STORAGE_KEYS.viewerShowGt, true));
  const [showPred, setShowPred] = useState(() =>
    loadBooleanPreference(STORAGE_KEYS.viewerShowPred, true)
  );
  const [showBoxes, setShowBoxes] = useState(() =>
    loadBooleanPreference(STORAGE_KEYS.viewerShowBoxes, true)
  );
  const [showLines, setShowLines] = useState(() =>
    loadBooleanPreference(STORAGE_KEYS.viewerShowLines, true)
  );
  const [showKeypoints, setShowKeypoints] = useState(() =>
    loadBooleanPreference(STORAGE_KEYS.viewerShowKeypoints, true)
  );

  useEffect(() => {
    const previousLabels = previousLabelsRef.current;
    setPreferredLabels((current) =>
      reconcileViewerLabelPreference({
        current,
        labels,
        previousLabels,
        hasStoredPreference: hasStoredLabelPreferenceRef.current
      })
    );
    previousLabelsRef.current = labels;
  }, [labelsKey, labels]);

  const activeLabels = useMemo(() => {
    return visibleViewerLabels(preferredLabels, labels);
  }, [labelsKey, labels, preferredLabels]);

  const setActiveLabels: Dispatch<SetStateAction<string[]>> = (value) => {
    setPreferredLabels((current) => {
      return applyViewerVisibleLabelSelection(current, labels, value);
    });
  };

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.viewerActiveLabels, JSON.stringify(preferredLabels));
    hasStoredLabelPreferenceRef.current = true;
  }, [preferredLabels]);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.viewerShowGt, showGt ? "1" : "0");
  }, [showGt]);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.viewerShowPred, showPred ? "1" : "0");
  }, [showPred]);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.viewerShowBoxes, showBoxes ? "1" : "0");
  }, [showBoxes]);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.viewerShowLines, showLines ? "1" : "0");
  }, [showLines]);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.viewerShowKeypoints, showKeypoints ? "1" : "0");
  }, [showKeypoints]);

  return {
    activeLabels,
    setActiveLabels,
    showGt,
    setShowGt,
    showPred,
    setShowPred,
    showBoxes,
    setShowBoxes,
    showLines,
    setShowLines,
    showKeypoints,
    setShowKeypoints
  };
}

export function reconcileViewerLabelPreference({
  current,
  labels,
  previousLabels,
  hasStoredPreference
}: {
  current: string[];
  labels: string[];
  previousLabels: string[];
  hasStoredPreference: boolean;
}) {
  if (
    !hasStoredPreference &&
    previousLabels.length === 0 &&
    current.length === 0 &&
    labels.length > 0
  ) {
    return labels;
  }
  const previousLabelSet = new Set(previousLabels);
  const hadEveryPreviousLabel =
    !hasStoredPreference &&
    previousLabels.length > 0 &&
    previousLabels.every((label) => current.includes(label));
  const additions = hadEveryPreviousLabel
    ? labels.filter((label) => !previousLabelSet.has(label))
    : [];
  const nextPreference = uniqueValues([...current, ...additions]);
  const currentLabelSet = new Set(labels);
  if (labels.length > 0 && !nextPreference.some((label) => currentLabelSet.has(label))) {
    return uniqueValues([...nextPreference, ...labels]);
  }
  return nextPreference;
}

export function visibleViewerLabels(preferredLabels: string[], labels: string[]) {
  const labelSet = new Set(labels);
  return preferredLabels.filter((label) => labelSet.has(label));
}

export function applyViewerVisibleLabelSelection(
  currentPreference: string[],
  labels: string[],
  value: SetStateAction<string[]>
) {
  const labelSet = new Set(labels);
  const currentVisible = currentPreference.filter((label) => labelSet.has(label));
  const nextVisible = typeof value === "function" ? value(currentVisible) : value;
  const hiddenPreference = currentPreference.filter((label) => !labelSet.has(label));
  const scopedVisible = nextVisible.filter((label) => labelSet.has(label));
  return uniqueValues([...hiddenPreference, ...scopedVisible]);
}

export function useWorkspaceShortcuts() {
  const [bindings, setBindings] = useState<ShortcutBindings>(() => loadShortcutBindings());

  useEffect(() => {
    function reloadBindings() {
      setBindings(loadShortcutBindings());
    }
    window.addEventListener("storage", reloadBindings);
    window.addEventListener(SHORTCUTS_CHANGED_EVENT, reloadBindings);
    return () => {
      window.removeEventListener("storage", reloadBindings);
      window.removeEventListener(SHORTCUTS_CHANGED_EVENT, reloadBindings);
    };
  }, []);

  function persist(next: ShortcutBindings) {
    localStorage.setItem(STORAGE_KEYS.shortcuts, JSON.stringify(next));
    setBindings(next);
    window.dispatchEvent(new Event(SHORTCUTS_CHANGED_EVENT));
  }

  function updateShortcut(actionId: ShortcutActionId, binding: string) {
    persist(normalizeShortcutBindings({ ...bindings, [actionId]: normalizeShortcutBinding(binding) }));
  }

  function resetShortcut(actionId: ShortcutActionId) {
    updateShortcut(actionId, DEFAULT_SHORTCUT_BINDINGS[actionId]);
  }

  function resetShortcuts() {
    persist(DEFAULT_SHORTCUT_BINDINGS);
  }

  function actionForEvent(event: KeyboardEvent) {
    const key = shortcutEventBinding(event);
    return SHORTCUT_ACTIONS.find((action) => bindings[action.id] === key)?.id ?? null;
  }

  return {
    actions: SHORTCUT_ACTIONS,
    bindings,
    actionForEvent,
    updateShortcut,
    resetShortcut,
    resetShortcuts
  };
}

export function fallbackLabelColor(label: string, index?: number) {
  const key = labelColorKey(label);
  const paletteIndex =
    index ?? Array.from(key).reduce((total, char) => total + char.charCodeAt(0), 0);
  return FALLBACK_LABEL_PALETTE[paletteIndex % FALLBACK_LABEL_PALETTE.length];
}

export function labelColorKey(label: string) {
  return label.trim().toLowerCase();
}

export function explicitLabelColor(
  labelColors: LabelColors,
  label: string,
  role: InstanceColorRole
) {
  return labelColors[labelColorKey(label)]?.[role];
}

export function settingControlValue(value: number, control: NumberSettingControl<string>) {
  return value * (control.scale ?? 1);
}

export function settingValueFromControl(value: number, control: NumberSettingControl<string>) {
  return value / (control.scale ?? 1);
}

export function loadSplitSize(storageKey: string, fallback: number, min: number, max: number) {
  const raw = localStorage.getItem(storageKey);
  const parsed = raw ? Number(raw) : fallback;
  return Number.isFinite(parsed) ? clampNumber(parsed, min, max) : fallback;
}

function loadLabelColors(): LabelColors {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.labelColors);
    if (!raw) {
      return DEFAULT_LABEL_COLORS;
    }
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    const validEntries = Object.entries(parsed)
      .map(([label, value]) => [labelColorKey(label), normalizeLabelColorEntry(value)] as const)
      .filter((entry): entry is readonly [string, Partial<Record<InstanceColorRole, string>>] =>
        Boolean(entry[0]) && Object.keys(entry[1]).length > 0
      );
    return {
      ...DEFAULT_LABEL_COLORS,
      ...Object.fromEntries(validEntries)
    };
  } catch {
    return DEFAULT_LABEL_COLORS;
  }
}

function normalizeLabelColorEntry(value: unknown): Partial<Record<InstanceColorRole, string>> {
  if (typeof value === "string" && isHexColor(value)) {
    return Object.fromEntries(
      INSTANCE_COLOR_ROLES.map((role) => [role.key, value])
    ) as Partial<Record<InstanceColorRole, string>>;
  }
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return {};
  }
  return Object.fromEntries(
    INSTANCE_COLOR_ROLES.map((role) => [role.key, (value as Record<string, unknown>)[role.key]])
      .filter((entry): entry is [InstanceColorRole, string] => typeof entry[1] === "string" && isHexColor(entry[1]))
  ) as Partial<Record<InstanceColorRole, string>>;
}

function loadOverlayStyle(): OverlayStyle {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.overlayStyle);
    if (!raw) {
      return DEFAULT_OVERLAY_STYLE;
    }
    return normalizeOverlayStyle(JSON.parse(raw) as Partial<OverlayStyle>);
  } catch {
    return DEFAULT_OVERLAY_STYLE;
  }
}

function loadInteractionSettings(): InteractionSettings {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.interaction);
    if (!raw) {
      return DEFAULT_INTERACTION_SETTINGS;
    }
    return normalizeInteractionSettings(JSON.parse(raw) as Partial<InteractionSettings>);
  } catch {
    return DEFAULT_INTERACTION_SETTINGS;
  }
}

function loadShortcutBindings(): ShortcutBindings {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.shortcuts);
    if (!raw) {
      return DEFAULT_SHORTCUT_BINDINGS;
    }
    return normalizeShortcutBindings(JSON.parse(raw) as Partial<ShortcutBindings>);
  } catch {
    return DEFAULT_SHORTCUT_BINDINGS;
  }
}

function loadSidebarCollapsed() {
  return localStorage.getItem(STORAGE_KEYS.sidebarCollapsed) === "1";
}

function loadBooleanPreference(key: string, fallback: boolean) {
  const value = localStorage.getItem(key);
  if (value === "1") {
    return true;
  }
  if (value === "0") {
    return false;
  }
  return fallback;
}

function loadActiveLabelPreference(labels: string[]) {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.viewerActiveLabels);
    if (!raw) {
      return labels;
    }
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed) || parsed.some((item) => typeof item !== "string")) {
      return labels;
    }
    return uniqueValues(parsed);
  } catch {
    return labels;
  }
}

function hasStoredActiveLabelPreference() {
  return localStorage.getItem(STORAGE_KEYS.viewerActiveLabels) !== null;
}

function normalizeOverlayStyle(value: Partial<OverlayStyle>): OverlayStyle {
  return {
    boxStrokeWidth: overlayStyleNumber(value.boxStrokeWidth, "boxStrokeWidth"),
    lineStrokeWidth: overlayStyleNumber(value.lineStrokeWidth, "lineStrokeWidth"),
    pointRadius: overlayStyleNumber(value.pointRadius, "pointRadius"),
    labelFontSize: overlayStyleNumber(value.labelFontSize, "labelFontSize"),
    labelStrokeWidth: overlayStyleNumber(value.labelStrokeWidth, "labelStrokeWidth"),
    labelBackgroundOpacity: overlayStyleNumber(
      value.labelBackgroundOpacity,
      "labelBackgroundOpacity"
    ),
    boxFillOpacity: overlayStyleNumber(value.boxFillOpacity, "boxFillOpacity"),
    activeStrokeWidth: overlayStyleNumber(value.activeStrokeWidth, "activeStrokeWidth"),
    opacity: overlayStyleNumber(value.opacity, "opacity"),
    directionHeadScale: overlayStyleNumber(value.directionHeadScale, "directionHeadScale"),
    predLineStyle: value.predLineStyle === "solid" ? "solid" : "dashed"
  };
}

function normalizeInteractionSettings(value: Partial<InteractionSettings>): InteractionSettings {
  const minZoom = interactionNumber(value.minZoom, "minZoom");
  const maxZoom = Math.max(
    minZoom + 0.25,
    interactionNumber(value.maxZoom, "maxZoom")
  );
  return {
    minZoom,
    maxZoom,
    wheelZoomSensitivity: interactionNumber(value.wheelZoomSensitivity, "wheelZoomSensitivity"),
    panSensitivity: interactionNumber(value.panSensitivity, "panSensitivity")
  };
}

function overlayStyleNumber(value: unknown, key: OverlayStyleNumberKey) {
  const control = OVERLAY_STYLE_CONTROL_MAP[key];
  return workspaceNumber(value, control.min, control.max, DEFAULT_OVERLAY_STYLE[key]);
}

function interactionNumber(value: unknown, key: InteractionSettingKey) {
  const control = INTERACTION_SETTING_CONTROL_MAP[key];
  return workspaceNumber(value, control.min, control.max, DEFAULT_INTERACTION_SETTINGS[key]);
}

function workspaceNumber(value: unknown, min: number, max: number, fallback: number) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? clampNumber(numeric, min, max) : fallback;
}

function isHexColor(value: string) {
  return /^#[0-9a-f]{6}$/i.test(value);
}

function uniqueValues(values: string[]) {
  const seen = new Set<string>();
  const result: string[] = [];
  values.forEach((value) => {
    const label = value.trim();
    const key = labelColorKey(label);
    if (!label || seen.has(key)) {
      return;
    }
    seen.add(key);
    result.push(label);
  });
  return result;
}

function normalizeShortcutBindings(value: Partial<ShortcutBindings>): ShortcutBindings {
  return {
    ...DEFAULT_SHORTCUT_BINDINGS,
    ...Object.fromEntries(
      SHORTCUT_ACTIONS.map((action) => [
        action.id,
        normalizeShortcutBinding(value[action.id] ?? action.defaultBinding)
      ])
    )
  };
}

export function normalizeShortcutBinding(value: string) {
  if (value === " ") {
    return "Space";
  }
  const normalized = value.trim();
  if (!normalized) {
    return "";
  }
  const parts = normalized.split("+").map((part) => part.trim()).filter(Boolean);
  const key = normalizeShortcutKey(parts.pop() ?? "");
  if (!key || isShortcutModifierKey(key)) {
    return "";
  }
  const modifiers = new Set<string>(parts.map(normalizeShortcutModifier).filter(Boolean));
  const uniqueModifiers = ["Ctrl", "Alt", "Shift", "Meta"].filter((part) =>
    modifiers.has(part)
  );
  return [...uniqueModifiers, key].join("+");
}

export function shortcutEventBinding(event: ShortcutKeyboardEvent) {
  const key = normalizeShortcutKey(event.key);
  if (!key || isShortcutModifierKey(key)) {
    return "";
  }
  return normalizeShortcutBinding(
    [
      event.ctrlKey ? "Ctrl" : "",
      event.altKey ? "Alt" : "",
      event.shiftKey ? "Shift" : "",
      event.metaKey ? "Meta" : "",
      key
    ]
      .filter(Boolean)
      .join("+")
  );
}

function normalizeShortcutKey(value: string) {
  if (value === " ") {
    return "Space";
  }
  const normalized = value.trim();
  if (!normalized) {
    return "";
  }
  const lower = normalized.toLowerCase();
  if (lower === "esc") {
    return "Escape";
  }
  if (lower === "space") {
    return "Space";
  }
  if (normalized.length === 1) {
    return normalized.toUpperCase();
  }
  return normalized;
}

function normalizeShortcutModifier(value: string) {
  const lower = value.trim().toLowerCase();
  if (lower === "ctrl" || lower === "control") {
    return "Ctrl";
  }
  if (lower === "alt" || lower === "option") {
    return "Alt";
  }
  if (lower === "shift") {
    return "Shift";
  }
  if (lower === "meta" || lower === "cmd" || lower === "command") {
    return "Meta";
  }
  return "";
}

function isShortcutModifierKey(key: string) {
  return key === "Control" || key === "Ctrl" || key === "Shift" || key === "Alt" || key === "Meta";
}

function clampNumber(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}
