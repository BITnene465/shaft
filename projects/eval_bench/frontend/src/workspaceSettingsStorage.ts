import type { SetStateAction } from "react";

import {
  DEFAULT_INTERACTION_SETTINGS,
  DEFAULT_LABEL_COLORS,
  DEFAULT_OVERLAY_STYLE,
  DEFAULT_SHORTCUT_BINDINGS,
  FALLBACK_LABEL_PALETTE,
  INSTANCE_COLOR_ROLES,
  INTERACTION_SETTING_CONTROLS,
  OVERLAY_STYLE_CONTROLS,
  SHORTCUT_ACTIONS,
  STORAGE_KEYS
} from "./workspaceSettingsSchema";
import type {
  InstanceColorRole,
  InteractionSettingKey,
  InteractionSettings,
  LabelColors,
  NumberSettingControl,
  OverlayStyle,
  OverlayStyleNumberKey,
  ShortcutBindings,
  ShortcutKeyboardEvent,
  ThemeMode
} from "./workspaceSettingsSchema";

const OVERLAY_STYLE_CONTROL_MAP = Object.fromEntries(
  OVERLAY_STYLE_CONTROLS.map((control) => [control.key, control])
) as Record<OverlayStyleNumberKey, NumberSettingControl<OverlayStyleNumberKey>>;
const INTERACTION_SETTING_CONTROL_MAP = Object.fromEntries(
  INTERACTION_SETTING_CONTROLS.map((control) => [control.key, control])
) as Record<InteractionSettingKey, NumberSettingControl<InteractionSettingKey>>;

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
  return roundNumber(value * (control.scale ?? 1), control.precision ?? precisionFromStep(control.step * (control.scale ?? 1)));
}

export function settingValueFromControl(value: number, control: NumberSettingControl<string>) {
  return roundToStep(value / (control.scale ?? 1), control.step);
}

export function loadSplitSize(storageKey: string, fallback: number, min: number, max: number) {
  const raw = localStorage.getItem(storageKey);
  const parsed = raw ? Number(raw) : fallback;
  return Number.isFinite(parsed) ? clampNumber(parsed, min, max) : fallback;
}

export function loadLabelColors(): LabelColors {
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

export function loadOverlayStyle(): OverlayStyle {
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

export function loadInteractionSettings(): InteractionSettings {
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

export function loadShortcutBindings(): ShortcutBindings {
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

export function loadSidebarCollapsed() {
  return localStorage.getItem(STORAGE_KEYS.sidebarCollapsed) === "1";
}

export function loadThemeMode(): ThemeMode {
  const value = localStorage.getItem(STORAGE_KEYS.themeMode);
  return value === "dark" ? "dark" : "light";
}

export function applyThemeMode(themeMode: ThemeMode) {
  document.documentElement.dataset.theme = themeMode;
  document.documentElement.style.colorScheme = themeMode;
}

export function loadBooleanPreference(key: string, fallback: boolean) {
  const value = localStorage.getItem(key);
  if (value === "1") {
    return true;
  }
  if (value === "0") {
    return false;
  }
  return fallback;
}

export function loadActiveLabelPreference(labels: string[]) {
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

export function hasStoredActiveLabelPreference() {
  return localStorage.getItem(STORAGE_KEYS.viewerActiveLabels) !== null;
}

export function normalizeOverlayStyle(value: Partial<OverlayStyle>): OverlayStyle {
  const normalized: OverlayStyle = {
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
  return migrateLegacyOverlayLabelStyle(normalized, value);
}

export function migrateLegacyOverlayLabelStyle(
  normalized: OverlayStyle,
  raw: Partial<OverlayStyle>
): OverlayStyle {
  const usedLegacyLabelDefaults =
    raw.labelFontSize === 14 &&
    raw.labelStrokeWidth === 4 &&
    raw.labelBackgroundOpacity === 0.86;
  const usedPreviousCompactLabelDefaults =
    raw.labelFontSize === 11 &&
    raw.labelStrokeWidth === 0.9 &&
    raw.labelBackgroundOpacity === 0.64;
  const usedPreviousLightLabelDefaults =
    raw.labelFontSize === 10 &&
    raw.labelStrokeWidth === 0.45 &&
    raw.labelBackgroundOpacity === 0.82;
  if (
    !usedLegacyLabelDefaults &&
    !usedPreviousCompactLabelDefaults &&
    !usedPreviousLightLabelDefaults
  ) {
    return normalized;
  }
  return {
    ...normalized,
    labelFontSize: DEFAULT_OVERLAY_STYLE.labelFontSize,
    labelStrokeWidth: DEFAULT_OVERLAY_STYLE.labelStrokeWidth,
    labelBackgroundOpacity: DEFAULT_OVERLAY_STYLE.labelBackgroundOpacity
  };
}

export function normalizeInteractionSettings(value: Partial<InteractionSettings>): InteractionSettings {
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
  return workspaceNumber(value, control.min, control.max, DEFAULT_OVERLAY_STYLE[key], control.step);
}

function interactionNumber(value: unknown, key: InteractionSettingKey) {
  const control = INTERACTION_SETTING_CONTROL_MAP[key];
  return workspaceNumber(value, control.min, control.max, DEFAULT_INTERACTION_SETTINGS[key], control.step);
}

function workspaceNumber(value: unknown, min: number, max: number, fallback: number, step: number) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? roundToStep(clampNumber(numeric, min, max), step) : fallback;
}

function roundToStep(value: number, step: number) {
  const rounded = Math.round(value / step) * step;
  return roundNumber(rounded, precisionFromStep(step));
}

function roundNumber(value: number, precision: number) {
  const factor = 10 ** precision;
  return Math.round(value * factor) / factor;
}

function precisionFromStep(step: number) {
  const [, fraction = ""] = String(step).split(".");
  return Math.min(6, fraction.length);
}

function isHexColor(value: string) {
  return /^#[0-9a-f]{6}$/i.test(value);
}

export function uniqueValues(values: string[]) {
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

export function normalizeShortcutBindings(value: Partial<ShortcutBindings>): ShortcutBindings {
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
