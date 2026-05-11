import { useEffect, useMemo, useState } from "react";
import type { CSSProperties } from "react";

export const DEFAULT_OVERLAY_COLORS = {
  gt: "#23c483",
  pred: "#f2a51f",
  fn: "#ff4d57",
  fp: "#d6409f",
  active: "#ffffff"
};

export type OverlayColorKey = keyof typeof DEFAULT_OVERLAY_COLORS;
export type OverlayColors = Record<OverlayColorKey, string>;
export type LabelColors = Record<string, string>;

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
  wheelZoomSensitivity: 0.00056,
  panSensitivity: 1
};

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
  overlayColors: "eval_bench_overlay_colors",
  overlayStyle: "eval_bench_overlay_style",
  labelColors: "eval_bench_label_colors",
  interaction: "eval_bench_interaction_settings",
  sidebarCollapsed: "eval_bench_sidebar_collapsed"
};

export function useWorkspaceSettings(labels: string[] = []) {
  const [overlayColors, setOverlayColors] = useState<OverlayColors>(() => loadOverlayColors());
  const [overlayStyle, setOverlayStyle] = useState<OverlayStyle>(() => loadOverlayStyle());
  const [labelColors, setLabelColors] = useState<LabelColors>(() => loadLabelColors());
  const [interactionSettings, setInteractionSettings] = useState<InteractionSettings>(() =>
    loadInteractionSettings()
  );
  const labelsKey = labels.join("|");
  const normalizedLabels = useMemo(
    () => uniqueValues([...labels, ...Object.keys(labelColors)]),
    [labelsKey, labelColors]
  );
  const mergedLabelColors = useMemo(
    () => buildLabelColors(normalizedLabels, labelColors),
    [labelColors, normalizedLabels]
  );
  const overlayVars = useMemo(
    () =>
      ({
        "--overlay-active": overlayColors.active,
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
    [overlayColors.active, overlayStyle]
  );

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.overlayColors, JSON.stringify(overlayColors));
  }, [overlayColors]);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.overlayStyle, JSON.stringify(overlayStyle));
  }, [overlayStyle]);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.labelColors, JSON.stringify(labelColors));
  }, [labelColors]);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.interaction, JSON.stringify(interactionSettings));
  }, [interactionSettings]);

  function updateOverlayColor(key: OverlayColorKey, value: string) {
    setOverlayColors((current) => ({ ...current, [key]: value }));
  }

  function updateOverlayStyle(key: OverlayStyleKey, value: number | string) {
    setOverlayStyle((current) => normalizeOverlayStyle({ ...current, [key]: value }));
  }

  function updateInteractionSetting(key: InteractionSettingKey, value: number) {
    setInteractionSettings((current) => normalizeInteractionSettings({ ...current, [key]: value }));
  }

  function updateLabelColor(label: string, value: string) {
    const normalizedLabel = label.trim();
    if (!normalizedLabel) {
      return;
    }
    setLabelColors((current) => ({ ...current, [normalizedLabel]: value }));
  }

  function removeLabelColor(label: string) {
    setLabelColors((current) => {
      const next = { ...current };
      delete next[label];
      return next;
    });
  }

  return {
    labels: normalizedLabels,
    overlayColors,
    overlayStyle,
    labelColors: mergedLabelColors,
    interactionSettings,
    overlayVars,
    updateOverlayColor,
    updateOverlayStyle,
    updateInteractionSetting,
    updateLabelColor,
    removeLabelColor,
    resetOverlayColors: () => setOverlayColors(DEFAULT_OVERLAY_COLORS),
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

export function fallbackLabelColor(label: string, index?: number) {
  const paletteIndex =
    index ?? Array.from(label).reduce((total, char) => total + char.charCodeAt(0), 0);
  return FALLBACK_LABEL_PALETTE[paletteIndex % FALLBACK_LABEL_PALETTE.length];
}

export function loadSplitSize(storageKey: string, fallback: number, min: number, max: number) {
  const raw = localStorage.getItem(storageKey);
  const parsed = raw ? Number(raw) : fallback;
  return Number.isFinite(parsed) ? clampNumber(parsed, min, max) : fallback;
}

function loadOverlayColors(): OverlayColors {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.overlayColors);
    if (!raw) {
      return DEFAULT_OVERLAY_COLORS;
    }
    const parsed = JSON.parse(raw) as Partial<OverlayColors>;
    return {
      ...DEFAULT_OVERLAY_COLORS,
      ...Object.fromEntries(
        Object.entries(parsed).filter(([, value]) => typeof value === "string" && isHexColor(value))
      )
    };
  } catch {
    return DEFAULT_OVERLAY_COLORS;
  }
}

function loadLabelColors(): LabelColors {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.labelColors);
    if (!raw) {
      return DEFAULT_LABEL_COLORS;
    }
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    const validEntries = Object.entries(parsed).filter(
      (entry): entry is [string, string] =>
        Boolean(entry[0]) && typeof entry[1] === "string" && isHexColor(entry[1])
    );
    return {
      ...DEFAULT_LABEL_COLORS,
      ...Object.fromEntries(validEntries)
    };
  } catch {
    return DEFAULT_LABEL_COLORS;
  }
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

function loadSidebarCollapsed() {
  return localStorage.getItem(STORAGE_KEYS.sidebarCollapsed) === "1";
}

function buildLabelColors(labels: string[], userColors: LabelColors): LabelColors {
  return Object.fromEntries(
    labels.map((label, index) => [label, userColors[label] ?? fallbackLabelColor(label, index)])
  );
}

function normalizeOverlayStyle(value: Partial<OverlayStyle>): OverlayStyle {
  return {
    boxStrokeWidth: workspaceNumber(value.boxStrokeWidth, 1, 10, DEFAULT_OVERLAY_STYLE.boxStrokeWidth),
    lineStrokeWidth: workspaceNumber(value.lineStrokeWidth, 1, 12, DEFAULT_OVERLAY_STYLE.lineStrokeWidth),
    pointRadius: workspaceNumber(value.pointRadius, 1, 12, DEFAULT_OVERLAY_STYLE.pointRadius),
    labelFontSize: workspaceNumber(value.labelFontSize, 9, 28, DEFAULT_OVERLAY_STYLE.labelFontSize),
    labelStrokeWidth: workspaceNumber(value.labelStrokeWidth, 0, 8, DEFAULT_OVERLAY_STYLE.labelStrokeWidth),
    labelBackgroundOpacity: workspaceNumber(
      value.labelBackgroundOpacity,
      0,
      1,
      DEFAULT_OVERLAY_STYLE.labelBackgroundOpacity
    ),
    boxFillOpacity: workspaceNumber(
      value.boxFillOpacity,
      0,
      0.5,
      DEFAULT_OVERLAY_STYLE.boxFillOpacity
    ),
    activeStrokeWidth: workspaceNumber(value.activeStrokeWidth, 2, 16, DEFAULT_OVERLAY_STYLE.activeStrokeWidth),
    opacity: workspaceNumber(value.opacity, 0.2, 1, DEFAULT_OVERLAY_STYLE.opacity),
    directionHeadScale: workspaceNumber(
      value.directionHeadScale,
      0.5,
      2.5,
      DEFAULT_OVERLAY_STYLE.directionHeadScale
    ),
    predLineStyle: value.predLineStyle === "solid" ? "solid" : "dashed"
  };
}

function normalizeInteractionSettings(value: Partial<InteractionSettings>): InteractionSettings {
  const minZoom = workspaceNumber(value.minZoom, 0.1, 1, DEFAULT_INTERACTION_SETTINGS.minZoom);
  const maxZoom = Math.max(
    minZoom + 0.25,
    workspaceNumber(value.maxZoom, 1, 20, DEFAULT_INTERACTION_SETTINGS.maxZoom)
  );
  return {
    minZoom,
    maxZoom,
    wheelZoomSensitivity: workspaceNumber(
      value.wheelZoomSensitivity,
      0.00005,
      0.003,
      DEFAULT_INTERACTION_SETTINGS.wheelZoomSensitivity
    ),
    panSensitivity: workspaceNumber(
      value.panSensitivity,
      0.2,
      3,
      DEFAULT_INTERACTION_SETTINGS.panSensitivity
    )
  };
}

function workspaceNumber(value: unknown, min: number, max: number, fallback: number) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? clampNumber(numeric, min, max) : fallback;
}

function isHexColor(value: string) {
  return /^#[0-9a-f]{6}$/i.test(value);
}

function uniqueValues(values: string[]) {
  return Array.from(new Set(values.filter(Boolean)));
}

function clampNumber(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}
