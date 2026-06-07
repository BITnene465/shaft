import { useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties, Dispatch, SetStateAction } from "react";

import {
  DEFAULT_INTERACTION_SETTINGS,
  DEFAULT_LABEL_COLORS,
  DEFAULT_OVERLAY_COLORS,
  DEFAULT_OVERLAY_STYLE,
  DEFAULT_SHORTCUT_BINDINGS,
  SHORTCUT_ACTIONS,
  SHORTCUTS_CHANGED_EVENT,
  STORAGE_KEYS
} from "./workspaceSettingsSchema";
import type {
  InstanceColorRole,
  InteractionSettingKey,
  InteractionSettings,
  LabelColors,
  OverlayStyle,
  OverlayStyleKey,
  ShortcutActionId,
  ShortcutBindings
} from "./workspaceSettingsSchema";
import {
  applyThemeMode,
  applyViewerVisibleLabelSelection,
  hasStoredActiveLabelPreference,
  labelColorKey,
  loadActiveLabelPreference,
  loadBooleanPreference,
  loadInteractionSettings,
  loadLabelColors,
  loadOverlayStyle,
  loadShortcutBindings,
  loadSidebarCollapsed,
  loadThemeMode,
  loadSplitSize,
  normalizeInteractionSettings,
  normalizeOverlayStyle,
  normalizeShortcutBinding,
  normalizeShortcutBindings,
  shortcutEventBinding,
  uniqueValues,
  visibleViewerLabels
} from "./workspaceSettingsStorage";

export * from "./workspaceSettingsSchema";
export * from "./workspaceSettingsStorage";

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

export function bootstrapThemePreference() {
  applyThemeMode(loadThemeMode());
}

export function useThemePreference() {
  const [themeMode, setThemeMode] = useState(() => loadThemeMode());

  useEffect(() => {
    applyThemeMode(themeMode);
    localStorage.setItem(STORAGE_KEYS.themeMode, themeMode);
  }, [themeMode]);

  return {
    themeMode,
    toggleThemeMode: () => setThemeMode((current) => (current === "dark" ? "light" : "dark"))
  };
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
