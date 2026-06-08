import { useCallback, useEffect, useMemo, useState } from "react";
import type { CSSProperties } from "react";

const TYPOGRAPHY_STORAGE_KEY = "eval_bench_typography_settings";
const TYPOGRAPHY_STORAGE_VERSION_KEY = "eval_bench_typography_settings_version";
const CURRENT_TYPOGRAPHY_STORAGE_VERSION = "12px-default";
const TYPOGRAPHY_CHANGED_EVENT = "eval-bench-typography-changed";
const CUSTOM_FONT_LINK_ID = "eval-bench-custom-font";
const CUSTOM_FONT_FACE_STYLE_ID = "eval-bench-custom-font-face";
const LEGACY_DEFAULT_BASE_FONT_SIZES = [
  11.5,
  12.5,
  13.5,
  14.5,
  15.5,
  16.5,
  17,
  18,
  19,
  20,
  20.5,
  22,
  24,
  26
];

export type TypographySettings = {
  fontFamily: string;
  monoFontFamily: string;
  fontCssUrl: string;
  customFontName: string;
  customFontFileUrl: string;
  baseFontSize: number;
};

export type TypographyPreset = {
  id: string;
  label: string;
  description: string;
  settings: TypographySettings;
};

export const DEFAULT_TYPOGRAPHY_SETTINGS: TypographySettings = {
  fontFamily:
    '"IBM Plex Sans", "SF Pro Text", "Noto Sans SC", ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
  monoFontFamily:
    '"JetBrains Mono", "SFMono-Regular", Consolas, "Liberation Mono", ui-monospace, monospace',
  fontCssUrl: "",
  customFontName: "",
  customFontFileUrl: "",
  baseFontSize: 12
};

export const TYPOGRAPHY_PRESETS: TypographyPreset[] = [
  {
    id: "dense-lab",
    label: "紧凑实验台",
    description: "比默认更紧凑，适合结果库、排行榜和组合报告长期浏览。",
    settings: {
      ...DEFAULT_TYPOGRAPHY_SETTINGS,
      baseFontSize: 11
    }
  },
  {
    id: "balanced-cn",
    label: "中文均衡",
    description: "中文阅读更稳，兼顾指标密度和长 run 名称。",
    settings: {
      ...DEFAULT_TYPOGRAPHY_SETTINGS,
      fontFamily:
        '"Noto Sans SC", "Source Han Sans SC", "Microsoft YaHei UI", "PingFang SC", ui-sans-serif, system-ui, sans-serif',
      monoFontFamily:
        '"JetBrains Mono", "Cascadia Mono", "SFMono-Regular", Consolas, ui-monospace, monospace',
      baseFontSize: 12
    }
  },
  {
    id: "inspection",
    label: "检查模式",
    description: "字号略大，适合在可视化样本和设置页做细节检查。",
    settings: {
      ...DEFAULT_TYPOGRAPHY_SETTINGS,
      baseFontSize: 14
    }
  }
];

export function bootstrapTypographySettings() {
  applyTypographySettings(loadTypographySettings());
}

function useSyncedTypographySettingsState() {
  const [typographySettings, setTypographySettings] = useState<TypographySettings>(() =>
    loadTypographySettings()
  );

  useEffect(() => {
    function syncTypographySettings() {
      const nextSettings = loadTypographySettings();
      setTypographySettings((current) =>
        sameTypographySettings(current, nextSettings) ? current : nextSettings
      );
    }
    window.addEventListener("storage", syncTypographySettings);
    window.addEventListener(TYPOGRAPHY_CHANGED_EVENT, syncTypographySettings);
    return () => {
      window.removeEventListener("storage", syncTypographySettings);
      window.removeEventListener(TYPOGRAPHY_CHANGED_EVENT, syncTypographySettings);
    };
  }, []);

  return [typographySettings, setTypographySettings] as const;
}

export function useTypographyPreferenceSync() {
  const [typographySettings] = useSyncedTypographySettingsState();

  useEffect(() => {
    applyTypographySettings(typographySettings);
  }, [typographySettings]);
}

export function useTypographySettings() {
  const [typographySettings, setTypographySettings] = useSyncedTypographySettingsState();
  const typographyVars = useMemo(
    () => typographyStyleVars(typographySettings),
    [typographySettings]
  );

  useEffect(() => {
    applyTypographySettings(typographySettings);
    localStorage.setItem(TYPOGRAPHY_STORAGE_KEY, JSON.stringify(typographySettings));
    localStorage.setItem(TYPOGRAPHY_STORAGE_VERSION_KEY, CURRENT_TYPOGRAPHY_STORAGE_VERSION);
    window.dispatchEvent(new CustomEvent(TYPOGRAPHY_CHANGED_EVENT));
  }, [typographySettings]);

  const updateTypographySettings = useCallback((patch: Partial<TypographySettings>) => {
    setTypographySettings((current) => normalizeTypographySettings({ ...current, ...patch }));
  }, []);

  const resetTypographySettings = useCallback(() => {
    setTypographySettings(DEFAULT_TYPOGRAPHY_SETTINGS);
  }, []);

  return useMemo(
    () => ({
      typographySettings,
      typographyVars,
      updateTypographySettings,
      resetTypographySettings
    }),
    [typographySettings, typographyVars, updateTypographySettings, resetTypographySettings]
  );
}

export function applyTypographySettings(settings: TypographySettings) {
  const normalized = normalizeTypographySettings(settings);
  const root = document.documentElement;
  root.style.setProperty("--app-font-family", effectiveFontFamily(normalized));
  root.style.setProperty("--mono-font", normalized.monoFontFamily);
  root.style.setProperty("--app-base-font-size", `${normalized.baseFontSize}px`);
  updateCustomFontLink(normalized.fontCssUrl);
  updateCustomFontFaceStyle(normalized);
}

function typographyStyleVars(settings: TypographySettings): CSSProperties {
  const normalized = normalizeTypographySettings(settings);
  return {
    "--app-font-family": effectiveFontFamily(normalized),
    "--mono-font": normalized.monoFontFamily,
    "--app-base-font-size": `${normalized.baseFontSize}px`
  } as CSSProperties;
}

function loadTypographySettings(): TypographySettings {
  try {
    const raw = localStorage.getItem(TYPOGRAPHY_STORAGE_KEY);
    if (!raw) {
      return DEFAULT_TYPOGRAPHY_SETTINGS;
    }
    if (localStorage.getItem(TYPOGRAPHY_STORAGE_VERSION_KEY) !== CURRENT_TYPOGRAPHY_STORAGE_VERSION) {
      return DEFAULT_TYPOGRAPHY_SETTINGS;
    }
    const parsed = JSON.parse(raw) as Partial<TypographySettings>;
    const normalized = normalizeTypographySettings(parsed);
    return isStoredOldDefaultTypography(parsed, normalized)
      ? DEFAULT_TYPOGRAPHY_SETTINGS
      : normalized;
  } catch {
    return DEFAULT_TYPOGRAPHY_SETTINGS;
  }
}

function normalizeTypographySettings(value: Partial<TypographySettings>): TypographySettings {
  return {
    fontFamily: normalizedFontStack(value.fontFamily, DEFAULT_TYPOGRAPHY_SETTINGS.fontFamily),
    monoFontFamily: normalizedFontStack(value.monoFontFamily, DEFAULT_TYPOGRAPHY_SETTINGS.monoFontFamily),
    fontCssUrl: normalizedFontCssUrl(value.fontCssUrl),
    customFontName: normalizedFontName(value.customFontName),
    customFontFileUrl: normalizedFontCssUrl(value.customFontFileUrl),
    baseFontSize: normalizedFontSize(value.baseFontSize)
  };
}

function normalizedFontStack(value: unknown, fallback: string) {
  if (typeof value !== "string") {
    return fallback;
  }
  const normalized = value.trim().replace(/["'<>]/g, "");
  return normalized ? normalized.slice(0, 420) : fallback;
}

function normalizedFontCssUrl(value: unknown) {
  if (typeof value !== "string") {
    return "";
  }
  const normalized = value.trim();
  if (!normalized) {
    return "";
  }
  if (/^(https?:\/\/|\/|\.\/|\.\.\/)/i.test(normalized)) {
    return normalized.slice(0, 800);
  }
  return "";
}

function normalizedFontName(value: unknown) {
  if (typeof value !== "string") {
    return "";
  }
  return value.trim().replace(/[;"{}]/g, "").slice(0, 96);
}

function normalizedFontSize(value: unknown) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return DEFAULT_TYPOGRAPHY_SETTINGS.baseFontSize;
  }
  return Math.max(10, Math.min(20, numeric));
}

function sameTypographySettings(left: TypographySettings, right: TypographySettings) {
  return (
    left.fontFamily === right.fontFamily &&
    left.monoFontFamily === right.monoFontFamily &&
    left.fontCssUrl === right.fontCssUrl &&
    left.customFontName === right.customFontName &&
    left.customFontFileUrl === right.customFontFileUrl &&
    left.baseFontSize === right.baseFontSize
  );
}

function isStoredOldDefaultTypography(
  raw: Partial<TypographySettings>,
  normalized: TypographySettings
) {
  return (
    normalized.fontFamily === DEFAULT_TYPOGRAPHY_SETTINGS.fontFamily &&
    normalized.monoFontFamily === DEFAULT_TYPOGRAPHY_SETTINGS.monoFontFamily &&
    normalized.fontCssUrl === "" &&
    normalized.customFontName === "" &&
    normalized.customFontFileUrl === "" &&
    LEGACY_DEFAULT_BASE_FONT_SIZES.includes(Number(raw.baseFontSize))
  );
}

function effectiveFontFamily(settings: TypographySettings) {
  if (!settings.customFontName || !settings.customFontFileUrl) {
    return settings.fontFamily;
  }
  return `"${settings.customFontName}", ${settings.fontFamily}`;
}

function updateCustomFontLink(fontCssUrl: string) {
  const existing = document.getElementById(CUSTOM_FONT_LINK_ID) as HTMLLinkElement | null;
  if (!fontCssUrl) {
    existing?.remove();
    return;
  }
  const link = existing ?? document.createElement("link");
  link.id = CUSTOM_FONT_LINK_ID;
  link.rel = "stylesheet";
  link.href = fontCssUrl;
  if (!existing) {
    document.head.appendChild(link);
  }
}

function updateCustomFontFaceStyle(settings: TypographySettings) {
  const existing = document.getElementById(CUSTOM_FONT_FACE_STYLE_ID) as HTMLStyleElement | null;
  if (!settings.customFontName || !settings.customFontFileUrl) {
    existing?.remove();
    return;
  }
  const style = existing ?? document.createElement("style");
  style.id = CUSTOM_FONT_FACE_STYLE_ID;
  style.textContent = `@font-face{font-family:"${settings.customFontName}";src:url("${settings.customFontFileUrl}");font-display:swap;unicode-range:U+0000-024F,U+2000-206F,U+20A0-22FF,U+2460-24FF,U+25A0-25FF;}`;
  if (!existing) {
    document.head.appendChild(style);
  }
}
