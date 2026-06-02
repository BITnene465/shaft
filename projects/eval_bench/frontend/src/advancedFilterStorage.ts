import type { AdvancedFilterControl } from "./advancedFilterTypes";
import { advancedFilterValues } from "./advancedFilterModel";

const ADVANCED_FILTER_OPEN_STORAGE_PREFIX = "eval_bench_advanced_filter_open";

export function advancedFilterOpenStateKey(title: string, controls: AdvancedFilterControl[]) {
  const controlIds = controls.map((control) => control.id).join(",");
  return `${ADVANCED_FILTER_OPEN_STORAGE_PREFIX}:${title}:${controlIds}`;
}

export function readAdvancedFilterOpenState(key: string) {
  if (typeof window === "undefined") {
    return false;
  }
  try {
    return window.localStorage.getItem(key) === "1";
  } catch {
    return false;
  }
}

export function writeAdvancedFilterOpenState(key: string, open: boolean) {
  if (typeof window === "undefined") {
    return;
  }
  try {
    window.localStorage.setItem(key, open ? "1" : "0");
  } catch {
    // Ignore storage failures; the filter still works for the current render.
  }
}

export function readAdvancedFilterDraftValues(
  key: string,
  controls: AdvancedFilterControl[],
): Record<string, string> {
  if (typeof window === "undefined") {
    return advancedFilterValues(controls);
  }
  try {
    const raw = window.sessionStorage.getItem(key);
    const stored = raw ? JSON.parse(raw) as Record<string, unknown> : {};
    return Object.fromEntries(
      controls.map((control) => [
        control.id,
        typeof stored[control.id] === "string" ? stored[control.id] : control.value,
      ])
    ) as Record<string, string>;
  } catch {
    return advancedFilterValues(controls);
  }
}

export function writeAdvancedFilterDraftValues(
  key: string,
  values: Record<string, string>,
  dirtyControlIds: ReadonlySet<string>,
) {
  if (typeof window === "undefined") {
    return;
  }
  try {
    if (dirtyControlIds.size === 0) {
      window.sessionStorage.removeItem(key);
      return;
    }
    window.sessionStorage.setItem(key, JSON.stringify(values));
  } catch {
    // Ignore storage failures; drafts still work for the current render.
  }
}

