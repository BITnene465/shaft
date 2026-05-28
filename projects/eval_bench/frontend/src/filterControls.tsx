import { useEffect, useId, useMemo, useRef, useState } from "react";
import type { KeyboardEvent as ReactKeyboardEvent, ReactNode } from "react";
import { Check, RotateCcw, Search, SlidersHorizontal, X } from "lucide-react";

import { FilterSelectControl, SearchInputControl, TextInputControl } from "./controlPrimitives";
import { ActionButton, DIALOG_FOCUSABLE_SELECTOR, PanelToggleButton } from "./ui";

const ADVANCED_FILTER_CONTROL_FOCUS_SELECTOR = [
  ".advanced-filter-controls input:not([disabled])",
  ".advanced-filter-controls select:not([disabled])",
  ".advanced-filter-controls textarea:not([disabled])",
  ".advanced-filter-controls button:not([disabled])"
].join(",");
const ADVANCED_FILTER_OPEN_STORAGE_PREFIX = "eval_bench_advanced_filter_open";

export function FilterSelect({
  label,
  value,
  values,
  labels,
  onChange,
  compact = false
}: {
  label: string;
  value: string;
  values: string[];
  labels?: Record<string, string>;
  onChange: (value: string) => void;
  compact?: boolean;
}) {
  return (
    <FilterSelectControl
      label={label}
      value={value}
      values={values}
      labels={labels}
      compact={compact}
      onChange={onChange}
    />
  );
}

export type AdvancedFilterControl =
  | {
      type: "search";
      id: string;
      label: string;
      value: string;
      placeholder?: string;
      onChange: (value: string) => void;
    }
  | {
      type: "text";
      id: string;
      label: string;
      value: string;
      placeholder?: string;
      onChange: (value: string) => void;
    }
  | {
      type: "number";
      id: string;
      label: string;
      value: string;
      min?: number;
      max?: number;
      step?: number;
      placeholder?: string;
      onChange: (value: string) => void;
    }
  | {
      type: "select";
      id: string;
      label: string;
      value: string;
      values: string[];
      labels?: Record<string, string>;
      onChange: (value: string) => void;
    };

export function AdvancedFilterBar({
  title,
  meta,
  controls,
  actions
}: {
  title: string;
  meta: string;
  controls: AdvancedFilterControl[];
  actions?: ReactNode;
}) {
  const openStateKey = advancedFilterOpenStateKey(title, controls);
  const draftStateKey = `${openStateKey}:draft`;
  const [open, setOpen] = useState(() => readAdvancedFilterOpenState(openStateKey));
  const [draftValues, setDraftValues] = useState<Record<string, string>>(() =>
    readAdvancedFilterDraftValues(draftStateKey, controls)
  );
  const panelId = useId();
  const rootRef = useRef<HTMLElement | null>(null);
  const popoverRef = useRef<HTMLDivElement | null>(null);
  const previouslyFocusedRef = useRef<HTMLElement | null>(null);
  const latestDraftValuesRef = useRef<Record<string, string>>(draftValues);
  const dirtyControlIdsRef = useRef<Set<string>>(
    advancedFilterDirtyControlIds(controls, draftValues)
  );
  const activeCount = controls.filter((control) => {
    return control.value.trim() !== defaultFilterValue(control);
  }).length;
  const activeFilters = controls
    .filter((control) => control.value.trim() !== defaultFilterValue(control))
    .map((control) => ({ control, value: displayFilterValue(control) }));
  const controlGroups = useMemo(() => groupAdvancedControls(controls), [controls]);
  const summary = activeCount > 0 ? `${activeCount} 个条件生效` : "未设条件";
  const appliedValuesKey = advancedFilterValuesKey(controls);
  const hasDraftChanges = controls.some((control) => draftValueFor(control) !== control.value);
  useEffect(() => {
    setOpen(readAdvancedFilterOpenState(openStateKey));
  }, [openStateKey]);
  useEffect(() => {
    const nextDraftValues = readAdvancedFilterDraftValues(draftStateKey, controls);
    dirtyControlIdsRef.current = advancedFilterDirtyControlIds(controls, nextDraftValues);
    latestDraftValuesRef.current = nextDraftValues;
    setDraftValues(nextDraftValues);
  }, [draftStateKey]);
  useEffect(() => {
    writeAdvancedFilterOpenState(openStateKey, open);
  }, [openStateKey, open]);
  useEffect(() => {
    const controlIds = new Set(controls.map((control) => control.id));
    for (const id of dirtyControlIdsRef.current) {
      if (!controlIds.has(id)) {
        dirtyControlIdsRef.current.delete(id);
      }
    }
    const nextDraftValues = syncDraftValuesWithApplied(
      controls,
      latestDraftValuesRef.current,
      dirtyControlIdsRef.current
    );
    latestDraftValuesRef.current = nextDraftValues;
    setDraftValues(nextDraftValues);
    writeAdvancedFilterDraftValues(draftStateKey, nextDraftValues, dirtyControlIdsRef.current);
  }, [appliedValuesKey, openStateKey]);
  useEffect(() => {
    if (!open) {
      return;
    }
    const focusTarget =
      popoverRef.current?.querySelector<HTMLElement>(ADVANCED_FILTER_CONTROL_FOCUS_SELECTOR) ??
      popoverRef.current?.querySelector<HTMLElement>(DIALOG_FOCUSABLE_SELECTOR);
    window.setTimeout(() => {
      (focusTarget ?? popoverRef.current)?.focus();
    }, 0);
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        closeAdvancedFilter();
        return;
      }
      if (event.key !== "Tab") {
        return;
      }
      const focusable = Array.from(
        popoverRef.current?.querySelectorAll<HTMLElement>(DIALOG_FOCUSABLE_SELECTOR) ?? []
      ).filter((element) => element.offsetParent !== null || element === document.activeElement);
      if (focusable.length === 0) {
        event.preventDefault();
        popoverRef.current?.focus();
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      const focusInsidePopover = popoverRef.current?.contains(document.activeElement);
      if (event.shiftKey && (!focusInsidePopover || document.activeElement === first)) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && (!focusInsidePopover || document.activeElement === last)) {
        event.preventDefault();
        first.focus();
      }
    }
    function onDocumentClick(event: MouseEvent) {
      const clickedInsideRoot = rootRef.current && event.composedPath().includes(rootRef.current);
      if (rootRef.current && !clickedInsideRoot) {
        window.setTimeout(() => closeAdvancedFilter({ restoreFocus: false }), 0);
      }
    }
    document.addEventListener("keydown", onKeyDown);
    document.addEventListener("click", onDocumentClick);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.removeEventListener("click", onDocumentClick);
    };
  }, [open]);
  function openAdvancedFilter() {
    previouslyFocusedRef.current = document.activeElement instanceof HTMLElement
      ? document.activeElement
      : null;
    setOpen(true);
  }
  function closeAdvancedFilter({ restoreFocus = true }: { restoreFocus?: boolean } = {}) {
    setOpen(false);
    if (restoreFocus) {
      window.setTimeout(() => {
        previouslyFocusedRef.current?.focus();
      }, 0);
    }
  }
  function toggleAdvancedFilter() {
    if (open) {
      closeAdvancedFilter();
      return;
    }
    openAdvancedFilter();
  }
  function resetAdvancedFilters() {
    const nextValues: Record<string, string> = {};
    for (const control of controls) {
      nextValues[control.id] = defaultFilterValue(control);
    }
    dirtyControlIdsRef.current.clear();
    latestDraftValuesRef.current = nextValues;
    writeAdvancedFilterDraftValues(draftStateKey, nextValues, dirtyControlIdsRef.current);
    setDraftValues(nextValues);
    applyAdvancedFilterValues(controls, nextValues);
  }
  function resetSingleAdvancedFilter(control: AdvancedFilterControl) {
    const nextValue = defaultFilterValue(control);
    dirtyControlIdsRef.current.delete(control.id);
    const nextValues = { ...latestDraftValuesRef.current, [control.id]: nextValue };
    latestDraftValuesRef.current = nextValues;
    writeAdvancedFilterDraftValues(draftStateKey, nextValues, dirtyControlIdsRef.current);
    setDraftValues(nextValues);
    resetAdvancedFilter(control);
  }
  function updateDraftValue(control: AdvancedFilterControl, value: string) {
    if (value === control.value) {
      dirtyControlIdsRef.current.delete(control.id);
    } else {
      dirtyControlIdsRef.current.add(control.id);
    }
    const nextValues = { ...latestDraftValuesRef.current, [control.id]: value };
    latestDraftValuesRef.current = nextValues;
    writeAdvancedFilterDraftValues(draftStateKey, nextValues, dirtyControlIdsRef.current);
    setDraftValues(nextValues);
  }
  function draftValueFor(control: AdvancedFilterControl) {
    return draftValues[control.id] ?? control.value;
  }
  function applyDraftFilters() {
    dirtyControlIdsRef.current.clear();
    writeAdvancedFilterDraftValues(
      draftStateKey,
      latestDraftValuesRef.current,
      dirtyControlIdsRef.current
    );
    applyAdvancedFilterValues(controls, latestDraftValuesRef.current);
  }
  function applyDraftFiltersFromKeyboard(event: ReactKeyboardEvent<HTMLDivElement>) {
    if (event.key !== "Enter" || !hasDraftChanges) {
      return;
    }
    const target = event.target;
    if (
      target instanceof HTMLButtonElement ||
      target instanceof HTMLTextAreaElement ||
      target instanceof HTMLAnchorElement
    ) {
      return;
    }
    event.preventDefault();
    applyDraftFilters();
  }
  return (
    <section
      ref={rootRef}
      className={[
        "advanced-filter-bar",
        open ? "open" : "",
        hasDraftChanges ? "dirty" : ""
      ].filter(Boolean).join(" ")}
      aria-label={`${title}: ${meta}`}
    >
      <div className="advanced-filter-compact">
        <PanelToggleButton
          active={open}
          className="advanced-filter-head"
          aria-controls={panelId}
          aria-haspopup="dialog"
          onClick={toggleAdvancedFilter}
        >
          <span className="advanced-filter-trigger-icon">
            <SlidersHorizontal size={15} />
          </span>
          <div>
            <strong>{title}</strong>
            <span>{summary}</span>
          </div>
        </PanelToggleButton>
        <div className="advanced-filter-summary" aria-live="polite">
          {activeFilters.length > 0 ? (
            activeFilters.slice(0, 4).map((filter) => (
              <ActionButton
                variant="mini"
                className="advanced-filter-token"
                key={filter.control.id}
                title={`清除 ${filter.control.label}: ${filter.value}`}
                aria-label={`清除 ${filter.control.label}: ${filter.value}`}
                icon={<X size={11} />}
                onClick={() => resetSingleAdvancedFilter(filter.control)}
              >
                <span>{filter.control.label}: {filter.value}</span>
              </ActionButton>
            ))
          ) : (
            <span className="advanced-filter-hint">{meta}</span>
          )}
          {activeFilters.length > 4 ? (
            <span className="advanced-filter-token muted">+{activeFilters.length - 4}</span>
          ) : null}
        </div>
        <div className="advanced-filter-actions">
          {actions}
          {activeCount > 0 ? (
            <ActionButton
              variant="mini"
              className="advanced-filter-clear"
              icon={<RotateCcw size={13} />}
              onClick={resetAdvancedFilters}
            >
              清空
            </ActionButton>
          ) : null}
        </div>
      </div>
      {open ? (
        <div
          ref={popoverRef}
          className="advanced-filter-popover"
          id={panelId}
          role="dialog"
          aria-label={`${title} 条件`}
          tabIndex={-1}
          onKeyDown={applyDraftFiltersFromKeyboard}
        >
          <div className="advanced-filter-popover-head">
            <div>
              <strong>{title}</strong>
              <span>{hasDraftChanges ? `${meta} / 有未应用修改` : meta}</span>
            </div>
            <ActionButton
              variant="mini"
              className="advanced-filter-apply"
              icon={<Check size={13} />}
              disabled={!hasDraftChanges}
              title="应用筛选条件，快捷键 Enter"
              aria-keyshortcuts="Enter"
              onClick={applyDraftFilters}
            >
              应用
            </ActionButton>
            <ActionButton
              variant="mini"
              className="advanced-filter-close"
              icon={<X size={13} />}
              onClick={() => closeAdvancedFilter()}
            >
              收起
            </ActionButton>
          </div>
          <div className="advanced-filter-directory">
            {controlGroups.map((group) => (
              <section className="advanced-filter-group" key={group.id}>
                <div className="advanced-filter-group-title">
                  <strong>{group.title}</strong>
                  <span>{group.controls.length.toLocaleString()} 项</span>
                </div>
                <div className="advanced-filter-controls">
                  {group.controls.map((control) =>
                    renderAdvancedControl(control, draftValueFor(control), (value) =>
                      updateDraftValue(control, value)
                    )
                  )}
                </div>
              </section>
            ))}
          </div>
        </div>
      ) : null}
    </section>
  );
}

function defaultFilterValue(control: AdvancedFilterControl) {
  if (control.type === "select") {
    return control.values.includes("all") ? "all" : control.values[0] ?? "";
  }
  return "";
}

function resetAdvancedFilter(control: AdvancedFilterControl) {
  control.onChange(defaultFilterValue(control));
}

function applyAdvancedFilterValues(
  controls: AdvancedFilterControl[],
  values: Record<string, string>,
) {
  for (const control of controls) {
    const nextValue = values[control.id] ?? control.value;
    if (nextValue !== control.value) {
      control.onChange(nextValue);
    }
  }
}

function displayFilterValue(control: AdvancedFilterControl) {
  if (control.type === "select") {
    return control.labels?.[control.value] ?? control.value;
  }
  return control.value;
}

function renderAdvancedControl(
  control: AdvancedFilterControl,
  value: string,
  onChange: (value: string) => void,
) {
  if (control.type === "search") {
    return (
      <SearchInputControl
        className="advanced-filter-search-control"
        icon={<Search size={15} />}
        key={control.id}
        label={control.label}
        value={value}
        onChange={onChange}
        placeholder={control.placeholder}
      />
    );
  }
  if (control.type === "number") {
    return (
      <TextInputControl
        className="advanced-filter-number-control"
        key={control.id}
        label={control.label}
        type="number"
        min={control.min}
        max={control.max}
        step={control.step}
        value={value}
        onChange={onChange}
        placeholder={control.placeholder}
      />
    );
  }
  if (control.type === "text") {
    return (
      <TextInputControl
        className="advanced-filter-text-control"
        key={control.id}
        label={control.label}
        value={value}
        onChange={onChange}
        placeholder={control.placeholder}
      />
    );
  }
  return (
    <FilterSelect
      key={control.id}
      label={control.label}
      value={value}
      values={control.values}
      labels={control.labels}
      onChange={onChange}
      compact
    />
  );
}

function groupAdvancedControls(controls: AdvancedFilterControl[]) {
  const groups = [
    { id: "query", title: "检索式", controls: [] as AdvancedFilterControl[] },
    { id: "scope", title: "范围目录", controls: [] as AdvancedFilterControl[] },
    { id: "rank", title: "排序与阈值", controls: [] as AdvancedFilterControl[] }
  ];
  for (const control of controls) {
    const normalized = `${control.id} ${control.label}`.toLowerCase();
    if (control.type === "search") {
      groups[0].controls.push(control);
    } else if (
      control.type === "number" ||
      normalized.includes("sort") ||
      normalized.includes("排序") ||
      normalized.includes("最低") ||
      normalized.includes("order")
    ) {
      groups[2].controls.push(control);
    } else {
      groups[1].controls.push(control);
    }
  }
  return groups.filter((group) => group.controls.length > 0);
}

function advancedFilterOpenStateKey(title: string, controls: AdvancedFilterControl[]) {
  const controlIds = controls.map((control) => control.id).join(",");
  return `${ADVANCED_FILTER_OPEN_STORAGE_PREFIX}:${title}:${controlIds}`;
}

function readAdvancedFilterOpenState(key: string) {
  if (typeof window === "undefined") {
    return false;
  }
  try {
    return window.localStorage.getItem(key) === "1";
  } catch {
    return false;
  }
}

function readAdvancedFilterDraftValues(
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

function writeAdvancedFilterDraftValues(
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

function writeAdvancedFilterOpenState(key: string, open: boolean) {
  if (typeof window === "undefined") {
    return;
  }
  try {
    window.localStorage.setItem(key, open ? "1" : "0");
  } catch {
    // Ignore storage failures; the filter still works for the current render.
  }
}

function advancedFilterValues(controls: AdvancedFilterControl[]) {
  return Object.fromEntries(controls.map((control) => [control.id, control.value])) as Record<string, string>;
}

function advancedFilterDirtyControlIds(
  controls: AdvancedFilterControl[],
  values: Record<string, string>,
) {
  return new Set(
    controls
      .filter((control) => (values[control.id] ?? control.value) !== control.value)
      .map((control) => control.id)
  );
}

function advancedFilterValuesKey(controls: AdvancedFilterControl[]) {
  return controls.map((control) => `${control.id}\u0000${control.value}`).join("\u0001");
}

function syncDraftValuesWithApplied(
  controls: AdvancedFilterControl[],
  currentDraftValues: Record<string, string>,
  dirtyControlIds: ReadonlySet<string>,
) {
  return Object.fromEntries(
    controls.map((control) => {
      return [
        control.id,
        dirtyControlIds.has(control.id)
          ? currentDraftValues[control.id] ?? control.value
          : control.value,
      ];
    })
  );
}
