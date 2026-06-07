import { useEffect, useId, useMemo, useRef, useState } from "react";
import type { KeyboardEvent as ReactKeyboardEvent, ReactNode } from "react";
import { Check, RotateCcw, SlidersHorizontal, X } from "lucide-react";

import type { AdvancedFilterControl } from "./advancedFilterTypes";
import { renderAdvancedControl } from "./advancedFilterFields";
import {
  advancedFilterDirtyControlIds,
  advancedFilterValuesKey,
  applyAdvancedFilterValues,
  defaultFilterValue,
  displayFilterValue,
  groupAdvancedControls,
  resetAdvancedFilter,
  syncDraftValuesWithApplied
} from "./advancedFilterModel";
import {
  advancedFilterOpenStateKey,
  readAdvancedFilterDraftValues,
  readAdvancedFilterOpenState,
  writeAdvancedFilterDraftValues,
  writeAdvancedFilterOpenState
} from "./advancedFilterStorage";
import { ActionButton, PanelToggleButton } from "./ui";
import { DIALOG_FOCUSABLE_SELECTOR } from "./uiDialog";
export type { AdvancedFilterControl } from "./advancedFilterTypes";
export { FilterSelect } from "./advancedFilterFields";

import "./filterTheme.css";
import "./filterControls.css";

const ADVANCED_FILTER_CONTROL_FOCUS_SELECTOR = [
  ".advanced-filter-controls input:not([disabled])",
  ".advanced-filter-controls select:not([disabled])",
  ".advanced-filter-controls textarea:not([disabled])",
  ".advanced-filter-controls button:not([disabled])"
].join(",");

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
      const clickedInsideSelectMenu =
        event.target instanceof Element && event.target.closest('[data-select-popover-menu="true"]');
      if (rootRef.current && !clickedInsideRoot && !clickedInsideSelectMenu) {
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
              <section className="advanced-filter-group" data-filter-group={group.id} key={group.id}>
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
