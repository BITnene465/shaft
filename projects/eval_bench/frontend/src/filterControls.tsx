import { useEffect, useId, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { RotateCcw, Search, SlidersHorizontal, X } from "lucide-react";

import { ActionButton, PanelToggleButton } from "./ui";

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
    <label className={compact ? "filter-select compact" : "filter-select"}>
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)} title={label}>
        {values.map((item) => (
          <option key={item} value={item}>
            {labels?.[item] ?? item}
          </option>
        ))}
      </select>
    </label>
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
  const [open, setOpen] = useState(false);
  const panelId = useId();
  const rootRef = useRef<HTMLElement | null>(null);
  const activeCount = controls.filter((control) => {
    return control.value.trim() !== defaultFilterValue(control);
  }).length;
  const activeFilters = controls
    .filter((control) => control.value.trim() !== defaultFilterValue(control))
    .map((control) => ({
      id: control.id,
      label: control.label,
      value: displayFilterValue(control)
    }));
  const controlGroups = useMemo(() => groupAdvancedControls(controls), [controls]);
  const summary = activeCount > 0 ? `${activeCount} 个条件生效` : "未设条件";
  useEffect(() => {
    if (!open) {
      return;
    }
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setOpen(false);
      }
    }
    function onDocumentClick(event: MouseEvent) {
      if (rootRef.current && !rootRef.current.contains(event.target as Node)) {
        window.setTimeout(() => setOpen(false), 0);
      }
    }
    document.addEventListener("keydown", onKeyDown);
    document.addEventListener("click", onDocumentClick);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.removeEventListener("click", onDocumentClick);
    };
  }, [open]);
  function resetAdvancedFilters() {
    for (const control of controls) {
      if (control.type === "select") {
        control.onChange(defaultFilterValue(control));
      } else {
        control.onChange("");
      }
    }
  }
  return (
    <section
      ref={rootRef}
      className={open ? "advanced-filter-bar open" : "advanced-filter-bar"}
      aria-label={`${title}: ${meta}`}
    >
      <div className="advanced-filter-compact">
        <PanelToggleButton
          active={open}
          className="advanced-filter-head"
          aria-controls={panelId}
          aria-haspopup="dialog"
          onClick={() => setOpen((value) => !value)}
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
              <span className="advanced-filter-token" key={filter.id} title={`${filter.label}: ${filter.value}`}>
                {filter.label}: {filter.value}
              </span>
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
        <div className="advanced-filter-popover" id={panelId} role="dialog" aria-label={`${title} 条件`}>
          <div className="advanced-filter-popover-head">
            <div>
              <strong>{title}</strong>
              <span>{meta}</span>
            </div>
            <ActionButton
              variant="mini"
              className="advanced-filter-close"
              icon={<X size={13} />}
              onClick={() => setOpen(false)}
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
                  {group.controls.map((control) => renderAdvancedControl(control))}
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

function displayFilterValue(control: AdvancedFilterControl) {
  if (control.type === "select") {
    return control.labels?.[control.value] ?? control.value;
  }
  return control.value;
}

function renderAdvancedControl(control: AdvancedFilterControl) {
  if (control.type === "search") {
    return (
      <label className="search-box advanced-search-box" key={control.id}>
        <Search size={15} />
        <input
          value={control.value}
          onChange={(event) => control.onChange(event.target.value)}
          placeholder={control.placeholder}
          aria-label={control.label}
        />
      </label>
    );
  }
  if (control.type === "number") {
    return (
      <label className="filter-select compact advanced-number-box" key={control.id}>
        <span>{control.label}</span>
        <input
          type="number"
          min={control.min}
          max={control.max}
          step={control.step}
          value={control.value}
          onChange={(event) => control.onChange(event.target.value)}
          placeholder={control.placeholder}
          aria-label={control.label}
        />
      </label>
    );
  }
  return (
    <FilterSelect
      key={control.id}
      label={control.label}
      value={control.value}
      values={control.values}
      labels={control.labels}
      onChange={control.onChange}
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
