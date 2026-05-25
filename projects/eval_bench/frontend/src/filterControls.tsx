import { useState } from "react";
import type { ReactNode } from "react";
import { RotateCcw, Search, SlidersHorizontal } from "lucide-react";

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
  const activeCount = controls.filter((control) => {
    return control.value.trim() !== defaultFilterValue(control);
  }).length;
  const summary = activeCount > 0 ? `${activeCount} 个条件生效` : "点击展开筛选";
  const hasActions = Boolean(actions) || activeCount > 0;
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
      className={open ? "advanced-filter-bar open" : "advanced-filter-bar"}
      aria-label={`${title}: ${meta}`}
    >
      <PanelToggleButton
        active={open}
        className="advanced-filter-head"
        onClick={() => setOpen((value) => !value)}
      >
        <SlidersHorizontal size={15} />
        <div>
          <strong>{title}</strong>
          <span>{summary}</span>
        </div>
      </PanelToggleButton>
      {hasActions ? (
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
      ) : null}
      {open ? (
        <div className="advanced-filter-controls">
          {controls.map((control) => {
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
          })}
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
