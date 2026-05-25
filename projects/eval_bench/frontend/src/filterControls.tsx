import type { ReactNode } from "react";
import { Search, SlidersHorizontal } from "lucide-react";

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
  return (
    <section className="advanced-filter-bar" aria-label={title}>
      <div className="advanced-filter-head">
        <SlidersHorizontal size={15} />
        <div>
          <strong>{title}</strong>
          <span>{meta}</span>
        </div>
      </div>
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
      {actions ? <div className="advanced-filter-actions">{actions}</div> : null}
    </section>
  );
}
