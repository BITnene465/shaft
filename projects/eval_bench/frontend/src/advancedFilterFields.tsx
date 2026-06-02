import { Search } from "lucide-react";

import { FilterSelectControl, SearchInputControl, TextInputControl } from "./controlPrimitives";
import type { AdvancedFilterControl } from "./advancedFilterTypes";

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

export function renderAdvancedControl(
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

