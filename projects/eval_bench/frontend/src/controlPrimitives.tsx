import type { InputHTMLAttributes, TextareaHTMLAttributes } from "react";

export function NumberSettingControl({
  label,
  value,
  min,
  max,
  step,
  onChange
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="number-setting-control">
      <span>
        {label}
        <strong>{Number.isInteger(value) ? value : value.toFixed(2)}</strong>
      </span>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

export function ColorControl({
  label,
  value,
  onChange
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <label className="color-control">
      <span>{label}</span>
      <input type="color" value={value} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
}

export type SelectOption = {
  value: string;
  label: string;
  disabled?: boolean;
};

type TextInputControlProps = Omit<
  InputHTMLAttributes<HTMLInputElement>,
  "onChange" | "value" | "type"
> & {
  label: string;
  value: string;
  className?: string;
  type?: "text" | "search" | "url" | "password";
  onChange: (value: string) => void;
};

export function TextInputControl({
  label,
  value,
  className,
  type = "text",
  onChange,
  ...props
}: TextInputControlProps) {
  return (
    <label className={className}>
      <span>{label}</span>
      <input
        {...props}
        type={type}
        value={value}
        onChange={(event) => onChange(event.target.value)}
      />
    </label>
  );
}

type NumberInputControlProps = Omit<
  InputHTMLAttributes<HTMLInputElement>,
  "onChange" | "value" | "type"
> & {
  label: string;
  value: number;
  className?: string;
  onChange: (value: number) => void;
};

export function NumberInputControl({
  label,
  value,
  className,
  onChange,
  ...props
}: NumberInputControlProps) {
  return (
    <label className={className}>
      <span>{label}</span>
      <input
        {...props}
        type="number"
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

type TextareaControlProps = Omit<
  TextareaHTMLAttributes<HTMLTextAreaElement>,
  "onChange" | "value"
> & {
  label: string;
  value: string;
  className?: string;
  onChange: (value: string) => void;
};

export function TextareaControl({
  label,
  value,
  className,
  onChange,
  ...props
}: TextareaControlProps) {
  return (
    <label className={className}>
      <span>{label}</span>
      <textarea value={value} onChange={(event) => onChange(event.target.value)} {...props} />
    </label>
  );
}

type StandaloneTextareaControlProps = Omit<
  TextareaHTMLAttributes<HTMLTextAreaElement>,
  "onChange" | "value"
> & {
  label: string;
  value: string;
  onChange: (value: string) => void;
};

export function StandaloneTextareaControl({
  label,
  value,
  onChange,
  ...props
}: StandaloneTextareaControlProps) {
  return (
    <textarea
      {...props}
      aria-label={props["aria-label"] ?? label}
      title={props.title ?? label}
      value={value}
      onChange={(event) => onChange(event.target.value)}
    />
  );
}

export function CheckboxFieldControl({
  label,
  checked,
  className,
  onChange
}: {
  label: string;
  checked: boolean;
  className?: string;
  onChange: (value: boolean) => void;
}) {
  return (
    <label className={className ? `checkbox-field ${className}` : "checkbox-field"}>
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
      <span>{label}</span>
    </label>
  );
}

export function FormSelectControl({
  label,
  value,
  options,
  disabled = false,
  required = false,
  className,
  hideLabel = false,
  onChange
}: {
  label: string;
  value: string;
  options: ReadonlyArray<SelectOption>;
  disabled?: boolean;
  required?: boolean;
  className?: string;
  hideLabel?: boolean;
  onChange: (value: string) => void;
}) {
  const labelClassName = [className, hideLabel ? "select-control-label-hidden" : ""]
    .filter(Boolean)
    .join(" ");
  return (
    <label className={labelClassName || undefined}>
      <span>{label}</span>
      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        disabled={disabled}
        required={required}
        title={label}
      >
        {options.map((option) => (
          <option key={option.value} value={option.value} disabled={option.disabled}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}

export function CompactSelectControl({
  label,
  value,
  options,
  disabled = false,
  dense = false,
  onChange
}: {
  label: string;
  value: string;
  options: ReadonlyArray<{ value: string; label: string }>;
  disabled?: boolean;
  dense?: boolean;
  onChange: (value: string) => void;
}) {
  return (
    <label className={dense ? "compact-select dense" : "compact-select"}>
      <span>{label}</span>
      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        disabled={disabled}
        title={label}
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}

export function FilterSelectControl({
  label,
  value,
  values,
  labels,
  compact = false,
  onChange
}: {
  label: string;
  value: string;
  values: string[];
  labels?: Record<string, string>;
  compact?: boolean;
  onChange: (value: string) => void;
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

export function ToggleButton({
  label,
  active,
  onChange
}: {
  label: string;
  active: boolean;
  onChange: (value: boolean) => void;
}) {
  return (
    <label className={active ? "control-check active" : "control-check"}>
      <input type="checkbox" checked={active} onChange={() => onChange(!active)} />
      {label}
    </label>
  );
}
