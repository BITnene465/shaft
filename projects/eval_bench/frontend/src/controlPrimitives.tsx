import type {
  InputHTMLAttributes,
  ReactNode,
  TextareaHTMLAttributes
} from "react";

export {
  CompactSelectControl,
  FilterSelectControl,
  FormSelectControl
} from "./selectPopoverControl";
export type { SelectOption } from "./selectPopoverControl";

import "./controlPrimitiveStyles.css";

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
  const displayValue = formatNumberInputValue(value, step);
  return (
    <label className="number-setting-control">
      <span>{label}</span>
      <input
        type="number"
        value={displayValue}
        min={min}
        max={max}
        step={step}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

function formatNumberInputValue(value: number, step: number) {
  if (!Number.isFinite(value)) {
    return "";
  }
  const [, fraction = ""] = String(step).split(".");
  const precision = Math.min(6, fraction.length);
  return precision === 0 ? String(Math.round(value)) : value.toFixed(precision);
}

export function RangeSettingControl({
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
    <label className="range-setting-control">
      <span className="sr-only">{label}</span>
      <input
        type="range"
        value={value}
        min={min}
        max={max}
        step={step}
        aria-label={label}
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

type TextInputControlProps = Omit<
  InputHTMLAttributes<HTMLInputElement>,
  "onChange" | "value" | "type"
> & {
  label: string;
  value: string;
  className?: string;
  type?: "text" | "search" | "url" | "password" | "number";
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

type StandaloneTextInputControlProps = Omit<
  InputHTMLAttributes<HTMLInputElement>,
  "onChange" | "value"
> & {
  label: string;
  value: string;
  onChange: (value: string) => void;
};

export function StandaloneTextInputControl({
  label,
  value,
  onChange,
  ...props
}: StandaloneTextInputControlProps) {
  return (
    <input
      {...props}
      aria-label={props["aria-label"] ?? label}
      title={props.title ?? label}
      value={value}
      onChange={(event) => onChange(event.target.value)}
    />
  );
}

type SearchInputControlProps = Omit<
  InputHTMLAttributes<HTMLInputElement>,
  "onChange" | "value" | "type"
> & {
  label: string;
  value: string;
  className?: string;
  icon?: ReactNode;
  action?: ReactNode;
  onChange: (value: string) => void;
};

export function SearchInputControl({
  label,
  value,
  className,
  icon,
  action,
  onChange,
  ...props
}: SearchInputControlProps) {
  return (
    <div className={className}>
      {icon}
      <input
        {...props}
        type="search"
        aria-label={props["aria-label"] ?? label}
        value={value}
        onChange={(event) => onChange(event.target.value)}
      />
      {action}
    </div>
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

export function StandaloneCheckboxControl({
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
    <input
      className={className}
      aria-label={label}
      type="checkbox"
      checked={checked}
      onChange={(event) => onChange(event.target.checked)}
    />
  );
}

export function StandaloneColorControl({
  label,
  value,
  onChange
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <input
      aria-label={label}
      title={label}
      type="color"
      value={value}
      onChange={(event) => onChange(event.target.value)}
    />
  );
}

export function InlineColorControl({
  label,
  caption,
  value,
  onChange
}: {
  label: string;
  caption?: string;
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <label>
      <small>{caption ?? label}</small>
      <StandaloneColorControl label={label} value={value} onChange={onChange} />
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
