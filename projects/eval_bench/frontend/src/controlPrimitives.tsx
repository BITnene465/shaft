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
