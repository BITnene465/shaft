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
