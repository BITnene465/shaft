import type { CSSProperties } from "react";

import "./compositeMicroMeter.css";

export function CompositeMicroMeter({
  className = "",
  label,
  value,
  meta,
  progress,
  idle = false,
  ariaLabel
}: {
  className?: string;
  label: string;
  value?: string;
  meta: string;
  progress: number;
  idle?: boolean;
  ariaLabel: string;
}) {
  const clampedProgress = Math.max(0, Math.min(1, progress));
  const style = {
    "--composite-meter-progress": clampedProgress.toFixed(4)
  } as CSSProperties;
  return (
    <div
      className={["composite-micro-meter", value ? "has-value" : "", idle ? "idle" : "", className]
        .filter(Boolean)
        .join(" ")}
      style={style}
      aria-label={ariaLabel}
    >
      <span>{label}</span>
      {value ? <strong>{value}</strong> : null}
      <b aria-hidden="true">
        <i />
      </b>
      <em>{meta}</em>
    </div>
  );
}
