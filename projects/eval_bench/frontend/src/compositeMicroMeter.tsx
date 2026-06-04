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
      <b className="composite-meter-ring" aria-hidden="true">
        <i />
      </b>
      <span className="composite-meter-copy">
        <em>{label}</em>
        {value ? <strong>{value}</strong> : null}
        <small>{meta}</small>
      </span>
    </div>
  );
}
