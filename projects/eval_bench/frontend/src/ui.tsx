import type { DetailsHTMLAttributes, ReactNode } from "react";

import { statusClassName, statusInfo } from "./statusModel";
import type { StatusDomain } from "./statusModel";
export * from "./uiActions";

export function MetricCard({
  icon,
  label,
  value
}: {
  icon: ReactNode;
  label: string;
  value: number;
}) {
  return (
    <div className="metric-card">
      <div className="metric-icon">{icon}</div>
      <div>
        <div className="metric-label">{label}</div>
        <div className="metric-value">{value.toLocaleString()}</div>
      </div>
    </div>
  );
}

export function PanelTitle({ title, meta }: { title: string; meta?: string }) {
  return (
    <div className="panel-title">
      <strong>{title}</strong>
      {meta ? <span>{meta}</span> : null}
    </div>
  );
}

export function SectionHeader({ title, subtitle }: { title: string; subtitle: string }) {
  return (
    <div className="section-header">
      <div>
        <h2>{title}</h2>
        <p>{subtitle}</p>
      </div>
    </div>
  );
}

export function EmptyState({ title, tone }: { title: string; tone?: "danger" }) {
  return <div className={tone === "danger" ? "empty-panel danger-text" : "empty-panel"}>{title}</div>;
}

export function Badge({ value, domain }: { value: string; domain?: StatusDomain }) {
  return <span className={statusClassName(value, domain)}>{statusInfo(value, domain).label}</span>;
}

export function ActionPanel({
  title,
  meta,
  children
}: {
  title: string;
  meta: string;
  children: ReactNode;
}) {
  return (
    <DisclosurePanel
      className="action-panel"
      summary={
        <>
          <span>{title}</span>
          <strong>{meta}</strong>
        </>
      }
    >
      {children}
    </DisclosurePanel>
  );
}

export function DisclosurePanel({
  className,
  summary,
  children,
  ...props
}: DetailsHTMLAttributes<HTMLDetailsElement> & {
  summary: ReactNode;
}) {
  return (
    <details {...props} className={className}>
      <summary>{summary}</summary>
      {children}
    </details>
  );
}

export function ConfigItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="config-item">
      <span>{label}</span>
      <strong title={value}>{value}</strong>
    </div>
  );
}
