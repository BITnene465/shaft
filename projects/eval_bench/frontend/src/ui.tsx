import {
  flexRender,
  getCoreRowModel,
  useReactTable
} from "@tanstack/react-table";
import type { ColumnDef } from "@tanstack/react-table";
import { useEffect, useId } from "react";
import type { ButtonHTMLAttributes, ReactNode } from "react";
import { X } from "lucide-react";

import { statusClassName, statusInfo } from "./statusModel";
import type { StatusDomain } from "./statusModel";

type ButtonVariant = "primary" | "secondary" | "mini";

function joinClassNames(...values: Array<string | false | null | undefined>) {
  return values.filter(Boolean).join(" ");
}

export function ActionButton({
  variant = "secondary",
  icon,
  compact,
  className,
  children,
  type = "button",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: ButtonVariant;
  icon?: ReactNode;
  compact?: boolean;
}) {
  return (
    <button
      {...props}
      type={type}
      className={joinClassNames(
        `${variant}-button`,
        compact && "compact",
        className,
      )}
    >
      {icon}
      {children}
    </button>
  );
}

export function CommandButton({
  variant = "primary",
  icon,
  children,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary";
  icon?: ReactNode;
}) {
  return (
    <ActionButton {...props} variant={variant} className="command-button" icon={icon}>
      <span>{children}</span>
    </ActionButton>
  );
}

export function IconActionButton({
  icon,
  title,
  dense = true,
  danger,
  className,
  type = "button",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  icon: ReactNode;
  title: string;
  dense?: boolean;
  danger?: boolean;
}) {
  return (
    <button
      {...props}
      type={type}
      title={title}
      aria-label={props["aria-label"] ?? title}
      className={joinClassNames(
        "icon-button",
        dense && "dense",
        danger && "danger",
        className,
      )}
    >
      {icon}
    </button>
  );
}

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

export function DataTable<T>({
  columns,
  data,
  emptyText,
  compact
}: {
  columns: ColumnDef<T>[];
  data: T[];
  emptyText: string;
  compact?: boolean;
}) {
  const table = useReactTable({ data, columns, getCoreRowModel: getCoreRowModel() });
  if (data.length === 0) {
    return <div className="empty-panel">{emptyText}</div>;
  }
  return (
    <div className={compact ? "table-shell compact" : "table-shell"}>
      <table>
        <thead>
          {table.getHeaderGroups().map((headerGroup) => (
            <tr key={headerGroup.id}>
              {headerGroup.headers.map((header) => (
                <th key={header.id}>
                  {header.isPlaceholder
                    ? null
                    : flexRender(header.column.columnDef.header, header.getContext())}
                </th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody>
          {table.getRowModel().rows.map((row) => (
            <tr key={row.id}>
              {row.getVisibleCells().map((cell) => (
                <td key={cell.id}>{flexRender(cell.column.columnDef.cell, cell.getContext())}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
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
    <details className="action-panel">
      <summary>
        <span>{title}</span>
        <strong>{meta}</strong>
      </summary>
      {children}
    </details>
  );
}

export function WorkspaceDialog({
  open,
  title,
  meta,
  wide,
  onClose,
  children
}: {
  open: boolean;
  title: string;
  meta?: string;
  wide?: boolean;
  onClose: () => void;
  children: ReactNode;
}) {
  const titleId = useId();
  useEffect(() => {
    if (!open) {
      return;
    }
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        onClose();
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose, open]);
  if (!open) {
    return null;
  }
  return (
    <div
      className="workspace-dialog-backdrop"
      onMouseDown={(event) => {
        if (event.currentTarget === event.target) {
          onClose();
        }
      }}
    >
      <section
        className={wide ? "workspace-dialog wide" : "workspace-dialog"}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
      >
        <header className="workspace-dialog-head">
          <div>
            <strong id={titleId}>{title}</strong>
            {meta ? <span>{meta}</span> : null}
          </div>
          <IconActionButton icon={<X size={14} />} title="关闭" onClick={onClose} />
        </header>
        <div className="workspace-dialog-body">{children}</div>
      </section>
    </div>
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
