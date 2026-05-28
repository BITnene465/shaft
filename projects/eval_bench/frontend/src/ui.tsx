import {
  flexRender,
  getCoreRowModel,
  useReactTable
} from "@tanstack/react-table";
import type { ColumnDef, RowData } from "@tanstack/react-table";
import { Link } from "@tanstack/react-router";
import { createElement, useEffect, useId, useRef } from "react";
import type {
  AnchorHTMLAttributes,
  ButtonHTMLAttributes,
  DetailsHTMLAttributes,
  ElementType,
  HTMLAttributes,
  ReactNode
} from "react";
import { AlertTriangle, X } from "lucide-react";

import { statusClassName, statusInfo } from "./statusModel";
import type { StatusDomain } from "./statusModel";

type ButtonVariant = "primary" | "secondary" | "mini";
type TableColumnWidth =
  | "select"
  | "actions"
  | "status"
  | "metric"
  | "number"
  | "date"
  | "id"
  | "compact"
  | "text"
  | "wide";
type TableColumnWrap = "nowrap" | "truncate" | "wrap";
type TableColumnAlign = "start" | "center" | "end";

declare module "@tanstack/react-table" {
  interface ColumnMeta<TData extends RowData, TValue> {
    width?: TableColumnWidth;
    wrap?: TableColumnWrap;
    align?: TableColumnAlign;
  }
}

export const DIALOG_FOCUSABLE_SELECTOR = [
  "a[href]",
  "button:not([disabled])",
  "textarea:not([disabled])",
  "input:not([disabled])",
  "select:not([disabled])",
  "[tabindex]:not([tabindex='-1'])"
].join(",");

function joinClassNames(...values: Array<string | false | null | undefined>) {
  return values.filter(Boolean).join(" ");
}

function tableColumnClassName(meta: ColumnDef<unknown>["meta"] | undefined) {
  return joinClassNames(
    "data-table-cell",
    meta?.width ? `table-col-${meta.width}` : "table-col-text",
    meta?.wrap ? `table-wrap-${meta.wrap}` : "table-wrap-truncate",
    meta?.align ? `table-align-${meta.align}` : "table-align-start"
  );
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

export function IconNavLink({
  icon,
  title,
  dense = true,
  className,
  ...props
}: {
  icon: ReactNode;
  title: string;
  dense?: boolean;
  className?: string;
  [key: string]: unknown;
}) {
  return createElement(
    Link as ElementType,
    {
      ...props,
      title,
      "aria-label": (props["aria-label"] as string | undefined) ?? title,
      className: joinClassNames("icon-button", dense && "dense", className)
    },
    icon,
  );
}

export function InlineNavLink({
  icon,
  children,
  className,
  ...props
}: {
  icon?: ReactNode;
  children: ReactNode;
  className?: string;
  [key: string]: unknown;
}) {
  return createElement(
    Link as ElementType,
    {
      ...props,
      className: joinClassNames("mini-link", className)
    },
    <>
      {icon}
      {children}
    </>,
  );
}

export function InlineAnchor({
  icon,
  children,
  className,
  ...props
}: AnchorHTMLAttributes<HTMLAnchorElement> & {
  icon?: ReactNode;
}) {
  return (
    <a {...props} className={joinClassNames("mini-link", className)}>
      {icon}
      {children}
    </a>
  );
}

export function NavigationCardAnchor({
  children,
  className,
  ...props
}: AnchorHTMLAttributes<HTMLAnchorElement>) {
  return (
    <a {...props} className={joinClassNames("navigation-card-anchor", className)}>
      {children}
    </a>
  );
}

export function NavigationCardFrame({
  children,
  className,
  ...props
}: HTMLAttributes<HTMLDivElement>) {
  return (
    <div {...props} className={joinClassNames("navigation-card-frame", className)}>
      {children}
    </div>
  );
}

export function PanelToggleButton({
  active,
  className,
  children,
  type = "button",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  active?: boolean;
}) {
  return (
    <button
      {...props}
      type={type}
      aria-expanded={active ?? props["aria-expanded"]}
      className={joinClassNames("panel-toggle-button", active && "active", className)}
    >
      {children}
    </button>
  );
}

export function SelectableRowButton({
  selected,
  className,
  children,
  type = "button",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  selected?: boolean;
}) {
  return (
    <button
      {...props}
      type={type}
      aria-current={selected ? "true" : props["aria-current"]}
      className={joinClassNames("sample-row", selected && "selected", className)}
    >
      {children}
    </button>
  );
}

export function SelectableTableRow({
  selected,
  className,
  children,
  ...props
}: HTMLAttributes<HTMLTableRowElement> & {
  selected?: boolean;
}) {
  return (
    <tr
      {...props}
      aria-current={selected ? "true" : props["aria-current"]}
      className={joinClassNames("selectable-row", selected && "selected", className)}
    >
      {children}
    </tr>
  );
}

export function OptionChipButton({
  active,
  className,
  children,
  type = "button",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  active?: boolean;
}) {
  return (
    <button
      {...props}
      type={type}
      aria-pressed={active ?? props["aria-pressed"]}
      className={joinClassNames("query-chip", active && "active", className)}
    >
      {children}
    </button>
  );
}

export function SelectableCardButton({
  active,
  className,
  children,
  type = "button",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  active?: boolean;
}) {
  return (
    <button
      {...props}
      type={type}
      aria-pressed={active ?? props["aria-pressed"]}
      className={joinClassNames("selectable-card-button", active && "active", className)}
    >
      {children}
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
                <th key={header.id} className={tableColumnClassName(header.column.columnDef.meta)}>
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
                <td key={cell.id} className={tableColumnClassName(cell.column.columnDef.meta)}>
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </td>
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
  const metaId = useId();
  const dialogRef = useRef<HTMLElement | null>(null);
  useEffect(() => {
    if (!open) {
      return;
    }
    const previouslyFocused = document.activeElement instanceof HTMLElement
      ? document.activeElement
      : null;
    const previousBodyOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const focusTarget = dialogRef.current?.querySelector<HTMLElement>(DIALOG_FOCUSABLE_SELECTOR);
    (focusTarget ?? dialogRef.current)?.focus();
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        onClose();
        return;
      }
      if (event.key !== "Tab") {
        return;
      }
      const focusable = Array.from(
        dialogRef.current?.querySelectorAll<HTMLElement>(DIALOG_FOCUSABLE_SELECTOR) ?? []
      ).filter((element) => element.offsetParent !== null || element === document.activeElement);
      if (focusable.length === 0) {
        event.preventDefault();
        dialogRef.current?.focus();
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = previousBodyOverflow;
      previouslyFocused?.focus();
    };
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
        ref={dialogRef}
        className={wide ? "workspace-dialog wide" : "workspace-dialog"}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={meta ? metaId : undefined}
        tabIndex={-1}
      >
        <header className="workspace-dialog-head">
          <div>
            <strong id={titleId}>{title}</strong>
            {meta ? <span id={metaId}>{meta}</span> : null}
          </div>
          <IconActionButton icon={<X size={14} />} title="关闭" onClick={onClose} />
        </header>
        <div className="workspace-dialog-body">{children}</div>
      </section>
    </div>
  );
}

export function DangerConfirmDialog({
  open,
  title,
  subject,
  description,
  confirmLabel = "确认删除",
  pending,
  onCancel,
  onConfirm
}: {
  open: boolean;
  title: string;
  subject: string;
  description: string;
  confirmLabel?: string;
  pending?: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  return (
    <WorkspaceDialog
      open={open}
      title={title}
      meta="危险操作确认"
      onClose={pending ? () => {} : onCancel}
    >
      <div className="danger-confirm-panel">
        <div className="danger-confirm-copy">
          <div className="danger-confirm-mark">
            <AlertTriangle size={22} />
          </div>
          <div>
            <span>目标对象</span>
            <strong title={subject}>{subject}</strong>
            <p>{description}</p>
          </div>
        </div>
        <div className="danger-confirm-actions">
          <ActionButton variant="secondary" disabled={pending} onClick={onCancel}>
            取消
          </ActionButton>
          <ActionButton
            variant="primary"
            className="danger-action-button"
            disabled={pending}
            onClick={onConfirm}
          >
            {pending ? "处理中" : confirmLabel}
          </ActionButton>
        </div>
      </div>
    </WorkspaceDialog>
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
