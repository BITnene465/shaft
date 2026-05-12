import * as Tabs from "@radix-ui/react-tabs";
import {
  flexRender,
  getCoreRowModel,
  useReactTable
} from "@tanstack/react-table";
import type { ColumnDef } from "@tanstack/react-table";
import type { ReactNode } from "react";

import { statusClassName, statusInfo } from "./statusModel";
import type { StatusDomain } from "./statusModel";

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

export function WorkspaceTabs({
  defaultValue,
  label,
  children
}: {
  defaultValue: string;
  label: string;
  children: ReactNode;
}) {
  return (
    <Tabs.Root defaultValue={defaultValue} className="workspace-tabs" aria-label={label}>
      {children}
    </Tabs.Root>
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

export function ConfigItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="config-item">
      <span>{label}</span>
      <strong title={value}>{value}</strong>
    </div>
  );
}
