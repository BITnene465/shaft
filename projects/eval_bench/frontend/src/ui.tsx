import * as Tabs from "@radix-ui/react-tabs";
import {
  flexRender,
  getCoreRowModel,
  useReactTable
} from "@tanstack/react-table";
import type { ColumnDef } from "@tanstack/react-table";
import type { ReactNode } from "react";

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

export function Badge({ value }: { value: string }) {
  const kind = value === "succeeded" ? "success" : value === "failed" ? "danger" : "neutral";
  return <span className={`badge ${kind}`}>{statusText(value)}</span>;
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

function statusText(value: string) {
  const labels: Record<string, string> = {
    queued: "排队中",
    running: "运行中",
    starting: "启动中",
    succeeded: "成功",
    failed: "失败",
    cancelled: "已取消",
    stopped: "已停止",
    registered: "已登记",
    imported: "已导入",
    archived: "已归档",
    detection: "检测",
    keypoint: "关键点"
  };
  return labels[value] ?? value;
}
