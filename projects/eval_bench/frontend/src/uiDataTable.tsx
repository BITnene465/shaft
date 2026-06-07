import {
  flexRender,
  getCoreRowModel,
  useReactTable
} from "@tanstack/react-table";
import type { ColumnDef, RowData } from "@tanstack/react-table";

import { joinClassNames } from "./uiActions";
import "./dataTable.css";

export type TableColumnWidth =
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
export type TableColumnWrap = "nowrap" | "truncate" | "wrap";
export type TableColumnAlign = "start" | "center" | "end";
export type TableColumnMeta = {
  width?: TableColumnWidth;
  wrap?: TableColumnWrap;
  align?: TableColumnAlign;
  className?: string;
};

declare module "@tanstack/react-table" {
  interface ColumnMeta<TData extends RowData, TValue> extends TableColumnMeta {}
}

export function tableColumnClassName(
  meta: TableColumnMeta | ColumnDef<unknown>["meta"] | undefined
) {
  return joinClassNames(
    "data-table-cell",
    meta?.width ? `table-col-${meta.width}` : "table-col-text",
    meta?.wrap ? `table-wrap-${meta.wrap}` : "table-wrap-truncate",
    meta?.align ? `table-align-${meta.align}` : "table-align-start",
    meta?.className
  );
}

export function DataTable<T>({
  columns,
  data,
  emptyText,
  compact,
  refreshing = false,
  refreshLabel = "表格更新中"
}: {
  columns: ColumnDef<T>[];
  data: T[];
  emptyText: string;
  compact?: boolean;
  refreshing?: boolean;
  refreshLabel?: string;
}) {
  const table = useReactTable({ data, columns, getCoreRowModel: getCoreRowModel() });
  if (data.length === 0) {
    return (
      <TableEmptyState
        emptyText={emptyText}
        compact={compact}
        refreshing={refreshing}
        refreshLabel={refreshLabel}
      />
    );
  }
  return (
    <div className={joinClassNames("table-shell", compact && "compact", refreshing && "refreshing")}>
      {refreshing ? (
        <span className="table-refresh-indicator" aria-live="polite">
          {refreshLabel}
        </span>
      ) : null}
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

export function TableEmptyState({
  emptyText,
  compact,
  refreshing = false,
  refreshLabel = "表格更新中"
}: {
  emptyText: string;
  compact?: boolean;
  refreshing?: boolean;
  refreshLabel?: string;
}) {
  return (
    <div
      className={joinClassNames(
        "table-shell",
        "empty",
        compact && "compact",
        refreshing && "refreshing"
      )}
    >
      {refreshing ? (
        <span className="table-refresh-indicator" aria-live="polite">
          {refreshLabel}
        </span>
      ) : null}
      <div className="empty-panel">{emptyText}</div>
    </div>
  );
}
