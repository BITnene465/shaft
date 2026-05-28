import type { ReactNode } from "react";

import { ActionButton } from "./ui";

export function clampListPageOffset(offset: number, total: number, pageSize: number) {
  if (total <= 0 || offset < total) {
    return Math.max(0, offset);
  }
  return Math.floor((total - 1) / pageSize) * pageSize;
}

export function updatePagedFilterValue<T>(
  currentValue: T,
  nextValue: T,
  setValue: (value: T) => void,
  ...resetOffsets: Array<(offset: number) => void>
) {
  if (Object.is(nextValue, currentValue)) {
    return;
  }
  for (const resetOffset of resetOffsets) {
    resetOffset(0);
  }
  setValue(nextValue);
}

export function PagerControl({
  className,
  offset,
  limit,
  total,
  meta,
  onPageChange
}: {
  className: string;
  offset: number;
  limit: number;
  total: number;
  meta?: ReactNode;
  onPageChange: (offset: number) => void;
}) {
  const start = total === 0 ? 0 : offset + 1;
  const end = Math.min(total, offset + limit);
  const previousOffset = Math.max(0, offset - limit);
  const nextOffset = offset + limit;
  return (
    <div className={className}>
      <span>
        {start.toLocaleString()}-{end.toLocaleString()} / {total.toLocaleString()}
        {meta}
      </span>
      <div>
        <ActionButton
          variant="mini"
          onClick={() => onPageChange(previousOffset)}
          disabled={offset <= 0}
        >
          上一页
        </ActionButton>
        <ActionButton
          variant="mini"
          onClick={() => onPageChange(nextOffset)}
          disabled={nextOffset >= total}
        >
          下一页
        </ActionButton>
      </div>
    </div>
  );
}

export function SamplePager({
  offset,
  limit,
  total,
  onPageChange
}: {
  offset: number;
  limit: number;
  total: number;
  onPageChange: (offset: number) => void;
}) {
  return (
    <PagerControl
      className="sample-pager"
      offset={offset}
      limit={limit}
      total={total}
      onPageChange={onPageChange}
    />
  );
}
