import { ActionButton } from "./ui";

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
  const start = total === 0 ? 0 : offset + 1;
  const end = Math.min(total, offset + limit);
  const previousOffset = Math.max(0, offset - limit);
  const nextOffset = offset + limit;
  return (
    <div className="sample-pager">
      <span>
        {start.toLocaleString()}-{end.toLocaleString()} / {total.toLocaleString()}
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
