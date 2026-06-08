import { memo, useCallback, useMemo } from "react";

import type { RunSampleSummary } from "./api";
import { AdvancedFilterBar, type AdvancedFilterControl } from "./filterControls";
import { basename } from "./formatters";
import { SelectableRowButton } from "./ui";

export function SampleFilters({
  errorFilter,
  labelFilter,
  labels,
  onErrorFilterChange,
  onLabelFilterChange
}: {
  errorFilter: string;
  labelFilter: string;
  labels: string[];
  onErrorFilterChange: (value: string) => void;
  onLabelFilterChange: (value: string) => void;
}) {
  const sampleFilterControls = useMemo<AdvancedFilterControl[]>(
    () => {
      const controls: AdvancedFilterControl[] = [
        {
          type: "select",
          id: "error",
          label: "状态",
          value: errorFilter,
          values: ["all", "fn", "fp", "missing", "clean"],
          labels: { all: "全部", fn: "漏检", fp: "误检", missing: "缺失预测", clean: "正常" },
          onChange: onErrorFilterChange
        },
        {
          type: "select",
          id: "label",
          label: "标签",
          value: labelFilter,
          values: ["all", ...labels],
          labels: { all: "全部" },
          onChange: onLabelFilterChange
        }
      ];
      return controls;
    },
    [errorFilter, labelFilter, labels, onErrorFilterChange, onLabelFilterChange]
  );
  return (
    <AdvancedFilterBar
      title="样本检索"
      meta={`${labels.length.toLocaleString()} labels`}
      controls={sampleFilterControls}
    />
  );
}

export function SampleList({
  samples,
  selectedIndex,
  refreshing = false,
  onSelect,
  emptyText
}: {
  samples: RunSampleSummary[];
  selectedIndex: number;
  refreshing?: boolean;
  onSelect: (index: number) => void;
  emptyText: string;
}) {
  if (samples.length === 0) {
    return <div className="sample-list empty">{emptyText}</div>;
  }
  return (
    <div className={refreshing ? "sample-list refreshing" : "sample-list"}>
      {refreshing ? (
        <span className="table-refresh-indicator" aria-live="polite">
          样本列表更新中
        </span>
      ) : null}
      {samples.map((sample) => (
        <RunSampleListRow
          key={sample.index}
          sample={sample}
          selected={sample.index === selectedIndex}
          onSelect={onSelect}
        />
      ))}
    </div>
  );
}

const RunSampleListRow = memo(function RunSampleListRow({
  sample,
  selected,
  onSelect
}: {
  sample: RunSampleSummary;
  selected: boolean;
  onSelect: (index: number) => void;
}) {
  const handleSelect = useCallback(() => onSelect(sample.index), [onSelect, sample.index]);
  return (
    <SelectableRowButton selected={selected} onClick={handleSelect}>
      <span className="sample-row-main">
        <strong>{sample.index + 1}</strong>
        <span title={sample.image}>{basename(sample.image)}</span>
      </span>
      <span className="sample-row-meta">
        真实 {sample.gt_instance_count.toLocaleString()} / 预测{" "}
        {sample.pred_instance_count.toLocaleString()}
      </span>
      <span className={sample.has_prediction ? "sample-status ok" : "sample-status missing"}>
        {sample.has_prediction ? "已预测" : "缺预测"}
      </span>
    </SelectableRowButton>
  );
});
