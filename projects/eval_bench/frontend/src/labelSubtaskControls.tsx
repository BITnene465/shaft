import { useState } from "react";
import type { FormEvent } from "react";

import { unique } from "./formatters";
import { ActionButton, OptionChipButton } from "./ui";

export function DetectionLabelSubtaskPanel({
  task,
  benchmarkId,
  labelOptions,
  selectedLabels,
  onChange,
  className
}: {
  task: string;
  benchmarkId: string;
  labelOptions: string[];
  selectedLabels: string[];
  onChange: (labels: string[]) => void;
  className?: string;
}) {
  const [draftLabel, setDraftLabel] = useState("");
  if (task !== "detection") {
    return null;
  }
  const selectedSet = new Set(selectedLabels);

  function toggleLabel(label: string) {
    if (selectedSet.has(label)) {
      onChange(selectedLabels.filter((item) => item !== label));
      return;
    }
    onChange(unique([...selectedLabels, label]));
  }

  function addDraftLabel(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const value = draftLabel.trim();
    if (!value) {
      return;
    }
    onChange(unique([...selectedLabels, value]));
    setDraftLabel("");
  }

  return (
    <div className={["label-subtask-panel", className].filter(Boolean).join(" ")}>
      <div className="label-subtask-head">
        <div>
          <strong>Detection 子任务</strong>
          <span>{benchmarkId || "未选择 benchmark"}</span>
        </div>
        <div className="label-subtask-actions">
          <ActionButton variant="mini" onClick={() => onChange(labelOptions)}>
            全部候选
          </ActionButton>
          <ActionButton variant="mini" onClick={() => onChange([])}>
            默认策略
          </ActionButton>
        </div>
      </div>
      <div className="label-subtask-chips">
        {labelOptions.map((label) => (
          <OptionChipButton
            key={label}
            active={selectedSet.has(label)}
            onClick={() => toggleLabel(label)}
          >
            {label}
          </OptionChipButton>
        ))}
        {labelOptions.length === 0 ? <span className="label-subtask-empty">暂无 label 索引</span> : null}
      </div>
      <form className="label-subtask-add" onSubmit={addDraftLabel}>
        <input
          value={draftLabel}
          onChange={(event) => setDraftLabel(event.target.value)}
          placeholder="自定义 label"
        />
        <ActionButton variant="mini" type="submit">
          添加
        </ActionButton>
      </form>
    </div>
  );
}
