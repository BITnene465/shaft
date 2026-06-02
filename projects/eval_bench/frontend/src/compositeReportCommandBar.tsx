import { Layers3 } from "lucide-react";

import type { CompositeSampleView } from "./api";
import { STAGE_MODES } from "./compositeReportModel";
import type { StageMode } from "./compositeReportModel";
import { errorMessage } from "./formatters";
import { OptionChipButton } from "./ui";

import "./compositeReportCommandBar.css";

export function CompositeReportCommandBar({
  activeSlots,
  readyLayerCount,
  missingLayerCount,
  composite,
  enabled,
  error,
  stageMode,
  onStageModeChange
}: {
  activeSlots: number;
  readyLayerCount: number;
  missingLayerCount: number;
  composite?: CompositeSampleView;
  enabled: boolean;
  error: unknown;
  stageMode: StageMode;
  onStageModeChange: (mode: StageMode) => void;
}) {
  return (
    <header className="composite-report-command">
      <div className="composite-report-title">
        <span>Composite Report</span>
        <strong>组合视图报告</strong>
        <em>自由组合多个评测结果，按任务层级同步查看 GT、预测和错误诊断。</em>
      </div>
      <div className="composite-report-controls">
        <ReportSignal
          activeSlots={activeSlots}
          readyLayerCount={readyLayerCount}
          missingLayerCount={missingLayerCount}
          composite={composite}
          enabled={enabled}
          error={error}
        />
        <StageModeSwitch stageMode={stageMode} onStageModeChange={onStageModeChange} />
      </div>
    </header>
  );
}

function ReportSignal({
  activeSlots,
  readyLayerCount,
  missingLayerCount,
  composite,
  enabled,
  error
}: {
  activeSlots: number;
  readyLayerCount: number;
  missingLayerCount: number;
  composite?: CompositeSampleView;
  enabled: boolean;
  error: unknown;
}) {
  return (
    <div className="composite-report-summary">
      <span>
        <Layers3 size={14} />
        {enabled ? "同步报告" : "等待组合"}
      </span>
      <strong>
        {activeSlots} layers · {readyLayerCount} ready · {missingLayerCount} missing
      </strong>
      <em title={error ? errorMessage(error) : composite?.image_key ?? ""}>
        {error ? errorMessage(error) : composite?.image_key ?? "至少选择两个可见图层"}
      </em>
    </div>
  );
}

function StageModeSwitch({
  stageMode,
  onStageModeChange
}: {
  stageMode: StageMode;
  onStageModeChange: (mode: StageMode) => void;
}) {
  return (
    <div className="composite-stage-mode" aria-label="组合报告视图模式">
      {STAGE_MODES.map((mode) => (
        <OptionChipButton
          active={stageMode === mode.value}
          className="composite-stage-mode-button"
          key={mode.value}
          onClick={() => onStageModeChange(mode.value)}
        >
          {mode.label}
        </OptionChipButton>
      ))}
    </div>
  );
}
