import { Boxes, Eye, EyeOff, RotateCcw, Trash2 } from "lucide-react";

import type { RunSummary } from "./api";
import {
  fallbackRun,
  layerIndex
} from "./compositeReportComposerModel";
import type { ReportGroup } from "./compositeReportComposerModel";
import { layerColor } from "./compositeLayerPalette";
import { runOptionLabel } from "./compositeReportModel";
import type { LayerSlot } from "./compositeReportModel";
import { CompositeReportEmptyState, CompositeReportPanelHeader } from "./compositeReportPanel";
import { TextInputControl } from "./controlPrimitives";
import { ActionButton, IconActionButton, OptionChipButton } from "./ui";

import "./compositeReportLayerPlan.css";

export function ReportLayerPlan({
  groups,
  runById,
  onReset,
  onApplyLayoutArrowPreset,
  onUpdateSlot,
  onRemoveSlot
}: {
  groups: ReportGroup[];
  runById: Map<string, RunSummary>;
  onReset: () => void;
  onApplyLayoutArrowPreset: () => void;
  onUpdateSlot: (id: string, patch: Partial<LayerSlot>) => void;
  onRemoveSlot: (id: string) => void;
}) {
  return (
    <aside className="report-layer-plan">
      <CompositeReportPanelHeader
        eyebrow="Report Layers"
        title="分层报告结构"
        action={
          <>
          <IconActionButton title="重置组合" icon={<RotateCcw size={14} />} onClick={onReset} />
          <ActionButton
            variant="secondary"
            compact
            icon={<Boxes size={14} />}
            onClick={onApplyLayoutArrowPreset}
          >
            Layout + Arrow
          </ActionButton>
          </>
        }
      />
      <div className="report-layer-tree">
        {groups.length === 0 ? (
          <CompositeReportEmptyState>从左侧结果池加入至少两个评测结果。</CompositeReportEmptyState>
        ) : (
          groups.map((group) => (
            <section className="report-layer-group" key={group.key}>
              <div className="report-layer-group-head">
                <strong>{group.title}</strong>
                <span>{group.subtitle}</span>
              </div>
              {group.slots.map((slot) => {
                const run = runById.get(slot.runId);
                return (
                  <article
                    className={slot.visible ? "report-layer-row" : "report-layer-row muted"}
                    key={slot.id}
                  >
                    <i style={{ background: layerColor(layerIndex(slot.layer)) }} />
                    <div className="report-layer-row-main">
                      <TextInputControl
                        className="report-layer-name"
                        label="Layer"
                        value={slot.layer}
                        onChange={(layer) => onUpdateSlot(slot.id, { layer })}
                      />
                      <div className="report-layer-run">
                        <strong title={slot.runId}>{slot.runId}</strong>
                        <span title={runOptionLabel(run ?? fallbackRun(slot.runId))}>
                          {run?.benchmark_split || run?.spec_task || "run"}
                        </span>
                      </div>
                      <div className="report-layer-switches">
                        <OptionChipButton
                          active={slot.showGt}
                          onClick={() => onUpdateSlot(slot.id, { showGt: !slot.showGt })}
                        >
                          GT
                        </OptionChipButton>
                        <OptionChipButton
                          active={slot.showPred}
                          onClick={() => onUpdateSlot(slot.id, { showPred: !slot.showPred })}
                        >
                          Pred
                        </OptionChipButton>
                      </div>
                    </div>
                    <div className="report-layer-row-actions">
                      <IconActionButton
                        title={slot.visible ? "隐藏图层" : "显示图层"}
                        icon={slot.visible ? <Eye size={14} /> : <EyeOff size={14} />}
                        onClick={() => onUpdateSlot(slot.id, { visible: !slot.visible })}
                      />
                      <IconActionButton
                        title="删除图层"
                        danger
                        icon={<Trash2 size={14} />}
                        onClick={() => onRemoveSlot(slot.id)}
                      />
                    </div>
                  </article>
                );
              })}
            </section>
          ))
        )}
      </div>
    </aside>
  );
}
