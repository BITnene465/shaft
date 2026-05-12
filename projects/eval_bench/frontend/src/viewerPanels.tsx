import { useMemo, useState } from "react";
import { X } from "lucide-react";

import type { EvalInstance, RunSampleDetail } from "./api";
import { StyleSlider, ColorControl, ToggleButton } from "./controlPrimitives";
import { formatMetric } from "./formatters";
import {
  countInstancesByLabel,
  formatBbox,
  objectMetricText,
  objectStatusLabel
} from "./viewerMetrics";
import type {
  LabelMetricRow,
  ObjectKind,
  ObjectRow,
  VisibleMetrics
} from "./viewerMetrics";
import {
  INSTANCE_COLOR_ROLES,
  OVERLAY_STYLE_CONTROLS,
  PRED_LINE_STYLE_OPTIONS,
  explicitLabelColor
} from "./workspaceSettings";
import type {
  InstanceColorRole,
  LabelColors,
  OverlayColors,
  OverlayStyle,
  OverlayStyleKey,
  ShortcutActionId
} from "./workspaceSettings";

export function DiagnosticStrip({
  diagnostics
}: {
  diagnostics: NonNullable<RunSampleDetail["diagnostics"]>;
}) {
  return (
    <div className="diagnostic-strip">
      <span>TP {diagnostics.matched_count.toLocaleString()}</span>
      <span>FP {diagnostics.false_positive_count.toLocaleString()}</span>
      <span>FN {diagnostics.false_negative_count.toLocaleString()}</span>
      <span>平均 IoU {formatMetric(diagnostics.mean_iou)}</span>
    </div>
  );
}

export function ViewerControlPanel({
  labels,
  activeLabels,
  colors,
  styleConfig,
  labelColors,
  showGt,
  showPred,
  showBoxes,
  showLines,
  showKeypoints,
  onToggleLabel,
  onStyleChange,
  onLabelColorChange,
  onLabelColorRemove,
  onResetStyle,
  onResetLabelColors,
  onShowGtChange,
  onShowPredChange,
  onShowBoxesChange,
  onShowLinesChange,
  onShowKeypointsChange
}: {
  labels: string[];
  activeLabels: string[];
  colors: OverlayColors;
  styleConfig: OverlayStyle;
  labelColors: LabelColors;
  showGt: boolean;
  showPred: boolean;
  showBoxes: boolean;
  showLines: boolean;
  showKeypoints: boolean;
  onToggleLabel: (label: string) => void;
  onStyleChange: (key: OverlayStyleKey, value: number | string) => void;
  onLabelColorChange: (label: string, role: InstanceColorRole, value: string) => void;
  onLabelColorRemove: (label: string, role?: InstanceColorRole) => void;
  onResetStyle: () => void;
  onResetLabelColors: () => void;
  onShowGtChange: (value: boolean) => void;
  onShowPredChange: (value: boolean) => void;
  onShowBoxesChange: (value: boolean) => void;
  onShowLinesChange: (value: boolean) => void;
  onShowKeypointsChange: (value: boolean) => void;
}) {
  function layerPresetValue() {
    if (showGt && showPred && showBoxes && showLines && showKeypoints) {
      return "all";
    }
    if (showGt && !showPred && showBoxes && showLines && showKeypoints) {
      return "gt";
    }
    if (!showGt && showPred && showBoxes && showLines && showKeypoints) {
      return "pred";
    }
    if (showGt && showPred && showBoxes && !showLines && !showKeypoints) {
      return "boxes";
    }
    if (showGt && showPred && !showBoxes && showLines && !showKeypoints) {
      return "lines";
    }
    return "custom";
  }

  function applyLayerPreset(value: string) {
    if (value === "gt") {
      onShowGtChange(true);
      onShowPredChange(false);
      onShowBoxesChange(true);
      onShowLinesChange(true);
      onShowKeypointsChange(true);
      return;
    }
    if (value === "pred") {
      onShowGtChange(false);
      onShowPredChange(true);
      onShowBoxesChange(true);
      onShowLinesChange(true);
      onShowKeypointsChange(true);
      return;
    }
    if (value === "boxes") {
      onShowGtChange(true);
      onShowPredChange(true);
      onShowBoxesChange(true);
      onShowLinesChange(false);
      onShowKeypointsChange(false);
      return;
    }
    if (value === "lines") {
      onShowGtChange(true);
      onShowPredChange(true);
      onShowBoxesChange(false);
      onShowLinesChange(true);
      onShowKeypointsChange(false);
      return;
    }
    onShowGtChange(true);
    onShowPredChange(true);
    onShowBoxesChange(true);
    onShowLinesChange(true);
    onShowKeypointsChange(true);
  }

  return (
    <div className="viewer-controls">
      <label className="compact-select">
        <span>视图</span>
        <select value={layerPresetValue()} onChange={(event) => applyLayerPreset(event.target.value)}>
          <option value="all">真值 + 预测 / 全部几何</option>
          <option value="gt">仅真值</option>
          <option value="pred">仅预测</option>
          <option value="boxes">只看框</option>
          <option value="lines">只看线</option>
          <option value="custom">自定义</option>
        </select>
      </label>
      <div className="layer-toggle-strip" aria-label="图层开关">
        <ToggleButton label="真值" active={showGt} onChange={onShowGtChange} />
        <ToggleButton label="预测" active={showPred} onChange={onShowPredChange} />
        <ToggleButton label="框" active={showBoxes} onChange={onShowBoxesChange} />
        <ToggleButton label="线" active={showLines} onChange={onShowLinesChange} />
        <ToggleButton label="点" active={showKeypoints} onChange={onShowKeypointsChange} />
      </div>
      <details className="control-popover">
        <summary>
          标签 <strong>{activeLabels.length}/{labels.length}</strong>
        </summary>
        <div className="label-select-grid">
          {labels.map((label) => {
            const active = activeLabels.includes(label);
            return (
              <button
                key={label}
                className={active ? "label-select active" : "label-select"}
                type="button"
                onClick={() => onToggleLabel(label)}
              >
                {label}
              </button>
            );
          })}
        </div>
      </details>
      <OverlayAppearancePanel
        styleConfig={styleConfig}
        onStyleChange={onStyleChange}
        onResetStyle={onResetStyle}
      />
      <LabelColorPanel
        labels={labels}
        overlayColors={colors}
        labelColors={labelColors}
        onChange={onLabelColorChange}
        onRemove={onLabelColorRemove}
        onReset={onResetLabelColors}
      />
    </div>
  );
}

export function handleViewerShortcutAction(
  actionId: ShortcutActionId,
  handlers: {
    clearSelection: () => void;
    toggleGt: () => void;
    togglePred: () => void;
    toggleBoxes: () => void;
    toggleLines: () => void;
    toggleKeypoints: () => void;
  }
) {
  if (actionId === "selection.clear") {
    handlers.clearSelection();
    return true;
  }
  if (actionId === "layer.toggleGt") {
    handlers.toggleGt();
    return true;
  }
  if (actionId === "layer.togglePred") {
    handlers.togglePred();
    return true;
  }
  if (actionId === "geometry.toggleBoxes") {
    handlers.toggleBoxes();
    return true;
  }
  if (actionId === "geometry.toggleLines") {
    handlers.toggleLines();
    return true;
  }
  if (actionId === "geometry.toggleKeypoints") {
    handlers.toggleKeypoints();
    return true;
  }
  return false;
}

function OverlayAppearancePanel({
  styleConfig,
  onStyleChange,
  onResetStyle,
  defaultOpen = false
}: {
  styleConfig: OverlayStyle;
  onStyleChange: (key: OverlayStyleKey, value: number | string) => void;
  onResetStyle: () => void;
  defaultOpen?: boolean;
}) {
  return (
    <details className="control-popover" open={defaultOpen}>
      <summary>
        样式 <strong>框 / 线 / 点</strong>
      </summary>
      <div className="control-title-row">
        <span>可视化参数</span>
        <button className="text-button" type="button" onClick={onResetStyle}>
          重置
        </button>
      </div>
      <div className="style-control-grid">
        {OVERLAY_STYLE_CONTROLS.map((control) => (
          <StyleSlider
            key={control.key}
            label={control.label}
            value={styleConfig[control.key]}
            min={control.min}
            max={control.max}
            step={control.step}
            onChange={(value) => onStyleChange(control.key, value)}
          />
        ))}
        <label className="compact-select dense">
          <span>预测线型</span>
          <select
            value={styleConfig.predLineStyle}
            onChange={(event) => onStyleChange("predLineStyle", event.target.value)}
          >
            {PRED_LINE_STYLE_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
      </div>
    </details>
  );
}

function LabelColorPanel({
  labels,
  overlayColors,
  labelColors,
  onChange,
  onRemove,
  onReset,
  defaultOpen = false
}: {
  labels: string[];
  overlayColors: OverlayColors;
  labelColors: LabelColors;
  onChange: (label: string, role: InstanceColorRole, value: string) => void;
  onRemove: (label: string, role?: InstanceColorRole) => void;
  onReset: () => void;
  defaultOpen?: boolean;
}) {
  const [draftLabel, setDraftLabel] = useState("");
  const [draftRole, setDraftRole] = useState<InstanceColorRole>("gt");
  const [draftColor, setDraftColor] = useState("#2563eb");
  const sortedLabels = useMemo(
    () => [...labels].sort((left, right) => left.localeCompare(right)),
    [labels]
  );

  function addLabelColor() {
    const label = draftLabel.trim();
    if (!label) {
      return;
    }
    onChange(label, draftRole, draftColor);
    setDraftLabel("");
  }

  return (
    <details className="control-popover" open={defaultOpen}>
      <summary>
        标签颜色 <strong>{sortedLabels.length}</strong>
      </summary>
      <div className="control-title-row">
        <span>按 label 匹配，大小写不敏感</span>
        <button className="text-button" type="button" onClick={onReset}>
          重置
        </button>
      </div>
      <div className="label-color-add-row">
        <input
          value={draftLabel}
          placeholder="输入 label"
          onChange={(event) => setDraftLabel(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              addLabelColor();
            }
          }}
        />
        <input
          aria-label="新增 label 颜色"
          type="color"
          value={draftColor}
          onChange={(event) => setDraftColor(event.target.value)}
        />
        <select
          aria-label="新增 label 颜色角色"
          value={draftRole}
          onChange={(event) => setDraftRole(event.target.value as InstanceColorRole)}
        >
          {INSTANCE_COLOR_ROLES.map((role) => (
            <option key={role.key} value={role.key}>
              {role.label}
            </option>
          ))}
        </select>
        <button className="secondary-button dense" type="button" onClick={addLabelColor}>
          添加
        </button>
      </div>
      <div className="label-color-grid">
        {sortedLabels.length === 0 ? (
          <div className="muted-line">还没有自定义 label 颜色。</div>
        ) : (
          sortedLabels.map((label) => (
            <div className="label-color-row" key={label}>
              <span className="label-color-name" title={label}>{label}</span>
              <div className="label-color-role-grid">
                {INSTANCE_COLOR_ROLES.map((role) => (
                  <ColorControl
                    key={role.key}
                    label={role.label}
                    value={explicitLabelColor(labelColors, label, role.key) ?? overlayColors[role.key]}
                    onChange={(value) => onChange(label, role.key, value)}
                  />
                ))}
              </div>
              <button
                className="icon-button dense"
                type="button"
                title={`移除 ${label} 颜色规则`}
                onClick={() => onRemove(label)}
              >
                <X size={13} />
              </button>
            </div>
          ))
        )}
      </div>
    </details>
  );
}

export function VisibleMetricStrip({ metrics }: { metrics: VisibleMetrics }) {
  return (
    <div className="diagnostic-strip">
      <span>真值 {metrics.gtCount.toLocaleString()}</span>
      <span>预测 {metrics.predCount.toLocaleString()}</span>
      <span>TP {metrics.matchedCount.toLocaleString()}</span>
      <span>FP {metrics.falsePositiveCount.toLocaleString()}</span>
      <span>FN {metrics.falseNegativeCount.toLocaleString()}</span>
      <span>平均 IoU {formatMetric(metrics.meanIou)}</span>
    </div>
  );
}

export function LabelMetricTable({ rows }: { rows: LabelMetricRow[] }) {
  return (
    <details className="label-metric-card">
      <summary>分标签指标</summary>
      {rows.length === 0 ? (
        <div className="muted-line">没有可见标签。</div>
      ) : (
        <div className="label-metric-table">
          <table>
            <thead>
              <tr>
                <th>标签</th>
                <th>真值</th>
                <th>预测</th>
                <th>TP</th>
                <th>FP</th>
                <th>FN</th>
                <th>P@.50</th>
                <th>R@.50</th>
                <th>平均 IoU</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.label}>
                  <td title={row.label}>{row.label}</td>
                  <td>{row.gtCount.toLocaleString()}</td>
                  <td>{row.predCount.toLocaleString()}</td>
                  <td>{row.matchedCount.toLocaleString()}</td>
                  <td>{row.falsePositiveCount.toLocaleString()}</td>
                  <td>{row.falseNegativeCount.toLocaleString()}</td>
                  <td>{formatMetric(row.precision)}</td>
                  <td>{formatMetric(row.recall)}</td>
                  <td>{formatMetric(row.meanIou)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </details>
  );
}

export function ObjectList({
  objects,
  activeObjectId,
  lockedObjectId,
  onHover,
  onLock
}: {
  objects: ObjectRow[];
  activeObjectId: string | null;
  lockedObjectId: string | null;
  onHover: (objectId: string | null) => void;
  onLock: (objectId: string | null) => void;
}) {
  return (
    <div className="object-list">
      <div className="instance-card-title">对象列表</div>
      {objects.length === 0 ? (
        <div className="muted-line">没有可见对象。</div>
      ) : (
        <div className="object-list-scroll">
          {objects.map((object) => (
            <button
              key={object.id}
              className={object.id === activeObjectId ? "object-row active" : "object-row"}
              type="button"
              onPointerEnter={() => onHover(object.id)}
              onPointerLeave={() => onHover(null)}
              onClick={() => onLock(object.id)}
            >
              <span className={`object-kind ${object.kind}`}>{objectKindLabel(object.kind)}</span>
              <span className="object-main">
                <span className="object-label">
                  {object.label}
                  <span className="object-index">#{object.index + 1}</span>
                </span>
                <span className="object-bbox">{formatBbox(object.bbox)}</span>
              </span>
              <span className={`object-status ${object.status}`}>
                {objectStatusLabel(object.status)}
              </span>
              <span className="object-match">{objectMetricText(object, formatMetric)}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export function InstanceStats({
  title,
  instances
}: {
  title: string;
  instances: EvalInstance[];
}) {
  const counts = countInstancesByLabel(instances);
  const entries = Object.entries(counts).sort(([left], [right]) => left.localeCompare(right));
  return (
    <div className="instance-card">
      <div className="instance-card-title">{title}</div>
      {entries.length === 0 ? (
        <div className="muted-line">没有实例。</div>
      ) : (
        <div className="label-chip-row">
          {entries.map(([label, count]) => (
            <span className="label-chip" key={label}>
              {label} {count}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function objectKindLabel(kind: ObjectKind) {
  return kind === "gt" ? "真值" : "预测";
}
