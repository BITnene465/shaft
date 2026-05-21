import type { EvalInstance } from "./api";
import { ToggleButton } from "./controlPrimitives";
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
import type { ShortcutActionId } from "./workspaceSettings";

export function ViewerControlPanel({
  labels,
  activeLabels,
  showGt,
  showPred,
  showBoxes,
  showLines,
  showKeypoints,
  onToggleLabel,
  onShowGtChange,
  onShowPredChange,
  onShowBoxesChange,
  onShowLinesChange,
  onShowKeypointsChange
}: {
  labels: string[];
  activeLabels: string[];
  showGt: boolean;
  showPred: boolean;
  showBoxes: boolean;
  showLines: boolean;
  showKeypoints: boolean;
  onToggleLabel: (label: string) => void;
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
