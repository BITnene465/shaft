import type { EvalInstance } from "./api";
import { CompactSelectControl, ToggleButton } from "./controlPrimitives";
import { formatMetric } from "./formatters";
import { DisclosurePanel, OptionChipButton, SelectableCardButton } from "./ui";
import {
  countInstancesByLabel,
  formatBbox,
  objectMetricText,
  objectStatusLabel
} from "./viewerMetrics";
import type {
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
      <CompactSelectControl
        label="视图"
        value={layerPresetValue()}
        options={[
          { value: "all", label: "真值 + 预测 / 全部几何" },
          { value: "gt", label: "仅真值" },
          { value: "pred", label: "仅预测" },
          { value: "boxes", label: "只看框" },
          { value: "lines", label: "只看线" },
          { value: "custom", label: "自定义" }
        ]}
        onChange={applyLayerPreset}
      />
      <div className="layer-toggle-strip" aria-label="图层开关">
        <ToggleButton label="真值" active={showGt} onChange={onShowGtChange} />
        <ToggleButton label="预测" active={showPred} onChange={onShowPredChange} />
        <ToggleButton label="框" active={showBoxes} onChange={onShowBoxesChange} />
        <ToggleButton label="线" active={showLines} onChange={onShowLinesChange} />
        <ToggleButton label="点" active={showKeypoints} onChange={onShowKeypointsChange} />
      </div>
      <DisclosurePanel
        className="control-popover"
        summary={
          <>
            标签 <strong>{activeLabels.length}/{labels.length}</strong>
          </>
        }
      >
        <div className="label-select-grid">
          {labels.map((label) => {
            const active = activeLabels.includes(label);
            return (
              <OptionChipButton
                key={label}
                active={active}
                className="label-select"
                onClick={() => onToggleLabel(label)}
              >
                {label}
              </OptionChipButton>
            );
          })}
        </div>
      </DisclosurePanel>
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
  const items = [
    { label: "真实", value: metrics.gtCount },
    { label: "预测", value: metrics.predCount }
  ];
  return (
    <div className="diagnostic-strip compact-counts" aria-label="可见对象数量">
      {items.map((item) => (
        <span key={item.label}>
          <em>{item.label}</em>
          <strong>{item.value.toLocaleString()}</strong>
        </span>
      ))}
    </div>
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
            <SelectableCardButton
              key={object.id}
              active={object.id === activeObjectId}
              className="object-row"
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
            </SelectableCardButton>
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
