import { CompositeMicroMeter } from "./compositeMicroMeter";
import type { CompositeObjectRef } from "./compositeObjectModel";
import { ActionButton } from "./ui";

import "./compositeObjectHud.css";

export function CompositeObjectHud({
  activeObject,
  activeObjectIndex,
  objectCount,
  relatedObjectCount,
  locked,
  onClear,
  onFocusedLayerChange
}: {
  activeObject: CompositeObjectRef | null;
  activeObjectIndex: number;
  objectCount: number;
  relatedObjectCount: number;
  locked: boolean;
  onClear: () => void;
  onFocusedLayerChange: (layer: string | null) => void;
}) {
  const clampedIndex = activeObjectIndex >= 0 ? activeObjectIndex + 1 : 0;
  const progress = objectCount > 0 && clampedIndex > 0 ? clampedIndex / objectCount : 0;

  if (!activeObject) {
    return (
      <div className="composite-object-hud idle" aria-label="组合对象交互状态">
        <div className="object-hud-main">
          <span>Object Probe</span>
          <strong>
            Hover 查看，Click 锁定，Alt/Shift + Wheel 或 [/] 巡航对象，Alt/Shift + Drag 平移，Right click 对象菜单
          </strong>
        </div>
        <CompositeMicroMeter
          className="object-hud-cruise"
          label={`${objectCount.toLocaleString()} objects`}
          meta="待选择"
          progress={progress}
          idle
          ariaLabel="对象巡航状态"
        />
      </div>
    );
  }
  return (
    <div className={`composite-object-hud status-${activeObject.status}`} aria-label="组合对象交互状态">
      <div className="object-hud-main">
        <span>{locked ? "Locked Object" : "Hovered Object"}</span>
        <strong title={`${activeObject.layer} / ${activeObject.label}`}>
          {activeObject.layer} · {activeObject.kind.toUpperCase()} #{activeObject.index + 1}
        </strong>
        <em title={activeObject.label}>{activeObject.status.toUpperCase()} · {activeObject.label}</em>
      </div>
      <CompositeMicroMeter
        className="object-hud-cruise"
        label={`${clampedIndex.toLocaleString()} / ${objectCount.toLocaleString()}`}
        meta={`${relatedObjectCount.toLocaleString()} related`}
        progress={progress}
        ariaLabel="对象巡航状态"
      />
      <div className="object-hud-actions">
        <ActionButton
          variant="secondary"
          compact
          onClick={() => onFocusedLayerChange(activeObject.layer)}
        >
          聚焦图层
        </ActionButton>
        <ActionButton variant="secondary" compact className="object-hud-clear" onClick={onClear}>
          清除
        </ActionButton>
      </div>
    </div>
  );
}
