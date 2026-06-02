import type { CSSProperties } from "react";

import type { CompositeObjectRef } from "./compositeObjectModel";
import { ActionButton } from "./ui";

import "./compositeObjectContextMenu.css";

export type CompositeObjectMenuRequest = {
  objectKey: string;
  clientX: number;
  clientY: number;
};

export function CompositeObjectContextMenu({
  request,
  object,
  locked,
  onLock,
  onInspect,
  onFocusLayer,
  onClear,
  onClose
}: {
  request: CompositeObjectMenuRequest | null;
  object: CompositeObjectRef | null;
  locked: boolean;
  onLock: (objectKey: string) => void;
  onInspect: (objectKey: string) => void;
  onFocusLayer: (layer: string) => void;
  onClear: () => void;
  onClose: () => void;
}) {
  if (!request || !object) {
    return null;
  }
  const style = {
    "--context-x": `${request.clientX}px`,
    "--context-y": `${request.clientY}px`
  } as CSSProperties;
  return (
    <div
      className={`composite-object-context-menu status-${object.status}`}
      style={style}
      role="menu"
      aria-label="组合对象右键菜单"
      onPointerDown={(event) => event.stopPropagation()}
      onContextMenu={(event) => event.preventDefault()}
    >
      <div className="object-context-head">
        <span>{locked ? "Locked" : "Object"}</span>
        <strong title={`${object.layer} / ${object.label}`}>
          {object.layer} · {object.kind.toUpperCase()} #{object.index + 1}
        </strong>
        <em title={object.label}>{object.status.toUpperCase()} · {object.label}</em>
      </div>
      <div className="object-context-actions">
        <ActionButton
          variant="secondary"
          compact
          onClick={() => {
            onLock(object.key);
            onClose();
          }}
        >
          {locked ? "解锁对象" : "锁定对象"}
        </ActionButton>
        <ActionButton
          variant="secondary"
          compact
          onClick={() => {
            onFocusLayer(object.layer);
            onClose();
          }}
        >
          聚焦图层
        </ActionButton>
        <ActionButton
          variant="secondary"
          compact
          onClick={() => {
            onInspect(object.key);
            onClose();
          }}
        >
          查看详情
        </ActionButton>
        <ActionButton
          variant="secondary"
          compact
          className="object-context-clear"
          onClick={() => {
            onClear();
            onClose();
          }}
        >
          清除选择
        </ActionButton>
      </div>
    </div>
  );
}
