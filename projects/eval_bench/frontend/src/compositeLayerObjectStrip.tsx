import type { WheelEvent } from "react";

import { useLayerObjectStripDrag } from "./compositeLayerObjectStripDrag";
import type { CompositeObjectRef } from "./compositeObjectModel";
import { OptionChipButton } from "./ui";

import "./compositeLayerObjectStrip.css";

export function LayerObjectStrip({
  objects,
  activeObjectKey,
  relatedObjectKeys,
  lockedObjectKey,
  onObjectHover,
  onObjectLock,
  onObjectInspect,
  onObjectWheel
}: {
  objects: CompositeObjectRef[];
  activeObjectKey: string | null;
  relatedObjectKeys: Set<string>;
  lockedObjectKey: string | null;
  onObjectHover: (objectKey: string | null) => void;
  onObjectLock: (objectKey: string | null) => void;
  onObjectInspect: (objectKey: string | null) => void;
  onObjectWheel: (event: WheelEvent<HTMLElement>) => void;
}) {
  const objectDrag = useLayerObjectStripDrag({
    onObjectHover,
    onObjectLock
  });

  if (objects.length === 0) {
    return null;
  }
  return (
    <div
      className={objectDrag.dragging ? "layer-object-strip dragging" : "layer-object-strip"}
      aria-label="图层错误对象"
      onWheelCapture={onObjectWheel}
      {...objectDrag.objectStripDragHandlers}
    >
      <span className={objectDrag.dragging ? "layer-object-drag-hint active" : "layer-object-drag-hint"}>
        {objectDrag.dragging ? "释放锁定对象" : "按住拖动扫选"}
      </span>
      {objects.map((object) => (
        <OptionChipButton
          active={activeObjectKey === object.key}
          data-object-key={object.key}
          className={[
            "layer-object-chip",
            object.status,
            relatedObjectKeys.has(object.key) ? "related" : "",
            lockedObjectKey === object.key ? "locked" : ""
          ]
            .filter(Boolean)
            .join(" ")}
          key={object.key}
          onPointerEnter={() => onObjectHover(object.key)}
          onPointerLeave={() => {
            if (!objectDrag.dragging) {
              onObjectHover(null);
            }
          }}
          onClick={(event) => {
            event.stopPropagation();
            if (objectDrag.shouldSuppressClick()) {
              return;
            }
            onObjectLock(object.key);
          }}
          onDoubleClick={(event) => {
            event.stopPropagation();
            onObjectInspect(object.key);
          }}
          title={`${object.layer} ${object.status.toUpperCase()} ${object.label}`}
        >
          <span>{object.kind.toUpperCase()}</span>
          <b>#{object.index + 1}</b>
          {object.status.toUpperCase()}
        </OptionChipButton>
      ))}
      <em>{objects.length.toLocaleString()} objects</em>
    </div>
  );
}
