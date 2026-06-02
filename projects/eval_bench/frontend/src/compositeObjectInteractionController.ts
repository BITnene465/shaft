import { useCallback, useEffect, useMemo, useState } from "react";
import type { WheelEvent } from "react";

import type { CompositeSampleLayer } from "./api";
import type { CompositeObjectMenuRequest } from "./compositeObjectContextMenu";
import { useCompositeObjectContextMenuLifecycle } from "./compositeObjectContextMenuLifecycle";
import { useCompositeObjectKeyboardNavigation } from "./compositeObjectKeyboardNavigation";
import {
  allCompositeObjectRefs,
  nextCompositeObjectKey,
  relatedCompositeObjectKeys,
  resolveCompositeObjectRef
} from "./compositeObjectInteraction";

export function useCompositeObjectInteraction({
  imageKey,
  layers,
  onFocusedLayerChange
}: {
  imageKey?: string;
  layers: CompositeSampleLayer[];
  onFocusedLayerChange: (layer: string | null) => void;
}) {
  const [hoveredObjectKey, setHoveredObjectKey] = useState<string | null>(null);
  const [lockedObjectKey, setLockedObjectKey] = useState<string | null>(null);
  const [contextMenu, setContextMenu] = useState<CompositeObjectMenuRequest | null>(null);
  const activeObjectKey = lockedObjectKey ?? hoveredObjectKey;
  const objectRefs = useMemo(() => allCompositeObjectRefs(layers), [layers]);
  const activeObjectIndex = objectRefs.findIndex((object) => object.key === activeObjectKey);
  const activeObject = useMemo(
    () => resolveCompositeObjectRef(layers, activeObjectKey),
    [activeObjectKey, layers]
  );
  const contextMenuObject = useMemo(
    () => resolveCompositeObjectRef(layers, contextMenu?.objectKey ?? null),
    [contextMenu, layers]
  );
  const relatedObjectKeys = useMemo(
    () => relatedCompositeObjectKeys(layers, activeObjectKey),
    [activeObjectKey, layers]
  );
  const closeContextMenu = useCallback(() => setContextMenu(null), []);

  useEffect(() => {
    setHoveredObjectKey(null);
    setLockedObjectKey(null);
    setContextMenu(null);
  }, [imageKey]);

  const navigateObject = useCallback(
    (direction: -1 | 1) => {
      const nextKey = nextCompositeObjectKey(layers, activeObjectKey, direction);
      if (!nextKey) {
        return;
      }
      const nextObject = resolveCompositeObjectRef(layers, nextKey);
      setHoveredObjectKey(null);
      setLockedObjectKey(nextKey);
      setContextMenu(null);
      if (nextObject) {
        onFocusedLayerChange(nextObject.layer);
      }
    },
    [activeObjectKey, layers, onFocusedLayerChange]
  );

  useCompositeObjectContextMenuLifecycle({ contextMenu, closeContextMenu });
  useCompositeObjectKeyboardNavigation({ navigateObject });

  function toggleObjectLock(objectKey: string | null) {
    setLockedObjectKey((current) => (current === objectKey ? null : objectKey));
    setContextMenu(null);
  }

  function clearObjectInteraction() {
    setHoveredObjectKey(null);
    setLockedObjectKey(null);
    setContextMenu(null);
  }

  function inspectObject(objectKey: string | null) {
    const object = resolveCompositeObjectRef(layers, objectKey);
    if (!object) {
      return;
    }
    setHoveredObjectKey(null);
    setLockedObjectKey(object.key);
    setContextMenu(null);
    onFocusedLayerChange(object.layer);
  }

  function openObjectContextMenu(request: CompositeObjectMenuRequest) {
    setHoveredObjectKey(null);
    setLockedObjectKey(request.objectKey);
    setContextMenu(request);
  }

  function handleObjectWheel(event: WheelEvent<HTMLElement>) {
    if (!event.altKey && !event.shiftKey) {
      return;
    }
    const wheelDelta =
      Math.abs(event.deltaY) >= Math.abs(event.deltaX) ? event.deltaY : event.deltaX;
    if (Math.abs(wheelDelta) < 1) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    navigateObject(wheelDelta > 0 ? 1 : -1);
  }

  return {
    activeObject,
    activeObjectKey,
    activeObjectIndex,
    clearObjectInteraction,
    closeContextMenu,
    contextMenu,
    contextMenuObject,
    inspectObject,
    lockedObjectKey,
    openObjectContextMenu,
    objectCount: objectRefs.length,
    onObjectWheel: handleObjectWheel,
    relatedObjectKeys,
    setHoveredObjectKey,
    toggleObjectLock
  };
}
