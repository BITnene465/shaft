import type { CompositeObjectRef } from "./compositeObjectModel";
import {
  localCanvasObjectIdToKey,
  parseCompositeObjectKey
} from "./compositeObjectModel";

export function overlayObjectIdForKey(
  objectRefs: CompositeObjectRef[],
  objectKey: string | null
) {
  if (!objectKey) {
    return null;
  }
  return objectRefs.find((object) => object.key === objectKey)?.overlayObjectId ?? null;
}

export function objectKeyForOverlayObject(
  objectRefs: CompositeObjectRef[],
  objectId: string | null
) {
  if (!objectId) {
    return null;
  }
  return objectRefs.find((object) => object.overlayObjectId === objectId)?.key ?? null;
}

export function relatedOverlayObjectIds(
  objectRefs: CompositeObjectRef[],
  relatedObjectKeys: Set<string>
) {
  return new Set(
    objectRefs
      .filter((object) => relatedObjectKeys.has(object.key))
      .map((object) => object.overlayObjectId)
  );
}

export function localObjectIdForKey(layer: string, objectKey: string | null) {
  if (!objectKey) {
    return null;
  }
  const object = parseCompositeObjectKey(objectKey);
  return object?.layer === layer ? `${object.kind}:${object.index}` : null;
}

export function relatedLocalObjectIds(layer: string, relatedObjectKeys: Set<string>) {
  const values = new Set<string>();
  relatedObjectKeys.forEach((objectKey) => {
    const localObjectId = localObjectIdForKey(layer, objectKey);
    if (localObjectId) {
      values.add(localObjectId);
    }
  });
  return values;
}

export function objectKeyForLocalObject(layer: string, objectId: string | null) {
  return objectId ? localCanvasObjectIdToKey(layer, objectId) : null;
}
