import { useState } from "react";

import { useWorkspaceSettings } from "./workspaceSettings";

export function useCompositeLayerCanvasController({
  labels,
  activeObjectId,
  onHover,
  onLock
}: {
  labels: string[];
  activeObjectId?: string | null;
  onHover?: (objectId: string | null) => void;
  onLock?: (objectId: string | null) => void;
}) {
  const workspaceSettings = useWorkspaceSettings(labels);
  const [localHoveredObjectId, setLocalHoveredObjectId] = useState<string | null>(null);
  const [localLockedObjectId, setLocalLockedObjectId] = useState<string | null>(null);
  const resolvedActiveObjectId = activeObjectId ?? localLockedObjectId ?? localHoveredObjectId;

  function handleHover(objectId: string | null) {
    if (onHover) {
      onHover(objectId);
      return;
    }
    setLocalHoveredObjectId(objectId);
  }

  function handleLock(objectId: string | null) {
    if (onLock) {
      onLock(objectId);
      return;
    }
    setLocalLockedObjectId((current) => (current === objectId ? null : objectId));
  }

  return {
    ...workspaceSettings,
    resolvedActiveObjectId,
    handleHover,
    handleLock
  };
}
