import { useEffect } from "react";

import type { CompositeSampleView } from "./api";
import { useCompositeObjectInteraction } from "./compositeObjectInteractionController";

export function useCompositeReportStageController({
  composite,
  focusedLayerKey,
  onFocusedLayerChange
}: {
  composite?: CompositeSampleView;
  focusedLayerKey: string | null;
  onFocusedLayerChange: (layer: string | null) => void;
}) {
  const layers = composite?.layers ?? [];
  const statuses = composite?.layer_statuses ?? [];
  const viewportSyncKey = composite ? `composite:${composite.image_key}` : null;
  const focusAvailable = Boolean(
    focusedLayerKey && statuses.some((status) => status.layer === focusedLayerKey)
  );
  const activeFocusedLayerKey = focusAvailable ? focusedLayerKey : null;
  const focusedLayers = activeFocusedLayerKey
    ? layers.filter((layer) => layer.layer === activeFocusedLayerKey)
    : layers;
  const focusedStatuses = activeFocusedLayerKey
    ? statuses.filter((status) => status.layer === activeFocusedLayerKey)
    : statuses;
  const objectInteraction = useCompositeObjectInteraction({
    imageKey: composite?.image_key,
    layers,
    onFocusedLayerChange
  });

  useEffect(() => {
    if (
      focusedLayerKey &&
      statuses.length > 0 &&
      !statuses.some((status) => status.layer === focusedLayerKey)
    ) {
      onFocusedLayerChange(null);
    }
  }, [focusedLayerKey, onFocusedLayerChange, statuses]);

  return {
    activeFocusedLayerKey,
    focusedLayers,
    focusedStatuses,
    layers,
    objectInteraction,
    statuses,
    viewportSyncKey
  };
}

export type CompositeReportStageState = ReturnType<typeof useCompositeReportStageController>;
