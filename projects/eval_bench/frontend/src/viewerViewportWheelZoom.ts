import { useEffect, useRef } from "react";
import type { MutableRefObject } from "react";

import { normalizedWheelDelta } from "./viewerGeometry";
import type { InteractionSettings } from "./workspaceSettings";

type WheelZoomOptions = {
  interactionSettings: InteractionSettings;
  stageRef: MutableRefObject<HTMLDivElement | null>;
  viewportRef: MutableRefObject<{ zoom: number; pan: { x: number; y: number } }>;
  onWheelZoom: (nextZoom: number, anchor: { x: number; y: number }) => void;
};

export function useViewerViewportWheelZoom({
  interactionSettings,
  stageRef,
  viewportRef,
  onWheelZoom
}: WheelZoomOptions) {
  const onWheelZoomRef = useRef(onWheelZoom);

  useEffect(() => {
    onWheelZoomRef.current = onWheelZoom;
  }, [onWheelZoom]);

  useEffect(() => {
    const node = stageRef.current;
    if (!node) {
      return undefined;
    }
    function handleWheel(event: WheelEvent) {
      event.preventDefault();
      const stageNode = stageRef.current;
      if (!stageNode) {
        return;
      }
      const rect = stageNode.getBoundingClientRect();
      const anchor = {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top
      };
      onWheelZoomRef.current(
        viewportRef.current.zoom *
          Math.exp(-normalizedWheelDelta(event) * interactionSettings.wheelZoomSensitivity),
        anchor
      );
    }
    node.addEventListener("wheel", handleWheel, { passive: false });
    return () => node.removeEventListener("wheel", handleWheel);
  }, [interactionSettings.wheelZoomSensitivity, stageRef, viewportRef]);
}
