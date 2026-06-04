import { useCallback, useEffect, useMemo, useState } from "react";

import type { RunSampleDetail } from "./api";
import { basename, isTextInputTarget, unique } from "./formatters";
import { CanvasStage } from "./viewerCanvas";
import { displayImageUrl } from "./viewerGeometry";
import { buildObjectRows, visibleSampleMetrics } from "./viewerMetrics";
import {
  InstanceStats,
  ObjectList,
  ViewerControlPanel,
  VisibleMetricStrip,
  handleViewerShortcutAction
} from "./viewerPanels";
import { ResizableSplit } from "./workspaceLayout";
import {
  useViewerLayerPreferences,
  useWorkspaceSettings,
  useWorkspaceShortcuts
} from "./workspaceSettings";
import { OptionChipButton } from "./ui";
import { VisualStatusBar } from "./visualStatusBar";
import type { VisualStatusItem } from "./visualStatusBar";
import { ViewerPointerSurface } from "./viewerPointerSurface";

export function SampleViewer({ detail }: { detail: RunSampleDetail }) {
  return <InteractiveSampleViewer detail={detail} />;
}

function InteractiveSampleViewer({ detail }: { detail: RunSampleDetail }) {
  const width = detail.sample.image_width ?? 1000;
  const height = detail.sample.image_height ?? 1000;
  const labels = useMemo(
    () => unique([...detail.gt_instances, ...detail.pred_instances].map((instance) => instance.label)),
    [detail.gt_instances, detail.pred_instances]
  );
  const {
    activeLabels,
    setActiveLabels,
    showGt,
    setShowGt,
    showPred,
    setShowPred,
    showBoxes,
    setShowBoxes,
    showLines,
    setShowLines,
    showKeypoints,
    setShowKeypoints
  } = useViewerLayerPreferences(labels);
  const [inspectorCollapsed, setInspectorCollapsed] = useState(false);
  const [hoveredObjectId, setHoveredObjectId] = useState<string | null>(null);
  const [lockedObjectId, setLockedObjectId] = useState<string | null>(null);
  const {
    overlayColors,
    overlayStyle,
    labelColors,
    interactionSettings,
    overlayVars
  } = useWorkspaceSettings(labels);
  const activeObjectId = lockedObjectId ?? hoveredObjectId;
  const activeLabelSet = useMemo(() => new Set(activeLabels), [activeLabels]);
  const visibleGtInstances = useMemo(
    () => detail.gt_instances.filter((instance) => activeLabelSet.has(instance.label)),
    [activeLabelSet, detail.gt_instances]
  );
  const visiblePredInstances = useMemo(
    () => detail.pred_instances.filter((instance) => activeLabelSet.has(instance.label)),
    [activeLabelSet, detail.pred_instances]
  );
  const objectRows = useMemo(
    () =>
      buildObjectRows({
        gtInstances: detail.gt_instances,
        predInstances: detail.pred_instances,
        labels: activeLabelSet,
        diagnostics: detail.diagnostics
      }),
    [activeLabelSet, detail.diagnostics, detail.gt_instances, detail.pred_instances]
  );
  const visibleMetrics = visibleSampleMetrics(detail, activeLabelSet);
  const { actionForEvent } = useWorkspaceShortcuts();
  const activeObject = objectRows.find((object) => object.id === activeObjectId);
  const statusItems: VisualStatusItem[] = [
    {
      label: "Sample",
      value: `#${detail.sample.index + 1}`,
      title: detail.sample.json_path
    },
    {
      label: "GT/Pred",
      value: `${visibleMetrics.gtCount.toLocaleString()}/${visibleMetrics.predCount.toLocaleString()}`
    },
    {
      label: "Match",
      value: (detail.diagnostics?.matched_count ?? 0).toLocaleString(),
      tone: "good"
    },
    {
      label: "FN/FP",
      value: `${(detail.diagnostics?.false_negative_count ?? 0).toLocaleString()}/${(detail.diagnostics?.false_positive_count ?? 0).toLocaleString()}`,
      tone:
        (detail.diagnostics?.false_negative_count ?? 0) +
          (detail.diagnostics?.false_positive_count ?? 0) >
        0
          ? "warn"
          : "good"
    },
    {
      label: "mIoU",
      value: detail.diagnostics ? detail.diagnostics.mean_iou.toFixed(3) : "-",
      tone: detail.diagnostics && detail.diagnostics.mean_iou >= 0.7 ? "good" : "warn"
    },
    {
      label: activeObject ? "Object" : "Image",
      value: activeObject
        ? `${activeObject.kind.toUpperCase()}#${activeObject.index + 1}`
        : `${width}x${height}`,
      title: activeObject?.label ?? detail.sample.image
    }
  ];

  useEffect(() => {
    setLockedObjectId(null);
    setHoveredObjectId(null);
  }, [detail.sample.index]);

  function toggleLabel(label: string) {
    setActiveLabels((current) => {
      if (current.includes(label)) {
        return current.filter((item) => item !== label);
      }
      return unique([...current, label]);
    });
  }

  function toggleLockedObject(objectId: string | null) {
    if (objectId === null) {
      setLockedObjectId(null);
      return;
    }
    setLockedObjectId((current) => (current === objectId ? null : objectId));
  }

  const previewObject = useCallback(
    (objectId: string | null) => {
      if (lockedObjectId) {
        return;
      }
      setHoveredObjectId((current) => (current === objectId ? current : objectId));
    },
    [lockedObjectId]
  );

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (isTextInputTarget(event.target)) {
        return;
      }
      const actionId = actionForEvent(event);
      if (!actionId) {
        return;
      }
      if (handleViewerShortcutAction(actionId, {
        clearSelection: () => {
          setLockedObjectId(null);
          setHoveredObjectId(null);
        },
        toggleGt: () => setShowGt((value) => !value),
        togglePred: () => setShowPred((value) => !value),
        toggleBoxes: () => setShowBoxes((value) => !value),
        toggleLines: () => setShowLines((value) => !value),
        toggleKeypoints: () => setShowKeypoints((value) => !value)
      })) {
        event.preventDefault();
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [actionForEvent]);

  const canvasStage = (
    <div className="viewer-stage-shell">
      <VisualStatusBar
        className="viewer-visual-status"
        title={basename(detail.sample.image)}
        subtitle={detail.run_id}
        items={statusItems}
        actions={
          <OptionChipButton onClick={() => setInspectorCollapsed((value) => !value)}>
            {inspectorCollapsed ? "检查器" : "收起"}
          </OptionChipButton>
        }
      />
      <ViewerPointerSurface>
        <CanvasStage
          width={width}
          height={height}
          imageUrl={displayImageUrl(detail.sample)}
          imageAlt={detail.sample.image}
          imageTileUrlTemplate={detail.sample.image_tile_url_template}
          imageTileSize={detail.sample.image_tile_size}
          gtInstances={detail.gt_instances}
          predInstances={detail.pred_instances}
          diagnostics={detail.diagnostics}
          visibleLabels={activeLabelSet}
          showGt={showGt}
          showPred={showPred}
          showBoxes={showBoxes}
          showLines={showLines}
          showKeypoints={showKeypoints}
          activeObjectId={activeObjectId}
          overlayColors={overlayColors}
          overlayStyle={overlayStyle}
          labelColors={labelColors}
          interactionSettings={interactionSettings}
          onHover={previewObject}
          onLock={toggleLockedObject}
        />
      </ViewerPointerSurface>
    </div>
  );
  const inspectorPanel = (
    <aside className="viewer-side-panel">
      <ViewerControlPanel
        labels={labels}
        activeLabels={activeLabels}
        showGt={showGt}
        showPred={showPred}
        showBoxes={showBoxes}
        showLines={showLines}
        showKeypoints={showKeypoints}
        onToggleLabel={toggleLabel}
        onShowGtChange={setShowGt}
        onShowPredChange={setShowPred}
        onShowBoxesChange={setShowBoxes}
        onShowLinesChange={setShowLines}
        onShowKeypointsChange={setShowKeypoints}
      />
      <VisibleMetricStrip metrics={visibleMetrics} />
      <div className="instance-workbench">
        <InstanceStats title="真值实例" instances={visibleGtInstances} />
        <InstanceStats title="预测实例" instances={visiblePredInstances} />
        <ObjectList
          objects={objectRows}
          activeObjectId={activeObjectId}
          lockedObjectId={lockedObjectId}
          onHover={previewObject}
          onLock={toggleLockedObject}
        />
      </div>
    </aside>
  );

  return (
    <div className="viewer-stack" style={overlayVars}>
      {inspectorCollapsed ? (
        <div className="viewer-canvas-layout side-collapsed">{canvasStage}</div>
      ) : (
        <ResizableSplit
          className="viewer-canvas-layout"
          storageKey="eval_bench_viewer_inspector_width"
          fixedPane="second"
          defaultSize={224}
          minSize={176}
          maxSize={560}
          first={canvasStage}
          second={inspectorPanel}
        />
      )}
    </div>
  );
}
