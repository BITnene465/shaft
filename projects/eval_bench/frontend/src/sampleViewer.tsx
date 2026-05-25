import { useEffect, useMemo, useState } from "react";

import type { RunSampleDetail } from "./api";
import { basename, isTextInputTarget, unique } from "./formatters";
import { CanvasStage } from "./viewerCanvas";
import { displayImageUrl } from "./viewerGeometry";
import {
  buildObjectRows,
  visibleLabelMetrics,
  visibleSampleMetrics
} from "./viewerMetrics";
import {
  InstanceStats,
  LabelMetricTable,
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
  const labelMetrics = visibleLabelMetrics(detail, activeLabelSet);
  const { actionForEvent } = useWorkspaceShortcuts();

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
      onHover={setHoveredObjectId}
      onLock={toggleLockedObject}
    />
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
      <LabelMetricTable rows={labelMetrics} />
      <div className="instance-workbench">
        <InstanceStats title="真值实例" instances={visibleGtInstances} />
        <InstanceStats title="预测实例" instances={visiblePredInstances} />
        <ObjectList
          objects={objectRows}
          activeObjectId={activeObjectId}
          lockedObjectId={lockedObjectId}
          onHover={setHoveredObjectId}
          onLock={toggleLockedObject}
        />
      </div>
    </aside>
  );

  return (
    <div className="viewer-stack" style={overlayVars}>
      <div className="viewer-toolbar">
        <div>
          <h2>{basename(detail.sample.image)}</h2>
          <p>{detail.sample.image}</p>
        </div>
        <div className="legend-row">
          <span className="legend-item gt">真值匹配</span>
          <span className="legend-item fn">漏检</span>
          <span className="legend-item pred">预测匹配</span>
          <span className="legend-item fp">误检</span>
          <button
            className="query-chip"
            type="button"
            onClick={() => setInspectorCollapsed((value) => !value)}
          >
            {inspectorCollapsed ? "显示检查器" : "收起检查器"}
          </button>
        </div>
      </div>
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
