import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { fetchCompositeSample } from "./api";
import type { RunSummary } from "./api";
import {
  filterReportRuns,
  groupSlots,
  pickLayerPreset
} from "./compositeReportComposerModel";
import type { LayerFilter } from "./compositeReportComposerModel";
import {
  SAMPLE_MAX_FALLBACK,
  inferLayerName,
  isRun,
  uniqueLayerKey
} from "./compositeReportModel";
import type { ActiveLayerConfig, LayerSlot } from "./compositeReportModel";
import {
  loadCompositeReportViewState,
  reconcileCompositeReportSlots,
  sameLayerSlots,
  saveCompositeReportViewState
} from "./compositeReportViewState";
import { useDashboardState } from "./dashboardState";

export function useCompositeReportController() {
  const stateQuery = useDashboardState();
  const runs = stateQuery.data?.runs ?? [];
  const reportRuns = useMemo(
    () => runs.filter((run) => run.report_path || run.report_count > 0 || run.prediction_count > 0),
    [runs]
  );
  const runById = useMemo(() => new Map(runs.map((run) => [run.run_id, run])), [runs]);
  const initialViewState = useMemo(() => loadCompositeReportViewState(), []);
  const [slots, setSlots] = useState<LayerSlot[]>(initialViewState.slots);
  const [sampleIndex, setSampleIndex] = useState(initialViewState.sampleIndex);
  const [focusedLayerKey, setFocusedLayerKey] = useState<string | null>(
    initialViewState.focusedLayerKey
  );
  const [query, setQuery] = useState(initialViewState.query);
  const [layerFilter, setLayerFilter] = useState<LayerFilter>(initialViewState.layerFilter);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  useEffect(() => {
    setSlots((current) => {
      const nextSlots = reconcileCompositeReportSlots(current, reportRuns);
      return sameLayerSlots(current, nextSlots) ? current : nextSlots;
    });
  }, [reportRuns]);

  const activeSlots = useMemo(
    () => slots.filter((slot) => slot.visible && slot.layer.trim() && slot.runId.trim()),
    [slots]
  );
  const activeLayerConfigs = useMemo<ActiveLayerConfig[]>(() => {
    const values: Record<string, string> = {};
    return activeSlots.map((slot, index) => {
      const key = uniqueLayerKey(slot.layer, values, index);
      values[key] = slot.runId;
      return { ...slot, key };
    });
  }, [activeSlots]);
  const selectedRuns = useMemo(
    () => activeSlots.map((slot) => runById.get(slot.runId)).filter(isRun),
    [activeSlots, runById]
  );
  const maxSampleIndex = selectedRuns.length
    ? Math.max(0, Math.max(...selectedRuns.map((run) => Math.max(1, run.prediction_count))) - 1)
    : SAMPLE_MAX_FALLBACK;
  const layerRuns = useMemo(() => {
    const values: Record<string, string> = {};
    activeLayerConfigs.forEach((slot) => {
      values[slot.key] = slot.runId;
    });
    return values;
  }, [activeLayerConfigs]);
  const compositeEnabled = Object.keys(layerRuns).length >= 2;
  const compositeQuery = useQuery({
    queryKey: ["composite-report-sample", layerRuns, sampleIndex],
    queryFn: ({ signal }) => fetchCompositeSample({ sampleIndex, layerRuns }, { signal }),
    enabled: compositeEnabled,
    placeholderData: (previousData) => previousData
  });
  const compositeMaxSampleIndex =
    compositeQuery.data?.image_count ? Math.max(0, compositeQuery.data.image_count - 1) : maxSampleIndex;

  useEffect(() => {
    if (sampleIndex > compositeMaxSampleIndex) {
      setSampleIndex(compositeMaxSampleIndex);
    }
  }, [compositeMaxSampleIndex, sampleIndex]);

  useEffect(() => {
    saveCompositeReportViewState({
      slots,
      sampleIndex,
      focusedLayerKey,
      query,
      layerFilter,
      sidebarOpen: false
    });
  }, [focusedLayerKey, layerFilter, query, sampleIndex, slots]);

  const filteredRuns = useMemo(
    () => filterReportRuns(reportRuns, query, layerFilter),
    [layerFilter, query, reportRuns]
  );
  const selectedRunIds = useMemo(() => new Set(slots.map((slot) => slot.runId)), [slots]);
  const groups = useMemo(() => groupSlots(slots, runById), [runById, slots]);
  const readyLayerCount = compositeQuery.data?.layer_statuses.filter((status) => status.available).length ?? 0;
  const missingLayerCount = compositeQuery.data
    ? compositeQuery.data.layer_statuses.length - readyLayerCount
    : 0;

  function addRun(run: RunSummary) {
    const layer = inferLayerName(run);
    setSlots((current) => [
      ...current,
      {
        id: `slot_${Date.now()}_${current.length}`,
        layer,
        runId: run.run_id,
        visible: true,
        showGt: true,
        showPred: true
      }
    ]);
  }

  function updateSlot(id: string, patch: Partial<LayerSlot>) {
    setSlots((current) => current.map((slot) => (slot.id === id ? { ...slot, ...patch } : slot)));
  }

  function removeSlot(id: string) {
    setSlots((current) => current.filter((slot) => slot.id !== id));
  }

  function resetComposition() {
    setSlots(reconcileCompositeReportSlots([], reportRuns));
    setSampleIndex(0);
    setFocusedLayerKey(null);
  }

  function applyLayoutArrowPreset() {
    const picked = pickLayerPreset(reportRuns, ["layout", "arrow"]);
    if (picked.length < 2) {
      return;
    }
    setSlots(
      picked.map((run, index) => ({
        id: `preset_${Date.now()}_${index}`,
        layer: inferLayerName(run),
        runId: run.run_id,
        visible: true,
        showGt: true,
        showPred: true
      }))
    );
    setSampleIndex(0);
    setFocusedLayerKey(null);
  }

  return {
    stateQuery,
    reportRuns,
    runById,
    activeSlots,
    activeLayerConfigs,
    filteredRuns,
    selectedRunIds,
    groups,
    readyLayerCount,
    missingLayerCount,
    compositeEnabled,
    compositeQuery,
    focusedLayerKey,
    query,
    layerFilter,
    sidebarOpen,
    setSampleIndex,
    setFocusedLayerKey,
    setQuery,
    setLayerFilter,
    setSidebarOpen,
    addRun,
    updateSlot,
    removeSlot,
    resetComposition,
    applyLayoutArrowPreset
  };
}

export type CompositeReportController = ReturnType<typeof useCompositeReportController>;
