import { Plus, Search } from "lucide-react";

import type { RunSummary } from "./api";
import { layerColor } from "./compositeLayerPalette";
import { layerIndex, LAYER_FILTERS } from "./compositeReportComposerModel";
import type { LayerFilter } from "./compositeReportComposerModel";
import { inferLayerName, runOptionLabel } from "./compositeReportModel";
import { CompositeReportPanelHeader } from "./compositeReportPanel";
import { SearchInputControl } from "./controlPrimitives";
import { formatMetric } from "./formatters";
import { Badge, IconActionButton, OptionChipButton } from "./ui";

import "./compositeReportRunPool.css";

export function ReportRunPool({
  runs,
  total,
  selectedRunIds,
  query,
  layerFilter,
  onQueryChange,
  onLayerFilterChange,
  onAddRun
}: {
  runs: RunSummary[];
  total: number;
  selectedRunIds: Set<string>;
  query: string;
  layerFilter: LayerFilter;
  onQueryChange: (value: string) => void;
  onLayerFilterChange: (value: LayerFilter) => void;
  onAddRun: (run: RunSummary) => void;
}) {
  return (
    <aside className="report-run-pool">
      <CompositeReportPanelHeader
        eyebrow="Result Pool"
        title="评测结果池"
        action={<Badge value={`${runs.length}/${total}`} />}
      />
      <SearchInputControl
        className="report-run-search"
        label="检索评测结果"
        value={query}
        icon={<Search size={15} />}
        placeholder="run / model / task / label"
        onChange={onQueryChange}
      />
      <div className="report-run-filter-tabs" aria-label="结果层类型">
        {LAYER_FILTERS.map((item) => (
          <OptionChipButton
            active={layerFilter === item.value}
            key={item.value}
            onClick={() => onLayerFilterChange(item.value)}
          >
            {item.label}
          </OptionChipButton>
        ))}
      </div>
      <div className="report-run-list">
        {runs.map((run) => {
          const selected = selectedRunIds.has(run.run_id);
          const layer = inferLayerName(run);
          return (
            <article
              className={selected ? "report-run-card selected" : "report-run-card"}
              key={run.run_id}
            >
              <div
                className="report-run-color"
                style={{ background: layerColor(layerIndex(layer)) }}
              />
              <div className="report-run-main">
                <strong title={run.run_id}>{run.run_id}</strong>
                <span title={runOptionLabel(run)}>{run.model_id}</span>
                <div className="report-run-meta">
                  <Badge value={layer} />
                  <em>{run.benchmark_split || run.spec_task || "split"}</em>
                  <em>{formatMetric(run.f1_iou50)}</em>
                </div>
              </div>
              <IconActionButton
                title={selected ? "已在报告中，继续加入一个新图层" : "加入组合报告"}
                icon={<Plus size={14} />}
                onClick={() => onAddRun(run)}
              />
            </article>
          );
        })}
      </div>
    </aside>
  );
}
