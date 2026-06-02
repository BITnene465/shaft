import { useState } from "react";

import type { RankBoard, SuiteRankBoard } from "./api";
import { OptionChipButton } from "./ui";

import "./rankBoardSummary.css";
import "./rankBoardFacets.css";

export function SuiteRankSummary({ board }: { board: SuiteRankBoard }) {
  return (
    <div className="rank-board-summary" aria-label="Suite 排行榜概要">
      <strong>Suite aggregate</strong>
      <span>{board.total.toLocaleString()} campaigns</span>
      <span>{board.evaluated_count.toLocaleString()} evaluated</span>
      <span>{facetTotal(board, "suites").toLocaleString()} suites</span>
      <span>{facetTotal(board, "models").toLocaleString()} models</span>
    </div>
  );
}

export function RankBoardSummary({
  board,
  runCount
}: {
  board: RankBoard;
  runCount: number;
}) {
  return (
    <div className="rank-board-summary" aria-label="排行榜概要">
      <strong>Leaderboard</strong>
      <span>{board.total.toLocaleString()} runs</span>
      <span>{board.evaluated_count.toLocaleString()} evaluated</span>
      <span>{facetTotal(board, "benchmarks").toLocaleString()} benchmarks</span>
      <span>{runCount.toLocaleString()} total</span>
    </div>
  );
}

export type RankFacetFilterValues = {
  task: string;
  benchmark: string;
  split: string;
  status: string;
  label: string;
  model: string;
  prompt: string;
  metricProfile: string;
};

export type RankFacetFilterHandlers = {
  task: (value: string) => void;
  benchmark: (value: string) => void;
  split: (value: string) => void;
  status: (value: string) => void;
  label: (value: string) => void;
  model: (value: string) => void;
  prompt: (value: string) => void;
  metricProfile: (value: string) => void;
};

export function RankFacetRail({
  board,
  filters,
  onFilterChange
}: {
  board: Pick<RankBoard, "facets">;
  filters: RankFacetFilterValues;
  onFilterChange: RankFacetFilterHandlers;
}) {
  const groups = [
    {
      title: "Tasks",
      items: board.facets.tasks ?? [],
      activeValue: filters.task,
      onSelect: onFilterChange.task
    },
    {
      title: "Benchmarks",
      items: board.facets.benchmarks ?? [],
      activeValue: filters.benchmark,
      onSelect: onFilterChange.benchmark
    },
    {
      title: "Splits",
      items: board.facets.splits ?? [],
      activeValue: filters.split,
      onSelect: onFilterChange.split
    },
    {
      title: "Status",
      items: board.facets.statuses ?? [],
      activeValue: filters.status,
      onSelect: onFilterChange.status
    },
    {
      title: "Labels",
      items: board.facets.labels ?? [],
      activeValue: filters.label,
      onSelect: onFilterChange.label
    },
    {
      title: "Models",
      items: board.facets.models ?? [],
      activeValue: filters.model,
      onSelect: onFilterChange.model
    },
    {
      title: "Prompts",
      items: board.facets.prompts ?? [],
      activeValue: filters.prompt,
      onSelect: onFilterChange.prompt
    },
    {
      title: "Metrics",
      items: board.facets.metric_profiles ?? [],
      activeValue: filters.metricProfile,
      onSelect: onFilterChange.metricProfile
    }
  ].filter((group) => group.items.length > 0);
  if (groups.length === 0) {
    return null;
  }
  return (
    <div className="rank-facet-rail">
      {groups.map((group) => (
        <RankFacetGroup
          key={group.title}
          title={group.title}
          items={group.items}
          activeValue={group.activeValue}
          onSelect={group.onSelect}
        />
      ))}
    </div>
  );
}

export function SuiteRankFacetRail({ board }: { board: Pick<SuiteRankBoard, "facets"> }) {
  const groups = [
    { title: "Suites", items: board.facets.suites ?? [] },
    { title: "Models", items: board.facets.models ?? [] },
    { title: "Prompts", items: board.facets.prompts ?? [] },
    { title: "Task splits", items: board.facets.task_splits ?? [] }
  ].filter((group) => group.items.length > 0);
  if (groups.length === 0) {
    return null;
  }
  return (
    <div className="rank-facet-rail suite-rank-facet-rail">
      {groups.map((group) => (
        <section className="rank-facet-group" key={group.title}>
          <span>{group.title}</span>
          <div>
            {group.items.slice(0, 8).map((item) => (
              <span className="rank-facet-readonly" key={item.value} title={item.value}>
                <span>{item.value}</span>
                <strong>{item.count.toLocaleString()}</strong>
              </span>
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}

function RankFacetGroup({
  title,
  items,
  activeValue,
  onSelect
}: {
  title: string;
  items: Array<{ value: string; count: number }>;
  activeValue: string;
  onSelect: (value: string) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const visibleItems = expanded ? items : items.slice(0, 5);
  const hiddenCount = Math.max(0, items.length - visibleItems.length);
  return (
    <section className={expanded ? "rank-facet-group expanded" : "rank-facet-group"}>
      <span>{title}</span>
      <div>
        {visibleItems.map((item) => {
          const active = activeValue === item.value;
          return (
            <OptionChipButton
              className="rank-facet-button"
              active={active}
              key={item.value}
              title={`${title}: ${item.value}`}
              onClick={() => onSelect(active ? "all" : item.value)}
            >
              <span>{item.value}</span>
              <strong>{item.count.toLocaleString()}</strong>
            </OptionChipButton>
          );
        })}
        {items.length > 5 ? (
          <OptionChipButton
            className="rank-facet-toggle"
            active={expanded}
            title={expanded ? `${title}: 收起 facet` : `${title}: 展开全部 facet`}
            onClick={() => setExpanded((value) => !value)}
          >
            {expanded ? "收起" : `展开全部 +${hiddenCount}`}
          </OptionChipButton>
        ) : null}
        {items.length === 0 ? <em>无</em> : null}
      </div>
    </section>
  );
}

function facetTotal(board: Pick<RankBoard, "facets">, key: string) {
  return board.facets[key]?.length ?? 0;
}
