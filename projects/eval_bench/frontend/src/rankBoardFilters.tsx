import { AdvancedFilterBar } from "./filterControls";
import { updatePagedFilterValue } from "./samplePager";

export type RankBoardFilterValues = {
  searchText: string;
  taskFilter: string;
  benchmarkFilter: string;
  benchmarkSplitFilter: string;
  statusFilter: string;
  labelFilter: string;
  modelFilter: string;
  promptFilter: string;
  metricProfileFilter: string;
  minScoreFilter: string;
};

export type RankBoardFilterOptions = {
  tasks: string[];
  benchmarks: string[];
  benchmarkSplits: string[];
  statuses: string[];
  labels: string[];
  models: string[];
  prompts: string[];
  metricProfiles: string[];
};

export type RankBoardFilterSetters = {
  setSearchText: (value: string) => void;
  setTaskFilter: (value: string) => void;
  setBenchmarkFilter: (value: string) => void;
  setBenchmarkSplitFilter: (value: string) => void;
  setStatusFilter: (value: string) => void;
  setLabelFilter: (value: string) => void;
  setModelFilter: (value: string) => void;
  setPromptFilter: (value: string) => void;
  setMetricProfileFilter: (value: string) => void;
  setMinScoreFilter: (value: string) => void;
  setPageOffset: (value: number) => void;
};

export function RankBoardFilterBar({
  values,
  options,
  setters
}: {
  values: RankBoardFilterValues;
  options: RankBoardFilterOptions;
  setters: RankBoardFilterSetters;
}) {
  return (
    <AdvancedFilterBar
      title="筛选"
      meta="任务、基准集、状态、标签、模型、Prompt、Metric 与最低分"
      controls={[
        {
          type: "search",
          id: "rank-query",
          label: "全文检索",
          value: values.searchText,
          onChange: (value) =>
            updatePagedFilterValue(
              values.searchText,
              value,
              setters.setSearchText,
              setters.setPageOffset
            ),
          placeholder: "搜索 run、模型、prompt、备注"
        },
        {
          type: "select",
          id: "rank-task",
          label: "任务",
          value: values.taskFilter,
          values: ["all", ...options.tasks],
          labels: { all: "全部" },
          onChange: (value) =>
            updatePagedFilterValue(
              values.taskFilter,
              value,
              setters.setTaskFilter,
              setters.setPageOffset
            )
        },
        {
          type: "select",
          id: "rank-benchmark",
          label: "基准集",
          value: values.benchmarkFilter,
          values: ["all", ...options.benchmarks],
          labels: { all: "全部" },
          onChange: (value) =>
            updatePagedFilterValue(
              values.benchmarkFilter,
              value,
              setters.setBenchmarkFilter,
              setters.setPageOffset
            )
        },
        {
          type: "select",
          id: "rank-benchmark-split",
          label: "Split",
          value: values.benchmarkSplitFilter,
          values: ["all", ...options.benchmarkSplits],
          labels: { all: "全部" },
          onChange: (value) =>
            updatePagedFilterValue(
              values.benchmarkSplitFilter,
              value,
              setters.setBenchmarkSplitFilter,
              setters.setPageOffset
            )
        },
        {
          type: "select",
          id: "rank-status",
          label: "状态",
          value: values.statusFilter,
          values: ["all", ...options.statuses],
          labels: { all: "全部" },
          onChange: (value) =>
            updatePagedFilterValue(
              values.statusFilter,
              value,
              setters.setStatusFilter,
              setters.setPageOffset
            )
        },
        {
          type: "select",
          id: "rank-label",
          label: "标签",
          value: values.labelFilter,
          values: ["all", ...options.labels],
          labels: { all: "全部" },
          onChange: (value) =>
            updatePagedFilterValue(
              values.labelFilter,
              value,
              setters.setLabelFilter,
              setters.setPageOffset
            )
        },
        {
          type: "select",
          id: "rank-model",
          label: "模型",
          value: values.modelFilter,
          values: ["all", ...options.models],
          labels: { all: "全部" },
          onChange: (value) =>
            updatePagedFilterValue(
              values.modelFilter,
              value,
              setters.setModelFilter,
              setters.setPageOffset
            )
        },
        {
          type: "select",
          id: "rank-prompt",
          label: "Prompt",
          value: values.promptFilter,
          values: ["all", ...options.prompts],
          labels: { all: "全部" },
          onChange: (value) =>
            updatePagedFilterValue(
              values.promptFilter,
              value,
              setters.setPromptFilter,
              setters.setPageOffset
            )
        },
        {
          type: "select",
          id: "rank-metric",
          label: "Metric",
          value: values.metricProfileFilter,
          values: ["all", ...options.metricProfiles],
          labels: { all: "全部" },
          onChange: (value) =>
            updatePagedFilterValue(
              values.metricProfileFilter,
              value,
              setters.setMetricProfileFilter,
              setters.setPageOffset
            )
        },
        {
          type: "number",
          id: "rank-min-score",
          label: "最低分",
          value: values.minScoreFilter,
          min: 0,
          max: 1,
          step: 0.01,
          placeholder: "0.70",
          onChange: (value) =>
            updatePagedFilterValue(
              values.minScoreFilter,
              value,
              setters.setMinScoreFilter,
              setters.setPageOffset
            )
        }
      ]}
    />
  );
}

