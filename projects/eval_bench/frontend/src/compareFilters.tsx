import { AdvancedFilterBar } from "./filterControls";
import { updatePagedFilterValue } from "./samplePager";

export type CompareFilterValues = {
  searchText: string;
  statusFilter: string;
  taskFilter: string;
  benchmarkFilter: string;
  benchmarkSplitFilter: string;
  labelFilter: string;
  modelFilter: string;
  promptFilter: string;
  historyBaselineFilter: string;
  historyCandidateFilter: string;
};

export type CompareFilterOptions = {
  statuses: string[];
  tasks: string[];
  benchmarks: string[];
  benchmarkSplits: string[];
  labels: string[];
  models: string[];
  prompts: string[];
};

export type CompareFilterSetters = {
  setSearchText: (value: string) => void;
  setStatusFilter: (value: string) => void;
  setTaskFilter: (value: string) => void;
  setBenchmarkFilter: (value: string) => void;
  setBenchmarkSplitFilter: (value: string) => void;
  setLabelFilter: (value: string) => void;
  setModelFilter: (value: string) => void;
  setPromptFilter: (value: string) => void;
  setHistoryBaselineFilter: (value: string) => void;
  setHistoryCandidateFilter: (value: string) => void;
  setPageOffset: (offset: number) => void;
  setHistoryOffset: (offset: number) => void;
};

export function CompareFilterBar({
  values,
  options,
  setters
}: {
  values: CompareFilterValues;
  options: CompareFilterOptions;
  setters: CompareFilterSetters;
}) {
  return (
    <AdvancedFilterBar
      title="对比高级检索"
      meta="筛选候选 run：状态、任务、基准集、label、模型、prompt 和备注全文"
      controls={[
        {
          type: "search",
          id: "compare-query",
          label: "全文检索",
          value: values.searchText,
          onChange: (value) =>
            updatePagedFilterValue(
              values.searchText,
              value,
              setters.setSearchText,
              setters.setPageOffset,
              setters.setHistoryOffset
            ),
          placeholder: "搜索 run、模型、prompt、备注"
        },
        {
          type: "select",
          id: "compare-status",
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
          id: "compare-task",
          label: "任务",
          value: values.taskFilter,
          values: ["all", ...options.tasks],
          labels: { all: "全部" },
          onChange: (value) =>
            updatePagedFilterValue(
              values.taskFilter,
              value,
              setters.setTaskFilter,
              setters.setPageOffset,
              setters.setHistoryOffset
            )
        },
        {
          type: "select",
          id: "compare-benchmark",
          label: "基准集",
          value: values.benchmarkFilter,
          values: ["all", ...options.benchmarks],
          labels: { all: "全部" },
          onChange: (value) =>
            updatePagedFilterValue(
              values.benchmarkFilter,
              value,
              setters.setBenchmarkFilter,
              setters.setPageOffset,
              setters.setHistoryOffset
            )
        },
        {
          type: "select",
          id: "compare-benchmark-split",
          label: "Split",
          value: values.benchmarkSplitFilter,
          values: ["all", ...options.benchmarkSplits],
          labels: { all: "全部" },
          onChange: (value) =>
            updatePagedFilterValue(
              values.benchmarkSplitFilter,
              value,
              setters.setBenchmarkSplitFilter,
              setters.setPageOffset,
              setters.setHistoryOffset
            )
        },
        {
          type: "select",
          id: "compare-label",
          label: "标签",
          value: values.labelFilter,
          values: ["all", ...options.labels],
          labels: { all: "全部" },
          onChange: (value) =>
            updatePagedFilterValue(
              values.labelFilter,
              value,
              setters.setLabelFilter,
              setters.setPageOffset,
              setters.setHistoryOffset
            )
        },
        {
          type: "select",
          id: "compare-model",
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
          id: "compare-prompt",
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
          type: "text",
          id: "compare-history-baseline",
          label: "历史基线",
          value: values.historyBaselineFilter,
          onChange: (value) =>
            updatePagedFilterValue(
              values.historyBaselineFilter,
              value,
              setters.setHistoryBaselineFilter,
              setters.setHistoryOffset
            ),
          placeholder: "baseline run id"
        },
        {
          type: "text",
          id: "compare-history-candidate",
          label: "历史候选",
          value: values.historyCandidateFilter,
          onChange: (value) =>
            updatePagedFilterValue(
              values.historyCandidateFilter,
              value,
              setters.setHistoryCandidateFilter,
              setters.setHistoryOffset
            ),
          placeholder: "candidate run id"
        }
      ]}
    />
  );
}

