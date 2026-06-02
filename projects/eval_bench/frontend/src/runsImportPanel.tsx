import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";

import type { BenchmarkSummary } from "./api";
import { importPredictions } from "./api";
import {
  CheckboxFieldControl,
  FormSelectControl,
  TextInputControl
} from "./controlPrimitives";
import { errorMessage } from "./formatters";
import { AppIcon } from "./iconLibrary";
import { DetectionLabelSubtaskPanel } from "./labelSubtaskControls";
import { ActionButton } from "./ui";

export function ImportPredictionsPanel({
  benchmarks,
  bare
}: {
  benchmarks: BenchmarkSummary[];
  bare?: boolean;
}) {
  const queryClient = useQueryClient();
  const [runId, setRunId] = useState("");
  const [benchmarkId, setBenchmarkId] = useState(benchmarks[0]?.benchmark_id ?? "");
  const [predictionRoot, setPredictionRoot] = useState("");
  const [task, setTask] = useState("detection");
  const [modelId, setModelId] = useState("");
  const [modelPath, setModelPath] = useState("imported");
  const [promptId, setPromptId] = useState("imported");
  const [benchmarkSplit, setBenchmarkSplit] = useState("auto");
  const [targetLabels, setTargetLabels] = useState<string[]>([]);
  const [specId, setSpecId] = useState("");
  const [strict, setStrict] = useState(false);
  const [overwrite, setOverwrite] = useState(false);
  const [evaluate, setEvaluate] = useState(true);
  const mutation = useMutation({
    mutationFn: importPredictions,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["dashboard-state"] });
      void queryClient.invalidateQueries({ queryKey: ["runs"] });
      void queryClient.invalidateQueries({ queryKey: ["rank-board"] });
      void queryClient.invalidateQueries({ queryKey: ["comparisons"] });
    }
  });
  const effectiveBenchmarkId = benchmarkId || benchmarks[0]?.benchmark_id || "";
  const selectedBenchmark = benchmarks.find(
    (benchmark) => benchmark.benchmark_id === effectiveBenchmarkId
  );
  const labelOptions = selectedBenchmark?.labels ?? [];
  const benchmarkSplitOptions = benchmarkImportSplitOptions(selectedBenchmark);
  useEffect(() => {
    if (task !== "detection") {
      setTargetLabels([]);
      return;
    }
    if (labelOptions.length > 0) {
      setTargetLabels((current) => current.filter((label) => labelOptions.includes(label)));
    }
  }, [task, effectiveBenchmarkId, labelOptions.join("\u0000")]);
  useEffect(() => {
    setBenchmarkSplit("auto");
  }, [effectiveBenchmarkId]);

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    mutation.mutate({
      run_id: runId.trim(),
      benchmark_id: effectiveBenchmarkId,
      prediction_root: predictionRoot.trim(),
      task,
      model_id: modelId.trim(),
      model_path: modelPath.trim() || "imported",
      prompt_id: promptId.trim() || "imported",
      spec_id: specId.trim() || undefined,
      split: benchmarkSplit === "auto" ? undefined : benchmarkSplit,
      target_labels: targetLabels,
      strict,
      overwrite,
      evaluate
    });
  }

  const benchmarkSelectOptions =
    benchmarks.length === 0
      ? [{ value: "", label: "暂无基准集" }]
      : benchmarks.map((benchmark) => ({
          value: benchmark.benchmark_id,
          label: benchmark.benchmark_id
        }));
  const content = (
    <form className="job-form import-form" onSubmit={submit}>
      <TextInputControl
        label="记录 ID"
        value={runId}
        onChange={setRunId}
        placeholder="model-a_val_import"
        required
      />
      <FormSelectControl
        label="基准集"
        value={effectiveBenchmarkId}
        options={benchmarkSelectOptions}
        required
        onChange={setBenchmarkId}
      />
      <TextInputControl
        className="wide-field"
        label="预测目录"
        value={predictionRoot}
        onChange={setPredictionRoot}
        placeholder="/path/to/prediction_json_dir"
        required
      />
      <FormSelectControl
        label="任务"
        value={task}
        options={[
          { value: "detection", label: "检测" },
          { value: "keypoint", label: "关键点" }
        ]}
        onChange={setTask}
      />
      <TextInputControl
        label="模型 ID"
        value={modelId}
        onChange={setModelId}
        placeholder="qwen3vl-best"
        required
      />
      <TextInputControl
        className="wide-field"
        label="模型路径"
        value={modelPath}
        onChange={setModelPath}
      />
      <TextInputControl label="Prompt" value={promptId} onChange={setPromptId} />
      <FormSelectControl
        label="Benchmark split"
        value={benchmarkSplit}
        options={benchmarkSplitOptions}
        onChange={setBenchmarkSplit}
      />
      <DetectionLabelSubtaskPanel
        className="full-field"
        task={task}
        benchmarkId={effectiveBenchmarkId}
        labelOptions={labelOptions}
        selectedLabels={targetLabels}
        onChange={setTargetLabels}
      />
      <TextInputControl label="规格" value={specId} onChange={setSpecId} placeholder="optional" />
      <CheckboxFieldControl label="严格导入" checked={strict} onChange={setStrict} />
      <CheckboxFieldControl label="覆盖已有 run" checked={overwrite} onChange={setOverwrite} />
      <CheckboxFieldControl label="导入后计算指标" checked={evaluate} onChange={setEvaluate} />
      <ActionButton
        type="submit"
        variant="primary"
        icon={<AppIcon name="importPrediction" size={16} />}
        disabled={mutation.isPending || benchmarks.length === 0}
      >
        导入
      </ActionButton>
      {mutation.data ? (
        <div className="form-result full-field">
          已导入 {mutation.data.imported_predictions.toLocaleString()} 条预测，缺失{" "}
          {mutation.data.missing_prediction_count.toLocaleString()} 条。{" "}
          <Link to="/runs/$runId" params={{ runId: mutation.data.run_id }}>
            打开 run
          </Link>
        </div>
      ) : null}
      {mutation.error ? (
        <div className="form-result error full-field">{errorMessage(mutation.error)}</div>
      ) : null}
    </form>
  );
  return bare ? content : <div className="workspace-card compact-form-card">{content}</div>;
}

function benchmarkImportSplitOptions(benchmark: BenchmarkSummary | undefined) {
  const values = new Set<string>();
  if (benchmark?.split) {
    values.add(benchmark.split);
  }
  Object.keys(benchmark?.split_manifests ?? {}).forEach((value) => {
    if (value.trim()) {
      values.add(value);
    }
  });
  return [
    { value: "auto", label: "自动推断" },
    ...Array.from(values)
      .sort((left, right) => left.localeCompare(right))
      .map((value) => ({ value, label: value }))
  ];
}
