import { useMemo, useState } from "react";
import type { FormEvent } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";

import { createBenchmark } from "./api";
import { CheckboxFieldControl, TextareaControl, TextInputControl } from "./controlPrimitives";
import { errorMessage } from "./formatters";
import { AppIcon } from "./iconLibrary";
import { parseBenchmarkSlices } from "./benchmarkModel";
import { ActionButton } from "./ui";

export function BenchmarkCreatePanel({ bare }: { bare?: boolean }) {
  const queryClient = useQueryClient();
  const [benchmarkId, setBenchmarkId] = useState("");
  const [sourceRoot, setSourceRoot] = useState("data/raw_data");
  const [sourceManifest, setSourceManifest] = useState("data/raw_data/splits/layout_val.txt");
  const [split, setSplit] = useState("val");
  const [suiteSlices, setSuiteSlices] = useState("");
  const [tasks, setTasks] = useState<string[]>(["detection", "keypoint"]);
  const [layers, setLayers] = useState("layout,arrow");
  const [overwrite, setOverwrite] = useState(false);
  const suiteSliceParse = useMemo(
    () => parseBenchmarkSlices(suiteSlices, tasks, layers),
    [suiteSlices, tasks, layers]
  );
  const mutation = useMutation({
    mutationFn: createBenchmark,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["dashboard-state"] });
      void queryClient.invalidateQueries({ queryKey: ["benchmarks"] });
    }
  });

  function toggleTask(task: string) {
    setTasks((current) => {
      if (current.includes(task)) {
        return current.filter((item) => item !== task);
      }
      return [...current, task];
    });
  }

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (suiteSliceParse.error) {
      return;
    }
    const slices = suiteSliceParse.slices;
    const suiteMode = slices.length > 0;
    const normalizedSplit = split.trim();
    const benchmarkSplit =
      suiteMode && (!normalizedSplit || normalizedSplit === "val")
        ? "suite"
        : normalizedSplit || "val";
    mutation.mutate({
      benchmark_id: benchmarkId.trim(),
      source_root: sourceRoot.trim(),
      source_manifest: suiteMode ? undefined : sourceManifest.trim(),
      split: benchmarkSplit,
      tasks,
      layers: layers
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean),
      slices: suiteMode ? slices : undefined,
      default_slice: slices[0]?.split,
      overwrite
    });
  }

  const content = (
    <form className="job-form benchmark-form" onSubmit={submit}>
      <TextInputControl
        label="基准集 ID"
        value={benchmarkId}
        onChange={setBenchmarkId}
        placeholder="grounding_layout_main"
        required
      />
      <TextInputControl
        className="wide-field"
        label="数据根目录"
        value={sourceRoot}
        onChange={setSourceRoot}
        required
      />
      <TextInputControl
        className="wide-field"
        label="Split 文件"
        value={sourceManifest}
        onChange={setSourceManifest}
        required={!suiteSlices.trim()}
      />
      <TextInputControl label="Split 名称" value={split} onChange={setSplit} required />
      <TextInputControl label="标注层" value={layers} onChange={setLayers} />
      <TextareaControl
        className="wide-field"
        label="Suite slices"
        value={suiteSlices}
        onChange={setSuiteSlices}
        rows={4}
        placeholder="grounding_arrow=data/raw_data/splits/grounding_arrow.txt | detection | arrow | arrow"
      />
      {suiteSliceParse.error ? (
        <div className="form-result error full-field">{suiteSliceParse.error}</div>
      ) : null}
      <CheckboxFieldControl
        label="检测"
        checked={tasks.includes("detection")}
        onChange={() => toggleTask("detection")}
      />
      <CheckboxFieldControl
        label="关键点"
        checked={tasks.includes("keypoint")}
        onChange={() => toggleTask("keypoint")}
      />
      <CheckboxFieldControl label="覆盖已有副本" checked={overwrite} onChange={setOverwrite} />
      <ActionButton
        type="submit"
        variant="primary"
        icon={<AppIcon name="submitCreate" size={16} />}
        disabled={
          mutation.isPending
          || (suiteSliceParse.slices.length === 0 && tasks.length === 0)
          || Boolean(suiteSliceParse.error)
        }
      >
        创建
      </ActionButton>
      {mutation.data ? (
        <div className="form-result full-field">
          已创建 {mutation.data.benchmark_id}，包含 {mutation.data.sample_count.toLocaleString()} 个样本。{" "}
          <Link to="/benchmarks/$benchmarkId" params={{ benchmarkId: mutation.data.benchmark_id }}>
            打开
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

