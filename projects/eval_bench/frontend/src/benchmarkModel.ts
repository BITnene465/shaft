import type { BenchmarkSummary, CreateBenchmarkSlicePayload } from "./api";
import { unique } from "./formatters";

export function parseBenchmarkSlices(
  value: string,
  defaultTasks: string[],
  defaultLayers: string
): { slices: CreateBenchmarkSlicePayload[]; error: string | null } {
  const fallbackLayers = splitCompactList(defaultLayers);
  const lines = value
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith("#"));
  if (value.trim() && lines.length === 0) {
    return { slices: [], error: "Suite slices 至少需要一行 split=manifest" };
  }
  const slices: CreateBenchmarkSlicePayload[] = [];
  const seenSplits = new Set<string>();
  for (const [index, line] of lines.entries()) {
    const [head, tasksText, layersText, labelsText] = line.split("|").map((item) => item.trim());
    const separatorIndex = head.indexOf("=");
    if (separatorIndex <= 0 || separatorIndex >= head.length - 1) {
      return {
        slices: [],
        error: `Suite slices 第 ${index + 1} 行必须使用 split=manifest 格式`
      };
    }
    const split = head.slice(0, separatorIndex).trim();
    const sourceManifest = head.slice(separatorIndex + 1).trim();
    if (seenSplits.has(split)) {
      return { slices: [], error: `Suite slices split 重复: ${split}` };
    }
    seenSplits.add(split);
    const parsedTasks = splitCompactList(tasksText);
    const parsedLayers = splitCompactList(layersText);
    const tasks = parsedTasks.length > 0 ? parsedTasks : defaultTasks;
    const invalidTasks = tasks.filter((task) => task !== "detection" && task !== "keypoint");
    if (invalidTasks.length > 0) {
      return { slices: [], error: `Suite slices 不支持的任务: ${invalidTasks.join(", ")}` };
    }
    slices.push({
      split,
      source_manifest: sourceManifest,
      tasks,
      layers: parsedLayers.length > 0 ? parsedLayers : fallbackLayers,
      target_labels: splitCompactList(labelsText)
    });
  }
  return { slices, error: null };
}

export function benchmarkSplitValues(benchmark: BenchmarkSummary | null | undefined): string[] {
  if (!benchmark) {
    return [];
  }
  return unique([benchmark.split, ...Object.keys(benchmark.split_manifests ?? {})].filter(Boolean));
}

function splitCompactList(value: string | undefined): string[] {
  return String(value ?? "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

