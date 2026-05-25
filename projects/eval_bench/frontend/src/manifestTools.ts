import type { BenchmarkSummary, PromptTemplate } from "./api";

export function formatManifest(value: unknown) {
  return `${JSON.stringify(value, null, 2)}\n`;
}

export function applyBenchmarkDefault(
  manifest: Record<string, unknown>,
  benchmarks: BenchmarkSummary[]
): Record<string, unknown> {
  const cloned = JSON.parse(JSON.stringify(manifest)) as Record<string, unknown>;
  const benchmarkIds = benchmarks.map((benchmark) => benchmark.benchmark_id);
  if (benchmarkIds.length === 0 || !isRecord(cloned.eval)) {
    return cloned;
  }
  const currentBenchmarkId = cloned.eval.benchmark_id;
  if (typeof currentBenchmarkId !== "string" || !benchmarkIds.includes(currentBenchmarkId)) {
    cloned.eval.benchmark_id = defaultBenchmarkForTask(cloned.eval.task, benchmarks).benchmark_id;
  }
  return cloned;
}

export function applyPromptTemplateToManifest(
  manifest: Record<string, unknown>,
  prompt: PromptTemplate
): Record<string, unknown> {
  const cloned = JSON.parse(JSON.stringify(manifest)) as Record<string, unknown>;
  const section = manifestPromptSection(cloned);
  if (!section) {
    cloned.eval = {};
    return applyPromptTemplateToManifest(cloned, prompt);
  }
  section.prompt_id = prompt.prompt_id;
  if (prompt.task) {
    section.task = prompt.task;
  }
  section.system_prompt = prompt.system_prompt;
  section.prompt_text = prompt.user_prompt;
  if (prompt.parser) {
    section.parser = prompt.parser;
  }
  if (prompt.metric_profile) {
    section.metric_profile = prompt.metric_profile;
  }
  if (prompt.visualization_profile) {
    section.visualization_profile = prompt.visualization_profile;
  }
  const targetLabels = targetLabelsFromPrompt(prompt);
  if (targetLabels.length > 0) {
    section.target_labels = targetLabels;
  } else {
    delete section.target_labels;
  }
  section.generation = mergeRecordDefaults(section.generation, prompt.generation);
  section.data = mergeRecordDefaults(section.data, prompt.data);
  section.prompt_template = {
    prompt_id: prompt.prompt_id,
    label: prompt.label,
    task: prompt.task
  };
  return cloned;
}

export function promptTemplateFromManifest(
  manifest: Record<string, unknown>,
  fallback?: PromptTemplate
): Partial<PromptTemplate> {
  const section = manifestPromptSection(manifest) ?? {};
  const promptId = promptStringValue(section.prompt_id) ?? fallback?.prompt_id ?? "custom.prompt";
  return {
    prompt_id: promptId,
    label: promptStringValue(section.label) ?? fallback?.label ?? promptId,
    task: promptStringValue(section.task) ?? fallback?.task ?? "detection",
    system_prompt: promptStringValue(section.system_prompt) ?? fallback?.system_prompt ?? "",
    user_prompt:
      promptStringValue(section.prompt_text) ??
      promptStringValue(section.user_prompt) ??
      fallback?.user_prompt ??
      "",
    parser: promptStringValue(section.parser) ?? fallback?.parser ?? null,
    metric_profile: promptStringValue(section.metric_profile) ?? fallback?.metric_profile ?? null,
    visualization_profile:
      promptStringValue(section.visualization_profile) ?? fallback?.visualization_profile ?? null,
    generation: isRecord(section.generation) ? section.generation : fallback?.generation ?? {},
    data: isRecord(section.data) ? section.data : fallback?.data ?? {},
    metadata: {
      ...(fallback?.metadata ?? {}),
      target_labels: isStringArray(section.target_labels)
        ? section.target_labels
        : targetLabelsFromPrompt(fallback),
      source: "dashboard_manifest"
    }
  };
}

export function targetLabelsFromPrompt(prompt?: Partial<PromptTemplate>) {
  const labels = prompt?.metadata?.target_labels;
  return isStringArray(labels) ? labels : [];
}

export function manifestEvalTask(manifest: Record<string, unknown> | null) {
  const section = manifest ? manifestPromptSection(manifest) : null;
  return promptStringValue(section?.task) ?? "";
}

export function manifestBenchmarkId(manifest: Record<string, unknown> | null) {
  const section = manifest ? manifestPromptSection(manifest) : null;
  return promptStringValue(section?.benchmark_id) ?? "";
}

export function manifestTargetLabels(manifest: Record<string, unknown> | null) {
  const section = manifest ? manifestPromptSection(manifest) : null;
  return isStringArray(section?.target_labels) ? section.target_labels : [];
}

export function updateManifestTargetLabels(
  manifest: Record<string, unknown>,
  labels: string[]
): Record<string, unknown> {
  const cloned = JSON.parse(JSON.stringify(manifest)) as Record<string, unknown>;
  const section = manifestPromptSection(cloned);
  if (!section) {
    cloned.eval = {};
    return updateManifestTargetLabels(cloned, labels);
  }
  const normalizedLabels = uniqueLabels(labels);
  if (normalizedLabels.length > 0) {
    section.target_labels = normalizedLabels;
  } else {
    delete section.target_labels;
  }
  return cloned;
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}

function uniqueLabels(labels: string[]) {
  const seen = new Set<string>();
  const result: string[] = [];
  labels.forEach((label) => {
    const value = label.trim();
    if (!value || seen.has(value)) {
      return;
    }
    seen.add(value);
    result.push(value);
  });
  return result;
}

function manifestPromptSection(manifest: Record<string, unknown>): Record<string, unknown> | null {
  if (isRecord(manifest.eval)) {
    return manifest.eval;
  }
  if (isRecord(manifest.preannotate)) {
    return manifest.preannotate;
  }
  return null;
}

function mergeRecordDefaults(current: unknown, defaults: Record<string, unknown>) {
  return {
    ...defaults,
    ...(isRecord(current) ? current : {})
  };
}

function promptStringValue(value: unknown) {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function defaultBenchmarkForTask(task: unknown, benchmarks: BenchmarkSummary[]): BenchmarkSummary {
  if (typeof task === "string" && task.trim()) {
    const matched = benchmarks.find((benchmark) => benchmark.tasks.includes(task.trim()));
    if (matched) {
      return matched;
    }
  }
  return benchmarks[0];
}
