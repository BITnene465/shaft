import type { RunSummary } from "./api";

export type RunArtifactTone = "complete" | "ready" | "draft";

export type RunArtifactReadiness = {
  label: string;
  percent: number;
  tone: RunArtifactTone;
};

export function recentRunsByCreatedAt(runs: RunSummary[], limit: number) {
  return [...runs]
    .sort((left, right) => {
      const leftTime = Date.parse(left.created_at ?? "");
      const rightTime = Date.parse(right.created_at ?? "");
      return (
        (Number.isFinite(rightTime) ? rightTime : 0) -
        (Number.isFinite(leftTime) ? leftTime : 0)
      );
    })
    .slice(0, limit);
}

export function runArtifactReadiness(run: RunSummary): RunArtifactReadiness {
  const hasPredictions = run.prediction_count > 0;
  const hasReport = Boolean(run.report_path || run.report_count > 0);
  const hasNote = run.note.trim().length > 0;
  const percent = Math.min(
    100,
    (hasPredictions ? 48 : 0) + (hasReport ? 40 : 0) + (hasNote ? 12 : 0)
  );
  if (hasReport) {
    return {
      label: hasNote ? "report + note" : "report ready",
      percent: Math.max(percent, 88),
      tone: "complete"
    };
  }
  if (hasPredictions) {
    return {
      label: hasNote ? "pred + note" : "prediction ready",
      percent: Math.max(percent, 48),
      tone: "ready"
    };
  }
  return {
    label: hasNote ? "note only" : "draft",
    percent: Math.max(percent, 8),
    tone: "draft"
  };
}

export function runAgeLabel(value: string | null, now = Date.now()) {
  if (!value) {
    return "-";
  }
  const createdAt = Date.parse(value);
  if (!Number.isFinite(createdAt)) {
    return "-";
  }
  const diffSeconds = Math.max(0, Math.floor((now - createdAt) / 1000));
  if (diffSeconds < 60) {
    return "now";
  }
  const diffMinutes = Math.floor(diffSeconds / 60);
  if (diffMinutes < 60) {
    return `${diffMinutes}m`;
  }
  const diffHours = Math.floor(diffMinutes / 60);
  if (diffHours < 48) {
    return `${diffHours}h`;
  }
  return `${Math.floor(diffHours / 24)}d`;
}
